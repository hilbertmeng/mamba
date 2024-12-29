# Code modified from: https://github.com/havenhq/mamba-chat/blob/main/train_mamba.py
import numpy as np
import argparse
import json
import os
import time
import math
import torch
# torch._dynamo.config.inline_inbuilt_nn_modules=True
import torch.distributed as dist

import warnings
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
from typing import Tuple
from functools import partial
from dataclasses import dataclass, field
from tqdm import tqdm

from torch.utils.data import Dataset, IterableDataset 
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import AdamW 

from transformers import AutoTokenizer, TrainingArguments
from transformers import Trainer
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.models.llama.configuration_llama import LlamaConfig

from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn

# from mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined

# torch.compiler.allow_in_graph([rms_norm_fn, mamba_split_conv1d_scan_combined])

import mamba_ssm
print(mamba_ssm.__file__)

def get_cosinedecay_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, decay_start=None, decay_end=13500, min_ratio=0.1, cos_max=1.0, last_epoch=-1):
    """
    LinearRampupCosineDecay
    """
    if decay_start is None: decay_start = num_warmup_steps + 1

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        _max, _min = cos_max, cos_max * min_ratio 
        decay_gap = _max - _min
        return _min + 0.5 * decay_gap * (1+math.cos(progress * math.pi))

    return LambdaLR(optimizer, lr_lambda, last_epoch)

class MambaCustomConfig(MambaConfig):
    """ custom config to make the model run with HF Trainer """
    def to_json_string(self,):
        return json.dumps(
        {
        "d_model" : int(self.d_model),
        "n_layer" : int(self.n_layer),
        "vocab_size" : int(self.vocab_size),
        "ssm_config" : self.ssm_cfg,
        "rms_norm" : self.rms_norm,
        "residual_in_fp32" : self.residual_in_fp32,
        "fused_add_norm" : self.fused_add_norm,
        "pad_vocab_size_multiple" : self.pad_vocab_size_multiple
        })  

def parse_function(example_proto, seq_len=2049,task_features=["input_ids"]):
    feature_desc = {key: tf.io.VarLenFeature(tf.int64) for key in task_features}
    example = tf.io.parse_single_example(example_proto, feature_desc)
    for name in list(example.keys()):
        t = example[name]
        if t.dtype == tf.int64:
            t = tf.cast(t, dtype=tf.int32)
        example[name] = tf.sparse.to_dense(t, default_value=0)[ :seq_len]
    return example

def convert(data, seq_len=2049):
    model_needed_inputs = {}
    model_needed_inputs['input_ids'] = data["input_ids"][:, : seq_len - 1]
    model_needed_inputs['labels'] = data["input_ids"][:, 1:seq_len]
    key = 'labels' if "labels" in data else 'input_ids'
    # weights = data[key] >= 0 if self.zero_loss else data[key] > 0
    # model_needed_inputs.weights = weights[:, 1:seq_len]
    # model_needed_inputs.paddings = tf.zeros_like(model_needed_inputs.ids)
    # model_needed_inputs.segment_ids = tf.ones_like(model_needed_inputs.ids)
    # pos = tf.range(seq_len - 1)
    # model_needed_inputs.segment_pos = model_needed_inputs.segment_ids * pos
    return model_needed_inputs

def load_pile_generator(data_path=None, num_files=2):
    seq_len = 2049
    pad_id = 0
    batch_size = 1
    task_features = ["input_ids"]
    padded_shapes = {key: seq_len for key in task_features}
    padding_values = {key: pad_id for key in task_features}

    if data_path is None: data_path = '/home/mengqy/Projects/lm-dataset/pile_train_dataset/' 
    print('data_path', data_path)
    fnames = [ data_path + p for p in os.listdir(data_path)]
    fnames.sort(key=lambda x:int(x.split('tfrecord.b')[-1]))
    ds = tf.data.Dataset.from_tensor_slices(fnames[:num_files])
    ds = ds.apply(tf.data.TFRecordDataset)
    ds = ds.map(partial(parse_function, seq_len=seq_len, task_features=task_features), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.padded_batch(batch_size=np.prod(batch_size),padded_shapes=padded_shapes,padding_values=padding_values,drop_remainder=True,)
    ds = ds.map(convert)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    ds = ds.as_numpy_iterator()
    return ds

class PileDataset(Dataset):
    def __init__(self, data_path, seq_len=2048, size='medium', debug=False):
        print('load dataset')
        if size == 'small':
            nfiles = 4
        elif size == 'medium':
            nfiles = 7
        self.generator = load_pile_generator(num_files=nfiles) 
        t0 = time.time()
        if debug:
           self.data = [self.generator.next() for _ in range(256*200)]
        else:
           self.data = list(self.generator)
        print('load dataset done', len(self.data), time.time()-t0)
        print('sample data:', self[0]['input_ids'].shape, self[0]['labels'].shape, self.data[0])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return dict(input_ids=self.data[i]['input_ids'][0], labels=self.data[i]['labels'][0])

@dataclass
class MambaTrainingArguments(TrainingArguments):
    num_training_steps: int = field(default=13500)
    decay_end: int = field(default=13500)
    min_ratio: float = field(default=0.1)

    # adam
    adam_beta1: float = field(default=0.9) 
    adam_beta2: float = field(default=0.95)
    adam_weight_decay: float = field(default=0.1)
    #ADAM_CLIP_THRESHOLD = 1.0 #TODO
    #ADAM_EPSILON_ROOT = 0.0 #TODO
    #def __init__(self, **kwargs):
    #    names = set([f.name for f in dataclasses.fields(self)])
    #    for k, v in kwargs.items():
    #        if k in names:
    #            setattr(self, k, v)


class MyTrainer(Trainer):

    def create_optimizer_and_scheduler(self, num_training_steps: int): 
        """
        Setup the optimizer and the learning rate scheduler.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through :obj:`optimizers`, or override this method in a subclass.
        """
        #if self.optimizer is not None:
        #    return self.optimizer
        # Prepare optimizer and schedule 

        warnings.warn(f"These default Huggingface TrainingArguments (warmup_ratio=0, optim=OptimizerNames.ADAMW_TORCH, optim_args=None, lr_scheduler_type=SchedulerType.LINEAR) are disabled due to overriding optimzer and lr_scheduler in {__file__}")

        no_decay = ["bias", "LayerNorm.weight", 'norm.weight', 'norm_f.weight'] # _no_weight_decay
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not (any(nd in n for nd in no_decay) or getattr(p, '_no_weight_decay', False)) ],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay) or getattr(p, '_no_weight_decay', False) ],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon, betas=(self.args.adam_beta1, self.args.adam_beta2), weight_decay=self.args.adam_weight_decay)

        if self.lr_scheduler is None:
            self.lr_scheduler = get_cosinedecay_schedule_with_warmup(self.optimizer, self.args.warmup_steps, self.args.num_training_steps, decay_end=self.args.decay_end, min_ratio=self.args.min_ratio)
            self._created_lr_scheduler = True
        return self.optimizer, self.lr_scheduler

    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids = inputs.pop("input_ids")
        lm_logits = model(input_ids).logits

        labels = input_ids.to(lm_logits.device)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        labels = labels[:, 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss()
        lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1))

        return lm_loss

    # def save_model(self, output_dir, _internal_call=None):
    #     if not os.path.exists(output_dir):
    #         os.makedirs(output_dir)
            
    #     torch.save(self.model.state_dict(), f"{output_dir}/pytorch_model.bin")
    #     self.tokenizer.save_pretrained(output_dir)
        
        # https://huggingface.co/state-spaces/mamba-130m/blob/main/config.json
        #json_str = """
        #            {
        #                "d_model": 768,
        #                "n_layer": 24,
        #                "vocab_size": 50277,
        #                "ssm_cfg": {},
        #                "rms_norm": true,
        #                "residual_in_fp32": true,
        #                "fused_add_norm": true,
        #                "pad_vocab_size_multiple": 8
        #            }"""
        #with open(f"{output_dir}/config.json", 'w') as f:
        #    f.write(json_str)

def compile_model(
    model,
) -> None:
    backend = os.environ.get("TORCH_COMPILE_BACKEND", "inductor")
    for m in reversed(list(model.modules())):
        print(type(m), m)
        if isinstance(m, RMSNorm): continue
        m.compile(backend=backend)
    # torch.compile(model.tok_embeddings, fullgraph=True, dynamic = True)
    # torch.compile(model.norm, fullgraph=True, dynamic = True)
    # torch.compile(model.output, fullgraph=True, dynamic = True)
    return model


def run(args):
    assert args.model_size in ['small', 'medium']
    assert args.model_name in ['Mamba1', 'Mamba2', 'Llama']
    model_size = args.model_size
    model_name = args.model_name
    reproduce = bool(args.reproduce)

    if model_size == 'small':
        lr = 6e-4
        hparams = dict(warmup_steps=48,num_training_steps=4800,max_steps=4800,decay_end=4800)
    elif model_size == 'medium':
        lr = 3e-4
        hparams = dict(warmup_steps=135,num_training_steps=13500,max_steps=13500,decay_end=13500)
    elif model_size == 'large':
        lr = 2.5e-4
        hparams = dict(warmup_steps=290,num_training_steps=29000,max_steps=29000,decay_end=29000)

    weight_decay = 0.1
    resume_from_checkpoint = bool(args.resume)
    if reproduce:
        lr = lr * 5
        min_lr = 1e-5 
        min_ratio = min_lr / lr
    else:
        min_ratio = 0.1 

    if model_name in ['Mamba1', 'Mamba2']: 
        gradient_checkpointing = False
        model_config ={
            "d_intermediate": 0,
            "vocab_size": 50432,
            "ssm_cfg": {
                "layer": "Mamba2", # "Mamba1"
                "use_mem_eff_path": True,
            },
            "attn_layer_idx": [],
            "attn_cfg": {},
            "rms_norm": True,
            "residual_in_fp32": True,
            "fused_add_norm": True,
            "pad_vocab_size_multiple": 16,
            "tie_embeddings": False 
            }

        if model_size == 'small':
            model_config.update({
                "d_model": 768,
                "n_layer": 24,
            })
        elif model_size == 'medium':
            model_config.update({
                "d_model": 1024,
                "n_layer": 48,
            })
        elif model_size == 'large':
            model_config.update({
                "d_model": 1536,
                "n_layer": 48,
            })
        use_minimal = False 
        if use_minimal:
           model_config["ssm_cfg"]["use_mem_eff_path"] = False
           model_config["ssm_cfg"]["use_minimal"] = True 

        model_config["ssm_cfg"]["layer"] = model_name
        model_config['ddense'] = bool(args.ddense)
        model_config['dense_type'] = args.dense_type
        model_config['fused_add_norm'] = bool(args.fused_add_norm)
        model_config['tie_embeddings'] = bool(args.tie_emb) 

        model_config = MambaCustomConfig(**model_config)
        model = MambaLMHeadModel(model_config, dtype=torch.float32, device="cuda")
    elif model_name == 'Llama': 
        gradient_checkpointing = False
        model_config = {
                'vocab_size': 50432, 
                'use_cache': False}
        if model_size == 'small':
            model_config.update({
                'hidden_size': 768, 
                'intermediate_size': 2048, 
                'num_hidden_layers': 12, 
                'num_attention_heads': 12, 
            })
        elif model_size == 'medium':
            model_config.update({
                'hidden_size': 1024, 
                'intermediate_size': 2816, 
                'num_hidden_layers': 24, 
                'num_attention_heads': 16, 
            })

        model_config['tie_embeddings'] = bool(args.tie_emb) 

        model_config = LlamaConfig(**model_config)
        model_config._flash_attn_2_enabled = True
        model = LlamaForCausalLM(model_config)
        model.training = True
        # _ = model.to('cuda')
        # _ = model.to(torch.bfloat16)

    # model = compile_model(model)

    num_devices = torch.cuda.device_count()
    gradient_accumulation_steps = args.gradient_accumulation_steps // num_devices

    for n, p in model.named_parameters():
        print(n, p.shape, p.std().item(), p.mean().item(), getattr(p, '_no_weight_decay', False))

    training_args = MambaTrainingArguments(
            learning_rate=lr,
            num_train_epochs=1,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            output_dir=args.output+args.run_name,
            save_total_limit=2,
            logging_steps=1,
            save_steps=100,
            run_name=args.run_name,
            resume_from_checkpoint=resume_from_checkpoint,
            min_ratio=min_ratio,
            bf16=True,
            adam_epsilon=1e-8, 
            adam_beta1=0.9,
            adam_beta2=0.95,
            max_grad_norm=1,
            adam_weight_decay=weight_decay,
            dataloader_num_workers=4,
            weight_decay=weight_decay,
            torch_compile=True,
            save_safetensors=False,
            gradient_checkpointing=gradient_checkpointing,
            **hparams,
        )

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.eos_token = "<|endoftext|>"
    tokenizer.pad_token = tokenizer.eos_token

    dataset = PileDataset(args.data_path, size=model_size, debug=args.debug)

    print(f'model params: {sum(p.numel() for p in model.parameters())/1024/1024} M')
    print('#'*50)
    print('run args:', args)
    print('#'*50)
    print('model config:', model_config)
    print('#'*50)
    print('training args:', training_args)
    print('#'*50)

    trainer = MyTrainer(
        model=model,
        train_dataset=dataset,
        tokenizer=tokenizer,
        args=training_args,
    )
    # print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(f'{args.output}{args.run_name}/checkpoint-{trainer.state.global_step}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Mamba2")
    parser.add_argument("--model_size", type=str, default='small') # ['small', 'medium', 'large']
    parser.add_argument("--output", type=str, default="output_models/")
    parser.add_argument("--run_name", type=str, default="mamba2_debug")
    parser.add_argument("--tokenizer", type=str, default="EleutherAI/gpt-neox-20b")
    # parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--debug", type=int, default=0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--optim", type=str, default="adamw_torch")
    parser.add_argument("--data_path", type=str, default="/home/mengqy/Projects/lm-dataset/val_with_eos.npy")
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--ddense", type=int, default=0)
    parser.add_argument("--dense_type", type=str, default='l')
    parser.add_argument("--fused_add_norm", type=int, default='0')
    parser.add_argument("--reproduce", type=int, default='1')
    parser.add_argument("--tie_emb", type=int, default='1')
    parser.add_argument("--resume", type=int, default='0')
    parser.add_argument('--local-rank', type=int, help='Local rank passed from torch.distributed.launch')

    args = parser.parse_args()

    if bool(args.debug):
        os.environ["WANDB_DISABLED"] = "true"

    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(args.local_rank)

    run(args)

