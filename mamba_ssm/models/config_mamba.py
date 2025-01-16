from dataclasses import dataclass, field
from transformers.configuration_utils import PretrainedConfig


@dataclass
#class MambaConfig:
class MambaConfig(PretrainedConfig):

    d_model: int = 2560
    d_intermediate: int = 0
    n_layer: int = 64
    vocab_size: int = 50277
    ssm_cfg: dict = field(default_factory=dict)
    attn_layer_idx: list = field(default_factory=list)
    attn_cfg: dict = field(default_factory=dict)
    rms_norm: bool = True
    residual_in_fp32: bool = True
    fused_add_norm: bool = True
    pad_vocab_size_multiple: int = 8
    tie_embeddings: bool = True
    ddense: bool = False
    dense_type: str = ''
    ddense_pre_norm: bool = False
    ddense_post_norm: bool = False
    ddense_tanh: bool = False
    d_model_deviation: float = 0
    cut_residual_lr_only: bool = False