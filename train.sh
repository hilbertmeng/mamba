#! /bin/bash

#20241219
#EXP_NAME="mamba2_mudd_pile_small_debug"; DDENSE="1"; DENSE_TYPE="l";DEBUG="1"
#EXP_NAME="mamba2_pile_small_dynamicdense"; DDENSE="1"; DENSE_TYPE="l"; DEBUG="0"
# EXP_NAME="mamba2_pile_small_base"; MODEL_NAME="mamba"; DDENSE="0"; DENSE_TYPE="l"; DEBUG="1"; FUSED_ADD_NORM="0"

#20241220
# EXP_NAME="mamba2_pile_small_mudd_zs"; DDENSE="1"; DENSE_TYPE="zs"; DEBUG="0"; FUSED_ADD_NORM="0"
#EXP_NAME="mamba2_pile_small_mudd_zxbct"; DDENSE="1"; DENSE_TYPE="zxbct"; DEBUG="0"; FUSED_ADD_NORM="0"
# EXP_NAME="llama_pile_small_base"; MODEL_NAME="llama"; DDENSE="0"; DENSE_TYPE="l"; DEBUG="1"; FUSED_ADD_NORM="0"

#20241227
# debug
# EXP_NAME="Mamba2_pile_medium"; MODEL_NAME="Mamba2"; MODEL_SIZE="medium"; DDENSE="0"; DENSE_TYPE="l"; FUSED_ADD_NORM="0"; DEBUG="1"; REP="1" # 28h
# EXP_NAME="Llama_pile_medium"; MODEL_NAME="Llama"; MODEL_SIZE="medium"; DDENSE="0"; DENSE_TYPE="l"; FUSED_ADD_NORM="0"; DEBUG="1"; # 34h 
# run
# EXP_NAME="Mamba2_pile_small_rep"; MODEL_NAME="Mamba2"; MODEL_SIZE="small"; DDENSE="0"; DENSE_TYPE="l"; FUSED_ADD_NORM="0"; DEBUG="0"; REP="1"; TIE_EMB="1" # 4h
# EXP_NAME="Mamba2_pile_small_rep_gpu1"; MODEL_NAME="Mamba2"; MODEL_SIZE="small"; DDENSE="0"; DENSE_TYPE="l"; FUSED_ADD_NORM="0"; DEBUG="0"; REP="1"; TIE_EMB="1" # 4h

#20241228
# debug
# EXP_NAME="Llama_pile_smal_rep"; MODEL_NAME="Llama"; MODEL_SIZE="small"; DDENSE="0"; DENSE_TYPE="l"; FUSED_ADD_NORM="0"; DEBUG="1"; REP="1";TIE_EMB='1';  # 
# EXP_NAME="Mamba2_pile_small_rep"; MODEL_NAME="Mamba2"; MODEL_SIZE="small"; DDENSE="1"; DENSE_TYPE="zs"; FUSED_ADD_NORM="0"; DEBUG="1"; REP="1"; TIE_EMB="1" # 4h
# run
EXP_NAME="Llama_pile_smal_rep"; MODEL_NAME="Llama"; MODEL_SIZE="small"; DDENSE="0"; DENSE_TYPE="l"; FUSED_ADD_NORM="0"; DEBUG="0"; REP="1";TIE_EMB='1';  # 


# DEVICE="0";NUM_GPUS="1"
DEVICE="0,1";NUM_GPUS="2"
RESUME="0"

# CMD="CUDA_VISIBLE_DEVICES=${DEVICE} python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} train_mamba2.py --run_name ${EXP_NAME} --model_name ${MODEL_NAME} --model_size ${MODEL_SIZE} --debug ${DEBUG} --ddense ${DDENSE} --dense_type ${DENSE_TYPE} --fused_add_norm ${FUSED_ADD_NORM} --reproduce ${REP}"
# echo $CMD

if [[ "${DEBUG}" == "0" ]]; then
    CUDA_VISIBLE_DEVICES=${DEVICE} python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} train_mamba2.py --run_name ${EXP_NAME} --model_name ${MODEL_NAME} --model_size ${MODEL_SIZE} --debug ${DEBUG} --ddense ${DDENSE} --dense_type ${DENSE_TYPE} --fused_add_norm ${FUSED_ADD_NORM} --reproduce ${REP} --tie_emb ${TIE_EMB} >> logs/${EXP_NAME}.log 2>&1 &
else
    EXP_NAME="${EXP_NAME}_debug"
    CUDA_VISIBLE_DEVICES=${DEVICE} python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} train_mamba2.py --run_name ${EXP_NAME} --model_name ${MODEL_NAME} --model_size ${MODEL_SIZE} --debug ${DEBUG} --ddense ${DDENSE} --dense_type ${DENSE_TYPE} --fused_add_norm ${FUSED_ADD_NORM} --reproduce ${REP} --tie_emb ${TIE_EMB} --resume ${RESUME}
fi
