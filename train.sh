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
# EXP_NAME="Llama_pile_smal_rep"; MODEL_NAME="Llama"; MODEL_SIZE="small"; DDENSE="0"; DENSE_TYPE="l"; FUSED_ADD_NORM="0"; DEBUG="0"; REP="1";TIE_EMB='1';  # 

#20241229
# debug
# EXP_NAME="Mamba2_pile_medium_rep_m2l"; MODEL_NAME="Mamba2"; MODEL_SIZE="large"; DDENSE="0"; DENSE_TYPE="l"; FUSED_ADD_NORM="0"; DEBUG="1"; REP="1"; TIE_EMB="1"; MEDIUM2LARGE="1"
# run
# EXP_NAME="Mamba2_pile_large_rep"; MODEL_NAME="Mamba2"; MODEL_SIZE="large"; DDENSE="0"; DENSE_TYPE="l"; FUSED_ADD_NORM="0"; DEBUG="0"; REP="1"; TIE_EMB="1"; MEDIUM2LARGE="1"

# EXP_NAME="Mamba2_pile_medium_rep_m2l"; MODEL_NAME="Mamba2"; MODEL_SIZE="medium"; DDENSE="0"; DENSE_TYPE="l"; FUSED_ADD_NORM="0"; DEBUG="0"; REP="1"; TIE_EMB="1"; MEDIUM2LARGE="1"

#20241230
# debug
# EXP_NAME="Mamba2_pile_small_rep_m2l_muddzs_sdlr10"; MODEL_NAME="Mamba2"; MODEL_SIZE="small"; DDENSE="1"; DENSE_TYPE="zs"; FUSED_ADD_NORM="0"; DEBUG="0"; REP="1"; TIE_EMB="1"; MEDIUM2LARGE="1" SCALE_DOWN_MUDD_LR="10"
# EXP_NAME="Mamba2_pile_small_rep_m2l_muddzs_sdlr20"; MODEL_NAME="Mamba2"; MODEL_SIZE="small"; DDENSE="1"; DENSE_TYPE="zs"; FUSED_ADD_NORM="0"; DEBUG="0"; REP="1"; TIE_EMB="1"; MEDIUM2LARGE="1" SCALE_DOWN_MUDD_LR="20"

# run
# EXP_NAME="Mamba2_pile_large_rep"; MODEL_NAME="Mamba2"; MODEL_SIZE="large"; DDENSE="0"; DENSE_TYPE="l"; FUSED_ADD_NORM="0"; DEBUG="0"; REP="1"; TIE_EMB="1"; MEDIUM2LARGE="1"; SCALE_DOWN_MUDD_LR="1"
# EXP_NAME="Mamba2_pile_medium_rep_m2l_muddzs_sdlr10"; MODEL_NAME="Mamba2"; MODEL_SIZE="medium"; DDENSE="1"; DENSE_TYPE="zs"; FUSED_ADD_NORM="0"; DEBUG="0"; REP="1"; TIE_EMB="1"; MEDIUM2LARGE="1" SCALE_DOWN_MUDD_LR="10"

# 20250101
# debug
# EXP_NAME="Mamba2_pile_small_rep_m2l_muddzs_sdlr10"; MODEL_NAME="Mamba2"; MODEL_SIZE="small"; DDENSE="1"; DENSE_TYPE="zs"; FUSED_ADD_NORM="0"; DEBUG="1"; REP="1"; TIE_EMB="1"; MEDIUM2LARGE="1" SCALE_DOWN_MUDD_LR="10"
# EXP_NAME="Mamba2_pile_medium_rep_m2l_muddrzxbct_sdlr10"; MODEL_NAME="Mamba2"; MODEL_SIZE="medium"; DDENSE="1"; DENSE_TYPE="rzxbct"; FUSED_ADD_NORM="0"; DEBUG="1"; REP="1"; TIE_EMB="1"; MEDIUM2LARGE="1" SCALE_DOWN_MUDD_LR="10"
# run 
# EXP_NAME="Mamba2_pile_medium_rep_m2l_muddzxbct_sdlr10"; MODEL_NAME="Mamba2"; MODEL_SIZE="medium"; DDENSE="1"; DENSE_TYPE="zxbct"; FUSED_ADD_NORM="0"; DEBUG="0"; REP="1"; TIE_EMB="1"; MEDIUM2LARGE="1" SCALE_DOWN_MUDD_LR="10"
# EXP_NAME="Mamba2_pile_medium_rep_m2l_muddrzxbct_sdlr10"; MODEL_NAME="Mamba2"; MODEL_SIZE="medium"; DDENSE="1"; DENSE_TYPE="rzxbct"; FUSED_ADD_NORM="0"; DEBUG="0"; REP="1"; TIE_EMB="1"; MEDIUM2LARGE="1" SCALE_DOWN_MUDD_LR="10"

# 20250104

# debug 
# EXP_NAME="Mamba2_pile_small_rep_m2l_muddzs_sdlr10"; MODEL_NAME="Mamba2"; MODEL_SIZE="small"; DDENSE="1"; DENSE_TYPE="zs"; FUSED_ADD_NORM="0"; DEBUG="1"; REP="1"; TIE_EMB="1"; MEDIUM2LARGE="1" SCALE_DOWN_MUDD_LR="10"

# 20250105
# debug
# EXP_NAME="Mamba2_pile_small_rep_muddzs_sdlr10"; MODEL_NAME="Mamba2"; MODEL_SIZE="small"; DDENSE="1"; DENSE_TYPE="zs"; FUSED_ADD_NORM="0"; DEBUG="1"; REP="1"; TIE_EMB="1"; MEDIUM2LARGE="0" SCALE_DOWN_MUDD_LR="10"
# run
# EXP_NAME="Mamba2_pile_small_rep_muddzs_sdlr10"; MODEL_NAME="Mamba2"; MODEL_SIZE="small"; DDENSE="1"; DENSE_TYPE="zs"; FUSED_ADD_NORM="0"; DEBUG="0"; REP="1"; TIE_EMB="1"; MEDIUM2LARGE="0" SCALE_DOWN_MUDD_LR="10"
# EXP_NAME="Mamba2_pile_small_rep_muddzxbct_sdlr10"; MODEL_NAME="Mamba2"; MODEL_SIZE="small"; DDENSE="1"; DENSE_TYPE="zxbct"; FUSED_ADD_NORM="0"; DEBUG="0"; REP="1"; TIE_EMB="1"; MEDIUM2LARGE="0" SCALE_DOWN_MUDD_LR="10"
# EXP_NAME="Mamba2_pile_small_rep_muddrzxbct_sdlr10"; MODEL_NAME="Mamba2"; MODEL_SIZE="small"; DDENSE="1"; DENSE_TYPE="rzxbct"; FUSED_ADD_NORM="0"; DEBUG="0"; REP="1"; TIE_EMB="1"; MEDIUM2LARGE="0" SCALE_DOWN_MUDD_LR="10"
# EXP_NAME="Mamba2_pile_small_rep_muddzs_sdlr5"; MODEL_NAME="Mamba2"; MODEL_SIZE="small"; DDENSE="1"; DENSE_TYPE="zs"; FUSED_ADD_NORM="0"; DEBUG="0"; REP="1"; TIE_EMB="1"; MEDIUM2LARGE="0" SCALE_DOWN_MUDD_LR="5"
# EXP_NAME="Mamba2_pile_small_rep_muddrzxbct_sdlr1_hidnorm1"; MODEL_NAME="Mamba2"; MODEL_SIZE="small"; DDENSE="1"; DENSE_TYPE="rzxbct"; FUSED_ADD_NORM="0"; DEBUG="0"; REP="1"; TIE_EMB="1"; MEDIUM2LARGE="0" SCALE_DOWN_MUDD_LR="1"; DDENSE_HID_NORM="1"; DDENSE_TANH="0"
# EXP_NAME="Mamba2_pile_small_rep_muddrzxbct_sdlr1_tanh1"; MODEL_NAME="Mamba2"; MODEL_SIZE="small"; DDENSE="1"; DENSE_TYPE="rzxbct"; FUSED_ADD_NORM="0"; DEBUG="0"; REP="1"; TIE_EMB="1"; MEDIUM2LARGE="0" SCALE_DOWN_MUDD_LR="1"; DDENSE_HID_NORM="0"; DDENSE_TANH="1"

# 20250106
# EXP_NAME="Mamba2_pile_small_rep_muddrzxbct_sdlr10_dev0p5"; MODEL_NAME="Mamba2"; MODEL_SIZE="small"; DDENSE="1"; DENSE_TYPE="rzxbct"; FUSED_ADD_NORM="0"; DEBUG="0"; REP="1"; TIE_EMB="1"; MEDIUM2LARGE="0" SCALE_DOWN_MUDD_LR="10"; DDENSE_HID_NORM="0"; DDENSE_TANH="0"; D_MODEL_DEVIATION="0.5"
# EXP_NAME="Mamba2_pile_small_rep_muddrzxbct_sdlr10_dev0p75"; MODEL_NAME="Mamba2"; MODEL_SIZE="small"; DDENSE="1"; DENSE_TYPE="rzxbct"; FUSED_ADD_NORM="0"; DEBUG="0"; REP="1"; TIE_EMB="1"; MEDIUM2LARGE="0" SCALE_DOWN_MUDD_LR="10"; DDENSE_HID_NORM="0"; DDENSE_TANH="0"; D_MODEL_DEVIATION="0.75"
# EXP_NAME="Mamba2_pile_small_rep_muddrzxbct_sdlr10_dev0p25"; MODEL_NAME="Mamba2"; MODEL_SIZE="small"; DDENSE="1"; DENSE_TYPE="rzxbct"; FUSED_ADD_NORM="0"; DEBUG="0"; REP="1"; TIE_EMB="1"; MEDIUM2LARGE="0" SCALE_DOWN_MUDD_LR="10"; DDENSE_HID_NORM="0"; DDENSE_TANH="0"; D_MODEL_DEVIATION="0.25"

# 20250107
#run
# EXP_NAME="Mamba2_pile_small_rep_muddrzxbct_sdlr10_expand4"; MODEL_NAME="Mamba2"; MODEL_SIZE="small"; DDENSE="1"; DENSE_TYPE="rzxbct"; FUSED_ADD_NORM="0"; DEBUG="0"; REP="1"; TIE_EMB="1"; MEDIUM2LARGE="0" SCALE_DOWN_MUDD_LR="10"; DDENSE_HID_NORM="0"; DDENSE_TANH="0"; D_MODEL_DEVIATION="0"; EXPAND="4"

# EXP_NAME="Mamba2_pile_small_rep_muddrzxbct_sdlr10_dev0p5_rerun"; MODEL_NAME="Mamba2"; MODEL_SIZE="small"; DDENSE="1"; DENSE_TYPE="rzxbct"; FUSED_ADD_NORM="0"; DEBUG="0"; REP="1"; TIE_EMB="1"; MEDIUM2LARGE="0" SCALE_DOWN_MUDD_LR="10"; DDENSE_HID_NORM="0"; DDENSE_TANH="0"; D_MODEL_DEVIATION="0.5"; EXPAND="2";  CUT_RESIDUAL_LR_ONLY="0"

# EXP_NAME="Mamba2_pile_small_rep_muddrzxbct_sdlr10_res_only"; MODEL_NAME="Mamba2"; MODEL_SIZE="small"; DDENSE="1"; DENSE_TYPE="rzxbct"; FUSED_ADD_NORM="0"; DEBUG="0"; REP="1"; TIE_EMB="1"; MEDIUM2LARGE="0" SCALE_DOWN_MUDD_LR="10"; DDENSE_HID_NORM="0"; DDENSE_TANH="0"; D_MODEL_DEVIATION="0"; EXPAND="2"; CUT_RESIDUAL_LR_ONLY="1"

# EXP_NAME="Mamba2_pile_small_rep_muddrzxxbct_sdlr10"; MODEL_NAME="Mamba2"; MODEL_SIZE="small"; DDENSE="1"; DENSE_TYPE="rzxxbct"; FUSED_ADD_NORM="0"; DEBUG="0"; REP="1"; TIE_EMB="1"; MEDIUM2LARGE="0" SCALE_DOWN_MUDD_LR="10"; DDENSE_HID_NORM="0"; DDENSE_TANH="0"; D_MODEL_DEVIATION="0"; EXPAND="2";  CUT_RESIDUAL_LR_ONLY="0"

# EXP_NAME="Mamba2_pile_small_rep_muddrzxbct_sdlr20"; MODEL_NAME="Mamba2"; MODEL_SIZE="small"; DDENSE="1"; DENSE_TYPE="rzxbct"; FUSED_ADD_NORM="0"; DEBUG="0"; REP="1"; TIE_EMB="1"; MEDIUM2LARGE="0" SCALE_DOWN_MUDD_LR="20"; DDENSE_HID_NORM="0"; DDENSE_TANH="0"; D_MODEL_DEVIATION="0"; EXPAND="2"; CUT_RESIDUAL_LR_ONLY="0"

# EXP_NAME="Mamba2_pile_small_rep_muddrzxs_sdlr10"; MODEL_NAME="Mamba2"; MODEL_SIZE="small"; DDENSE="1"; DENSE_TYPE="rzxs"; FUSED_ADD_NORM="0"; DEBUG="0"; REP="1"; TIE_EMB="1"; MEDIUM2LARGE="0" SCALE_DOWN_MUDD_LR="10"; DDENSE_HID_NORM="0"; DDENSE_TANH="0"; D_MODEL_DEVIATION="0"; EXPAND="2"; CUT_RESIDUAL_LR_ONLY="0"

# 20250108
EXP_NAME="Mamba2_pile_small_rep_muddrzxbct_pre_norm"; MODEL_NAME="Mamba2"; MODEL_SIZE="small"; DDENSE="1"; DENSE_TYPE="rzxbct"; FUSED_ADD_NORM="0"; DEBUG="0"; REP="1"; TIE_EMB="1"; MEDIUM2LARGE="0" SCALE_DOWN_MUDD_LR="1"; DDENSE_HID_NORM="1"; DDENSE_TANH="0"; D_MODEL_DEVIATION="0"; EXPAND="2"; CUT_RESIDUAL_LR_ONLY="0"


DATA_PATH="/home/mengqy/data/pile_train_dataset/"

# DEVICE="0";NUM_GPUS="1"; PORT="12372"
# DEVICE="0,1,4,5";NUM_GPUS="4"; PORT="32373"


DEVICE="0,1";NUM_GPUS="2"; PORT="22373"
# DEVICE="2,3";NUM_GPUS="2"; PORT="32378"
# DEVICE="4,5";NUM_GPUS="2"; PORT="22375"

# DEVICE="6,7";NUM_GPUS="2"; PORT="12376"


# DEVICE="0,1,2,3";NUM_GPUS="4"; PORT="12371"
# DEVICE="4,5,6,7";NUM_GPUS="4"; PORT="12379"
# DEVICE="0,1,2,3,4,5,6,7";NUM_GPUS="8"
RESUME="0"

# CMD="CUDA_VISIBLE_DEVICES=${DEVICE} python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} train_mamba2.py --run_name ${EXP_NAME} --model_name ${MODEL_NAME} --model_size ${MODEL_SIZE} --debug ${DEBUG} --ddense ${DDENSE} --dense_type ${DENSE_TYPE} --fused_add_norm ${FUSED_ADD_NORM} --reproduce ${REP}"
# echo $CMD

if [[ "${DEBUG}" == "0" ]]; then
    # MASTER_PORT=12370 MASTER_ADDR="127.0.0.1"
    NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=${DEVICE} python -m torch.distributed.launch --rdzv-backend=c10d --nnodes=1 --rdzv-endpoint=localhost:${PORT} --nproc_per_node=${NUM_GPUS} train_mamba2.py --run_name ${EXP_NAME} --model_name ${MODEL_NAME} --model_size ${MODEL_SIZE} --debug ${DEBUG} --ddense ${DDENSE} --dense_type ${DENSE_TYPE} --fused_add_norm ${FUSED_ADD_NORM} --reproduce ${REP} --tie_emb ${TIE_EMB} --resume ${RESUME} --data_path ${DATA_PATH} --medium2large ${MEDIUM2LARGE} --scale_down_mudd_lr ${SCALE_DOWN_MUDD_LR} --ddense_hid_norm ${DDENSE_HID_NORM} --ddense_tanh ${DDENSE_TANH} --d_model_deviation ${D_MODEL_DEVIATION} --expand ${EXPAND} --cut_residual_lr_only ${CUT_RESIDUAL_LR_ONLY} >> logs/${EXP_NAME}.log 2>&1 &
else
    EXP_NAME="${EXP_NAME}_debug"
    NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=${DEVICE} python -m torch.distributed.launch --rdzv-backend=c10d --nnodes=1 --rdzv-endpoint=localhost:${PORT} --nproc_per_node=${NUM_GPUS} train_mamba2.py --run_name ${EXP_NAME} --model_name ${MODEL_NAME} --model_size ${MODEL_SIZE} --debug ${DEBUG} --ddense ${DDENSE} --dense_type ${DENSE_TYPE} --fused_add_norm ${FUSED_ADD_NORM} --reproduce ${REP} --tie_emb ${TIE_EMB} --resume ${RESUME} --data_path ${DATA_PATH} --medium2large ${MEDIUM2LARGE} --scale_down_mudd_lr ${SCALE_DOWN_MUDD_LR}  --ddense_hid_norm ${DDENSE_HID_NORM} --ddense_tanh ${DDENSE_TANH} --d_model_deviation ${D_MODEL_DEVIATION} --expand ${EXPAND} --cut_residual_lr_only ${CUT_RESIDUAL_LR_ONLY}
fi
