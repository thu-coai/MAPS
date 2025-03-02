#!/bin/bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
echo $PATH
cd ft_cog
which python

NUM_GPUS_PER_WORKER=8
MP_SIZE=1

script_path=$(realpath $0)
script_dir=$(dirname $script_path)
main_dir=$(dirname $script_dir)
# MODEL_TYPE="../checkpoints/sat/cogagent-vqa"
# MODEL_TYPE="/workspace/share/zhuerle/checkpoints/cogagent-vqa"
MODEL_TYPE="../checkpoints/sat/THUDM/CogAgent/cogagent-vqa"
MODEL_NAME="cogagent-vqa"
VERSION="vqa"
# TOKENIZER_TYPE="../checkpoints/lmsys/vicuna-7b-v1.5"
TOKENIZER_TYPE="../checkpoints/huggingface/lmsys/vicuna-7b-v1.5"
EXP_NOTE="test"
SAVE_DIR="ft_cog/checkpoints/${EXP_NOTE}"
MAX_LEN=2000

MODEL_ARGS="--from_pretrained $MODEL_TYPE \
    --max_length $MAX_LEN \
    --lora_rank 50 \
    --use_lora \
    --local_tokenizer $TOKENIZER_TYPE \
    --version $VERSION"
# TIPS: max_length include low-resolution image sequence (which has 256 tokens) 

OPTIONS_SAT="SAT_HOME=~/.sat_models"
# OPTIONS_NCCL="NCCL_DEBUG=info NCCL_IB_DISABLE=0 NCCL_NET_GDR_LEVEL=2 LOCAL_WORLD_SIZE=$NUM_GPUS_PER_WORKER"
OPTIONS_NCCL="NCCL_IB_DISABLE=0 NCCL_NET_GDR_LEVEL=2 LOCAL_WORLD_SIZE=$NUM_GPUS_PER_WORKER" # exclude NCCL_DEBUG=info
# HOST_FILE_PATH="hostfile"

train_data="datasets/grid_v11_240831_split/train"
valid_data="datasets/grid_v11_240831_split/valid"

gpt_options=" \
       --experiment-name finetune-${MODEL_NAME} \
       --model-parallel-size ${MP_SIZE} \
       --mode finetune \
       --train-iters 2000 \
       --resume-dataloader \
       ${MODEL_ARGS} \
       --train-data ${train_data} \
       --valid-data ${valid_data} \
       --distributed-backend nccl \
       --lr-decay-style cosine \
       --warmup .02 \
       --checkpoint-activations \
       --vit_checkpoint_activations \
       --save-interval 2000 \
       --eval-interval 200 \
       --save ${SAVE_DIR} \
        --eval-iters 10 \
       --eval-batch-size 1 \
       --split 1. \
       --deepspeed_config test_config_zel.json \
       --skip-init \
       --seed 2023  \
"
# --strict-eval \


cur_time=$(date "+%Y%m%d-%H%M%S")

run_cmd="${OPTIONS_NCCL} ${OPTIONS_SAT} deepspeed --master_port 16666 finetune_cogagent.py ${gpt_options}"
echo ${run_cmd}
eval ${run_cmd}

set +x
