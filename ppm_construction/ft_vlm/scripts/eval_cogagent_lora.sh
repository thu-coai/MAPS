#!/bin/bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
echo $PATH
which python

NUM_GPUS_PER_WORKER=8
MP_SIZE=1

script_path=$(realpath $0)
script_dir=$(dirname $script_path)
main_dir=$(dirname $script_dir)
MODEL_NAME="cogagent-vqa-09-02-16-42"   # v11

MODEL_TYPE="ft_vlm/checkpoints/test/finetune-${MODEL_NAME}"

test_data_name="grid_v11_240831_split_test"
test_data="datasets/grid_v11_240831_split/test"

VERSION="vqa"
TOKENIZER_TYPE="../checkpoints/huggingface/lmsys/vicuna-7b-v1.5"
EXP_NOTE="test"
SAVE_DIR="ft_vlm/checkpoints/${EXP_NOTE}"
MAX_LEN=1000

MODEL_ARGS="--from_pretrained $MODEL_TYPE \
    --max_length $MAX_LEN \
    --lora_rank 50 \
    --local_tokenizer $TOKENIZER_TYPE \
    --version $VERSION"
# TIPS: max_length include low-resolution image sequence (which has 256 tokens) 

OPTIONS_SAT="SAT_HOME=~/.sat_models"
# OPTIONS_NCCL="NCCL_DEBUG=info NCCL_IB_DISABLE=0 NCCL_NET_GDR_LEVEL=2 LOCAL_WORLD_SIZE=$NUM_GPUS_PER_WORKER"
OPTIONS_NCCL="NCCL_IB_DISABLE=0 NCCL_NET_GDR_LEVEL=2 LOCAL_WORLD_SIZE=${NUM_GPUS_PER_WORKER}" # exclude NCCL_DEBUG=info
# HOST_FILE_PATH="hostfile"

# train_data="../CogVLM/archive_split/train"
# valid_data="../CogVLM/archive_split/valid"
train_data="datasets/circuit_240204_split/train"

eval_path="ft_vlm/results/test_${test_data_name}/${MODEL_NAME}.json"

gpt_options="\
    --experiment-name finetune-${MODEL_NAME} \
    --model-parallel-size ${MP_SIZE} \
    --mode finetune \
    --train-iters 0 \
    --resume-dataloader \
    ${MODEL_ARGS} \
    --train-data ${train_data} \
    --test-data ${test_data} \
    --distributed-backend nccl \
    --lr-decay-style cosine \
    --warmup .02 \
    --checkpoint-activations \
    --vit_checkpoint_activations \
    --save-interval 1000 \
    --eval-interval 200 \
    --save ${SAVE_DIR} \
    --strict-eval \
    --eval-batch-size 1 \
    --split 1. \
    --deepspeed_config test_config_zel.json \
    --skip-init \
    --seed 2023 \
    --eval_results_path ${eval_path} \
    --get_image_from_cur_dir \
"
# --only_inference \
# --img_suffix .png \ # currently can't be used

# run_cmd="${OPTIONS_NCCL} ${OPTIONS_SAT} deepspeed --master_port 16666 --hostfile ${HOST_FILE_PATH} finetune_cogagent_demo.py ${gpt_options}"
run_cmd="${OPTIONS_NCCL} ${OPTIONS_SAT} deepspeed --master_port 16666 inference_cogagent.py ${gpt_options}"
echo ${run_cmd}
eval ${run_cmd}

set +x
