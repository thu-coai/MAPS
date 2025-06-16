ROOT_PATH=./ppm_construction/data_syn
NOTE=grid_v11_240831
cd "$ROOT_PATH"

# Use the absolute path of the current directory after cd for subsequent commands
ABS_ROOT_PATH="$(pwd)"

python generate.py \
    --note v11 \
    --gen_num 200 \
    --save_path "$ABS_ROOT_PATH/data/$NOTE.json" \
    --num_proc 64 \
    > "$ABS_ROOT_PATH/logs/$NOTE.log"