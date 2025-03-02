ROOT_PATH=./ppm_construction/data_syn
NOTE=grid_v11_240831
cd $ROOT_PATH
python $ROOT_PATH/generate.py \
    --note v11 \
    --gen_num 20000 \
    --save_path $ROOT_PATH/data/$NOTE.json \
    --num_proc 64 \
    > $ROOT_PATH/logs/$NOTE.log