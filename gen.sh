NUM_GPUS=2
model_path="1231czx/qwen_self_corr_warmup2_clean_ep1"
#1231czx/qwmathbase_grpo2_step80

#for i in $(seq 0 $((NUM_GPUS - 1))); do
#        CUDA_VISIBLE_DEVICES=$i python ./gen_hf.py \
#            --model_name_or_path $model_path \
#            --K 64 \
#            --temperature 1.0 \
#            --local_index $i \
#            --my_world_size 8 &
#done


 CUDA_VISIBLE_DEVICES=0 python ./gen_hf.py \
            --model_name_or_path $model_path \
            --K 64 \
            --temperature 1.0 \
            --local_index 0 \
            --my_world_size 2 &

 CUDA_VISIBLE_DEVICES=1 python ./gen_hf.py \
            --model_name_or_path $model_path \
            --K 64 \
            --temperature 1.0 \
            --local_index 1 \
            --my_world_size 2 &
