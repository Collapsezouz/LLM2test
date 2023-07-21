export model_path=/nas/share/model/huggingface/model/models--meta-llama--Llama-2-7b-hf

export data_path=/nas/dataset/llm/phbs_llm/instruction_data.jsonl
export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7
export gpu_num=6

export output_dir=./logs/llm_stage1_0406
export deepspeed_config=llm_model/configs/deepspeed.json

deepspeed llm_model/alpaca/train.py \
    --deepspeed $deepspeed_config \
    --model_name_or_path $model_path \
    --data_path $data_path \
    --bf16 True \
    --output_dir $output_dir\
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 5 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 512 \
    --gradient_checkpointing True > ./logs/llama2_finetuning1.log 2>&1

