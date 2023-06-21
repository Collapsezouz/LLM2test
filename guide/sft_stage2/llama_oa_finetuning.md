办公场景的指令微调。

场景：
- 会议纪要


# llama_oa
## v0.1
前置模型: /nas/tmp/phbsxgpt/llama_plugin_v1.2
数据集: /nas/dataset/llm/phbs_llm/meeting/summary_dataset_clean.jsonl


使用deepspeed库的训练过程:
```
export model_path=/nas/tmp/phbsxgpt/llama_plugin_v1.2
export CUDA_VISIBLE_DEVICES=6,7

export deepspeed_config=llm_model/configs/deepspeed.json
export data_path=/nas/dataset/llm/phbs_llm/meeting/summary_dataset_clean.jsonl
export output_dir=./logs/llama_oa_v0.1

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
    --save_steps 1000 \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --instruct_version=2 > ./logs/ft_log/llama_oa_v0.1.log 2>&1

## 训练完成后移动目录
清理状态文件夹: 
cd ./logs/llama_oa_v0.1
find . -name global_step* | grep './checkpoint-' | xargs -i rm {} -rf
cd ../..
复制到nas目录: cp -r ./logs/llama_oa_v0.1 /nas/tmp/phbsxgpt/
复制完成后, 删除原目录: rm ./logs/llama_oa_v0.1 -rf
```
