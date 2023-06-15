基于llama模型做信息抽取


# llama_plugin
## v1.0
`旧版`
前置模型: /nas/tmp/phbsxgpt/llama_ie_v1.1
数据集: /nas/dataset/llm/phbs_llm/plugin_instruct/plugin_v1_3k.jsonl


使用deepspeed库的训练过程:
```
export model_path=/nas/tmp/phbsxgpt/llama_ie_v1.1
export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7

export deepspeed_config=llm_model/configs/deepspeed.json
export data_path=/nas/dataset/llm/phbs_llm/plugin_instruct/plugin_v1_3k.jsonl
export output_dir=./logs/llama_plugin_v1.0

deepspeed llm_model/alpaca/train.py \
    --deepspeed="${deepspeed_config}" \
    --model_name_or_path="${model_path}" \
    --data_path="${data_path}" \
    --bf16=True \
    --output_dir="${output_dir}"\
    --num_train_epochs=1 \
    --per_device_train_batch_size=1 \
    --gradient_accumulation_steps=2 \
    --evaluation_strategy="no" \
    --save_strategy="steps" \
    --save_steps=1000 \
    --save_total_limit=3 \
    --learning_rate=2e-5 \
    --weight_decay=0. \
    --warmup_ratio=0.03 \
    --lr_scheduler_type="cosine" \
    --logging_steps=1 \
    --tf32=True \
    --model_max_length=2048 \
    --gradient_checkpointing=True \
    --instruct_version=2 > ./logs/ft_log/llama_plugin_v1.log 2>&1

## 训练完成后移动目录
清理状态文件夹: 
cd ./logs/llama_plugin_v1.0
find . -name global_step* | grep './checkpoint-' | xargs -i rm {} -rf
cd ../..
复制到nas目录: cp -r ./logs/llama_plugin_v1.0 /nas/tmp/phbsxgpt/llama_plugin_prev
复制完成后, 删除原目录: rm ./logs/llama_plugin_v1.0 -rf
```

## v1.1
`基于旧版`
前置模型: /nas/tmp/phbsxgpt/llama_plugin_prev/llama_plugin_v1.0
数据集: /nas/dataset/llm/phbs_llm/plugin_instruct/plugin_v1.1_12.jsonl


使用deepspeed库的训练过程:
```
export model_path=/nas/tmp/phbsxgpt/llama_plugin_v1.0
export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7

export deepspeed_config=llm_model/configs/deepspeed.json
export data_path=/nas/dataset/llm/phbs_llm/plugin_instruct/plugin_v1.1_12.jsonl
export output_dir=./logs/llama_plugin_v1.1

deepspeed llm_model/alpaca/train.py \
    --deepspeed="${deepspeed_config}" \
    --model_name_or_path="${model_path}" \
    --data_path="${data_path}" \
    --bf16=True \
    --output_dir="${output_dir}"\
    --num_train_epochs=1 \
    --per_device_train_batch_size=1 \
    --gradient_accumulation_steps=2 \
    --evaluation_strategy="no" \
    --save_strategy="steps" \
    --save_steps=1000 \
    --save_total_limit=3 \
    --learning_rate=2e-5 \
    --weight_decay=0. \
    --warmup_ratio=0.03 \
    --lr_scheduler_type="cosine" \
    --logging_steps=1 \
    --tf32=True \
    --model_max_length=2048 \
    --gradient_checkpointing=True \
    --instruct_version=2 > ./logs/ft_log/llama_plugin_v1.1.log 2>&1

## 训练完成后移动目录
清理状态文件夹: 
cd ./logs/llama_plugin_v1.1
find . -name global_step* | grep './checkpoint-' | xargs -i rm {} -rf
cd ../..
复制到nas目录: cp -r ./logs/llama_plugin_v1.1 /nas/tmp/phbsxgpt/llama_plugin_prev
复制完成后, 删除原目录: rm ./logs/llama_plugin_v1.1 -rf
```

## v1.0
`改数据集的key 重新训练`
前置模型: /nas/tmp/phbsxgpt/llama_ie_v1.1
数据集: /nas/dataset/llm/phbs_llm/plugin_instruct/plugin_v1.0_3k_0605.jsonl


使用deepspeed库的训练过程:
```
export model_path=/nas/tmp/phbsxgpt/llama_ie_v1.1
export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7

export deepspeed_config=llm_model/configs/deepspeed.json
export data_path=/nas/dataset/llm/phbs_llm/plugin_instruct/plugin_v1.0_3k_0605.jsonl
export output_dir=./logs/llama_plugin_v1.0

deepspeed llm_model/alpaca/train.py \
    --deepspeed="${deepspeed_config}" \
    --model_name_or_path="${model_path}" \
    --data_path="${data_path}" \
    --bf16=True \
    --output_dir="${output_dir}"\
    --num_train_epochs=1 \
    --per_device_train_batch_size=1 \
    --gradient_accumulation_steps=2 \
    --evaluation_strategy="no" \
    --save_strategy="steps" \
    --save_steps=1000 \
    --save_total_limit=3 \
    --learning_rate=2e-5 \
    --weight_decay=0. \
    --warmup_ratio=0.03 \
    --lr_scheduler_type="cosine" \
    --logging_steps=1 \
    --tf32=True \
    --model_max_length=2048 \
    --gradient_checkpointing=True \
    --instruct_version=2 > ./logs/ft_log/llama_plugin_v1_0605.log 2>&1

## 训练完成后移动目录
清理状态文件夹: 
cd ./logs/llama_plugin_v1.0
find . -name global_step* | grep './checkpoint-' | xargs -i rm {} -rf
cd ../..
复制到nas目录: cp -r ./logs/llama_plugin_v1.0 /nas/tmp/phbsxgpt/
复制完成后, 删除原目录: rm ./logs/llama_plugin_v1.0 -rf
```

## v1.1_query
前置模型: /nas/tmp/phbsxgpt/llama_plugin_v1.0
数据集: /nas/dataset/llm/phbs_llm/plugin_instruct/plugin_query_8k.jsonl


使用deepspeed库的训练过程:
```
export model_path=/nas/tmp/phbsxgpt/llama_plugin_v1.0
export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7

export deepspeed_config=llm_model/configs/deepspeed.json
export data_path=/nas/dataset/llm/phbs_llm/plugin_instruct/plugin_query_8k.jsonl
export output_dir=./logs/llama_plugin_v1.1_query

deepspeed llm_model/alpaca/train.py \
    --deepspeed="${deepspeed_config}" \
    --model_name_or_path="${model_path}" \
    --data_path="${data_path}" \
    --bf16=True \
    --output_dir="${output_dir}"\
    --num_train_epochs=1 \
    --per_device_train_batch_size=1 \
    --gradient_accumulation_steps=2 \
    --evaluation_strategy="no" \
    --save_strategy="steps" \
    --save_steps=1000 \
    --save_total_limit=3 \
    --learning_rate=2e-5 \
    --weight_decay=0. \
    --warmup_ratio=0.03 \
    --lr_scheduler_type="cosine" \
    --logging_steps=1 \
    --tf32=True \
    --model_max_length=2048 \
    --gradient_checkpointing=True \
    --instruct_version=2 > ./logs/ft_log/llama_plugin_v1.1_query.log 2>&1

## 训练完成后移动目录
清理状态文件夹: 
cd ./logs/llama_plugin_v1.0
find . -name global_step* | grep './checkpoint-' | xargs -i rm {} -rf
cd ../..
复制到nas目录: cp -r ./logs/llama_plugin_v1.1_query /nas/tmp/phbsxgpt/
复制完成后, 删除原目录: rm ./logs/llama_plugin_v1.1_query -rf
```

## MOSS_v1.0_query
前置模型: /cache/model/huggingface/models--fnlp--moss-moon-003-sft-plugin/snapshots/8364711b02c73e75f75b99e9ef0f97214ebed035
数据集: /nas/dataset/llm/phbs_llm/plugin_instruct/plugin_query_8k.jsonl

使用deepspeed库的训练过程:
```
export model_path=/cache/model/huggingface/models--fnlp--moss-moon-003-sft-plugin/snapshots/8364711b02c73e75f75b99e9ef0f97214ebed035
export CUDA_VISIBLE_DEVICES=6,7

export deepspeed_config=llm_model/configs/deepspeed_stage3_16b.json
export data_path=/nas/dataset/llm/phbs_llm/plugin_instruct/plugin_query_8k.jsonl
export output_dir=./logs/MOSS_plugin_v1.0_query

deepspeed llm_model/alpaca/train_moss.py \
    --deepspeed="${deepspeed_config}" \
    --model_name_or_path="${model_path}" \
    --data_path="${data_path}" \
    --bf16=True \
    --output_dir="${output_dir}"\
    --num_train_epochs=1 \
    --per_device_train_batch_size=1 \
    --gradient_accumulation_steps=2 \
    --evaluation_strategy="no" \
    --save_strategy="steps" \
    --save_steps=200 \
    --save_total_limit=1 \
    --learning_rate=2e-5 \
    --weight_decay=0. \
    --warmup_ratio=0.03 \
    --lr_scheduler_type="cosine" \
    --logging_steps=1 \
    --tf32=True \
    --model_max_length=2048 \
    --gradient_checkpointing=True \
    --instruct_version=2 > ./logs/ft_log/MOSS_plugin_v1.1_query.log 2>&1

## 训练完成后移动目录
清理状态文件夹: 
cd ./logs/MOSS_plugin_v1.0_query
find . -name global_step* | grep './checkpoint-' | xargs -i rm {} -rf
cd ../..
复制到nas目录: cp -r ./logs/MOSS_plugin_v1.0_query /nas/tmp/phbsxgpt/
复制完成后, 删除原目录: rm ./logs/MOSS_plugin_v1.0_query -rf
```