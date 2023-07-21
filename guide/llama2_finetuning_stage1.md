# hf模型根目录: /nas/share/model/huggingface
# 7b-chat模型目录: /nas/share/model/huggingface/models--meta-llama--Llama-2-7b-chat-hf/snapshots/0ede8dd71e923db6258295621d817ca8714516d4

项目地址: 

huggingface文档: https://huggingface.co/meta-llama/Llama-2-7b


# Init
```
地址: https://github.com/huggingface/transformers/tree/c612628045822f909020f7eb6784c79700813eda

# 从已有的目录切换分支
cd /home/app/transformers
git checkout c612628045822f909020f7eb6784c79700813eda

pip install -e /home/app/transformers

# 验证安装
python -c "from transformers import LlamaForCausalLM;print(LlamaForCausalLM)"
```

* 模型目录: 
/nas/share/model/huggingface/models--meta-llama--Llama-2-7b-chat-hf/snapshots/0ede8dd71e923db6258295621d817ca8714516d4

llama原版模型文件: /nas/share/model/huggingface/models--meta-llama--Llama-2-7b-chat-hf

* 合成模型文件
```
# 13b版本

# 7b版本
export llama2_ori_path=/nas/share/model/huggingface/model
export llama2_path=/nas/share/model/huggingface/model/models--meta-llama--Llama-2-7b-hf
export delta_path=/data/share/model/huggingface/models--meta-llama--Llama-2-7b-chat-hf/snapshots/0ede8dd71e923db6258295621d817ca8714516d4

> cd /home/app/transformers
python src/transformers/models/llama2/convert_llama2_weights_to_hf.py \
    --input_dir $llama2_ori_path --model_size 7B --output_dir $llama2_path

# Train

* [训练配置1](../llm_model/configs/deepspeed.json)
  主要情况: 6张3090显卡, zero stage=2, offload_optimizer=cpu
  显存使用约20-24GB不等
  训练速度: 12条数据10s左右

* [训练配置2](../lm_model/configs/deepspeed_stage3.json)
  主要情况: 6张3090显卡, zero stage=2, offload_optimizer=cpu
  显存使用约11GB
  训练速度: 12条数据29s左右

* 其它配置参考
run on A10 instances (ex: , 4 x A10 24GB; , 2 x A10), make the following: https://github.com/databrickslabs/dolly/blob/master/config/ds_z3_bf16_config.json



* 调试过程记录:  
```
export model_path=/nas/share/model/huggingface/model/models--meta-llama--Llama-2-7b-hf

export data_path=/nas/dataset/llm/phbs_llm/instruction_data.jsonl
export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7
export gpu_num=6

export output_dir=./logs/llm_stage1_0406

##  [未调试成功] 使用torch分布式框架(fsdp实现模型并行, 需要在torch 1.12版本以上)
torchrun --nnodes=1 --nproc_per_node=$gpu_num --master_port=9520 \
    llm_model/alpaca/train.py \
    --model_name_or_path $model_path \
    --data_path $data_path \
    --bf16 True \
    --output_dir $output_dir \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1200 \
    --save_total_limit 100 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 True \
    --model_max_length 1024 \
    --gradient_checkpointing True > ./logs/torchrun_alpaca1.log 2>&1

## [调试成功] 使用deepspeed
export deepspeed_config=llm_model/configs/deepspeed.json
export data_path=/nas/dataset/llm/phbs_llm/instruction_data_3w.jsonl
export output_dir=./logs/llm_stage1_3w_0406

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
    --gradient_checkpointing True > ./logs/llama_finetuning6.log 2>&1

## 训练完成后移动目录
cp -r ./logs/llm_stage1_3w_0406 /nas/tmp/phbsxgpt/
```

* 断点调试代码
```
export model_path=/home/app/tmp/llama2/hf_7b
export data_path=/nas/dataset/llm/phbs_llm/instruction_data_sample.jsonl
export deepspeed_config=llm_model/configs/deepspeed.json
export output_dir=./logs/ft_log/tmp_debug

export DEBUG_PORT=5679

export CUDA_VISIBLE_DEVICES=7

deepspeed llm_model/alpaca/train_debug.py \
    --deepspeed $deepspeed_config \
    --model_name_or_path $model_path \
    --data_path $data_path \
    --bf16 True \
    --output_dir $output_dir\
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1200 \
    --save_total_limit 100 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 512 \
    --gradient_checkpointing True
```

* debug dataset
```
export model_path=/home/app/tmp/llama2/hf_7b
export data_path=/nas/dataset/llm/phbs_llm/instruction_data_sample.jsonl
export deepspeed_config=llm_model/configs/deepspeed.json
export output_dir=./logs/ft_log/tmp_debug
export DEBUG_PORT=5679

> python -m llm_model.alpaca.dataset_debug \
    --model_name_or_path $model_path \
    --data_path $data_path
```


## 基于原版llama训练
* 微调模型版本: "new holder" ##/nas/tmp/phbsxgpt/llm_stage1_3w_0406

前置模型: /home/app/tmp/llama2/hf_7b
数据集: /nas/dataset/llm/phbs_llm/instruction_data_3w.jsonl


使用deepspeed库的训练过程:
```
export model_path=/home/app/tmp/llama2/hf_7b
export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7

export deepspeed_config=llm_model/configs/deepspeed.json
export data_path=/nas/dataset/llm/phbs_llm/instruction_data_3w.jsonl
export output_dir=./logs/llm_stage1_3w_0406

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
    --save_total_limit 3 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 512 \
    --gradient_checkpointing True > ./logs/llama2_finetuning6.log 2>&1

## 训练完成后移动目录
清理状态文件夹: 
cd ./logs/llm_stage1_3w_0406
find . -name global_step* | grep './checkpoint-' | xargs -i rm {} -rf
cd ../..
复制到nas目录: cp -r ./logs/llm_stage1_3w_0406 /nas/tmp/phbsxgpt/
复制完成后, 删除原目录: rm ./logs/llm_stage1_3w_0406 -rf
```












## 基于chatllama_zh训练
* 微调模型版本: /nas/tmp/phbsxgpt/chatllama_zh_stage1_6w_0406

* 前置步骤: [模型转换](./convert.md)

### v1
前置模型: /nas/model/llama/hf_chatllama_zh_7b
数据集: /nas/dataset/llm/phbs_llm/instruction_data_6w.jsonl

使用deepspeed库的训练过程:
```
export model_path=/nas/model/llama/hf_chatllama_zh_7b
export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7

export deepspeed_config=llm_model/configs/deepspeed.json
export data_path=/nas/dataset/llm/phbs_llm/instruction_data_6w.jsonl
export output_dir=./logs/chatllama_zh_stage1_6w_0406

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
    --save_total_limit 3 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 512 \
    --gradient_checkpointing True > ./logs/chatllama_zh_finetuning2.log 2>&1

## 训练完成后移动目录
清理状态文件夹: 
cd ./logs/chatllama_zh_stage1_6w_0406
find . -name global_step* | grep './checkpoint-' | xargs -i rm {} -rf
cd ../..
复制到nas目录: cp -r ./logs/chatllama_zh_stage1_6w_0406 /nas/tmp/phbsxgpt/
复制完成后, 删除原目录: rm ./logs/chatllama_zh_stage1_6w_0406 -rf
```

### v1.1
前置模型-v1: /nas/tmp/phbsxgpt/chatllama_zh_stage1_6w_0406
数据集: /nas/dataset/llm/phbs_llm/business_instruction_data.jsonl

使用deepspeed库的训练过程:
```
export model_path=/nas/tmp/phbsxgpt/chatllama_zh_stage1_6w_0406
export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7

export deepspeed_config=llm_model/configs/deepspeed.json
export data_path=/nas/dataset/llm/phbs_llm/business_instruction_data.jsonl
export output_dir=./logs/chatllama_zh_stage1_v1.1

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
    --save_total_limit 3 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 512 \
    --gradient_checkpointing True > ./logs/chatllama_zh_v1.1_ft.log 2>&1

## 训练完成后移动目录
清理状态文件夹: 
cd ./logs/chatllama_zh_stage1_v1.1
find . -name global_step* | grep './checkpoint-' | xargs -i rm {} -rf
cd ../..
复制到nas目录: cp -r ./logs/chatllama_zh_stage1_v1.1 /nas/tmp/phbsxgpt/
复制完成后, 删除原目录: rm ./logs/chatllama_zh_stage1_v1.1 -rf
```


### v1.1.1
前置模型: /nas/model/llama/hf_chatllama_zh_7b_v230420
数据集: /nas/dataset/llm/phbs_llm/merge_instruction_data_v1.1.jsonl

使用deepspeed库的训练过程:
```
export model_path=/nas/model/llama/hf_chatllama_zh_7b_v230420
export CUDA_VISIBLE_DEVICES=1,3,4,5,6,7

export deepspeed_config=llm_model/configs/deepspeed.json
export data_path=/nas/dataset/llm/phbs_llm/merge_instruction_data_v1.1.jsonl
export output_dir=./logs/chatllama_zh_stage1_v1.1.1

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
    --save_total_limit 3 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 512 \
    --gradient_checkpointing True > ./logs/ft_log/chatllama_zh_v1.1.1_ft.log 2>&1

## 训练完成后移动目录
清理状态文件夹: 
cd ./logs/chatllama_zh_stage1_v1.1.1
find . -name global_step* | grep './checkpoint-' | xargs -i rm {} -rf
cd ../..
复制到nas目录: cp -r ./logs/chatllama_zh_stage1_v1.1.1 /nas/tmp/phbsxgpt/
复制完成后, 删除原目录: rm ./logs/ft_log/chatllama_zh_stage1_v1.1 -rf
```


### v1.1.2
前置模型: /nas/model/llama/hf_chatflow_7b_v230518
数据集: /nas/dataset/llm/phbs_llm/business_instruction_data.jsonl

使用deepspeed库的训练过程:
```
export model_path=/nas/model/llama/hf_chatflow_7b_v230518
export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7

export deepspeed_config=llm_model/configs/deepspeed.json
export data_path=/nas/dataset/llm/phbs_llm/business_instruction_data.jsonl
export output_dir=./logs/chatflow_7b_stage1_v1.1.2

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
    --save_total_limit 3 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 512 \
    --gradient_checkpointing True > ./logs/ft_log/chatflow_7b_stage1_v1.1.2.log 2>&1

## 训练完成后移动目录
清理状态文件夹: 
cd ./logs/chatflow_7b_stage1_v1.1.2
find . -name global_step* | grep './checkpoint-' | xargs -i rm {} -rf
cd ../..
复制到nas目录: cp -r ./logs/chatflow_7b_stage1_v1.1.2 /nas/tmp/phbsxgpt/
复制完成后, 删除原目录: rm ./logs/chatflow_7b_stage1_v1.1.2 -rf
```



## 基于vicuna训练
* 微调模型版本: /nas/tmp/phbsxgpt/vicuna_stage1_6w_0410

前置模型: /nas/model/llama/vicuna_7b
数据集: /nas/dataset/llm/phbs_llm/instruction_data_6w.jsonl


使用deepspeed库的训练过程:
```
export model_path=/nas/model/llama/vicuna_7b
export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7

export deepspeed_config=llm_model/configs/deepspeed.json
export data_path=/nas/dataset/llm/phbs_llm/instruction_data_6w.jsonl
export output_dir=./logs/vicuna_stage1_6w_0410


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
    --save_total_limit 3 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 512 \
    --gradient_checkpointing True > ./logs/vicuna_finetuning1.log 2>&1

## 训练完成后移动目录
清理状态文件夹: 
cd ./logs/vicuna_stage1_6w_0410
find . -name global_step* | grep './checkpoint-' | xargs -i rm {} -rf
cd ../..
复制到nas目录: cp -r ./logs/vicuna_stage1_6w_0410 /nas/tmp/phbsxgpt/
复制完成后, 删除原目录: rm ./logs/vicuna_stage1_6w_0410 -rf
```



## 基于kaola训练
* 微调模型版本: /nas/tmp/phbsxgpt/kaola_stage1_6w_0410

前置模型: /nas/model/llama/kaola_7b
数据集: /nas/dataset/llm/phbs_llm/instruction_data_6w.jsonl


使用deepspeed库的训练过程:
```
export model_path=/nas/model/llama/kaola_7b
export CUDA_VISIBLE_DEVICES=0,1,5,6,7

export deepspeed_config=llm_model/configs/deepspeed.json
export data_path=/nas/dataset/llm/phbs_llm/instruction_data_6w.jsonl
export output_dir=./logs/kaola_stage1_6w_0410


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
    --save_total_limit 3 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 512 \
    --gradient_checkpointing True > ./logs/kaola_finetuning1.log 2>&1

## 训练完成后移动目录
清理状态文件夹: 
cd ./logs/kaola_stage1_6w_0410
find . -name global_step* | grep './checkpoint-' | xargs -i rm {} -rf
cd ../..
复制到nas目录: cp -r ./logs/kaola_stage1_6w_0410 /nas/tmp/phbsxgpt/
复制完成后, 删除原目录: rm ./logs/kaola_stage1_6w_0410 -rf
```