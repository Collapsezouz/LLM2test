项目地址: https://github.com/lm-sys/FastChat#vicuna-weights

huggingface文档: https://huggingface.co/docs/transformers/main/model_doc/llama


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

* 模型目录(增量版): 
/data/share/model/huggingface/models--lmsys--vicuna-13b-delta-v0/snapshots/163cfea0909591641713e1ad405d4852992192fc
/data/share/model/huggingface/models--lmsys--vicuna-7b-delta-v0/snapshots/142d4e179e45304c430e5bf2cf5f9816f1c2be7a

llama原版模型文件: /data/share/model/huggingface/models--decapoda-research--llama-13b-hf

* 合成模型文件
```
# 13b版本
export llama_ori_path=/nas/model/llama
export llama_path=/nas/model/llama/hf_13b
export delta_path=/data/share/model/huggingface/models--lmsys--vicuna-13b-delta-v0/snapshots/163cfea0909591641713e1ad405d4852992192fc
export vicuna_path=/nas/model/llama/vicuna_13b
> cd /home/app/transformers
python src/transformers/models/llama/convert_llama_weights_to_hf.py \
    --input_dir $llama_ori_path --model_size 13B --output_dir $llama_path

> cd /home/app/open/FastChat
python3 -m fastchat.model.apply_delta \
    --base $llama_path \
    --target $vicuna_path \
    --delta $delta_path

# 7b版本
export llama_ori_path=/nas/model/llama
export llama_path=/nas/model/llama/hf_7b
export delta_path=/data/share/model/huggingface/models--lmsys--vicuna-7b-delta-v0/snapshots/142d4e179e45304c430e5bf2cf5f9816f1c2be7a
export vicuna_path=/nas/model/llama/vicuna_7b

> cd /home/app/transformers
python src/transformers/models/llama/convert_llama_weights_to_hf.py \
    --input_dir $llama_ori_path --model_size 7B --output_dir $llama_path

> cd /home/app/open/FastChat
python3 -m fastchat.model.apply_delta \
    --base $llama_path \
    --target $vicuna_path \
    --delta $delta_path
```

```
> cd /home/app/transformers
python src/transformers/models/llama/convert_llama_weights_to_hf.py \
    --input_dir /nas/model/llama --model_size 7B --output_dir /nas/model/llama/hf_7b
```

# Train

* [训练配置1](../llm_model/configs/deepspeed.json)
  主要情况: 6张3090显卡, zero stage=2, offload_optimizer=cpu
  显存使用约20-24GB不等
  训练速度: 12条数据10s左右

* [训练配置2](../lm_model/configs/deepspeed_stage3.json)
  主要情况: 6张3090显卡, zero stage=2, offload_optimizer=cpu
  显存使用约11GB
  训练速度: 12条数据29s左右


* 调试过程记录:  
```
export model_path=/nas/model/llama/vicuna_13b

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
export model_path=/home/app/tmp/llama/hf_7b
export data_path=/nas/dataset/llm/phbs_llm/instruction_data_sample.jsonl
export deepspeed_config=llm_model/configs/deepspeed_stage3.json

export DEBUG_PORT=5679

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


## 基于原版llama训练
* 微调模型版本: /nas/tmp/phbsxgpt/llm_stage1_3w_0406

前置模型: /home/app/tmp/llama/hf_7b
数据集: /nas/dataset/llm/phbs_llm/instruction_data_3w.jsonl


使用deepspeed库的训练过程:
```
export model_path=/home/app/tmp/llama/hf_7b
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
    --gradient_checkpointing True > ./logs/llama_finetuning6.log 2>&1

## 训练完成后移动目录
复制到nas目录: cp -r ./logs/llm_stage1_3w_0406 /nas/tmp/phbsxgpt/
复制完成后, 删除原目录: rm ./logs/llm_stage1_3w_0406 -rf
```


## 基于chatllama_zh训练
* 微调模型版本: /nas/tmp/phbsxgpt/chatllama_zh_stage1_6w_0406

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
复制到nas目录: cp -r ./logs/chatllama_zh_stage1_6w_0406 /nas/tmp/phbsxgpt/
复制完成后, 删除原目录: rm ./logs/chatllama_zh_stage1_6w_0406 -rf
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
复制到nas目录: cp -r ./logs/vicuna_stage1_6w_0410 /nas/tmp/phbsxgpt/
复制完成后, 删除原目录: rm ./logs/vicuna_stage1_6w_0410 -rf
```