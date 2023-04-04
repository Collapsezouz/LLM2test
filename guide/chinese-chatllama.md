* 项目地址: https://github.com/ydli-ai/Chinese-ChatLLaMA.git



# Init

* 模型文件
/data/share/model/huggingface/models--P01son--ChatLLaMA-zh-7B/snapshots/580463a1bb714d4324cf6e167bef9778fa8ab1d8

```
> ls /data/share/model/huggingface/models--P01son--ChatLLaMA-zh-7B/snapshots/580463a1bb714d4324cf6e167bef9778fa8ab1d8
README.md  chatllama_7b.bin  tokenizer.model
```

* TencentPretrain
```
> cd /home/app/open
> git clone https://github.com/Tencent/TencentPretrain.git
```



# Fine-tuning
* 数据集: /nas/dataset/llm/phbs_llm/instruction_data.jsonl
  
69546条数据，使用私有项目[llm-dataset](http://git.dev.sendbp.com/phbs/corpus/llm-dataset)的174个种子通过chatgpt生成的数据集, 样例:  
```
{"idx": 0, "instruction": "将输入的英文文本翻译成中文，并输出翻译结果。", "input": "\"The quick brown fox jumps over the lazy dog.\"", "output": "\"那只敏捷的棕色狐狸跳过了那只懒惰的狗。\"", "resp_idx": 0, "type": "longest_sentence|news_categories_multi_labels|ascending_sorting"}
```

* 修改TecentPretrain配置

Check out the `tencentpretrain/utils/constants.py` file, and modify L4: `special_tokens_map.json` to `llama_special_tokens_map.json`


* 预处理指令数据集：

```
> mkdir -p /nas/dataset/llm/phbs_llm/instruction_data

>
export INSTRUCTION_PATH=/nas/dataset/llm/phbs_llm/instruction_data.jsonl
export LLaMA_PATH=/data/share/model/huggingface/models--P01son--ChatLLaMA-zh-7B/snapshots/580463a1bb714d4324cf6e167bef9778fa8ab1d8
export OUTPUT_DATASET_PATH=/nas/dataset/llm/phbs_llm/instruction_data.pt

> cd /home/app/open/TencentPretrain
python3 preprocess.py --corpus_path $INSTRUCTION_PATH --spm_model_path $LLaMA_PATH/tokenizer.model \
                      --dataset_path $OUTPUT_DATASET_PATH --data_processor alpaca --seq_length 512
```
调试代码: `DEBUG_PORT=5679 REMOTE_DEBUG=1 python -m tests.tencentpretrain.prepare_data --corpus_path $INSTRUCTION_PATH --spm_model_path $LLaMA_PATH/tokenizer.model --dataset_path $OUTPUT_DATASET_PATH --data_processor alpaca --seq_length 512`


指令微调：

```
mkdir -p /nas/tmp/hhw/phbs_llama

export LLaMA_PATH=/data/share/model/huggingface/models--P01son--ChatLLaMA-zh-7B/snapshots/580463a1bb714d4324cf6e167bef9778fa8ab1d8

export LLaMA_PATH=/home/app/tmp/ChatLLaMA-zh-7B
export OUTPUT_DATASET_PATH=/nas/dataset/llm/phbs_llm/instruction_data.pt
export Gpu_Num=6
export Total_Step=`python -c 'print(int((69546+3)/'$Gpu_Num'))'`

export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7
export NCCL_DEBUG=info
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

deepspeed pretrain.py --deepspeed --deepspeed_config /home/app/ov-nlg-model/deepspeed_config.json \
                      --pretrained_model_path $LLaMA_PATH/chatllama_7b.bin \
                      --dataset_path $OUTPUT_DATASET_PATH --spm_model_path $LLaMA_PATH/tokenizer.model \
                      --config_path models/llama/7b_config.json \
                      --output_model_path /nas/tmp/hhw/phbs_llama/phbs_llama_7b \
                      --world_size $Gpu_Num --data_processor lm \
                      --total_steps $Total_Step --save_checkpoint_steps 2000 --batch_size $Gpu_Num > /home/app/ov-nlg-model/logs/phbs_llama_finetuning5.log 2>&1
```

* debug:  
```
export LLaMA_PATH=/home/app/tmp/ChatLLaMA-zh-7B
export OUTPUT_DATASET_PATH=/nas/dataset/llm/phbs_llm/instruction_data.pt
export Gpu_Num=4
export Total_Step=`python -c 'print(int((69546+3)/'$Gpu_Num'))'`

export CUDA_VISIBLE_DEVICES=4,5,6,7
export NCCL_DEBUG=info
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

> cd /home/app/ov-nlg-model

DEBUG_PORT=5679 REMOTE_DEBUG=1 deepspeed tests/tp_pretrain.py --deepspeed --deepspeed_config /home/app/ov-nlg-model/deepspeed_config.json \
    --pretrained_model_path $LLaMA_PATH/chatllama_7b.bin \
    --dataset_path $OUTPUT_DATASET_PATH --spm_model_path $LLaMA_PATH/tokenizer.model \
    --config_path models/llama/7b_config.json \
    --output_model_path /nas/tmp/hhw/phbs_llama/phbs_llama_7b \
    --world_size $Gpu_Num --data_processor lm \
    --total_steps $Total_Step --save_checkpoint_steps 2000 --batch_size $Gpu_Num
```