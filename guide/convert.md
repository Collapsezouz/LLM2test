# convert tencentpretrain to huggingface

* ChatLLaMA模型路径(tencentpretrain): /data/share/model/huggingface/models--P01son--ChatLLaMA-zh-7B/snapshots/580463a1bb714d4324cf6e167bef9778fa8ab1d8/chatllama_7b.bin
* 原版llama模型路径: /nas/model/llama
* huggingface模型输出: /nas/model/llama/hf_chatllama_zh_7b

```
export tp_model_dir=/data/share/model/huggingface/models--P01son--ChatLLaMA-zh-7B/snapshots/580463a1bb714d4324cf6e167bef9778fa8ab1d8

python llm_model/scripts/convert_llama_tencentpretrain_to_hf.py \
    --tp_model_dir $tp_model_dir \
    --tp_model_name chatllama_7b \
    --input_dir /nas/model/llama \
    --model_size 7B \
    --output_dir /nas/model/llama/hf_chatllama_zh_7b
```