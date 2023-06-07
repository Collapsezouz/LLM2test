# convert tencentpretrain to huggingface
* 原版llama模型路径: /nas/model/llama
  

* ChatLLaMA模型路径(tencentpretrain) v230328: /data/share/model/huggingface/models--P01son--ChatLLaMA-zh-7B/snapshots/580463a1bb714d4324cf6e167bef9778fa8ab1d8/chatllama_7b.bin
* huggingface模型输出: /nas/model/llama/hf_chatllama_zh_7b_v230328

* ChatLLaMA模型路径(tencentpretrain) v230420: /data/share/model/huggingface/models--P01son--ChatLLaMA-zh-7B/snapshots/d9aadd7fc00ab0d89995b3c680661a7e9f2163fe/chatllama_7b.bin
* huggingface模型输出: /nas/model/llama/hf_chatllama_zh_7b_v230420


```
### 7b-v230328 
export tp_model_dir=/data/share/model/huggingface/models--P01son--ChatLLaMA-zh-7B/snapshots/580463a1bb714d4324cf6e167bef9778fa8ab1d8
export hf_model_output=/nas/model/llama/hf_chatllama_zh_7b_v230328

### 7b-v230420
export tp_model_dir=/data/share/model/huggingface/models--P01son--ChatLLaMA-zh-7B/snapshots/d9aadd7fc00ab0d89995b3c680661a7e9f2163fe
export hf_model_output=/nas/model/llama/hf_chatllama_zh_7b_v230420
ln -sf /nas/model/llama/hf_chatllama_zh_7b_v230420 /nas/model/llama/hf_chatllama_zh_7b

python llm_model/scripts/convert_llama_tencentpretrain_to_hf.py \
    --tp_model_dir $tp_model_dir \
    --tp_model_name chatllama_7b \
    --input_dir /nas/model/llama \
    --model_size 7B \
    --output_dir $hf_model_output


### 7b-v230518
export tp_model_dir=/data/share/model/huggingface/models--Linly-AI--ChatFlow-7B/snapshots/23fa44d4b7abecbeb76c76db88bdf785dba78b3b
export hf_model_output=/nas/model/llama/hf_chatflow_7b_v230518
python llm_model/scripts/convert_llama_tencentpretrain_to_hf.py \
    --tp_model_dir $tp_model_dir \
    --tp_model_name chatflow_7b \
    --input_dir /nas/model/llama \
    --model_size 7B \
    --output_dir $hf_model_output

### 13b-v230524
export tp_model_dir=/data/share/model/huggingface/models--Linly-AI--ChatFlow-13B/snapshots/1dbad879c089a9fe70e8c89138c27b9b56695fbe
export hf_model_output=/nas/model/llama/hf_chatflow_13b_v230524
python llm_model/scripts/convert_llama_tencentpretrain_to_hf.py \
    --tp_model_dir $tp_model_dir \
    --tp_model_name chatflow_13b \
    --input_dir /nas/model/llama \
    --model_size 13B \
    --output_dir $hf_model_output
```