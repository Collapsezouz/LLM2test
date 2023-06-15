# 下载配置
_hf_hub_download_args = {
    'cache_dir': '/data/share/model/huggingface',
    'resume_download': True
}

CACHE_DIR = _hf_hub_download_args['cache_dir']

# default_model = "EleutherAI/pythia-70m"
default_model = "gpt2"

default_llama_model_path = '/nas/model/llama/hf_7b'
default_custom_llama_model_path = '/home/app/expert-cpt/logs/chatllama_zh_stage1_v1.1.1'
