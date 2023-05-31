# export DEBUG_PORT=5679
# export CUDA_VISIBLE_DEVICES=1
# REMOTE_DEBUG=1 python -m tests.hf.pipline generate
# REMOTE_DEBUG=1 python -m tests.hf.pipline llama
import torch, time
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
from transformers.generation.streamers import BaseStreamer
from huggingface_hub import snapshot_download
from transformers import GenerationConfig

from .. import logger

# 下载配置
_hf_hub_download_args = {
    'cache_dir': '/data/share/model/huggingface',
    'resume_download': True
}

CACHE_DIR = _hf_hub_download_args['cache_dir']

_default_model = "EleutherAI/pythia-70m"
_default_llama_model_path = '/home/app/expert-cpt/logs/chatllama_zh_stage1_v1.1.1'

_default_input_text = '''我需要为一家医疗器械公司制定销售策略，该如何入手？

- 输出:
'''

class TestTokenIteratorStreamer(BaseStreamer):
    def put(self, value):
        is_input_ids = isinstance(value.tolist()[0], list)
        if not is_input_ids:
            logger.debug('TestTokenIteratorStreamer put %s', value.tolist())

    def end(self):
        logger.debug('TestTokenIteratorStreamer end')


def test_generate(model_name=None, model_path=None):
    if not model_path:
        model_name = model_name or _default_model
        model_path = snapshot_download(
            repo_id=model_name,
            cache_dir=CACHE_DIR,
            local_files_only = True
        )
        logger.debug("model_path: %s", model_path)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=False)
    model_kwargs = {
        'torch_dtype': torch.bfloat16
    }
    begin_ts = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        device_map=None,
    #     device_map="auto",
    #     offload_folder="offload", offload_state_dict = True,
    #     torch_dtype=torch.float16,
    #     torch_dtype=torch.bfloat16,
    #     revision="step2000",
        **model_kwargs,
        local_files_only=True
    )

    logger.info("loaded model cost %s seconds.", time.monotonic() - begin_ts)
    generator = pipeline('text-generation', model=model, tokenizer=tokenizer, 
                         return_tensors='pt')

    streamer = TestTokenIteratorStreamer()
    rst = generator('''hello''', streamer=streamer, max_new_tokens=10)
    logger.info("rst: %s", rst)
    logger.info("streamer: %s", streamer)
    first_token_id = rst[0]['generated_token_ids'][0]
    first_char = tokenizer.decode([first_token_id])
    logger.info('first_char: %s', first_char)


def test_llama(text=None, max_new_tokens:int=200, model_name=None, model_path=None, use_fast:bool=False, device_map:str="auto"):
    if model_name:
        model_path = snapshot_download(
            repo_id=model_name,
            cache_dir=CACHE_DIR,
            local_files_only = True
        )
        logger.debug("model_path: %s", model_path)
    elif not model_path:
        model_path = _default_llama_model_path
    device_map = device_map or None
    text = text or _default_input_text
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=use_fast, local_files_only=True)
    model_kwargs = {
        'torch_dtype': torch.bfloat16
    }
    begin_ts = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        device_map=device_map,
    #     offload_folder="offload", offload_state_dict = True,
    #     torch_dtype=torch.float16,
    #     torch_dtype=torch.bfloat16,
    #     revision="step2000",
        **model_kwargs,
        local_files_only=True
    )

    logger.info("loaded model cost %s seconds.", time.monotonic() - begin_ts)
    generation_config = GenerationConfig.from_pretrained(model_path, local_files_only=True)
    logger.info("generation_config: %s", generation_config)

    generator = pipeline(
        'text-generation', 
        model=model, 
        tokenizer=tokenizer,
        # return_tensors='pt'
    )

    streamer = TestTokenIteratorStreamer()
    rst = generator(
        text, 
        streamer=streamer,
        max_new_tokens=max_new_tokens)
    logger.info("rst: %s", rst)
    logger.info("streamer: %s", streamer)

    
if __name__ == "__main__":
    _d, component = dict(globals()).items(), {}
    for k, v in _d:
        if k.startswith('test_'):
            component[k] = v
            component[k[5:]] = v

    import fire
    fire.Fire(component)