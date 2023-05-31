# export DEBUG_PORT=5679
# export CUDA_VISIBLE_DEVICES=1
# REMOTE_DEBUG=1 python -m tests.hf.generate generate --text='["hello", "中国的首都是"]'
# python -m tests.hf.generate encode --text=中国
# python -m tests.hf.generate decode --tokens='[31373, 50256]'
# python -m tests.hf.generate batch_encode --text='["hello", "中国的首都是"]'
import torch, time
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download
from transformers import GenerationConfig

from . import test_config
from .. import logger

_default_input_text = '''我需要为一家医疗器械公司制定销售策略，该如何入手？

- 输出:
'''


def test_generate(text=None, model_name=None, model_path=None, device_map=None):
    if not model_path:
        model_name = model_name or test_config.default_model
        model_path = snapshot_download(
            repo_id=model_name,
            cache_dir=test_config.CACHE_DIR,
            local_files_only = True
        )
        logger.debug("model_path: %s", model_path)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=False)
    # model_kwargs = {
    #     'torch_dtype': torch.bfloat16
    # }
    # begin_ts = time.monotonic()
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_path, 
    #     device_map=device_map,
    #     **model_kwargs,
    #     local_files_only=True
    # )
    # logger.info("loaded model cost %s seconds.", time.monotonic() - begin_ts)

    text = text or [_default_input_text]
    logger.info("text: %s", text)
    if isinstance(text, (list, tuple)):
        tokens = tokenizer.batch_encode_plus(text)
    else:
        tokens = tokenizer.encode(text)
    logger.info("tokens: %s", tokens)


def test_decode(tokens=None, model_name=None, model_path=None):
    if not model_path:
        model_name = model_name or test_config.default_model
        model_path = snapshot_download(
            repo_id=model_name,
            cache_dir=test_config.CACHE_DIR,
            local_files_only = True
        )
        logger.debug("model_path: %s", model_path)
    
    logger.info("tokens: %s", tokens)
    if not tokens:
        logger.warning("tokens is None")
        return
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=False)
    text = tokenizer.decode(tokens)
    logger.info("decoded text: %s", text)


def test_encode(text=None, model_name=None, model_path=None):
    if not model_path:
        model_name = model_name or test_config.default_model
        model_path = snapshot_download(
            repo_id=model_name,
            cache_dir=test_config.CACHE_DIR,
            local_files_only = True
        )
        logger.debug("model_path: %s", model_path)
    
    logger.info("text: %s", text)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=False)
    tokens = tokenizer.encode(text)
    logger.info("tokens: %s", tokens)


def test_batch_encode(text=None, model_name=None, model_path=None, padding=False, truncation=False, max_length:int=50):
    if not model_path:
        model_name = model_name or test_config.default_model
        model_path = snapshot_download(
            repo_id=model_name,
            cache_dir=test_config.CACHE_DIR,
            local_files_only = True
        )
        logger.debug("model_path: %s", model_path)
    
    logger.info("text: %s", text)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=False)
    logger.info("vocab_size: %s, all_special_tokens: %s", tokenizer.vocab_size, tokenizer.all_special_tokens)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokens = tokenizer.batch_encode_plus(text, padding=padding, truncation=truncation, max_length=max_length)
    logger.info("tokens: %s", tokens)

    
if __name__ == "__main__":
    _d, component = dict(globals()).items(), {}
    for k, v in _d:
        if k.startswith('test_'):
            component[k] = v
            component[k[5:]] = v

    import fire
    fire.Fire(component)