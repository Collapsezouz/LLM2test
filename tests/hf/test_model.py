import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download
from . import test_config
from tests import logger

def get_tokenizer(model_name=None, model_path=None):
    if not model_path:
        model_name = model_name or test_config.default_model
        model_path = snapshot_download(
            repo_id=model_name,
            cache_dir=test_config.CACHE_DIR,
            local_files_only = True
        )
        logger.debug("model_path: %s", model_path)
    logger.debug("load tokenizer %s", model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, use_fast=False)
    return tokenizer

def load_model(model_name=None, model_path=None, device_map=None, revision=None, model_kwargs:dict=None):
    if not model_path:
        model_name = model_name or test_config.default_model
        model_path = snapshot_download(
            repo_id=model_name,
            cache_dir=test_config.CACHE_DIR,
            local_files_only = True
        )
        logger.debug("model_path: %s", model_path)
    model_kwargs = model_kwargs or {}
    if 'torch_dtype' not in model_kwargs: model_kwargs['torch_dtype'] = torch.bfloat16
    if revision: model_kwargs['revision'] = revision
    begin_ts = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        device_map=device_map,
    #     offload_folder="offload", offload_state_dict = True,
        **model_kwargs,
        local_files_only=True
    )
    logger.info("loaded model cost %s seconds.", time.monotonic() - begin_ts)
    return model