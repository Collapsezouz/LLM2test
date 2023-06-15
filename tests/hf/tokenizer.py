from transformers import AutoTokenizer
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
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=False)
    return tokenizer