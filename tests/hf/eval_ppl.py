# https://huggingface.co/docs/transformers/perplexity
import torch, time, tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download
from transformers import GenerationConfig

from . import test_config
from .. import logger


en_texts = [
    "hello world",
    "Running this with the stride length equal to the max input length is equivalent to the suboptimal, non-sliding-window strategy we discussed above. ",
    "This is a closer approximation to the true decomposition of the sequence probability and will typically yield a more favorable score. The downside is that it requires a separate forward pass for each token in the corpus.",
    "When working with approximate models, however, we typically have a constraint on the number of tokens the model can process. "
]

def test_ppl(texts=None, batch_size:int=8, max_seq_len:int=4096, model_name=None, model_path=None, device_map=None):
    # 未调通
    texts = texts or en_texts

    if not model_path:
        model_name = model_name or test_config.default_model
        model_path = snapshot_download(
            repo_id=model_name,
            cache_dir=test_config.CACHE_DIR,
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
        device_map=device_map,
        **model_kwargs,
        local_files_only=True
    )
    logger.info("loaded model cost %s seconds.", time.monotonic() - begin_ts)

    stride = 128
    i, nlls = 0, []
    for i in tqdm.tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i+batch_size]
        tokens = tokenizer.batch_encode_plus(texts)
        for begin_loc in range(0, max_seq_len, stride):
            end_loc = min(begin_loc + stride, max_seq_len)



if __name__ == "__main__":
    _d, component = dict(globals()).items(), {}
    for k, v in _d:
        if k.startswith('test_'):
            component[k] = v
            component[k[5:]] = v

    import fire
    fire.Fire(component)