# REMOTE_DEBUG=1 DEBUG_PORT=5679 python -m tests.alpaca.prepare_data preprocess --model_path=/nas/model/llama/hf_13b
import transformers

from llm_model.alpaca.utils import load_dataset
from llm_model.alpaca.train import preprocess, PROMPT_DICT, smart_tokenizer_and_embedding_resize, \
        DEFAULT_PAD_TOKEN, DEFAULT_EOS_TOKEN, DEFAULT_BOS_TOKEN, DEFAULT_UNK_TOKEN
from tests import logger

_default_data_path = '/nas/dataset/llm/phbs_llm/instruction_data.jsonl'
_default_model_path = '/nas/model/llama/vicuna_13b'


def _tokenizer_fn(model_path, max_length:int=2048):
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path,
        model_max_length=max_length,
        padding_side="right",
        use_fast=False,
        local_files_only=True
    )
    if 'pad_token' not in tokenizer.special_tokens_map:
        logger.debug('ori special_tokens_map: %s', tokenizer.special_tokens_map)
        num_new_tokens = tokenizer.add_special_tokens({
            "pad_token": DEFAULT_PAD_TOKEN,
            "eos_token": DEFAULT_EOS_TOKEN,
            "bos_token": DEFAULT_BOS_TOKEN,
            "unk_token": DEFAULT_UNK_TOKEN,
        })
        logger.debug('_tokenizer_fn add_special_tokens %s token, special_tokens_map: %s', num_new_tokens, tokenizer.special_tokens_map)
    return tokenizer

def test_preprocess(data_path=None, model_path=None, start:int=0, end:int=3, max_length:int=2048):
    data_path = data_path or _default_data_path
    model_path = model_path or _default_model_path
    logger.debug("test_preprocess\tdata: %s\tmodel: %s", data_path, model_path)

    list_data_dict = load_dataset(data_path)
    logger.info("load %s item, example: %s", len(list_data_dict), list_data_dict[:1])
    list_data_dict = list_data_dict[start:end]

    tokenizer = _tokenizer_fn(model_path, max_length=max_length)

    prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
    sources = [
        prompt_input.format_map(example) if example.get("input", "") not in ('', '<noinput>') else prompt_no_input.format_map(example)
        for example in list_data_dict
    ]
    targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

    data_dict = preprocess(sources, targets, tokenizer)
    logger.info('data_dict:')
    for k, v in data_dict.items():
        logger.info("\t%s: %s", k, v[:2])


if __name__ == "__main__":
    _d, component = dict(globals()).items(), {}
    for k, v in _d:
        if k.startswith('test_'):
            component[k] = v
            component[k[5:]] = v

    import fire
    fire.Fire(component)