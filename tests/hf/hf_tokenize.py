# python -m tests.hf.hf_tokenize tokenize
# python -m tests.hf.hf_tokenize gpt2
# python -m tests.hf.hf_tokenize llama
# from transformers import GPT2Tokenizer
from .test_model import get_tokenizer
from . import test_config
from tests import logger

# _model_path = '/data/share/model/huggingface/models--gpt2/snapshots/e7da7f221d5bf496a48136c0cd264e630fe9fcc8'


def test_tokenize(text='护', model_name=None, model_path=None, add_special_tokens:bool=False, **kwargs):
    logger.debug("test_tokenize model name=%s, path=%s", model_name, model_path)
    tokenizer = get_tokenizer(model_name=model_name, model_path=model_path)
    # tokenizer = GPT2Tokenizer.from_pretrained(_model_path, use_fast=False) 
    logger.debug('special_tokens_map: %s', tokenizer.special_tokens_map)
    tokens = tokenizer.encode(text, add_special_tokens=add_special_tokens)
    de_text = tokenizer.decode(tokens)
    logger.info('bytes: %s', text.encode('utf8'))
    logger.info('tokens: %s, decode text: %s', tokens, de_text)
    # https://baike.baidu.com/item/UTF-8/481798

    for i, token in enumerate(tokens):
        token_str = tokenizer.decode([token])
        if getattr(tokenizer, 'decoder', None):
            token_bytes = [ord(x) for x in tokenizer.decoder.get(token)]
            logger.debug('\ttoken_id %s -> %s, %s', token, token_str, token_bytes)
        else:
            logger.debug('\ttoken_id %s -> %s(len=%s, ord=%s)', 
                token, token_str, len(token_str), ord(token_str[0]) if token_str else '')


def test_gpt2(text='护', model_name=None, model_path=None, **kwargs):
    if not model_name and not model_path:
        model_name = 'gpt2'
    test_tokenize(text=text, model_name=model_name, model_path=model_path, **kwargs)


def test_llama(text='hello护士护', model_name=None, model_path=None, **kwargs):
    if not model_name and not model_path:
        model_path = test_config.default_llama_model_path
    test_tokenize(text=text, model_name=model_name, model_path=model_path, **kwargs)


if __name__ == "__main__":
    _d, component = dict(globals()).items(), {}
    for k, v in _d:
        if k.startswith('test_'):
            component[k] = v
            component[k[5:]] = v

    import fire
    fire.Fire(component)