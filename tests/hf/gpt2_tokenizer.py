# python -m tests.hf.gpt2_tokenizer tokenize
from transformers import GPT2Tokenizer

from tests import logger

_model_path = '/data/share/model/huggingface/models--gpt2/snapshots/e7da7f221d5bf496a48136c0cd264e630fe9fcc8'


def test_tokenize(text='护'):
    tokenizer = GPT2Tokenizer.from_pretrained(_model_path, use_fast=False) 
    tokens = tokenizer.encode(text)
    de_text = tokenizer.decode(tokens)
    logger.info('bytes: %s', text.encode('utf8'))
    logger.info('tokens: %s, decode text: %s', tokens, de_text)
    # https://baike.baidu.com/item/UTF-8/481798

    for i, token in enumerate(tokens):
        token_str = tokenizer.decode([token])
        token_bytes = [ord(x) for x in tokenizer.decoder.get(token)]
        logger.debug('\ttoken_id %s -> %s, %s', token, token_str, token_bytes)
    

if __name__ == "__main__":
    _d, component = dict(globals()).items(), {}
    for k, v in _d:
        if k.startswith('test_'):
            component[k] = v
            component[k[5:]] = v

    import fire
    fire.Fire(component)