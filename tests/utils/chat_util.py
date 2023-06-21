# python -m tests.utils.chat_util parse
# export DEBUG_PORT=5679
# REMOTE_DEBUG=1 python -m tests.utils.chat_util train_data
# REMOTE_DEBUG=1 python -m tests.utils.chat_util predict_input
import json
from llm_model.utils.chat_util import *
from tests.hf.test_model import get_tokenizer, test_config
from tests import logger
from ._mock_data import mock_chat_data



def test_parse():
    for i, chat_data in enumerate(mock_chat_data):
        if i: logger.info("")
        logger.info('chat_data: %s', chat_data)
        encoder = ChatTextEncoder()
        msg_iter = encoder.parse_chat_obj(chat_data)
        for msg in msg_iter:
            logger.info("# block %s-%s", msg.type, msg.round_idx)
            for block_item in msg.block_list:
                logger.info("%s", block_item.text)


def test_train_data(model_path=None, max_input_tokens:int=2560, max_context_tokens:int=768, max_output_token:int=1536):
    """训练集

    Args:
        model_path (str, optional): 模型目录路径
        max_input_tokens (int, optional): 模型输入的最大token长度, 包含system+plugins+chat_input(prev_chat+user+quote+call+call_result)
        max_context_tokens (int, optional): 模型上下文的最大token长度, 包含system+plugins
        max_output_token (int, optional): 模型输出的最大token长度
    """
    tokenizer = get_tokenizer(model_path=(model_path or test_config.default_llama_model_path))
    for i, chat_data in enumerate(mock_chat_data):
        if i: logger.info("")
        logger.info('chat_data: %s', chat_data)
        encoder = ChatTextEncoder()
        train_data = encoder.train_data(chat_data)
        j = 0
        for input_dialog, output_msg in train_data:
            j += 1 # 模型第几次调用, 从1开始
            round_idx = output_msg.round_idx # 第几轮对话, 从0开始
            input_text = input_dialog.to_text()
            # output_text = output_msg.to_text()
            output_text = encoder.encode_output_msg(output_msg, fixed_block_struct=True)
            logger.info('\n---Model Input %s-%s---\n%s---Model Output---\n%s', round_idx, j, input_text, output_text)
            # encode input
            chat_tokenizer = ChatTokenizer(dialog=input_dialog, tokenizer=tokenizer)
            ctx_tokens, chat_tokens = chat_tokenizer.truncate_tokens(
                max_input_tokens=max_input_tokens,
                max_context_tokens=max_context_tokens
            )
            tokens = ctx_tokens + chat_tokens
            logger.info('input tokens %s: %s', len(tokens), tokens)
            de_text = tokenizer.decode(tokens)
            logger.info('input tokens decode: %s', de_text)
            # encode output
            output_tokens = tokenizer.encode(output_text, add_special_tokens=False)
            ori_output_tokens_len = len(output_tokens)
            if len(output_tokens) > max_output_token:
                output_tokens = output_tokens[:max_output_token]
            logger.info("output_tokens len: %s -> %s", ori_output_tokens_len, len(output_tokens))
            de_output_text = tokenizer.decode(output_tokens)
            logger.info("output truncated: %s", de_output_text)

def test_predict_input(model_path=None, max_input_tokens:int=2048, max_context_tokens:int=1024):
    tokenizer = get_tokenizer(model_path=(model_path or test_config.default_llama_model_path))
    for i, chat_data in enumerate(mock_chat_data):
        if i: logger.info("")
        logger.info('chat_data: %s', chat_data)
        encoder = ChatTextEncoder()
        dialog = encoder.predict_input(chat_data)
        input_text = dialog.to_text()
        logger.info('Predict Input: %s', input_text)
        chat_tokenizer = ChatTokenizer(dialog=dialog, tokenizer=tokenizer)
        ctx_tokens, chat_tokens = chat_tokenizer.truncate_tokens(
            max_input_tokens=max_input_tokens, 
            max_context_tokens=max_context_tokens
        )
        tokens = ctx_tokens + chat_tokens
        logger.info('tokens %s: %s', len(tokens), tokens)
        de_text = tokenizer.decode(tokens)
        logger.info('de_text: %s', de_text)


if __name__ == "__main__":
    _d, component = dict(globals()).items(), {}
    for k, v in _d:
        if k.startswith('test_'):
            component[k] = v
            component[k[5:]] = v

    import fire
    fire.Fire(component)