# python -m tests.utils.instruct_util parse
# python -m tests.utils.instruct_util encode
from llm_model.utils.instruct_util import *
from tests import logger

def test_parse():
    data = [
        (
            {
                'system': 'system prompt',
                'user': '指令问句',
                'quote': 'quote content'
            },
            {},
            InstructItem(system='system prompt', instruction='指令问句', quote='quote content')
        ),
        (
            {
                'instruct': {
                    'system': 'system prompt',
                    'ask': '指令问句',
                    'input': 'quote content'
                }
            },
            {
                'sub_keys': ['instruct'],
                'instruction': ('ask', 'question'),
                'quote': 'input'
            },
            InstructItem(system='system prompt', instruction='指令问句', quote='quote content')
        )
    ]
    util = InstructUtil()
    for obj, opt, expected in data:
        instruct_item = util.parse_instruct_obj(obj, opt=opt)
        logger.info('parse %s opt%s => %s', data, opt, instruct_item)
        if expected:
            assert expected == instruct_item


def test_encode():
    data = [
        (
            { # instruct obj
                'system': 'system prompt',
                'user': '指令问句',
                'quote': 'quote content'
            },
            { # encode opt
                'quote_key': 'quote',
                'instruction_key': ('instruction', 'user'),
                'user_block_name': 'Human',
                'quote_block_name': 'Quote'
            },
            ( # expected result, None表示不断言
                None,
                None 
            )
        ),
        (
            {'question': '中国的首都', 'quote': '', '_send_text': 'api.ov-nlg:920d6ee2ab2444e3979c5fcd62592f0c'},
            {
                "instruct_pattern": "{system}{instruction}\n- 输出:",
                "quote_pattern": "{system}{instruction}\n\n- 输入:\n{quote}\n\n- 输出:",
                "system_key": "system",
                "instruction_key": ['ask', 'question', 'user'],
                "quote_key": "quote"
            },
            (None, None)
        )
    ]
    util = InstructUtil()
    for obj, opt, expected in data:
        logger.info('obj=%s, opt=%s', obj, opt)
        for version in (1, 2):
            text = util.encode(obj, version=version, encode_opt=opt)
            logger.info('prompt(ver=%s): %s', version, text)
            expected_result = expected[version-1] if expected else None
            if expected_result:
                assert expected_result == text
        logger.info("")


if __name__ == "__main__":
    _d, component = dict(globals()).items(), {}
    for k, v in _d:
        if k.startswith('test_'):
            component[k] = v
            component[k[5:]] = v

    import fire
    fire.Fire(component)