# python -m tests.utils.instruct_util parse
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
    for obj, opt, expected in data:
        instruct_item = parse_instruct_obj(obj, opt=opt)
        logger.info('parse %s opt%s => %s', data, opt, instruct_item)
        if expected:
            assert expected == instruct_item


if __name__ == "__main__":
    _d, component = dict(globals()).items(), {}
    for k, v in _d:
        if k.startswith('test_'):
            component[k] = v
            component[k[5:]] = v

    import fire
    fire.Fire(component)