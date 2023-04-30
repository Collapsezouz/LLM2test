# python -m tests.alpaca.instruct_data instruct_encode
from tests import logger

def _instruct_encode(instruct:dict):
    _input = instruct.get("input", "")
    if _input in ('<noinput>',): _input = ''
    _system, _instruction, _output = \
        instruct.get("system", ""), instruct.get("instruction", ""), instruct.get("output", "")
    prompt = ''
    if _system:
        prompt += _system + "\n"
    prompt += "<!User>:\n" + _instruction + "\n"
    if _input:
        prompt += '<!Input>:\n' + _input + "\n"
    prompt += '<!Machine>:\n'
    return prompt

def _get_instruct_list():
    return [
        {
            "system": "你是信息抽取机器人。输入的内容是从PDF提取的结果, 用<span x=? y=?>?</span>表示PDF的一个文字块，x为横坐标, y为纵坐标。",
            "instruction": "提取出所有融资租赁公司的名称, 按列表形式返回",
            "input": "<page num=75>\n<span x=242 y=632>名称：公司A</span>\n<span x=242 y=674>名称：公司B</span>\n</page>",
            "output": "* 公司A\n* 公司B",
        },
        {
            "instruction": "[指令内容]",
            "output": "[输出内容]"
        },
        {
            "instruction": "[指令内容]",
            "input": "[输入内容]",
            "output": "[输出内容]"
        }
    ]


def test_instruct_encode():
    _list = _get_instruct_list()
    for i, instruct in enumerate(_list):
        logger.info('%s %s', i, instruct)
        prompt = _instruct_encode(instruct)
        logger.info("prompt: %s\n", prompt)


if __name__ == "__main__":
    _d, component = dict(globals()).items(), {}
    for k, v in _d:
        if k.startswith('test_'):
            component[k] = v
            component[k[5:]] = v

    import fire
    fire.Fire(component)