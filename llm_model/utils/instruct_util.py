from llm_model.utils.obj_util import dict_multi_get
from smart.utils.dict import dict_find
from typing import Any, NamedTuple


class InstructItem(NamedTuple):
    system:str
    instruction:str
    quote:str = None


def instruct_encode(instruct:dict, version=None, encode_opt:dict=None):
    if version is None: version = (encode_opt or {}).get('version', 2)
    if version == 1:
        return instruct_encode_v1(instruct, encode_opt=encode_opt)
    else:
        return instruct_encode_v2(instruct, encode_opt=encode_opt)


def instruct_encode_v1(instruct:dict, encode_opt:dict=None):
    _quote = dict_multi_get(instruct, ('input', 'quote'), default_val='')
    quote_empty_vals = encode_opt.get('quote_empty_vals', ('<noinput>', '无', '空'))
    if _quote in quote_empty_vals: _quote = ''
    _system = instruct.get("system", "")
    _instruction = dict_multi_get(instruct, ('instruction', 'ask', 'user'), '')
    system_prompt = _system + "\n" if _system else ''
    if _quote:
        pattern = (encode_opt or {}).get(
            'quote_pattern', "{system}{instruction}\n\n- 输入:\n{quote}\n\n- 输出:")
        return pattern.format(
            system=system_prompt, 
            instruction=_instruction,
            quote=_quote
        )
    else:
        pattern = (encode_opt or {}).get(
            'instruct_pattern', "{system}{instruction}\n\n- 输出:")
        return pattern.format(
            system=system_prompt,
            instruction=_instruction
        )


def instruct_encode_v2(instruct:dict, encode_opt:dict=None):
    _input = instruct.get("input", "")
    if _input in ('<noinput>',): _input = ''
    _system, _instruction = instruct.get("system", ""), instruct.get("instruction", "")
    prompt = ''
    if _system:
        prompt += _system + "\n"
    prompt += "<!User>:\n" + _instruction + "\n"
    if _input:
        prompt += '<!Input>:\n' + _input + "\n"
    prompt += '<!Machine>:\n'
    return prompt


def parse_instruct_obj(item:dict, opt=None):
    """解析指令对象

    Args:
        item (dict): instruct obj
        instruct_key (dict, optional): 指令key. Defaults to None.
    """
    opt = opt or {}
    sub_keys = opt.get('sub_keys')
    if sub_keys:
        obj = dict_find(item, sub_keys)
    else:
        obj = item
    if not all((obj, isinstance(obj, dict))):
        return None
    system = dict_multi_get(obj, opt.get('system', 'system'))
    instruction = dict_multi_get(obj, opt.get('instruction', ('user', 'instruction')))
    quote = dict_multi_get(obj, opt.get('quote', 'quote'))
    return InstructItem(system, instruction, quote=quote)
