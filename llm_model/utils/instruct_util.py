from llm_model.utils.obj_util import dict_multi_get
from smart.utils.dict import dict_find
from typing import NamedTuple


class InstructItem(NamedTuple):
    system:str
    instruction:str
    quote:str = None


class InstructUtil:
    def parse_instruct_obj(self, item:dict, opt=None):
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
        system = dict_multi_get(obj, opt.get('system_key', 'system'))
        instruction = dict_multi_get(obj, opt.get('instruction_key', ('user', 'instruction', 'ask')))
        quote = dict_multi_get(obj, opt.get('quote_key', ('quote', 'input')))
        return InstructItem(system, instruction, quote=quote)

    def encode(self, instruct, version=None, encode_opt:dict=None):
        """encode instruct

        Args:
            instruct (dict|InstructItem): instruct obj
            version (int, optional): encode version. Defaults to None.
            encode_opt (dict, optional): encode opt. Defaults to None.

        Returns:
            _type_: _description_
        """
        if version is None: version = (encode_opt or {}).get('version', 2)
        if isinstance(instruct, InstructItem):
            item = instruct
        else:
            item = self.parse_instruct_obj(instruct, opt=encode_opt)
        if version == 1:
            return self._encode_v1(item, encode_opt=encode_opt)
        else:
            return self._encode_v2(item, encode_opt=encode_opt)

    def _encode_v1(self, instruct:InstructItem, encode_opt:dict=None):
        quote_empty_vals = encode_opt.get('quote_empty_vals', ('<noinput>', '无', '空'))
        _quote = instruct.quote
        if _quote in quote_empty_vals: _quote = ''
        _system = instruct.system or ''
        _instruction = instruct.instruction or ''
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

    def _encode_v2(self, instruct:InstructItem, encode_opt:dict=None):
        quote_empty_vals = encode_opt.get('quote_empty_vals', ('<noinput>', '无', '空'))
        _quote = instruct.quote
        if _quote in quote_empty_vals: _quote = ''
        _system = instruct.system or ''
        _instruction = instruct.instruction or ''
        prompt = ''
        if _system:
            prompt += _system + "\n"
        user_block_name = encode_opt.get('user_block_name', 'User')
        prompt += '<!' + user_block_name + '>:\n' + _instruction + '\n'
        if _quote:
            quote_block_name = encode_opt.get('quote_block_name', 'Input')
            prompt += '<!' + quote_block_name + '>:\n' + _quote + '\n'
        prompt += '<!Machine>:\n'
        return prompt


