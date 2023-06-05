import json, typing
from enum import Enum, auto
from smart.utils.yaml import yaml_dumps
from smart.utils.list import list_safe_iter
# from collections import namedtuple
from typing import NamedTuple


def dict_multi_get(obj:dict, keys:list, default_val=None):
    if not obj: return default_val
    for key in list_safe_iter(keys):
        if key in obj:
            return obj[key]
    return default_val


class ChatBlockKey(Enum):
    system = auto() # 系统prompt
    plugins = auto() # 对话启用的插件
    user = auto() # 用户输入的内容
    quote = auto() # 引用内容
    call = auto() # 调用插件
    call_result = auto() # 插件返回结果
    thought = auto() # 思维链
    machine = auto() # 机器生成的文本回答

    def __eq__(self, __value: object) -> bool:
        return self.name == (__value.name if isinstance(__value, ChatBlockKey) else __value)
    
    def __hash__(self) -> int:
        return self.name.__hash__()


# ChatBlockItem = namedtuple('ChatBlockItem', ['key', 'text', 'value'])
class ChatBlockItem(NamedTuple):
    key:ChatBlockKey
    text:str
    value:typing.Any

class ChatRoundData:
    Type_System = 'system'
    Type_Chat = 'chat'

    def __init__(self, block_list, round_idx:int=-1, type=None) -> None:
        self.block_list:typing.List[ChatBlockItem] = block_list
        self.round_idx = round_idx
        self.type = type or self.Type_Chat
    
    def get_text_list(self):
        return [
            block.text
            for block in (self.block_list or [])
        ]


class ChatTextEncoder:
    Default_Block_Tag_Map = {
        'system': 'System', # 系统prompt
        'plugins': 'plugins', # 对话启用的插件
        'user': 'User', # 用户输入的内容
        'quote': 'Quote', # 引用内容
        'call': 'Call', # 调用插件
        'call_result': 'Response', # 插件返回结果
        'thought': 'thought', # 思维链
        'machine': 'Machine', # 机器生成的文本回答
    }
    Block_key_Alias_Map = {
        'instruct': ChatBlockKey.user,
        'ask': ChatBlockKey.user,
        'human': ChatBlockKey.user,
        'input': ChatBlockKey.quote,
        'output': ChatBlockKey.machine,
        'assistant': ChatBlockKey.machine,
    }

    def __init__(self, block_tag_map:dict=None) -> None:
        self._block_tag_map = self.Default_Block_Tag_Map if block_tag_map is None else block_tag_map
        self._cfg_plugins_format = 'json'
        self._chat_input_keys_list = [
            (('user', 'ask', 'instruct'), ''), # keys, default_val
            (('quote',), None)
        ]
    
    def _value_encode(self, value, format=None):
        if isinstance(value, str):
            return value
        
        if format == 'json':
            return json.dumps(value, ensure_ascii=False)
        elif format in ('yaml', 'yml'):
            return yaml_dumps(value)
        elif format == 'text_list':
            return "\n".join((str(line) for line in value))
        else:
            return str(value)
    
    def cast_block_key(self, key, default_val=None):
        if key in self.Block_key_Alias_Map:
            key = self.Block_key_Alias_Map[key]
        if isinstance(key, ChatBlockKey): 
            return key
        return getattr(ChatBlockKey, key, default_val)

    def block_key2tag(self, key):
        if isinstance(key, ChatBlockKey): key = key.name
        return self._block_tag_map.get(key, key)

    def block_prefix(self, key):
        return '<!' + self.block_key2tag(key) + '>:\n'
    
    def block_end(self, key):
        return "\n"
        # return "<|eob|>\n"
    
    def block_encode(self, key, value):
        if key in ('plugins', ChatBlockKey.plugins):
            return self.encode_plugins(value, only_val=False)
        _text = self.block_prefix(key)
        _text += self._value_encode(value)
        _text += self.block_end(key)
        return _text
    
    def encode_plugins(self, plugins, only_val=False):
        if plugins is None:
            return ''
        _text = '' if only_val else self.block_prefix(ChatBlockKey.plugins)
        _text += self._value_encode(plugins, format=self._cfg_plugins_format)
        if not only_val: _text += self.block_end(ChatBlockKey.plugins)
        return _text
    
    def parse_chat_obj(self, chat_obj:dict, reverse_chat:bool=False, no_system:bool=False):
        if not chat_obj:
            return []
        
        if not no_system:
            system_block_list = []
            for key in ('system', 'plugins'):
                value = chat_obj.get(key)
                if not value:
                    continue
                block_key = self.cast_block_key(key, default_val=key)
                block_text = self.block_encode(block_key, value)
                block = ChatBlockItem(block_key, block_text, value)
                system_block_list.append(block)
            yield ChatRoundData(system_block_list, type=ChatRoundData.Type_System)
        
        chat_list = chat_obj.get('chat') or []
        if reverse_chat: chat_list = reversed(chat_list)
        round_size = list(chat_list)
        for idx, chat_item in enumerate(chat_list):
            round_idx = round_size-idx-1 if reverse_chat else idx
            block_list = []
            for keys, default_val in self._chat_input_keys_list:
                key = keys[0]
                value = dict_multi_get(chat_item, keys, default_val=default_val)
                if value is None:
                    continue
                block_key = self.cast_block_key(key, default_val=key)
                block_text = self.block_encode(block_key, value)
                block = ChatBlockItem(block_key, block_text, value)
                block_list.append(block)
            outputs_list = chat_item.get('output') or []
            for outputs in outputs_list:
                has_call = False
                for output in outputs:
                    if 'call' in output:
                        # 插件调用
                        block_key = ChatBlockKey.call
                        value = output.get('call')
                        has_call = True
                    elif 'thought' in output:
                        # 思维链
                        block_key = ChatBlockKey.thought
                        value = output.get('thought')
                    else:
                        # 机器生成文本回答
                        block_key = ChatBlockKey.machine
                        value = output.get('text')
                    block_text = self.block_encode(block_key, value)
                    block = ChatBlockItem(block_key, block_text, value)
                    block_list.append(block)
                if has_call:
                    for output in outputs:
                        # 插件返回结果
                        if 'call' not in output: continue
                        block_key = ChatBlockKey.call_result
                        value = dict_multi_get(output, ('text', 'call_result'), default_val='')
                        block_text = self.block_encode(block_key, value)
                        block = ChatBlockItem(block_key, block_text, value)
                        block_list.append(block)
            yield ChatRoundData(block_list, round_idx=round_idx, type=ChatRoundData.Type_Chat)

    def predict_input(self, chat_obj):
        pass

    def __input_output_idx_tuple_iter(self, is_output_iter):
        output_pos_begin = None
        for i, is_output in enumerate(is_output_iter):
            if is_output:
                if output_pos_begin is None:
                    output_pos_begin = i
                continue
            if output_pos_begin is not None:
                yield output_pos_begin, i
                output_pos_begin = None
        if output_pos_begin is not None:
            yield output_pos_begin, i+1

    def train_data(self, chat_obj):
        round_data_iter = self.parse_chat_obj(chat_obj)
        prev_round_data_list:typing.List[ChatRoundData] = []
        for round_data in round_data_iter:
            if round_data.type == ChatRoundData.Type_System:
                prev_round_data_list.append(round_data)
                continue
            input_output_idx_tuple_iter = self.__input_output_idx_tuple_iter((
                block.key in (ChatBlockKey.call, ChatBlockKey.machine, ChatBlockKey.thought)
                for block in round_data.block_list
            ))
            for input_idx, output_idx in input_output_idx_tuple_iter:
                input_block_list = round_data.block_list[:input_idx]
                output_block_list = round_data.block_list[input_idx:output_idx]
                input_data = ChatRoundData(
                    block_list=input_block_list, 
                    round_idx=round_data.round_idx, 
                    type=round_data.type)
                output_data = ChatRoundData(
                    block_list=output_block_list, 
                    round_idx=round_data.round_idx, 
                    type=round_data.type)
                yield [*prev_round_data_list, input_data], output_data
            prev_round_data_list.append(round_data)


def chat_input():
    pass