import json, typing
from enum import Enum, auto
from transformers import PreTrainedTokenizerBase
from smart.utils.yaml import yaml_dumps
from smart.utils.list import list_safe_iter
# from collections import namedtuple
from llm_model.utils.obj_util import items_remove_tail, dict_multi_get
# from typing import Any, NamedTuple



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
class ChatBlockItem:
    def __init__(self, key:ChatBlockKey, text:str, value=None, tokens:list=None) -> None:
        self.key = key
        self.text = text
        self.value = value
        self.tokens = tokens

class ChatMessage:
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
    
    def to_text(self):
        return ''.join(self.get_text_list())

class ChatDialog:
    def __init__(self, context:list = None, chat:list=None) -> None:
        self.context:typing.List[ChatMessage] = context or []
        self.chat:typing.List[ChatMessage]  = chat or []
    
    def to_messages(self):
        return self.context + self.chat
    
    def context_text(self):
        return ''.join([
            msg.to_text()
            for msg in self.context
        ])
    
    def chat_text(self, start:int=0, end:int=None):
        return ''.join([
            msg.to_text()
            for msg in self.chat[start:end]
        ])
    
    def to_text(self):
        return self.context_text() + self.chat_text()

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
        if not value:
            return ''
        elif isinstance(value, str):
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
        if isinstance(key, ChatBlockKey): 
            return key
        if key in self.Block_key_Alias_Map:
            key = self.Block_key_Alias_Map[key]
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
    
    def block_list_encode(self, block_list:typing.List[ChatBlockItem], merge_val:bool=True):
        if not merge_val:
            return ''.join((
                self.block_encode(block.key, block.value)
                for block in block_list
            ))
        else:
            if not block_list:
                return ''
            prev_key = None
            _text = ''
            for block in block_list:
                if block.key != prev_key:
                    _text += self.block_prefix(block.key)
                _text += self._value_encode(block.value)
                _text += self.block_end(block.key)
                prev_key = block.key
            return _text
    
    def encode_plugins(self, plugins, only_val=False):
        if plugins is None:
            return ''
        _text = '' if only_val else self.block_prefix(ChatBlockKey.plugins)
        _text += self._value_encode(plugins, format=self._cfg_plugins_format)
        if not only_val: _text += self.block_end(ChatBlockKey.plugins)
        return _text
    
    def is_output_block_key(self, key):
        """是否是输出类型的block key

        Args:
            key (ChatBlockKey|str): block key

        Returns:
            bool: true代表output类型的block key, false代表input类型
        """
        if not isinstance(key, ChatBlockKey): key = self.cast_block_key(key)
        return key in (ChatBlockKey.call, ChatBlockKey.machine, ChatBlockKey.thought)
    
    def is_output_block(self, block:ChatBlockItem):
        return self.is_output_block_key(block.key)
    
    def parse_chat_obj(self, chat_obj:dict, reverse_chat:bool=False, no_system:bool=False):
        if not chat_obj:
            yield from []
            return
        
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
            yield ChatMessage(system_block_list, type=ChatMessage.Type_System)
        
        chat_list = chat_obj.get('chat') or []
        if reverse_chat: chat_list = list(reversed(chat_list))
        round_size = len(chat_list)
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
            yield ChatMessage(block_list, round_idx=round_idx, type=ChatMessage.Type_Chat)

    def predict_input(self, chat_obj):
        msg_iter = self.parse_chat_obj(chat_obj, reverse_chat=True)
        context_list:typing.List[ChatMessage] = []
        chat_list:typing.List[ChatMessage] = []
        is_last_round = True
        for msg in msg_iter:
            if msg.type == ChatMessage.Type_System:
                context_list.append(msg)
                continue
            if is_last_round:
                input_block_list = items_remove_tail(msg.block_list, self.is_output_block)
                message = ChatMessage(
                    block_list=input_block_list,
                    round_idx=msg.round_idx,
                    type=msg.type
                )
                chat_list.insert(0, message)
                is_last_round = False
            else:
                chat_list.insert(0, msg)
        return ChatDialog(
            context=context_list,
            chat=chat_list
        )

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
        msg_iter = self.parse_chat_obj(chat_obj)
        context_list:typing.List[ChatMessage] = []
        prev_chat_list:typing.List[ChatMessage] = []
        for msg in msg_iter:
            if msg.type == ChatMessage.Type_System:
                context_list.append(msg)
                continue
            input_output_idx_tuple_iter = self.__input_output_idx_tuple_iter((
                self.is_output_block_key(block.key)
                for block in msg.block_list
            ))
            for input_idx, output_idx in input_output_idx_tuple_iter:
                input_block_list = msg.block_list[:input_idx]
                output_block_list = msg.block_list[input_idx:output_idx]
                last_msg = ChatMessage(
                    block_list=input_block_list, 
                    round_idx=msg.round_idx, 
                    type=msg.type)
                output_msg = ChatMessage(
                    block_list=output_block_list, 
                    round_idx=msg.round_idx, 
                    type=msg.type)
                input_dialog = ChatDialog(
                    context=context_list,
                    chat=[*prev_chat_list, last_msg]
                )
                yield input_dialog, output_msg
            prev_chat_list.append(msg)

    def encode_output_msg(self, output:ChatMessage, fixed_block_struct=True):
        """output_msg转换成文本序列

        Args:
            output (ChatMessage): 模型输出
            fixed_block_struct (bool, optional): 是否固定输出模版结构. Defaults to True.
        """
        if not output:
            return ''
        if not fixed_block_struct:
            return output.to_text()
        group_list = {}
        for block in output.block_list:
            if block.key not in group_list:
                group_list[block.key] = []
            group_list[block.key].append(block)
        output_text = ''
        for key in (ChatBlockKey.thought, ChatBlockKey.call, ChatBlockKey.machine):
            block_list = group_list.get(key)
            if block_list and len(block_list):
                output_text += self.block_list_encode(block_list=block_list, merge_val=True)
            else:
                output_text += self.block_encode(key, '')
        return output_text

    
    def parse_output(self, output_text):
        pass


class ChatTokenizer:
    Default_Max_Context_Tokens_Ratio = 0.5

    def __init__(self, dialog:ChatDialog, tokenizer:PreTrainedTokenizerBase) -> None:
        self.dialog = dialog
        self.tokenizer = tokenizer
        self._default_max_context_tokens_ratio = self.Default_Max_Context_Tokens_Ratio

    def get_context_tokens(self, max_context_tokens:int=None, add_bos_token:bool=True):
        dialog, tokenizer = self.dialog, self.tokenizer
        if not max_context_tokens:
            text = dialog.context_text()
            return tokenizer.encode(text)
        context_tokens = []
        for i, msg in enumerate(dialog.context):
            for j, block in enumerate(msg.block_list):
                if block.tokens is None:
                    is_first_block = (i == 0) and (j == 0)
                    text = block.text
                    if text:
                        tokens = tokenizer.encode(
                            text, add_special_tokens=(is_first_block and add_bos_token))
                    else:
                        tokens = []
                    block.tokens = tokens
                if block.tokens:
                    context_tokens.extend(block.tokens)
                if max_context_tokens and len(context_tokens) >= max_context_tokens: break
            if max_context_tokens and len(context_tokens) >= max_context_tokens: break
        if max_context_tokens and len(context_tokens) > max_context_tokens:
            context_tokens = context_tokens[:max_context_tokens]
        elif not context_tokens:
            # add bos_token_id if context is empty
            context_tokens = tokenizer.encode('')
        return context_tokens
    
    def get_chat_tokens(self, max_chat_tokens:int=None, add_eos_token:bool=False):
        dialog, tokenizer = self.dialog, self.tokenizer
        if not max_chat_tokens:
            text = dialog.chat_text()
            return tokenizer.encode(text, add_special_tokens=False)
        r_chat_tokens = [] # reversed chat tokens
        if add_eos_token:
            r_chat_tokens.insert(0, tokenizer.eos_token_id)
        for msg in reversed(dialog.chat):
            for block in reversed(msg.block_list):
                if block.tokens is None:
                    text = block.text
                    if text:
                        tokens = tokenizer.encode(text, add_special_tokens=False)
                        block.tokens = tokens
                if block.tokens:
                    r_chat_tokens.extend(reversed(block.tokens))
                if len(r_chat_tokens) >= max_chat_tokens: break
            if len(r_chat_tokens) >= max_chat_tokens: break
        
        chat_tokens = list(reversed(r_chat_tokens[:max_chat_tokens]))
        return chat_tokens
    
    def calc_max_chat_tokens(self, max_input_tokens:int=None, context_tokens_len:int=None):
        if not max_input_tokens:
            return None
        if context_tokens_len:
            return max_input_tokens - context_tokens_len

    def truncate_tokens(self, max_input_tokens:int=None, max_context_tokens:int=None
                        , add_eos_token:bool=False, add_bos_token:bool=True):
        if not max_input_tokens or max_input_tokens < 0:
            return self.get_context_tokens(), self.get_chat_tokens()

        if not max_context_tokens:
            max_context_tokens = int(max_input_tokens * self._default_max_context_tokens_ratio)
        assert max_input_tokens >= max_context_tokens

        context_tokens = self.get_context_tokens(max_context_tokens, add_bos_token=add_bos_token)
        max_chat_tokens = self.calc_max_chat_tokens(
            max_input_tokens=max_input_tokens,
            context_tokens_len=len(context_tokens)
        )
        chat_tokens = self.get_chat_tokens(max_chat_tokens, add_eos_token=add_eos_token)
        return context_tokens, chat_tokens