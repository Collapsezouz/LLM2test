import time, torch

from .hf_task import HFModelTask, auto_load, logger
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.streamers import BaseStreamer
from transformers import pipeline
from llm_model.utils.chat_util import ChatTextEncoder
from llm_model.utils.instruct_util import InstructUtil
from smart.utils.dict import dict_find
from . import utils

class TokenIteratorStreamer(BaseStreamer):
    def put(self, value):
        is_input_ids = isinstance(value.tolist()[0], list)
        if not is_input_ids:
            logger.debug('TokenIteratorStreamer put %s', value.tolist())

    def end(self):
        logger.debug('TokenIteratorStreamer end')


@auto_load.task('llm_model.hf_chat_generation')
class HFChatGenerationTask(HFModelTask):
    def load_model(self, model_name=None, model_path=None, enable_bfloat16:bool=True, use_fast:bool=False, model_kwargs=None):
        if model_path is None:
            model_opts = self.init_model(model_name=model_name, model_path=model_path)
            model_path = model_opts.get('model_path')
        assert model_path, 'model_path is None'
        begin_ts = time.monotonic()
        model_kwargs = model_kwargs or {}
        if enable_bfloat16:
            model_kwargs['torch_dtype'] = torch.bfloat16 # torch.float16
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            **model_kwargs,
            local_files_only=True
        )
        ts_1 = time.monotonic()
        logger.info("loaded hf model cost %s seconds.", ts_1 - begin_ts)
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            use_fast=use_fast,
            local_files_only=True)
        ts_2 = time.monotonic()
        logger.info("loaded hf tokenizer cost %s seconds.", ts_2 - ts_1)
        return {
            'tokenizer': tokenizer,
            'model': hf_model
        }

    def generate(self, model=None, tokenizer=None, dialog_key='dialog', instruct_opt={},
                output_tokens_key='pred_tokens', output_text_key='pred_text',
                max_tokens:int=None, max_new_tokens:int=None, output_full_text:bool=False, pipeline_opt:dict=None, streamer=None):
        pipeline_opt = pipeline_opt or {}
        generator = pipeline('text-generation', model=model, tokenizer=tokenizer, **pipeline_opt)
        generator_opt = {}
        if streamer == 'text':
            from transformers.generation.streamers import TextStreamer
            generator_opt['streamer'] = TextStreamer(tokenizer)

        for i, item in enumerate(self.recv_data()):
            input_chat_obj = item.get(dialog_key)
            _max_new_tokens = dict_find(item, ('pred_opt', 'max_new_tokens'), default_val=max_new_tokens)
            _max_tokens = dict_find(item, ('pred_opt', 'max_tokens'), default_val=max_tokens)
            chat_mode, input_text, input_tokens = False, None, None
            # input_tokens = item.get(input_tokens_key)
            if input_chat_obj:
                chat_mode = True
                chat_encoder = ChatTextEncoder()
                chat_dialog = chat_encoder.predict_input(input_chat_obj)
                input_text = chat_dialog.to_text()
            else:
                instruct_util = InstructUtil()
                instruct_item = instruct_util.parse_instruct_obj(item, opt=instruct_opt)
                input_text = instruct_util.encode(instruct_item, encode_opt=instruct_opt)

            _generator_opt = dict(generator_opt)
            _generator_opt.update(item.get('pred_opt') or {})
            if _max_new_tokens: _generator_opt['max_new_tokens'] = _max_new_tokens
            # if _max_tokens: _generator_opt['max_tokens'] = _max_tokens
            if not input_text:
                logger.warning('hf_chat_generation null item %s', item)
                self.send_data(item)
                continue
            logger.debug('hf_chat_generation %s input: %s', i, input_text)
            pred_rst = generator(input_text, **_generator_opt)
            logger.debug('hf_chat_generation %s output: %s', i, pred_rst)
            output_tokens = dict_find(pred_rst, (0, 'generated_token_ids'))
            output_text = dict_find(pred_rst, (0, 'generated_text'))
            if output_text is not None:
                if not output_full_text and input_text:
                    output_text = output_text[len(input_text):]
                item[output_text_key] = output_text
            else:
                if not output_full_text and input_tokens:
                    output_tokens = output_tokens[len(input_tokens):]
                item[output_tokens_key] = output_tokens
            self.send_data(item)