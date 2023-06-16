import time, torch

from .hf_task import HFModelTask, auto_load, logger
from transformers import AutoModelForCausalLM, AutoTokenizer

from smart.utils.dict import dict_find
from . import utils
from llm_model.alpaca.instruct_util import instruct_encode


@auto_load.task('llm_model.hf_instruct_job')
class HFInstructJobTask(HFModelTask):
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

    def generate(self, model=None, input_tokens_key='tokens', output_tokens_key='pred_tokens', 
                max_tokens:int=None, max_new_tokens:int=None, output_full_text:bool=False):
        pipeline_opt = pipeline_opt or {}
        # generator = pipeline('text-generation', model=model, tokenizer=tokenizer, **pipeline_opt)
        generator_opt = {}

        for i, item in enumerate(self.recv_data()):
            input_tokens = item.get(input_tokens_key)
            if not input_tokens:
                logger.warning('hf_instruct_job null item %s', item)
                self.send_data(item)
                continue

            _max_new_tokens = dict_find(item, ('pred_opt', 'max_new_tokens'), default_val=max_new_tokens)
            _max_tokens = dict_find(item, ('pred_opt', 'max_tokens'), default_val=max_tokens)
            if _max_new_tokens is None or _max_new_tokens <= 0:
                _max_new_tokens = _max_tokens - len(input_tokens) - 1

            _generator_opt = dict(generator_opt)
            if _max_new_tokens: _generator_opt['max_new_tokens'] = _max_new_tokens

            logger.debug('hf_text_generation %s input: %s', i, prompt_input)
            pred_rst = generator(prompt_input, **_generator_opt)
            logger.debug('hf_text_generation %s output: %s', i, pred_rst)
            output_tokens = dict_find(pred_rst, (0, 'generated_token_ids'))
            output_text = dict_find(pred_rst, (0, 'generated_text'))
            if output_text is not None:
                if not output_full_text and prompt_input:
                    output_text = output_text[len(prompt_input):]
                item[output_text_key] = output_text
            else:
                if not output_full_text and input_tokens:
                    output_tokens = output_tokens[len(input_tokens):]
                item[output_tokens_key] = output_tokens
            self.send_data(item)
    
    def encode_instruct(self, input_text_key='text', instruct_version:int=None, item_iter=None, item_iter_fn=None):
        _item_iter = item_iter or (item_iter_fn or self.recv_data)()
        def _out_item_iter_fn():
            for i, item in enumerate(_item_iter):
                _instruction, _system, _input = \
                    item.get('instruction'), item.get('system'), item.get('input'),
                _text = instruct_encode({
                    'instruction': _instruction,
                    'system': _system,
                    'input': _input
                }, version=instruct_version)
                item[input_text_key] = _text
                yield item
        return {
            'item_iter': _out_item_iter_fn(),
            'item_iter_fn': _out_item_iter_fn
        }