import time, torch

from .hf_task import HFModelTask, auto_load, logger
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.streamers import BaseStreamer
from transformers import pipeline

from smart.utils.dict import dict_find

class TokenIteratorStreamer(BaseStreamer):
    def put(self, value):
        is_input_ids = isinstance(value.tolist()[0], list)
        if not is_input_ids:
            logger.debug('TokenIteratorStreamer put %s', value.tolist())

    def end(self):
        logger.debug('TokenIteratorStreamer end')


@auto_load.task('llm_model.hf_text_generation')
class HFTextGenerationTask(HFModelTask):
    def load_model(self, model_name=None, model_path=None, enable_bfloat16:bool=True, model_kwargs=None):
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
        logger.info("loaded hf_model cost %s seconds.", time.monotonic() - begin_ts)
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        return {
            'tokenizer': tokenizer,
            'model': hf_model
        }

    def generate(self, model=None, tokenizer=None, prompt_pattern=None, input_text_key='text', input_tokens_key='tokens', output_tokens_key='pred_tokens', output_text_key='pred_text',
                max_tokens:int=None, max_new_tokens:int=None, output_full_text:bool=False, pipeline_opt:dict=None, streamer=None):
        pipeline_opt = pipeline_opt or {}
        generator = pipeline('text-generation', model=model, tokenizer=tokenizer, **pipeline_opt)
        generator_opt = {}
        if streamer == 'text':
            from transformers.generation.streamers import TextStreamer
            generator_opt['streamer'] = TextStreamer(tokenizer)

        for i, item in enumerate(self.recv_data()):
            input_tokens = item.get(input_tokens_key)
            input_text = item.get(input_text_key)
            _max_new_tokens = dict_find(item, ('pred_opt', 'max_new_tokens'), default_val=max_new_tokens)
            _max_tokens = dict_find(item, ('pred_opt', 'max_tokens'), default_val=max_tokens)
            _generator_opt = dict(generator_opt)
            if _max_new_tokens: _generator_opt['max_new_tokens'] = _max_new_tokens
            # if _max_tokens: _generator_opt['max_tokens'] = _max_tokens
            if not input_text:
                logger.warning('hf_text_generation null item %s', item)
                self.send_data(item)
                continue
            if prompt_pattern:
                prompt_input = prompt_pattern.format(input=input_text)
            else:
                prompt_input = input_text
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