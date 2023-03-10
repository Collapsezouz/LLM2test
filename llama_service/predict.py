import torch
from smart.auto.tree import TreeMultiTask
from llama.tokenizer import Tokenizer
from llama_service.model_service import LLaMAService
from llama_service.__utils import logger, auto_load


@auto_load.task('llama_predict')
class LLaMAPredictTask(TreeMultiTask):
    Opt_Max_Input_Limit = 500
    Opt_Max_Gen_Limit = 1000
    Default_Temperature = 0.8
    Default_Top_P = 0.95

    generator:LLaMAService = None
    tokenizer_obj:Tokenizer = None

    def predict(self, max_input_len:int=None, max_gen_len:int=None, 
                temperature:float=None, top_p:float=None, send_step:int=10, prepend_input:bool=False):
        generator:LLaMAService = self.generator
        tokenizer = self.tokenizer_obj
        assert generator is not None
        assert tokenizer is not None

        max_input_len = max_input_len or self.Opt_Max_Gen_Limit
        max_gen_len = max_gen_len or self.Opt_Max_Gen_Limit
        temperature = self.Default_Temperature if temperature is None else temperature
        top_p = top_p or self.Default_Top_P
        params = self.generator.model.params
        max_seq_len = params.max_seq_len

        eos_id = tokenizer.eos_id

        item_iter = self.recv_data()

        for i, item in enumerate(item_iter):
            tokens = item.pop('tokens', None)
            if not tokens:
                logger.info("LLaMAPredictTask.predict skip empty item %s", item)
                continue
            _temperature = item.get('temperature', temperature)
            _top_p = item.get('top_p', top_p)
            _max_gen_len = min(item.get('max_gen_len', max_gen_len), self.Opt_Max_Input_Limit)
            if len(tokens) > max_input_len:
                tokens = tokens[-max_input_len:]
            
            pred_tokens_iter = generator.generate(tokens, 
                    max_input_len=max_input_len, 
                    max_gen_len=_max_gen_len,
                    max_seq_len=max_seq_len,
                    temperature=_temperature,
                    top_p=_top_p)
            _token_batch = []
            if prepend_input:
                _token_batch.extend(tokens)
            for j, pred_tokens in enumerate(pred_tokens_iter):
                _token_batch.append(pred_tokens[0])
                if (j+1) % send_step == 0 and len(_token_batch):
                    pred_item = item.copy()
                    pred_item['pred_sub_tokens'] = _token_batch
                    self.send_data(pred_item)
                    _token_batch = []
                if eos_id in pred_tokens:
                    break
            pred_item = item.copy()
            pred_item['pred_sub_tokens'] = _token_batch
            pred_item['pred_end'] = 1
            self.send_data(pred_item)