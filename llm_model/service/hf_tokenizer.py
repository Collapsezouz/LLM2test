from .hf_task import HFModelTask
from llama_service.__utils import logger, auto_load
from transformers import AutoTokenizer, PreTrainedTokenizer

@auto_load.task('llm_model.hf_tokenizer')
class HFTokenizerTask(HFModelTask):

    def init(self, model_name=None, model_path=None, use_fast:bool=False):
        if model_path is None:
            model_opts = self.init_model(model_name=model_name, model_path=model_path)
            model_path = model_opts.get('model_path')
        assert model_path, 'model_path is None'
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            use_fast=use_fast,
            local_files_only=True
        )
        logger.debug('hf_tokenizer init %s vocab, special_token: %s', tokenizer.vocab_size, tokenizer.special_tokens_map)
        return {
            'tokenizer': tokenizer
        }

    def encode(self, tokenizer=None, encode_opt:dict=None, input_text_key='text', input_tokens_key='tokens', item_iter=None, item_iter_fn=None):
        assert tokenizer is not None, 'tokenizer is None'
        tokenizer:PreTrainedTokenizer
        logger.debug('hf_tokenizer.encode %s -> %s, opt=%s', input_text_key, input_tokens_key, encode_opt)
        _item_iter = item_iter or (item_iter_fn or self.recv_data)()
        encode_opt = encode_opt or {}
        def _out_item_iter_fn():
            for i, item in enumerate(_item_iter):
                if i == 0: logger.debug("hf_tokenizer.encode recv first item %s", item)
                text = item.get(input_text_key)
                if text:
                    tokens = tokenizer.encode(text, **encode_opt)
                    item[input_tokens_key] = tokens
                yield item
        
        return {
            'item_iter_fn': _out_item_iter_fn
        }
    
    def decode(self, tokenizer=None, decode_opt:dict=None, output_tokens_key='pred_tokens', output_text_key='pred_text', item_iter=None, item_iter_fn=None):
        assert tokenizer is not None, 'tokenizer is None'
        tokenizer:PreTrainedTokenizer
        logger.debug('hf_tokenizer.decode %s -> %s, opt=%s', output_tokens_key, output_text_key, decode_opt)
        _item_iter = item_iter or (item_iter_fn or self.recv_data)()
        decode_opt = decode_opt or {}
        def _out_item_iter_fn():
            for i, item in enumerate(_item_iter):
                if i == 0: logger.debug("hf_tokenizer.decode recv first item %s", item)
                tokens = item.get(output_tokens_key)
                if tokens:
                    decode_text = tokenizer.decode(tokens, **decode_opt)
                    item[output_text_key] = decode_text
                yield item
        
        return {
            'item_iter_fn': _out_item_iter_fn
        }
    
    # def decode_sub(self, tokenizer=None, item_iter=None, item_iter_fn=None):
    #     assert tokenizer is not None, 'tokenizer is None'
    #     tokenizer:PreTrainedTokenizer

    #     _item_iter = item_iter or (item_iter_fn or self.recv_data)()
    #     def _out_item_iter_fn():
    #         for item in _item_iter:
    #             pred_sub_tokens = item.get('pred_sub_tokens')
    #             pred_sub_text = ''
    #             if pred_sub_tokens:
    #                 pred_proto = tokenizer.sp_model.decode(pred_sub_tokens, out_type='immutable_proto')
    #                 if pred_proto:
    #                     try:
    #                         for piece in pred_proto.pieces:
    #                             prefix = piece.piece[:-len(piece.surface)]
    #                             pred_sub_text += prefix.replace('▁', ' ')
    #                             break
    #                     except:
    #                         logger.warning("hf_tokenizer.decode_sub fail parse piece: %s", pred_proto)
    #                 pred_sub_text += pred_proto.text
    #             item['pred_sub_text'] = pred_sub_text
    #             logger.debug("hf_tokenizer.decode_sub %s %s", pred_sub_tokens, pred_sub_text)
    #             yield item
    #     return {
    #         # 'item_iter': _out_item_iter_fn(),
    #         'item_iter_fn': _out_item_iter_fn
    #     }