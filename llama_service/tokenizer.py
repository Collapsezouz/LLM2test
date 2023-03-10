from smart.auto.tree import TreeMultiTask
from llama.tokenizer import Tokenizer
from llama_service.__utils import logger, auto_load

@auto_load.task('llama_tokenize')
class LLaMATokenizeTask(TreeMultiTask):
    tokenizer_obj = None

    @staticmethod
    def init_tokenizer(tokenizer_path=None) -> Tokenizer:
        if LLaMATokenizeTask.tokenizer_obj is None:
            assert bool(tokenizer_path)
            LLaMATokenizeTask.tokenizer_obj = Tokenizer(model_path=tokenizer_path)
        return LLaMATokenizeTask.tokenizer_obj

    def encode(self, tokenizer_path=None, item_text_key='text', item_iter=None, item_iter_fn=None):
        tokenizer = self.init_tokenizer(tokenizer_path=tokenizer_path)
        logger.debug('LLaMATokenizeTask.encode start %s', tokenizer)
        _item_iter = item_iter or (item_iter_fn or self.recv_data)()
        def _out_item_iter_fn():
            for i, item in enumerate(_item_iter):
                logger.debug("llama_tokenize.encode recv %s %s", i, item)
                text = item.get(item_text_key)
                if text:
                    tokens = tokenizer.encode(text, bos=True, eos=False)
                    item['tokens'] = tokens
                yield item
        
        return {
            # 'item_iter': _out_item_iter_fn(),
            'item_iter_fn': _out_item_iter_fn
        }
    
    def decode_sub(self, tokenizer_path=None, item_iter=None, item_iter_fn=None):
        tokenizer = self.init_tokenizer(tokenizer_path=tokenizer_path)
        logger.debug('LLaMATokenizeTask.decode_sub start %s', tokenizer)
        _item_iter = item_iter or (item_iter_fn or self.recv_data)()
        def _out_item_iter_fn():
            for item in _item_iter:
                pred_sub_tokens = item.get('pred_sub_tokens')
                pred_sub_text = ''
                if pred_sub_tokens:
                    pred_proto = tokenizer.sp_model.decode(pred_sub_tokens, out_type='immutable_proto')
                    if pred_proto:
                        try:
                            for piece in pred_proto.pieces:
                                prefix = piece.piece[:-len(piece.surface)]
                                pred_sub_text += prefix.replace('▁', ' ')
                                break
                        except:
                            logger.warning("LLaMATokenizeTask.decode_sub fail parse piece: %s", pred_proto)
                    pred_sub_text += pred_proto.text
                item['pred_sub_text'] = pred_sub_text
                logger.debug("llama_tokenize.decode_sub %s %s", pred_sub_tokens, pred_sub_text)
                yield item
        return {
            # 'item_iter': _out_item_iter_fn(),
            'item_iter_fn': _out_item_iter_fn
        }