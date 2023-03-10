from typing import List

import torch

from llama.tokenizer import Tokenizer
from llama.model import Transformer
from llama_service.__utils import logger


class LLaMAService:
    def __init__(self, model: Transformer, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate(self, tokens, max_input_len:int=None, max_gen_len:int=None, max_seq_len:int=None, temperature:float=None, top_p:float=None):
        max_input_len = max_input_len or self.Opt_Max_Gen_Limit
        max_gen_len = max_gen_len or self.Opt_Max_Gen_Limit
        params = self.model.params
        max_seq_len = params.max_seq_len

        max_prompt_size = len(tokens)
        total_len = min(max_seq_len, max_gen_len + max_prompt_size)
        input_tensor = torch.full((1, total_len), self.tokenizer.pad_id).cuda().long()
        input_tensor[0, :len(tokens)] = torch.tensor(tokens).long()
        input_text_mask = input_tensor != self.tokenizer.pad_id
        start_pos = len(tokens)
        prev_pos = 0
        for cur_pos in range(start_pos, total_len):
            logits = self.model.forward(input_tensor[:, prev_pos:cur_pos], prev_pos)
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = torch.where(
                input_text_mask[:, cur_pos], input_tensor[:, cur_pos], next_token
            )
            input_tensor[:, cur_pos] = next_token
            yield next_token.tolist()
            prev_pos = cur_pos

        # pred_tokens_list = input_tensor.tolist()
        # pred_tokens = pred_tokens_list[0] if len(pred_tokens_list) else []
        # return pred_tokens


def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token
