import sys, os
try:
    import llm_model
except:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), *(['..']*2))))
    from smart.utils.log import auto_load_logging_config, set_default_logging_config
    auto_load_logging_config() or set_default_logging_config()

LOCAL_RANK = os.environ.get('LOCAL_RANK', '')

if LOCAL_RANK in ('0', ''):
    from smart.utils.remote_debug import enable_remote_debug
    enable_remote_debug()

from llm_model.alpaca.train import *


def debug_dataset():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments))
    model_args, data_args, *_ = parser.parse_args_into_dataclasses()
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        padding_side="right",
        use_fast=False,
    )
    # if tokenizer.pad_token is None:
    if 'pad_token' not in tokenizer.special_tokens_map:
        tokenizer.add_special_tokens(
            {
                "pad_token": DEFAULT_PAD_TOKEN,
                "eos_token": DEFAULT_EOS_TOKEN,
                "bos_token": DEFAULT_BOS_TOKEN,
                "unk_token": DEFAULT_UNK_TOKEN,
            }
        )

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    logging.debug('data_module: %s', data_module)

if __name__ == "__main__":
    debug_dataset()
