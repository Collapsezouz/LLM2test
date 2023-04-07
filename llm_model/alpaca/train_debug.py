import sys, os
try:
    import llm_model
except:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), *(['..']*2))))
    from smart.utils.log import auto_load_logging_config, set_default_logging_config
    auto_load_logging_config() or set_default_logging_config()

LOCAL_RANK = os.environ.get('LOCAL_RANK')

if LOCAL_RANK == '0':
    from smart.utils.remote_debug import enable_remote_debug
    enable_remote_debug()

from llm_model.alpaca.train import train

if __name__ == "__main__":
    train()