import sys
import os
os.chdir('/home/app/open/TencentPretrain')

sys.path.insert(0, '/home/app/open/TencentPretrain')
sys.path.append('/home/app/ov-nlg-model')

LOCAL_RANK = os.environ.get('LOCAL_RANK')

import logging
logger = logging.getLogger('test')

if LOCAL_RANK == '0':
    from smart.utils.remote_debug import enable_by_env
    enable_by_env()

from smart.utils.log import auto_load_logging_config, set_default_logging_config
auto_load_logging_config(base_dir='/home/app/ov-nlg-model') or set_default_logging_config()

logger.debug('test pretrain %s, LOCAL_RANK: %s', os.getpid(), os.environ.get('LOCAL_RANK'))

import torch
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = True
torch.cuda.init()
torch.cuda.empty_cache() # 清空缓存
# torch.cuda.max_memory_allocated()
# torch.cuda.memory_allocated()
# torch.cuda.memory_reserved()
torch.cuda.memory_summary()

import pretrain
pretrain.main()