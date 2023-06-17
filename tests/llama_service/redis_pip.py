# python -m tests.llama_service.redis_pip get_text --key 'api.ov-nlg:d9073cda583f4b94b1e5aefdebfd4757'
# python -m tests.llama_service.redis_pip queue_pop --key 'api.ov-nlg:d9073cda583f4b94b1e5aefdebfd4757'
# python -m tests.llama_service.redis_pip queue_send
# python -m tests.llama_service.redis_pip queue_pop
from redis import Redis
from smart.utils.yaml import yaml_load_file
from smart.utils.dict import dict_find
from llm_model.utils.text_util import byte_safe_decode
from tests import logger
import json
from smart.utils.serialize import TypeObjSerializer
from . import _mock_data


def test_queue_send(config='llm_model/service/hf_text_generation.yml'):
    cfg_dict = yaml_load_file(config)
    req_key = dict_find(cfg_dict, ('configs', 'pip_req', 'key'))
    logger.debug('test_queue_send %s', req_key)
    redis_cfg:dict = dict_find(cfg_dict, ('configs', 'redis_svr'))
    redis = Redis(
        host=redis_cfg.get('host'),
        port=redis_cfg.get('port'),
        password=redis_cfg.get('password'),
        db=redis_cfg.get('db'),
    )
    value_str = json.dumps(_mock_data.chat_data, ensure_ascii=False)
    redis.rpush(req_key, value_str)
    logger.info('sended: %s', value_str)

def test_get_text(key=None, config='llm_model/service/hf_text_generation.yml'):
    cfg_dict = yaml_load_file(config)
    redis_cfg:dict = dict_find(cfg_dict, ('configs', 'redis_svr'))
    # logger.debug('redis: %s', redis_cfg)
    redis = Redis(
        host=redis_cfg.get('host'),
        port=redis_cfg.get('port'),
        password=redis_cfg.get('password'),
        db=redis_cfg.get('db'),
    )
    value = redis.get(key)
    text, trim_byte = byte_safe_decode(value)
    logger.info('redis get result: %s, %s', text, trim_byte)

def test_queue_pop(key=None, config='llm_model/service/hf_chat_generation.yml', timeout:int=10):
    if not key:
        key = dict_find(_mock_data.chat_data, ('_send_queue',))
    logger.info('test_queue_pop %s', key)
    cfg_dict = yaml_load_file(config)
    redis_cfg:dict = dict_find(cfg_dict, ('configs', 'redis_svr'))
    # logger.debug('redis: %s', redis_cfg)
    redis = Redis(
        host=redis_cfg.get('host'),
        port=redis_cfg.get('port'),
        password=redis_cfg.get('password'),
        db=redis_cfg.get('db'),
    )
    value = redis.blpop(key, timeout)
    logger.info('redis queue_pop: %s', value)
    # if value:
    #     type, item = TypeObjSerializer.decode(value[0])
    #     # item = json.loads(value)
    #     logger.info('item: %s', item)


if __name__ == "__main__":
    _d, component = dict(globals()).items(), {}
    for k, v in _d:
        if k.startswith('test_'):
            component[k] = v
            component[k[5:]] = v

    import fire
    fire.Fire(component)