# python -m tests.llama_service.redis_pip get_text --key 'api.ov-nlg:d9073cda583f4b94b1e5aefdebfd4757'
from redis import Redis
from smart.utils.yaml import yaml_load_file
from smart.utils.dict import dict_find
from llm_model.utils.text_util import byte_safe_decode
from tests import logger


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


if __name__ == "__main__":
    _d, component = dict(globals()).items(), {}
    for k, v in _d:
        if k.startswith('test_'):
            component[k] = v
            component[k[5:]] = v

    import fire
    fire.Fire(component)