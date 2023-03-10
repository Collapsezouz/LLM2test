# python -m tests.llama_service.predict send_recv
import time, json
from tests import logger
from smart.utils.yaml import yaml_load_file
from smart.utils.dict import dict_find
from redis import Redis

_test_text_list = [
    "中游主要是地理信息的支撑系统开发，包括数据的"
]

def _get_service_config(config_path=None):
    config_dict = yaml_load_file(config_path or './llama_service/service.yml')
    return config_dict

def test_send_recv(text=None, max_gen_len:int=50, temperature:float=None, top_p:float=None):
    service_config = _get_service_config()
    item_text_key = dict_find(service_config, ('configs', 'encode_args', 'item_text_key')) or 'text'
    item_send_key = dict_find(service_config, ('configs', 'pip_resp', 'item_send_key')) or '_send_queue'
    pip_req_key = dict_find(service_config, ('configs', 'pip_req', 'key')) or '_send_queue'
    redis_config = dict_find(service_config, ('configs', 'redis_svr')) or {}
    redis_kwargs = {
        'host': redis_config.get('host'),
        'port': redis_config.get('port'),
        'password': redis_config.get('password'),
        'db': redis_config.get('db'),
        **(redis_config.get('redis_kwargs') or {})
    }
    redis = Redis(**redis_kwargs)

    _text_list = [text] if text else _test_text_list
    _resp_key_list = []
    for i, _text in enumerate(_text_list):
        logger.info("%s# %s", i, _text)
        _send_key = "test.llama_service.predict" + str(time.time()) + ":" + str(i)
        _resp_key_list.append(_send_key)
        _req_item = {}
        _req_item[item_text_key] = _text
        _req_item[item_send_key] = _send_key
        if max_gen_len: _req_item['max_gen_len'] = max_gen_len
        if temperature is not None: _req_item['temperature'] = temperature
        if top_p: _req_item['top_p'] = top_p
        item_str = json.dumps(_req_item, ensure_ascii=False)
        redis.rpush(pip_req_key, item_str)
        logger.info("send %s %s", i, item_str)
    
    for i, _resp_key in enumerate(_resp_key_list):
        logger.info("\nrecv %s %s", i, _resp_key)
        cursor = 0
        resp_text = b''
        for i in range(200):
            sub_text = redis.getrange(_resp_key, cursor, -1)
            # if isinstance(sub_text, bytes): 
            #     sub_text = sub_text.decode('utf8')
            if len(sub_text):
                logger.info("%s:%s", len(sub_text), sub_text)
                resp_text += sub_text
                cursor += len(sub_text)
            if b'\0' in sub_text:
                break
            time.sleep(0.5)
        try:
            resp_text = resp_text.decode('utf8')
        except:
            pass
        logger.info("recv_text: %s", resp_text)

if __name__ == "__main__":
    _d, component = dict(globals()).items(), {}
    for k, v in _d:
        if k.startswith('test_'):
            component[k] = v
            component[k[5:]] = v

    import fire
    fire.Fire(component)