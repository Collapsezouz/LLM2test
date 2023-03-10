from auto_tasks.redis.redis_pip import RedisPipTask
from redis import Redis
from llama_service.__utils import logger, auto_load


@auto_load.task('ov_nlg_redis_pip')
class OvNlgRedisPipTask(RedisPipTask):
    def send_text(self, redis:Redis, item_send_key='_send_queue', item_iter=None, item_iter_fn=None, recv_args={}, 
            item_ttl:int=None, item_sub_key=None, item_sub_end=None):
        item_iter = item_iter or (item_iter_fn or self.recv_data)(**recv_args)
        for i, item in enumerate(item_iter): 
            send_key = item.get(item_send_key)
            if not send_key:
                logger.warning('OvNlgRedisPipTask item miss item_send_key: %s', item)
                continue
            sub_text = item.get(item_sub_key)
            sub_end = item.get(item_sub_end)
            send_data = str(sub_text) if sub_text else ''
            if sub_end:
                send_data += '\0'
            if len(send_data):
                redis.append(send_key, send_data)
                if item_ttl:
                    redis.expire(send_key, item_ttl)