

Redis_Svr = {
    'host': '10.2.223.230',
    'port': 6390,
    'password': '58d326d7e5e6e8739f51ddbe25be049fa4f75f9f',
    'db': 3,
    'redis_kwargs': {
        'socket_timeout': 90,
        'socket_connect_timeout': 60,
        'retry_on_timeout': True
    }
}

Pip_Req = {
    'key': 'cedb_model.ov_nlg.pred_req',
    'redis_pool_interval': 60,
    'is_daemon': True
}

Pip_Resp = {
    'key': 'cedb_model.ov_nlg.pred_resp',
    'item_queue_key': '_send_queue',
    'queue_ttl': 86400
}