import uuid, time
from llama_service.__utils import logger, auto_load
from smart.auto import TreeMultiTask
from redis import Redis


@auto_load.task('llm_model.service_test')
class HFServiceTestTask(TreeMultiTask):
    def test_sample(self):
        req_id = uuid.uuid1().hex
        self.send_data({
            'idx': 0,
            'ask': '中国的首都',
            '_send_queue': 'tmp.llm_model.service_test:'+req_id+'.0',
            'pred_opt': {
                'max_new_tokens': 20
            }
        })
        self.send_data({
            'idx': 1,
            'ask': '将以下文档总结成几个要点。',
            '_send_queue': 'tmp.llm_model.service_test:'+req_id+'.1',
            'quote': '''IPO进程“停摆”近一年之后，中国老牌企业软件公司用友集团旗下子公司——用友金融信息技术股份有限公司（NEEQ：839483）于近日宣布，拟公开发行股票并在北京证券交易所上市，继续完成受疫情而搁置的上市进程。
这也是首家宣布拟在北交所上市的金融科技公司。
2016年11月，用友金融挂牌新三板，主要提供金融行业数字化产品及解决方案，用友网络(24.530, -1.66, -6.34%)直接持有用友金融74.53%股份。
2021年9月，用友金融在新三板停牌，并在华泰联合证券的辅导下，通过了中国证监会北京监管局的辅导验收，并准备转入“精选层”，在北交所成立后，准备通过北交所上市，并已经接受多轮问询。
2022年4月，在回复北交所第三轮问询时，用友金融表示，由于疫情等原因，预计无法在规定时间内完成回复，并申请延期至今。''',
            # 'pred_opt': {
            #     'max_new_tokens': 10,
            # }
        })
    
    def print_resp(self, redis:Redis, item_send_key='_send_queue'):
        item_iter = self.recv_data()

        for i, item in enumerate(item_iter):
            ask, quote = item.get('ask'), item.get('quote')
            _resp_key = item.get(item_send_key)
            logger.info("service_test item %s ask: %s, quote: %s, resp: %s", i, ask, quote, _resp_key)
            start=0
            while True:
                val = redis.getrange(_resp_key, start, -1)
                time.sleep(.5)
                if val:
                    start += len(val)
                    logger.info('%s', val)
                    if val[-1] in ('\0', b'\0', 0):
                        break