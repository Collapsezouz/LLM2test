import uuid, time
from llama_service.__utils import logger, auto_load
from smart.auto import TreeMultiTask
from redis import Redis


@auto_load.task('llm_model.service_test')
class HFServiceTestTask(TreeMultiTask):
    def instruct_sample(self):
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

    def tokenize_sample(self):
        self.send_data({
            'idx': 0,
            'text': '你好',
            'pred_opt': {
                'max_new_tokens': 10,
                'temperature': 1.2,
                'top_p': 0.9
            }
        })
        self.send_data({
            'idx': 1,
            'text': 'Hello world!',
            'pred_opt': {
                'max_new_tokens': 10,
            }
        })

    def plugin_sample(self):
        req_id = uuid.uuid1().hex
        self.send_data({
            'idx': 0,
            'dialog': {
                'system': '用户正在使用aida系统, 当前界面: 表格模版报告',
                'plugins': {
                    "aida": {
                        "name": "数字助理系统",
                        "desc": "AI自动生成表格和智能写作的系统",
                        "api": {
                            "odb/metadata": {
                                "desc": "新建或修改在线表格"
                            },
                            "odb/data": {
                                "desc": "新增或修改表格行列数据"
                            }
                        }
                    },
                    # "browser_search": {
                    #     "name": "浏览器搜索"
                    # },
                    # "es_search": {
                    #     "name": "ElasticSearch搜索",
                    #     "desc": "ES表格搜索"
                    # },
                    # "graph_search": {
                    #     "name": "图数据库搜索",
                    #     "desc": "知识图谱搜索"
                    # },
                    # "calculator": {
                    #     "name": "计算器"
                    # }
                },
                'chat': [
                    {
                        "ask": "营收超过100亿元的科创板公司的2020年的营业成本, 营业收入和营业利润",
                        # "quote": "",
                        "output": [
                            [
                                {
                                    "call": 'plugin_desc("aida", api=["odb/*"])',
                                    "call_result": '''Object Column {
    name:Str // display name
    key:Str
}
Object RowData: Map<key:Str, value:Str> // key is Column.key, value is 单元格的值
Object MetaData { // 表格的元数据定义
    column: List[Column]
}
Object View { // 多维表格试图
    row: List[Column] // 定义行字段
    column: List[RowData] // 定义列字段
    value: Str // 视图单元格值的列名, Column.key or Column.name
}
Api "odb/save" { // 保存表格结构、视图和数据
    params: {
        metadata: MetaData
        view: View
        data: List[RowData]
    }
}
Api "odb/put_row" { // 保存行数据
    params: {
        metadata: MetaData
        view: View
        data: List[RowData] // RowData中的_id是空表示插入新行, _id非空表示修改旧行数据
    }
}'''
                                }
                            ],
                            [
                                {
                                    "call": '''aida(api="odb_save", params={
    "metadata": {
        "column": [
            {"name": "公司", "key":"company"},
            {"name": "属性", "key":"attr"},
            {"name": "时间(年)", "key":"year"},
            {"name": "值", "key":"value"}
        ]
    },
    "view": {
        "row": [
            {"name": "公司", "key":"company"}
        ],
        "column": [
            {"attr":"营业成本", "year":"2020"},
            {"attr":"营业收入", "year":"2020"},
            {"attr":"营业利润", "year":"2020"},
        ],
        "value": "value"
    }
})''',
                                    "call_result": '<link module="odb" id="dfAernD1"/>'
                                }
                            ],
                            [
                                {"text":"为您创建了一张表格。请启用数据查询类的插件，自动填充表格数据。"}
                            ]
                        ],
                    }
                ]
            },
            '_send_text': 'tmp.llm_model.service_test.text:'+req_id+'.0',
            '_send_queue': 'tmp.llm_model.service_test:'+req_id+'.0',
            'pred_opt': {
                'max_new_tokens': 1000
            }
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

    def test_job_sample(self):
        self.send_data({
            'idx': 0,
            'instruct': '中国的首都',
            'pred_opt': {
                'max_new_tokens': 20
            }
        })
        self.send_data({
            'idx': 1,
            'instruct': '将以下文档总结成几个要点。',
            'input': '''IPO进程“停摆”近一年之后，中国老牌企业软件公司用友集团旗下子公司——用友金融信息技术股份有限公司（NEEQ：839483）于近日宣布，拟公开发行股票并在北京证券交易所上市，继续完成受疫情而搁置的上市进程。
这也是首家宣布拟在北交所上市的金融科技公司。
2016年11月，用友金融挂牌新三板，主要提供金融行业数字化产品及解决方案，用友网络(24.530, -1.66, -6.34%)直接持有用友金融74.53%股份。
2021年9月，用友金融在新三板停牌，并在华泰联合证券的辅导下，通过了中国证监会北京监管局的辅导验收，并准备转入“精选层”，在北交所成立后，准备通过北交所上市，并已经接受多轮问询。
2022年4月，在回复北交所第三轮问询时，用友金融表示，由于疫情等原因，预计无法在规定时间内完成回复，并申请延期至今。''',
            'pred_opt': {
                'max_new_tokens': 200,
                'temperature': 1.2,
                'top_p': 0.9
            }
        })
        self.send_data({
            'idx': 2,
            'system': '你是信息抽取机器人。输入的内容是从PDF提取的结果, 用<span x=? y=?>?</span>表示PDF的一个文字块，x为横坐标, y为纵坐标。',
            'instruct': '提取出所有融资租赁公司的名称, 按列表形式返回',
            'input': '''<page num=75>
<span x=242 y=632>名称：公司A</span>
<span x=242 y=674>名称：公司B</span>
</page>''',
            'pred_opt': {
                'max_new_tokens': 100,
                'temperature': 0
            }
        })

    def mock_queue_resp(self):
        self.send_data({
            'dialog': {
                'system': 'You are helpful assistant.', 
                'chat': [{'ask': '中国的首都'}]
            }, 
            '_send_queue': 'tmp.llm_model.service_test:0', 
            'pred_text': '<.System:\n北京'
        })