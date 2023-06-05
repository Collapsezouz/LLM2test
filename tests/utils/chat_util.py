# python -m tests.utils.chat_util parse
# export DEBUG_PORT=5679
# REMOTE_DEBUG=1 python -m tests.utils.chat_util train_data
import json
from llm_model.utils.chat_util import *
from tests import logger

_mock_chat_data = [
{
    'system': '用户正在使用aida系统, 当前界面: 表格模版报告',
    'plugins': '''{
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
    }
}''',
    'chat': [
        {
            "ask": "营收超过100亿元的科创板公司的2020年的营业成本, 营业收入和营业利润",
            # "quote": "",
            "output": [
                [
                    {
                        "call": 'plugin_desc("aida", api=["odb/*"])',
                        "text": '''Object Column {
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
                        "text": '<link module="odb" id="dfAernD1"/>'
                    }
                ],
                [
                    {"text":"为您创建了一张表格。请启用数据查询类的插件，自动填充表格数据。"}
                ]
            ],
        }
    ],
    '_send_text': 'tmp.llm_model.service_test.text:1',
    '_send_queue': 'tmp.llm_model.service_test:1',
    'pred_opt': {
        'max_new_tokens': 1000
    }
}, 
{
    'system': '用户正在使用aida系统，当前界面: 表格模版报告',
    'plugins': '''{
    "kg_search": {
        "name": "图数据库搜索",
        "desc": "生成sparql查询",
        "api": {
            "ontology": {
                "desc": "查询本体语义描述"
            },
            "query": {
                "desc": "查询sparql数据库"
            }
        }
    }
  }
}''',
    'chat': [
        {
            "ask": "厦门国际的注册资本",
            "output": [
                [
                    {
                        "call": 'plugin_desc("kg_search")',
                        "text": '''Api ontology { // 查询本体
    params: {
        entity: List[Str] // 实体, 可选: company, people
    }
}
Api query { // 查询sparql数据库
    params: {
        text: Str // 问句
        sql: Str // Sparql查询语句
    }
}'''
                    }
                ],
                [
                    {
                        "call": 'kg_search(api="ontology", entity=["company"])',
                        "text": '''Entity company { // 公司实体
    name: Str // 公司名称
    entity_class: Str // 实体类型
    entity_tag: Str // 实体标签
}
Entity extendAttr { // 扩展属性实体
    name: Str // 属性名
    time: Str // 时间
    year: Str // 从time提取出来的年份
    month: Str // 从time提取出来的月份
    day: Str // 从time提取出来的日
    value: Str // 属性值
    value_num: Str // 从value提取出来的数值
    value_unit: Str // 从value提取出来的值单位
    value_currency: Str // 从value提取出来的货币单位
    percent_m: Str // 环比增长
    percent_y: Str // 同比增长
}
Relation<company, extendAttr> {
    extend_attr // 公司实体的扩展属性实体
}'''
                    }
                ],
                [
                    {
                        "call": """kg_search(api='query', text='厦门国际的注册资本', sql='''SELECT ?extendAttr ?entity ?attr ?value
{
?company :name ?entity.
?company :extend_attr ?extendAttr.
?extendAttr :name ?attr.
?extendAttr :value ?value
filter(regex(xsd:string(?entity),"厦门国贸"))
filter(regex(xsd:string(?attr),"注册资本"))
}"""
                    }
                ]
            ]
        }
    ],
    '_send_text': 'tmp.llm_model.service_test.text:2',
    '_send_queue': 'tmp.llm_model.service_test:2',
    'pred_opt': {
        'max_new_tokens': 1000
    }
}]


def test_parse():
    for i, chat_data in enumerate(_mock_chat_data):
        if i: logger.info("")
        logger.info('chat_data: %s', chat_data)
        encoder = ChatTextEncoder()
        round_data_iter = encoder.parse_chat_obj(chat_data)
        for round_data in round_data_iter:
            logger.info("# block %s-%s", round_data.type, round_data.round_idx)
            for block_item in round_data.block_list:
                logger.info("%s", block_item.text)


def test_train_data():
    for i, chat_data in enumerate(_mock_chat_data):
        if i: logger.info("")
        logger.info('chat_data: %s', chat_data)
        encoder = ChatTextEncoder()
        train_data = encoder.train_data(chat_data)
        for input_list, output in train_data:
            input_text = ''.join([
                ''.join(input.get_text_list())
                for input in input_list
            ])
            output_text = ''.join(output.get_text_list())
            logger.info('\n---Model Input---\n%s---Model Output---\n%s', input_text, output_text)



if __name__ == "__main__":
    _d, component = dict(globals()).items(), {}
    for k, v in _d:
        if k.startswith('test_'):
            component[k] = v
            component[k[5:]] = v

    import fire
    fire.Fire(component)