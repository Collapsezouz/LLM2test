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
},
{
    'system': '用户正在使用aida系统, 当前界面: 表格模版报告',
    'plugins': '''{
    "web_search": {
        "name": "网络搜索",
        "api": {
            "": {
                "args": {
                    "keyword": "关键词, 按' '分割"
                }
            }
        }
    }
}''',
    'chat': [
        {
            "ask": "你知道乔布斯影响了多少人吗？",
            "output": [
                [
                    {
                        "thought":"这是一个关于特定实体的问题，我需要搜索一下关于乔布斯影响力广泛的相关信息"
                    },
                    {
                        "call": "web_search(\"乔布斯 影响力\")",
                        "call_result": "<|1|>: '领导力│乔布斯身上的领导力解析。             乔布斯身上的领导力解析 。在开始这篇论文之前，让我们先来平静下内心或者敬仰，或者难过，或者惋惜的心情，开始我们的理智的分析乔布斯这位世界级大师的传奇人生，以及在他领导下的苹果公司的飞速发展，看看这位传奇人物有着怎样的胸怀气魄以及非凡的领导才能。2007年，史蒂夫·乔布斯被《财富》杂志评为了年度最有影响力的商人。          领导力│乔布斯身上的领导力解析。'\n<|2|>: '知名IT网站日前评出了2009年度“全球最具影响力科技领袖50强”，苹果CEO史蒂夫·乔布斯(Steve Jobs)高居榜首。'\n<|3|>: '1994年，乔布斯问工作室的同事：“谁是世界上最有影响力的人？” 有人回答，“南非总统曼德拉”. 知道曼德拉事迹的人知道，这个答案没有错。'\n"
                    }
                ],
                [
                    {
                        "text": "乔布斯的影响力非常广泛。他影响了很多人的想法和思维方式。据知名IT网站评选，乔布斯曾被评为全球最具影响力的科技领袖，史蒂夫·乔布斯还被《财富》杂志评为了年度最有影响力的商人。他的领导才能和胸怀气魄也广受赞誉。 <sup><|2|></sup>"
                    }
                ]
            ]
        },
        {
            "ask": "他的创新精神和领导才能是如何影响了苹果公司？",
            "output": [
                [
                    {
                        "thought":"这是一个关于乔布斯和苹果公司的问题，我需要搜索一下乔布斯在苹果公司的领导才能和创新精神以及它们对苹果公司的影响。"
                    },
                    {
                        "call": "web_search(\"乔布斯 苹果 公司 领导才能 创新精神 影响\")",
                        "call_result": "<|1|>: '乔布斯的反应显示了他的领导能力与苹果公司真正的文化价值是非常相符的——作为一个公司，要不停地促进世界向前发展。他向公众表态：“我们会很快修复好 ...'\n<|2|>: '他的传奇人生和独特魅力辉映苹果的激情与浪漫，他的非凡成就影响整个IT产业和电子世界。 根据新兴产业发展的“三螺旋理论”，在“技术创新－资本市场－企业家 ...'\n<|3|>: '当年负责制作苹果“1984”广告的创意导演李·克劳回忆说：“‘1984’表达了苹果的理念和目标：让人民而非政府或大公司掌握操纵技术，让计算机变成人人都可以用的产品，而不是强权者控制人们生活的工具。\\u3000\\u3000苹果之所以叫苹果，就是因为乔布斯觉得技术应该为人所用，而不是为人所惧，应该用一个大家觉得简单亲切的名字，而不是复杂艰深的名字。当年负责制作苹果“1984”广告的创意导演李·克劳回忆说：“‘1984’表达了苹果的理念和目标：让人民而非政府或大公司掌握操纵技术，让计算机变成人人都可以用的产品，而不是强权者控制人们生活的工具。\\u3000\\u3000Macintosh的出现引发了个人计算机世界的革命，iphone的出现则引发了手机世界的革命。\\u3000\\u3000乔布斯是美国发明家、企业家、苹果公司创始人，他引领了个人电脑、动画电影、音乐、手机、平板电脑、数字出版等六大产业的\n"
                    }
                ],
                [
                    {
                        "text": "乔布斯在苹果公司的领导才能和创新精神对公司的影响是显著的。他的反应和行动显示了他的领导能力与苹果公司真正的文化价值是非常相符的，即要不停地促进世界向前发展。他的非凡成就和独特魅力也影响了整个IT产业和电子世界。在苹果公司的产品理念上，乔布斯始终坚持让计算机变成人人都可以用的产品，而不是强权者控制人们生活的工具。他引领了个人电脑、动画电影、音乐、手机、平板电脑、数字出版等六大产业的革命，使苹果公司成为了全球最有影响力的公司之一。 <sup><|3|></sup>"
                    }
                ]
            ]
        }
    ],
    '_send_text': 'tmp.llm_model.service_test.text:3',
    '_send_queue': 'tmp.llm_model.service_test:3',
    'pred_opt': {
        'max_new_tokens': 2000
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
        j = 0
        for input_list, output in train_data:
            j += 1 # 模型第几次调用, 从1开始
            round_idx = output.round_idx # 第几轮对话, 从0开始
            input_text = ''.join([
                ''.join(input.get_text_list())
                for input in input_list
            ])
            output_text = ''.join(output.get_text_list())
            logger.info('\n---Model Input %s-%s---\n%s---Model Output---\n%s', round_idx, j, input_text, output_text)



if __name__ == "__main__":
    _d, component = dict(globals()).items(), {}
    for k, v in _d:
        if k.startswith('test_'):
            component[k] = v
            component[k[5:]] = v

    import fire
    fire.Fire(component)