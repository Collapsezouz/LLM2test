通过开源数据集做指令微调

# Dataset
* csl数据集
来源: https://github.com/ydli-ai/CSL
文件路径: /nas/dataset/llm/prompt_dataset/csl_camera_readly.tsv

* dolly-15k
文件路径: /nas/dataset/llm/prompt_dataset/databricks-dolly-15k.jsonl

* wombat
文件路径: /nas/dataset/llm/prompt_dataset/wombat_train.json

* firefly
文件路径: /nas/dataset/llm/firefly-train-1.1M/firefly-train-1.1M.jsonl

* ShareGPT
文件路径: /nas/dataset/llm/ShareGPT52K

* moss-002-sft-data, moss-003-sft-data
文件路径: /nas/dataset/llm/fnlp/moss-002-sft-data
文件路径: /nas/dataset/llm/fnlp/moss-003-sft-data

* belle
HuggingFace Dataset:
- BelleGroup/multiturn_chat_0.8M
- BelleGroup/train_1M_CN
- BelleGroup/generated_chat_0.4M

* news_commentary
HuggingFace Detaset: news_commentary


## Advanced Dataset
* ceval-exam


* bigcode/the-stack
来源: https://huggingface.co/datasets/bigcode/the-stack


## Other Dataset

* TigerResearch pretraining
TigerResearch/pretrain_zh: 中文开源预训练集 - 55G，包含中文书籍、中文互联网、中文百科(基于 GPT3 的 pretrain 的数据分布，采集中文书籍，互联网，和百科类数据，并通过数据源质量分过滤和 tf-idf soft deduping，从 20TB 数据过滤到 2TB，保持语言和类目的比例，并在此基础上随机抽样 100G 数据开源)
TigerResearch/pretrain_en: 51G，包含英文书籍、英文互联网、英文百科
TigerResearch/tigerbot-research-plugin: 共2W篇完整研报，按段落保存, 发布时间区间: 2022-09-30 至 2023-05-19
TigerResearch/tigerbot-earning-plugin: 2500篇财报，抽取后按段落保存, 发布时间区间为: 2022-02-28 至 2023-05-10
TigerResearch/tigerbot-wiki-plugin: 百科类
TigerResearch/tigerbot-law-plugin: 法律11大类，共5.5W+条款 - 宪法, 刑法, 行政法, 司法解释, 民法商法, 民法典, 行政法规, 社会法, 部门规章, 经济法, 诉讼与非诉讼程序法


* TigerResearch sft数据集
TigerResearch/tigerbot-zhihu-zh-10k: 基于开源搜集的知乎数据生成的sft问答对
TigerResearch/tigerbot-wiki-qa-bart-en-10k: 英文wiki类的问答数据
TigerResearch/tigerbot-dolly-Brainstorming-en-1.7k: 基于dolly数据集加工的头脑风暴Brainstorming相关分类的的sft
TigerResearch/tigerbot-stackexchange-qa-en-0.5m: 基于stackexchange问答站点dump数据生成sft数据集
TigerResearch/tigerbot-cmu-wiki-en: 基于cmu开放在的wiki问答数据集整理的sft数据
TigerResearch/tigerbot-firefly-zh-20k: 基于firefly数据集生成的问答sft数据
TigerResearch/tigerbot-kaggle-recipes-en-2k: 食谱类sft数据集
TigerResearch/tigerbot-OIG-multichat-en-50k: OIG多轮对话数据集过滤5W条并加工成角色扮演sft数据集
TigerResearch/tigerbot-kaggle-leetcodesolutions-en-2k: 基于leetcode-solutions数据集，加工生成的代码类sft数据集
TigerResearch/tigerbot-riddle-qa-1k: 中文-猜谜语sft数据集
TigerResearch/tigerbot-book-qa-1k: 中文书籍-名著相关知识问答数据
TigerResearch/tigerbot-HC3-zh-12k: 基于公开的HC3数据集加工生成的常识问答sft数据集
TigerResearch/tigerbot-mt-note-generation-en: 病历生成相关的sft数据集
TigerResearch/tigerbot-superclue-c3-zh-5k: 基于cluebenchmark公开的数据集加工生成的阅读理解sft数据集
TigerResearch/tigerbot-gsm-8k-en: 基于gsm8k数据集加工而来。GSM8K（Grade School Math 8K）是一个包含 8.5K 高质量语言多样化小学数学单词问题的数据集。创建数据集是为了支持对需要多步推理的基本数学问题的问答任务。
TigerResearch/tigerbot-wiki-qa-zh-1k: 中文百科问答数据。
TigerResearch/tigerbot-youtube-howto-en-50k: 基于开源数据加工的sft，youtube中如何做(howto)系列
TigerResearch/tigerbot-dolly-classification-en-2k: 基于dolly数据集加工的分类classification相关分类的的sft




