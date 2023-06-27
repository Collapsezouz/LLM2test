海宇企业图谱生成模型（OceanVerse Enteriprise KG Graph Natual Language Generating Model）

# 开发
## Init
初始化项目:  
```
> pip install -r requirements.txt
```

## deepspeed安装
> pip install deepspeed
> apt install libaio-dev
> ds_report


# 运行服务
* text generation服务
```
> cd /home/app/expert-cpt
> smart_auto llm_model.service.chatllama_zh.stage1_v1-1-1 predict_service > ./logs/llm_service_predict_v111.log 2>&1 &

ps -ef | grep smart_auto | grep 'llm_model.service'

tail -f ./logs/llm_service_predict.log
```


* chat generation服务
```
> cd /home/app/expert-cpt
> smart_auto llm_model.service.hf_chat_generation predict_service> ./logs/chat_generation_v111.log 2>&1 &
```

* 停止服务: `ps -ef | grep smart_auto | grep 'llm_model.service' | awk '{print $2}' | xargs kill -9`



# QA
常见问题记录.  

* OpenSSL: module 'lib' has no attribute 'X509_V_FLAG_CB_ISSUER_CHECK'
问题：
```
> pip install 'pymongo>=4,<5'

> python
import pymongo
【日志】
...
  File "/opt/conda/lib/python3.7/site-packages/pymongo/ssl_support.py", line 22, in <module>
    import pymongo.pyopenssl_context as _ssl
  File "/opt/conda/lib/python3.7/site-packages/pymongo/pyopenssl_context.py", line 27, in <module>
    from OpenSSL import SSL as _SSL
  File "/opt/conda/lib/python3.7/site-packages/OpenSSL/__init__.py", line 8, in <module>
    from OpenSSL import crypto, SSL
  File "/opt/conda/lib/python3.7/site-packages/OpenSSL/crypto.py", line 1556, in <module>
    class X509StoreFlags(object):
  File "/opt/conda/lib/python3.7/site-packages/OpenSSL/crypto.py", line 1577, in X509StoreFlags
    CB_ISSUER_CHECK = _lib.X509_V_FLAG_CB_ISSUER_CHECK
AttributeError: module 'lib' has no attribute 'X509_V_FLAG_CB_ISSUER_CHECK'
```
解决方法: `pip install pyOpenSSL --upgrade`


head -n 
gpu5
1.12.1

/nas/dataset/llm/phbs_llm/ie_instuct
