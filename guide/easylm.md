# Init

* 拷贝代码
```
> cd /home/app/open
> git clone https://github.com/young-geng/EasyLM.git
```


* 安装依赖包

```
> 
pip install "jax[cpu]"
pip install flax
pip install mlxu
pip install dill

> pip list | egrep "jax|flax|mlxu|dill"
dill                              0.3.6
flax                              0.6.4
jax                               0.3.25
jaxlib                            0.3.25
mlxu                              0.1.10
```

## 转换模型文件
koala模型增量文件:  
/data/share/model/huggingface/models--young-geng--koala/snapshots/ea9d26214360de4276173a272b58f25e3936fd7f


* 将原版llama模型文件转为easylm格式
  
```
临时注释EasyLM/jax_utils.py两行代码:
# from jax.sharding import PartitionSpec as PS
# from jax.sharding import Mesh

> mkdir -p /nas/model/llama/easylm_model
> cd /home/app/open/EasyLM
python -m EasyLM.models.llama.convert_torch_to_easylm \
    --checkpoint_dir='/nas/model/llama/7B' \
    --output_file='/nas/model/llama/easylm_model/llama_7b.msgpack' \
    --streaming=True
```

* 合并llama模型文件, 得到完整的kaola模型文件
```
> cd /home/app/open/EasyLM
python -m EasyLM.scripts.diff_checkpoint \
    --recover_diff=True \
    --load_base_checkpoint='params::/nas/model/llama/easylm_model/llama_7b.msgpack' \
    --load_target_checkpoint='params::/data/share/model/huggingface/models--young-geng--koala/snapshots/ea9d26214360de4276173a272b58f25e3936fd7f/koala_7b_diff_v2' \
    --output_file='/nas/model/llama/easylm_model/koala_7b_v2.msgpack' \
    --streaming=True
```

* 转换成huggingface格式  
```
> cd /home/app/open/EasyLM
python -m EasyLM.models.llama.convert_easylm_to_hf \
    --load_checkpoint='params::/nas/model/llama/easylm_model/koala_7b_v2.msgpack' \
    --tokenizer_path='/nas/model/llama/tokenizer.model' \
    --model_size='7b' \
    --output_dir='/nas/model/llama/kaola_7b'
```