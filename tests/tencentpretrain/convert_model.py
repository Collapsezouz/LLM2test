import torch
import json

_default_model_path = '/data/share/model/huggingface/models--P01son--ChatLLaMA-zh-7B/snapshots/580463a1bb714d4324cf6e167bef9778fa8ab1d8/chatllama_7b.bin'


def test_print_params(path=None, out=None):
    path = path or _default_model_path
    model = torch.load(path)
    info = {}
    for k in model.keys():
        tensor:torch.Tensor = model[k]
        info[k] = [list(tensor.shape), str(tensor.dtype)]
    if out:
        with open(out, mode='w') as f:
            json.dump(info, f)


if __name__ == "__main__":
    _d, component = dict(globals()).items(), {}
    for k, v in _d:
        if k.startswith('test_'):
            component[k] = v
            component[k[5:]] = v

    import fire
    fire.Fire(component)