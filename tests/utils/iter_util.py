# python3 -m tests.utils.iter_util remove_head
# python3 -m tests.utils.iter_util remove_tail
from llm_model.utils.iter_util import *
from tests import logger


def test_remove_head():
    val_target_tuple = [
        ([2,2,3,2,1,2,2], [2], [3,2,1,2,2])
    ]
    for val_target in val_target_tuple:
        item_list, remove_vals, expected_val, *_ = val_target
        result = items_remove_head(item_list, lambda x:x in remove_vals)
        logger.info("remove_head %s, %s => %s", remove_vals, item_list, result)
        assert result == expected_val


def test_remove_tail():
    val_target_tuple = [
        ([2,3,2,1,2,2], [2], [2, 3, 2, 1])
    ]
    for val_target in val_target_tuple:
        item_list, remove_vals, expected_val, *_ = val_target
        result = items_remove_tail(item_list, lambda x:x in remove_vals)
        logger.info("remove_tail %s, %s => %s", remove_vals, item_list, result)
        assert result == expected_val


if __name__ == "__main__":
    _d, component = dict(globals()).items(), {}
    for k, v in _d:
        if k.startswith('test_'):
            component[k] = v
            component[k[5:]] = v

    import fire
    fire.Fire(component)