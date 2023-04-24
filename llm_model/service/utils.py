from smart.utils.list import list_safe_iter

def item_get_by_multi_key(item:dict, keys, default_val=None):
    if not item:
        return default_val
    
    for key in list_safe_iter(keys):
        if key in item:
            return item.get(key, default_val)
        
    return default_val