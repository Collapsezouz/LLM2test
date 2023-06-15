


def items_remove_head(item_list, filter_fn):
    begin_idx = 0
    while begin_idx < len(item_list):
        item = item_list[begin_idx]
        if not filter_fn(item):
            break
        begin_idx += 1
    return item_list[begin_idx:]


def items_remove_tail(item_list, filter_fn):
    end_idx = len(item_list)-1
    while end_idx >= 0:
        item = item_list[end_idx]
        if not filter_fn(item):
            break
        end_idx -= 1
    return item_list[:end_idx+1]
