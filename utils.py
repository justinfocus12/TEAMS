

def find_true_in_dict(d):
    # Thanks Bing Chat
    if isinstance(d, dict):
        for v in d.values():
            if find_true_in_dict(v):
                return True
    elif isinstance(d, bool) or isinstance(d,int):
        return bool(d)
    return False
