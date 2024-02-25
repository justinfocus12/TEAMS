

def find_true_in_dict(d):
    # Thanks Bing Chat
    if isinstance(d, dict):
        for v in d.values():
            if find_true_in_dict(v):
                return True
    elif isinstance(d, bool) or isinstance(d,int):
        return bool(d)
    return False

def concat_dict_of_lists(d0,d1):
    # d0 and d1 must be two dictionaries of lists, corresponding exactly
    assert set(d0.keys()) == set(d1.keys())
    for key in d0.keys():
        d0[key] += d1[key]
    return 


def concat_dict_of_arrays(d0,d1,axis=-1):
    # d0 and d1 must be two dictionaries of lists, corresponding exactly
    assert set(d0.keys()) == set(d1.keys())
    for key in d0.keys():
        d0[key] = np.concatenate((d0[key], d1[key]), axis=axis)
    return 
