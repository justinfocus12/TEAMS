import numpy as np
from scipy.special import logsumexp

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

# ------------ Statistical functions -----------------
def compute_logsf_empirical(x,logw=None):
    # x: scalar data samples
    # logw: log-weights
    n = len(x)
    if logw is None:
        logw = np.zeros(n)
    # Assume pre-processing has been done so all values are finite 
    assert np.all(np.isfinite(x)) and np.all(np.isfinite(logw)) and len(x) == len(logw)
    logZ = logsumexp(logw)
    order = np.argsort(x)
    xord,logword = x[order],logw[order]
    logsf_emp = np.logaddexp.accumulate(logword[::-1])[::-1] - logZ # log of sf(x) = p(X >= x)
    return xord,logsf_emp

def compute_block_maxima(x,T):
    # T should be an integer, and x is a timeseries
    nx = len(x)
    nb = int(nx/T) # number of blocks
    print(f'{x.shape = }')
    print(f'{nb = }')
    print(f'{T = }')
    m = np.max(x[:(nb*T)].reshape((nb,T)), axis=1)
    return m

def compute_return_time_block_maxima(x,T):
    block_maxima = compute_block_maxima(x,T)
    rlev,lsf = compute_logsf_empirical(block_maxima)
    rtime = convert_logsf_to_rtime(lsf,T)
    return rlev,rtime,lsf

def convert_logsf_to_rtime(logsf, T):
    # log-survival function to return period
    rtime = -T / np.log(-np.expm1(logsf))
    return rtime

def compute_returnstats_and_histogram(f, time_block_size, bounds=None):
    if bounds is None:
        bounds = [np.min(f),np.max(f)]
    bins = np.linspace(bounds[0]-1e-10,bounds[1]+1e-10,30)
    hist,bin_edges = np.histogram(f, density=False, bins=bins)
    rlev,rtime,logsf = compute_return_time_block_maxima(f, time_block_size)
    idx = np.searchsorted(rlev, bin_edges[:-1])
    print(f'{idx = }')
    return bin_edges[:-1], hist, rtime[idx], logsf[idx]

