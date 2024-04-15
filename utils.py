import numpy as np
from scipy.special import logsumexp
from scipy.stats import genextreme as spgex, beta as spbeta

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

def clopper_pearson_confidence_interval(nsucc, ntot, alpha):
    lower = spbeta.ppf(alpha/2, nsucc, ntot-nsucc+1)
    upper = spbeta.ppf(1-alpha/2, nsucc+1, ntot-nsucc)
    return lower,upper

def pmf2ccdf(hist,bin_edges,return_errbars=False,alpha=None,N_errbars=None): 
    N = np.sum(hist)
    ccdf = np.cumsum(hist[::-1])[::-1] 
    ccdf_norm = np.where(ccdf>0, ccdf, np.nan) / N
    if not return_errbars:
        return ccdf_norm
    if N_errbars is None:
        N_errbars = N
    # Also return clopper-pearson confidence intervals
    lower,upper = clopper_pearson_confidence_interval(ccdf*N_errbars/N, N_errbars, alpha)
    return ccdf_norm,lower,upper

def compute_ccdf_errbars_bootstrap(x,bin_edges,boot_size,n_boot=1000,seed=91830):
    hist,_ = np.histogram(x,bins=bin_edges)
    N = np.sum(hist)
    assert n_boot < N
    ccdf = pmf2ccdf(hist,bin_edges)
    ccdf_boot = np.zeros((n_boot,len(bin_edges)-1))
    rng = default_rng(seed=seed)
    x_resamp = rng.choice(np.arange(N),replace=True,size=(n_boot,boot_size))
    for i_boot in range(n_boot):
        hist_boot,_ = np.histogram(x[i_boot,:],bins=bin_edges)
        ccdf_boot[i_boot],_ = pmf2ccdf(hist_boot,bin_edges)
    return ccdf,ccdf_boot


def compute_block_maxima(x,T):
    # T should be an integer, and x is a timeseries
    nx = len(x)
    nb = int(nx/T) # number of blocks
    m = np.max(x[:(nb*T)].reshape((nb,T)), axis=1)
    return m

def fit_gev_distn(block_maxima):
    neg_shape,loc,scale = spgex.fit(block_maxima)
    return -neg_shape,loc,scale

def gev_return_time(x,T,shape,loc,scale):
    logsf = spgex.logsf(x,-shape,loc=loc,scale=scale)
    print(f'{logsf = }')
    rtime = convert_logsf_to_rtime(logsf,T)
    return logsf,rtime

def compute_return_time_block_maxima(x,T):
    block_maxima = compute_block_maxima(x,T)
    rlev,logsf = compute_logsf_empirical(block_maxima)
    rtime = convert_logsf_to_rtime(logsf,T)
    # Also do a GEV fit 
    shape,loc,scale = fit_gev_distn(block_maxima)
    print(f'{rlev = }')
    print(f'{(shape,loc,scale) = }')
    logsf_gev,rtime_gev = gev_return_time(rlev,T,shape,loc,scale)
    return rlev,rtime,logsf,rtime_gev,logsf_gev,shape,loc,scale

def convert_logsf_to_rtime(logsf, T):
    # log-survival function to return period
    rtime = -T / np.log(-np.expm1(logsf))
    rtime[rtime <= T] = np.nan
    return rtime

def convert_sf_to_rtime(sf, T):
    rtime = -T / np.log1p(-sf)
    rtime = np.where(rtime <= T, np.nan, rtime)
    return rtime

def compute_returnstats_and_histogram(f, time_block_size, bounds=None):
    if bounds is None:
        bounds = [np.min(f),np.max(f)]
    bins = np.linspace(bounds[0]-1e-10,bounds[1]+1e-10,30)
    hist,bin_edges = np.histogram(f, density=False, bins=bins)
    rlev,rtime,logsf,rtime_gev,logsf_gev,shape,loc,scale = compute_return_time_block_maxima(f, time_block_size)
    idx = np.searchsorted(rlev, bin_edges[:-1])
    print(f'{idx = }')
    logsf_gev,rtime_gev = gev_return_time(bin_edges[:-1],time_block_size,shape,loc,scale)
    return bin_edges[:-1], hist, rtime[idx], logsf[idx], rtime_gev, logsf_gev, shape, loc, scale

