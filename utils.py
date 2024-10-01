import numpy as np
from numpy.random import default_rng
from scipy.special import logsumexp, gamma as GammaFunction
from scipy.stats import genextreme as spgex, beta as spbeta
from scipy.optimize import fsolve,bisect

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
def compute_logsf_empirical_with_multiplicities(x,logw=None,mults=None):
    # x: scalar data samples
    # logw: log-weights
    x = np.array(x)
    assert 1 == x.ndim
    n = len(x)
    if logw is None:
        logw = np.zeros(n)
    logw = np.array(logw)
    if mults is None:
        mults = np.ones(n, dtype=int)
    mults = np.array(mults)
    # Assume pre-processing has been done so all values are finite 
    assert np.all(np.isfinite(x)) and np.all(np.isfinite(logw)) and len(x) == len(logw) == len(mults)
    order = np.argsort(x)
    order = order[np.where(mults[order] >= 1)[0]]
    xord,logword = x[order],logw[order]
    logZ = logsumexp(logw[order], b=mults[order])
    logsf_emp = np.logaddexp.accumulate(logword[::-1] + np.log(mults[order][::-1]))[::-1] - logZ # log of sf(x) = p(X >= x)
    return xord,logsf_emp

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

def pmf2ccdf(hist,bin_edges,return_errbars=False,alpha=0.05,N_errbars=None): 
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

def ccdf2rlev_of_rtime(rlev,ccdf_norm,lower,upper):
    # Also invert to find return level as a function of return period
    logsf_grid = np.linspace(np.log(ccdf_norm[1]), np.log(ccdf_norm[-1]), 30) # decreasing
    rlev_inverted = np.interp(logsf_grid[::-1], np.log(ccdf_norm[::-1]), rlev[::-1])[::-1]
    rlev_inverted_upper = np.interp(logsf_grid[::-1], np.log(lower[::-1]), rlev[::-1])[::-1]
    rlev_inverted_lower = np.interp(logsf_grid[::-1], np.log(upper[::-1]), rlev[::-1])[::-1]
    return logsf_grid,rlev_inverted,rlev_inverted_lower,rlev_inverted_upper


def compute_ccdf_errbars_bootstrap(x,bin_edges,boot_size=None,n_boot=5000,seed=91830):
    hist,_ = np.histogram(x,bins=bin_edges)
    N = np.sum(hist)
    print(f'{N = }')
    if boot_size is None: boot_size = N
    assert boot_size <= N
    ccdf = pmf2ccdf(hist,bin_edges)
    ccdf_boot = np.zeros((n_boot,len(bin_edges)-1))
    rng = default_rng(seed=seed)
    for i_boot in range(n_boot):
        if i_boot % 100 == 0: print(f'{i_boot = }')
        x_resamp = x[rng.choice(np.arange(N),replace=True,size=boot_size)]
        hist_boot,_ = np.histogram(x_resamp,bins=bin_edges)
        ccdf_boot[i_boot] = pmf2ccdf(hist_boot,bin_edges)
    return ccdf,ccdf_boot


def compute_block_maxima(x,T):
    # T should be an integer, and x is an nd-array where time is the first dimension
    shp = x.shape
    nt = shp[0]
    nb = int(nt/T) # number of blocks
    m = np.max(x[:(nb*T)].reshape((nb,T) + shp[1:]), axis=1)
    return m

def fit_gev_distn(block_maxima, method="PWM"):
    if method == "MLE":
        neg_shape,loc,scale = spgex.fit(block_maxima)
        shape = -neg_shape
    elif method == "PWM":
        shape,loc,scale = estimate_gev_params_one_ensemble(block_maxima, np.zeros(len(block_maxima)), method="PWM")

    return shape,loc,scale

def estimate_gev_params_one_ensemble(Xall,logWall,max_num_uniform=1e5,min_level=None,method="PWM"):

    if min_level is None: min_level = -np.inf
    idx = np.where(Xall > min_level)[0]
    X = Xall[idx]
    log_weights = logWall[idx]
    logwnorm = log_weights - logsumexp(log_weights)

    if method == "MLE":
        # Replicate samples by inflating weights to all be approximate integers
        num_uniform = int(min(1/np.exp(np.min(logwnorm)), max_num_uniform)+0.5)
        print(f"num_uniform = {num_uniform}")

        weights_inflated = np.maximum(1, (wnorm*num_uniform)).astype(int)
        X_inflated = np.repeat(X, weights_inflated)
        print(f"len(bmi) = {len(X_inflated)}")
        shape,loc,scale = spgex.fit(X_inflated)
        shape *= -1 # switch conventions 
        print(f"from {len(X_inflated)} in range ({min(X_inflated),max(X_inflated)}): shape,loc,scale = {shape,loc,scale}")
    elif method == "PWM":
        # Use the method of Hosking et al 1985
        # Estimate the first three PWMs (beta0, beta1, beta2) by (b0, b1, b2)
        order = np.argsort(X)
        logWord = logwnorm[order]
        Xord = X[order]
        logFord = np.logaddexp.accumulate(logWord) # - np.exp(logWord/2 # TODO is this the proper estimator?
        b0 = np.exp(logsumexp(logWord, b=Xord)) #np.sum(Word * Xord)
        b1 = np.exp(logsumexp(logWord + logFord, b=Xord))#np.sum(Word * Xord * Ford)
        b2 = np.exp(logsumexp(logWord + 2*logFord, b=Xord))#np.sum(Word * Xord * Ford**2)
        # Solve for the shape, location, and scale parameters. Don't use the linear approximation, but
        b_ratio = (3*b2 - b0)/(2*b1 - b0)
        if b_ratio <= 0.0:
            # TODO come up with the best possible alternative...xi is a very large number, probably 
            raise Exception(f"The L-moment method has no solution; {b_ratio = }")
        # Choose initialization for solver
        tol = 1e-2
        psf0 = pwm_shape_func(0,b_ratio) 
        if psf0 == 0:
            shape = 0.0
        elif psf0 < 0: # shape > 0
            lower = 0.0
            upper = 1.0
            while pwm_shape_func(upper,b_ratio) < 0.0:
                upper *= 2.0
        else: # shape < 0
            lower = -1.0
            upper = 0.0
            while pwm_shape_func(lower,b_ratio) > 0.0:
                lower *= 2.0
        shape,root_result = bisect(pwm_shape_func, lower, upper, args=(b_ratio,), full_output=True, disp=True)

        g = GammaFunction(1 - shape)
        if shape == 0:
            scale = (2*b1 - b0)/np.log(2)
            loc = b0 - 0.5772*scale
        else:
            scale = shape*(2*b1 - b0)/((2**shape-1) * g)
            loc = b0 + scale*(1 - g)/shape


    gev_params = np.array([shape,loc,scale])
    #print(f"After fitting with method {method}, (shape,loc,scale) = \n{gev_params}")
        
    return gev_params

def pwm_shape_func(shape,b_ratio): # The function to solve: (3**shape-1)/(2**shape-1) - (3*b2-b0)/(2*b1-b0)
    if np.abs(shape) < 1e-6:
        return np.log(3)/np.log(2)*(1 + np.log(3/2)/2*shape - np.log(6)/4*shape**2) - b_ratio
        # Use local quadratic approximation
    return (3**shape - 1)/(2**shape - 1) - b_ratio

def hosking_shape_fprime_log(k, log_b_ratio):
    return np.log(3)/(3**k-1) - np.log(2)/(2**k-1)

def gev_return_time(x,T,shape,loc,scale):
    logsf = spgex.logsf(x,-shape,loc=loc,scale=scale)
    print(f'{logsf = }')
    rtime = convert_logsf_to_rtime(logsf,T)
    return logsf,rtime

def compute_return_time_block_maxima(x,T):
    block_maxima = compute_block_maxima(x,T)
    return compute_return_time_block_maxima_preblocked(block_maxima,T)

def compute_return_time_block_maxima_preblocked(block_maxima,T):
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

def compute_returnstats_preblocked(block_maxima, time_block_size, bounds=None):
    if bounds is None:
        bounds = [np.min(block_maxima),np.max(block_maxima)]
    bins = np.linspace(bounds[0]-1e-10,bounds[1]+1e-10,30)
    rlev,rtime,logsf,rtime_gev,logsf_gev,shape,loc,scale = compute_return_time_block_maxima_preblocked(block_maxima, time_block_size)
    idx = np.searchsorted(rlev, bins[:-1])
    print(f'{idx = }')
    logsf_gev,rtime_gev = gev_return_time(bins[:-1],time_block_size,shape,loc,scale)
    return bins[:-1], rtime[idx], logsf[idx], rtime_gev, logsf_gev, shape, loc, scale

def weighted_quantile(a, q, w, logscale=False):
    order = np.argsort(a)
    if logscale:
        log_cumweight = np.logaddexp.accumulate(w[order])
        i = np.argmax(log_cumweight - log_cumweight[-1] >= np.log(q))
    else:
        cumweight = np.cumsum(w[order])
        i = np.argmax(cumweight >= q*cumweight[-1])
    return a[order[i]]

