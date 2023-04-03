import numpy as np
import scipy.stats

from utils import build_registry

_PVAL_MAP, pvalue, get_pvalue = build_registry()


@pvalue('e_gauss')
def e_gauss(x, null_mean=0, var=1, alpha=0.05):
    trials, hypotheses, sample_size = x.shape
    lam = np.sqrt(2 * np.log(1 / alpha) / (var * sample_size))
    log_e = (lam * (x - null_mean) - (np.square(lam) * var / 2)).sum(axis=2)
    return np.exp(-1 * log_e)


@pvalue('p_gauss_cdf')
def p_gauss_cdf(x, null_mean=0, var=1):
    trials, hypotheses, sample_size = x.shape
    return 1. - scipy.stats.norm.cdf(x.sum(axis=2),
                                     loc=null_mean * sample_size,
                                     scale=np.sqrt(var * sample_size))


@pvalue('e_nn')
def e_nn(x):
    pass


@pvalue('identity')
def identity(x):
    return x
