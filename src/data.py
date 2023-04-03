from typing import Optional
import numpy as np

from utils import build_registry, H_K

_DATA_MAP, data, get_data_method = build_registry()


@data('gaussian')
def gaussian(K: int,
             p: float,
             mu: float,
             n: int,
             trials: int,
             var: float,
             rho: float,
             seed: Optional[float],
             cov_mode: str = 'uniform',
             neg: bool = False) -> np.ndarray:
    assert rho >= 0 and rho <= 1
    # null_mean is 0
    assert p >= 0 and p <= 1
    rng = np.random.default_rng(seed)
    non_null_ct = int(np.floor(p * K))
    alternate = np.concatenate(
        [np.ones(non_null_ct), np.zeros(K - non_null_ct)])
    alternates = np.stack([rng.permutation(alternate)
                           for _ in range(trials)])[:, :, np.newaxis]
    assert np.sum(alternates) == trials * non_null_ct

    if cov_mode == 'uniform':
        cov = np.full((K, K), rho * (-1 if neg else 1) * var / (K - 1))
        cov[list(zip(np.arange(K), np.arange(K)))] = var
    else:
        assert cov_mode == 'toeplitz'
        cov = np.diag(np.full(K, var)).astype(float)
        upper_tri = np.zeros((K, K))
        for diff in range(1, K):
            diff_indices = list(zip(np.arange(K - diff), np.arange(diff, K)))
            upper_tri[diff_indices] = np.power(rho, diff) * var
        off_diag = upper_tri.astype(float) + upper_tri.T.astype(float)
        cov += (off_diag * var)
    Z = rng.multivariate_normal(np.zeros(K), cov,
                                size=(trials, n)).transpose(0, 2, 1)
    Z += (mu * alternates)

    # Z is trials x K x n, alternates is trials x K x 1
    return Z, alternates[:, :, 0]


@data('bandit_growth')
def bandit_growth(K: int, mu: float, theta: float, n: int, trials: int,
                  seed: Optional[float]):
    assert theta >= 0 and theta <= 1
    rng = np.random.default_rng(seed)
    s = rng.exponential(scale=mu, size=(trials, K, 1))
    alternate_p = (theta * (K - np.arange(1, K + 1))) / (K + 1)
    alternates = rng.binomial(1, alternate_p, size=(trials, ))[:, :,
                                                               np.newaxis]
    Z = rng.normal(0, scale=1, size=(trials, K, n)) + (s * alternates)
    return Z.transpose(1, 0, 2)


@data('guo_rao_2008')
def guo_rao_2008(K: int, alpha: float, pi_0: float, trials: int,
                 seed: Optional[float]):
    """P-value joint distribution that is tight with the BY FDR bound (under
    arbitrary dependence)

    :param K: number of hypotheses
    :param alpha: FDR of BY applied to p-values
    :param pi_0: proportion of null hypotheses (take ceiling if
        np.ceil(pi_0 * non- integer))
    :param trials: number of trials
    :param seed: rng seed
    :return: trials x K array of p-values
    """
    rng = np.random.default_rng(seed)
    m_0 = int(np.ceil(pi_0 * K))

    thresh = alpha / H_K(K)
    coef = thresh / K
    null_p = (m_0 * coef) / np.arange(1, m_0 + 1)
    alt_p = np.full((K - m_0, ), coef)
    rem_p = 1 - (np.sum(null_p) + np.sum(alt_p))
    p = np.concatenate([null_p, alt_p, np.array([rem_p])])
    N = rng.choice(np.arange(K + 1) + 1, p=p, size=(trials, ))
    u = rng.uniform(size=(trials, 2))
    P = np.zeros(shape=(trials, K))

    null_indices = np.arange(m_0)
    alt_indices = np.arange(m_0, K)

    for i in range(trials):
        N_val = N[i]
        U_Kp1 = thresh + (u[i, 0] * (1 - thresh))
        if N_val <= m_0:
            N_indices = rng.choice(null_indices, size=N_val, replace=False)
            N_mask = np.zeros(K).astype(bool)
            N_mask[N_indices] = True
            P[i, N_mask] = coef * (N_val + u[i, 1] - 1)
            P[i, ~N_mask] = U_Kp1
        elif N_val < K + 1:
            N_indices = rng.choice(alt_indices,
                                   size=N_val - m_0,
                                   replace=False)
            N_mask = np.zeros(K).astype(bool)
            N_mask[N_indices] = True
            U_N = coef * (N_val + u[i, 1] - 1)
            P[i, :m_0] = U_N
            P[i, N_mask] = U_N
            P[i, (~N_mask) & (np.arange(K) > m_0)] = U_Kp1
        else:
            P[i, :m_0] = U_Kp1
            P[i, m_0:] = 1.

    alternates = np.tile(np.concatenate([np.zeros(m_0),
                                         np.ones(K - m_0)]), (trials, 1))
    return P, alternates
