from typing import Optional, Tuple

import numpy as np

from utils import build_registry, H_K

_ALG_MAP, alg, get_alg = build_registry()


def get_pvalue_levels(pvalues: np.ndarray,
                      alpha: float) -> Tuple[np.ndarray, np.ndarray]:
    """Round p-values to nearest multiple of alpha * i / K for integer i.
    :param pvalues: 1d array of p-values (i.e., in [0, 1])
    :param alpha: alpha level in [0, 1]
    :return: tuple of 1d arrays same size as pvalues. floor_level[i] <= pvalues[i] <= ceil_level[i]"""
    K, = pvalues.shape
    p_increment = alpha / K
    p_steps = pvalues / p_increment
    floor_level, ceil_level = np.floor(p_steps), np.ceil(p_steps)
    return floor_level, ceil_level


def get_evalue_pm(pvalues: np.ndarray,
                  alpha: float) -> Tuple[np.ndarray, np.ndarray]:
    """Convert one over e-values, convert back to e-values, and then round to
    nearest K / alpha i for integer i.

    :param pvalues: 1d array of 1 / e-values
    :param alpha: alpha level in [0, 1]
    :return: tuple of 1d arrays same size as pvalues. if pvalues = 1 / e_values, then e_floors[i] <= e_values[i] <= e_ceils[i]
    """
    K, = pvalues.shape
    floor_levels, ceil_levels = get_pvalue_levels(pvalues, alpha)
    e_floors, e_ceils = np.divide(K,
                                  alpha * ceil_levels,
                                  out=np.full(ceil_levels.shape, np.inf),
                                  where=ceil_levels != 0), np.divide(
                                      K,
                                      alpha * floor_levels,
                                      out=np.full(floor_levels.shape, np.inf),
                                      where=floor_levels != 0)
    assert np.all(e_ceils >= e_floors)
    return e_floors, e_ceils


@alg('BY')
class BY:
    """The Benjamini-Yekutieli procedure for controlling the FDR for p-values
    under arbitrary dependence."""

    def __init__(self, alpha: float) -> 'BY':
        """Initialize the Benjamini-Yekutieli procedure.

        :param alpha: level of desired FDR control
        :return: BY that will control FDR at level alpha
        """
        self.alpha = alpha
        self.alpha_G_inv = 1 / self.alpha

    def run_fdr(self, pvalues: np.ndarray) -> np.ndarray:
        """Run the Benjamini-Yekutieli procedure on a list of p-values.

        :param pvalues: 1d array of p-values (i.e., in [0, 1])
        :return: 1d boolean array of rejections
        """
        K, = pvalues.shape
        p_indices = np.argsort(pvalues)
        level = self.alpha / (K * H_K(K))
        mask = np.zeros(K).astype(bool)
        for i in range(K - 1, -1, -1):
            p_idx = p_indices[i]
            pvalue = pvalues[p_idx]
            order = i + 1
            if pvalue <= order * level or np.isclose(pvalue, order * level):
                mask[p_indices[:order]] = True
                break
        self.e_values = np.divide(1,
                                  pvalues,
                                  out=np.full(pvalues.shape, np.inf),
                                  where=pvalues != 0)
        return mask


@alg('eBH')
class eBH:
    """The e-BH procedure for controlling FDR for e-values under arbitrary
    dependence."""

    def __init__(self,
                 alpha: float,
                 stoch_round: bool = False,
                 dynamic: Optional[str] = None,
                 umi: Optional[str] = None,
                 correction: Optional[str] = None,
                 seed: Optional[float] = None):
        """Initialize the e-BH procedure.

        :param alpha: level of desired FDR control
        :param stoch_round: whether to use stochastic rounding to a constant grid (i.e., R_1-eBH)
        :param dynamic: whether to use stochastic rounding to a the e-BH threshold (i.e., R_2-eBH)
        :param umi: whether to divide each e-value by a uniform random variable U. 'single' for a single U and 'ind' for independent U_i (i.e., U-eBH or J-eBH).
        :param correction: whether to calibrate input p-values using BY calibrator (rather than assuming they are inverted e-values). 'BY' for the BY calibrator. Otherwise, the p-values are assumed to be inverted e-values.
        :param seed: seed for random number generator
        :return: eBH that will control FDR at level alpha with specified options.
        """
        self.alpha = alpha
        self.stoch_round = stoch_round
        self.dynamic = dynamic
        self.umi = umi
        self.correction = correction
        self.seed = seed

    def run_fdr(self, pvalues):
        """Run the eBH procedure on a list of inverted e-values.

        :param pvalues: 1d array of p-values (i.e., in [0, 1])
        :return: 1d boolean array of rejections
        """
        rng = np.random.default_rng(self.seed)
        K, = pvalues.shape
        alpha_rej = self.alpha
        if self.correction == 'BY':
            _, ceil_level = np.maximum(
                get_pvalue_levels(pvalues, self.alpha / H_K(K)), 1)
            e_values = K / (self.alpha * ceil_level)
            e_values[ceil_level > K] = 0
            pvalues = self.alpha * ceil_level / K
        else:
            e_values = np.divide(1,
                                 pvalues,
                                 out=np.full(pvalues.shape, np.inf),
                                 where=pvalues != 0)

        # Stochastic rounding
        if self.stoch_round:
            e_ceils, e_floors = get_evalue_pm(pvalues, self.alpha)
            diffs = e_ceils - e_floors
            coin_probs = np.divide(e_values - e_floors,
                                   diffs,
                                   out=np.full(e_values.shape, 0.),
                                   where=(diffs > 0) & (e_ceils != np.inf))
            coins = rng.binomial(1, coin_probs).round().astype(bool)

            # Explicitly set floor/ceil values to avoid multiplying e_ceil = inf by coin = 0
            new_e_values = np.zeros(e_values.shape)
            new_e_values[coins] = e_ceils[coins]
            new_e_values[~coins] = e_floors[~coins]
            e_values = new_e_values

        if self.umi is not None:
            if self.umi == 'single':
                uni = rng.uniform(0., 1.)
            else:
                uni = rng.uniform(0., 1., e_values.shape)
            self.umi_uni = uni
            e_values = np.divide(e_values,
                                 uni,
                                 out=np.full(pvalues.shape, np.inf),
                                 where=uni != 0)

        k_star = 1
        for i, e in zip(np.arange(1, K + 1), np.sort(e_values)[::-1]):
            thresh = K / (i * alpha_rej)
            if e >= thresh or np.isposinf(e) or np.isclose(e, thresh):
                k_star = i

        alpha_G_inv = K / (k_star * alpha_rej)
        self.alpha_G_inv = alpha_G_inv

        self.e_values = e_values
        if self.dynamic == 'single':
            uni = np.full(e_values.shape, rng.uniform(0., 1.))
        else:
            uni = rng.uniform(0., 1., size=e_values.shape)
        self.dynamic_uni = uni
        # Multiply by uniform RV if dynamic
        self.dynamic_rej_thresh = alpha_G_inv * uni
        self.rej_thresh = alpha_G_inv

        self.dynamic_rej_mask = (e_values >= self.dynamic_rej_thresh
                                 ) | np.isposinf(e_values) | np.isclose(
                                     e_values, self.dynamic_rej_thresh)
        self.rej_mask = (e_values >=
                         self.rej_thresh) | np.isposinf(e_values) | np.isclose(
                             e_values, self.rej_thresh)
        assert max(
            np.sum(self.rej_mask), 1
        ) == k_star, f'''(rejections, k_star): {(np.sum(self.rej_mask), k_star)}
rej_indices: {np.where(self.rej_mask)}
e_values: {e_values[self.rej_mask]}
sorted_evalues: {np.sort(e_values)[::-1]}
'''
        res_rej_mask = self.rej_mask if self.dynamic is None else self.dynamic_rej_mask
        return res_rej_mask


@alg('UeBH')
class UeBH(eBH):
    """Helper class for directly creating the Ue-BH procedure"""

    def __init__(self, alpha: float, seed: Optional[float] = None) -> 'UeBH':
        """Initialize the e-BH procedure.

        :param alpha: level of desired FDR control
        :param seed: seed for random number generator
        :return: instance of UeBH"""
        super(self, UeBH).__init(self, alpha=alpha, umi='single', seed=seed)
