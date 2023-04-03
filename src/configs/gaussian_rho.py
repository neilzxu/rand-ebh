from itertools import product
import os

import matplotlib
from matplotlib.colors import LinearSegmentedColormap

import matplotlib.pyplot as plt
import numpy as np  # NOQA
import pandas as pd
from exp import load_results, register_experiment, run_exp, setup_df
import metrics
from utils import make_diff_heatmap, make_line_plot, save_figure, make_heatmap


def make_datapoints(results):
    def get_dps(result):
        bool_alternates = np.floor(
            result['alternates']).astype(int).astype(bool)
        fdps = metrics.fdp(bool_alternates, result['rejsets'])[:, -1]
        tdps = metrics.tdp(bool_alternates, result['rejsets'])[:, -1]
        for alg in result['instances']:
            assert np.sum(alg.dynamic_rej_mask) >= np.sum(alg.rej_mask)

        return fdps, tdps, [
            alg.alpha_G_inv for alg in result['instances']
        ], result['pvalues'], [alg.e_values for alg in result['instances']], [
            alg.dynamic_uni for alg in result['instances']
        ]

    datapoints = [
        {
            'Method': result['name'],
            '$\\mu$': result['data_spec']['kwargs']['mu'],
            '$\\rho$': result['data_spec']['kwargs']['rho'],
            '$p$': result['data_spec']['kwargs']['p'],
            '$K$': result['data_spec']['kwargs']['K'],
            'Neg': result['data_spec']['kwargs']['neg'],
            'FDR': fdp,
            'Power': tdp,
            'alpha_levels': np.array(alpha),
            'pvalues': pvalues,
            'e_values': np.stack(e_values),
        } for result in results
        for fdp, tdp, alpha, pvalues, e_values, uni in zip(*get_dps(result))
    ]
    return datapoints


def build_alg_specs(alpha: float, null_mean: float, var: float, K: int,
                    trials: int, start_seed: float):
    evalue_spec = {
        'method': 'e_gauss',
        'kwargs': {
            'null_mean': 0,
            'var': var,
            'alpha': 2 * alpha / K
        }
    }
    pvalue_spec = {
        'method': 'p_gauss_cdf',
        'kwargs': {
            'null_mean': 0,
            'var': var
        }
    }

    shared_kwargs = {
        'alpha': alpha,
        'SEEDS': [start_seed + i for i in range(trials)]
    }

    eBH_spec = {
        'method': 'eBH',
        'kwargs': {
            **shared_kwargs
        },
        'pvalue': evalue_spec
    }
    eBH_round_spec = {
        'method': 'eBH',
        'kwargs': {
            'stoch_round': True,
            **shared_kwargs
        },
        'pvalue': evalue_spec
    }
    eBH_dynamic_spec = {
        'method': 'eBH',
        'kwargs': {
            'dynamic': 'uni',
            **shared_kwargs
        },
        'pvalue': evalue_spec
    }
    eBH_round_dynamic_spec = {
        'method': 'eBH',
        'kwargs': {
            'stoch_round': True,
            'dynamic': 'ind',
            **shared_kwargs
        },
        'pvalue': evalue_spec
    }
    eBH_umi_spec = {
        'method': 'eBH',
        'kwargs': {
            'umi': 'single',
            **shared_kwargs
        },
        'pvalue': evalue_spec
    }
    eBH_round_umi_spec = {
        'method': 'eBH',
        'kwargs': {
            'stoch_round': True,
            'umi': 'single',
            **shared_kwargs
        },
        'pvalue': evalue_spec
    }
    BH_spec = {
        'method': 'eBH',
        'kwargs': {
            **shared_kwargs
        },
        'pvalue': pvalue_spec
    }
    BY_spec = {
        'method': 'eBH',
        'kwargs': {
            'correction': 'BY',
            **shared_kwargs
        },
        'pvalue': pvalue_spec
    }
    BY_dynamic_spec = {
        'method': 'eBH',
        'kwargs': {
            'correction': 'BY',
            'umi': 'single',
            **shared_kwargs
        },
        'pvalue': pvalue_spec
    }
    BY_umi_spec = {
        'method': 'eBH',
        'kwargs': {
            'correction': 'BY',
            'umi': 'ind',
            **shared_kwargs
        },
        'pvalue': evalue_spec
    }
    return [
        eBH_spec, eBH_round_spec, eBH_dynamic_spec, eBH_round_dynamic_spec,
        eBH_umi_spec, eBH_round_umi_spec, BH_spec, BY_spec, BY_dynamic_spec,
        BY_umi_spec
    ]


def debug_data(df):
    alpha_G_inv = df.loc[0, 'alpha_levels'][:, np.newaxis]
    e_values = df.loc[0, 'e_values']
    uni = np.random.uniform(0., 1., size=e_values.shape)
    print(uni.shape, e_values.shape, alpha_G_inv.shape)
    recovery_e = (uni <= (e_values / alpha_G_inv)).astype(int) * alpha_G_inv
    r_values = np.maximum(e_values, recovery_e)
    avg_r_values = np.mean(r_values, axis=0)
    print(
        f'Excess: {np.sum(avg_r_values >= 1)} / ({np.sum(avg_r_values >= 1) / len(avg_r_values)})'
    )

    print(f'Mean: {np.mean(avg_r_values)}')


@register_experiment('gaussian_rho')
def gaussian_rho_exp(processes: int,
                     out_dir: str,
                     result_dir: str,
                     save_out: bool = True,
                     run_mode: str = 'from_spec') -> None:
    trials = 500
    K = 100
    rhos = np.arange(0, 1., .1)
    null_mean = 0
    var = 1
    mus = np.linspace(1, 4., 10)
    ps = [0.3]
    n = 1
    alpha = 0.05
    start_seed = 322

    data_specs = [{
        'method': 'gaussian',
        'kwargs': {
            'mu': mu,
            'p': p,
            'K': K,
            'n': n,
            'rho': rho,
            'var': var,
            'trials': trials,
            'cov_mode': 'toeplitz' if not neg else 'uniform',
            'neg': neg,
            'seed': start_seed + idx
        }
    } for idx, (mu, p, rho,
                neg) in enumerate(product(mus, ps, rhos, [True, False]))]
    alg_names = [
        'e-BH', 'e-BH (round)', 'e-BH (dynamic)', 'e-BH (round, dynamic)',
        'e-BH (UMI)', 'e-BH (round, UMI)', 'BH', 'BY', 'BY (dynamic)',
        'BY (UMI)'
    ]

    all_names = alg_names * len(data_specs)
    all_flat_args = []
    for data_idx, data_spec in enumerate(data_specs):
        alg_specs = build_alg_specs(alpha,
                                    null_mean,
                                    var,
                                    K=K,
                                    trials=trials,
                                    start_seed=2 * data_idx * trials +
                                    start_seed)

        all_flat_args += [(data_idx, alg_spec) for alg_spec in alg_specs]
    if run_mode == 'from_spec':
        all_results = run_exp(out_dir=out_dir,
                              data_specs=data_specs,
                              data_alg_specs=all_flat_args,
                              data_alg_names=all_names,
                              processes=processes,
                              save_result=save_out,
                              long_exp=False)
    if save_out:
        all_results = load_results(out_dir)
        df = setup_df(all_results, make_datapoints)
    else:
        df = setup_df(all_results, make_datapoints)

    name_map = {
        'e-BH': 'e-BH',
        'e-BH (round)': 'R$_1$-eBH',
        'e-BH (dynamic)': 'R$_2$-eBH',
        'e-BH (round, dynamic)': 'R-eBH',
        'e-BH (UMI)': 'U-eBH',
        'e-BH (round, UMI)': 'R$_1$+U-eBH',
        'BH': 'BH',
        'BY': 'BY',
        'BY (dynamic)': 'R-BY',
        'BY (UMI)': 'BY+$U/X$'
    }
    df['Method'] = df['Method'].map(name_map)
    height = 4
    font_scale = 50
    gridspec_kws = {'wspace': 0.2}
    for neg in [True, False]:
        neg_df = df[df['Neg'] == neg]
        formatter = matplotlib.ticker.LogFormatter(2, labelOnlyBase=False)

        ebh_base_fg = make_diff_heatmap(
            neg_df[(neg_df['$\\mu$'] >= 1) & (neg_df['$\\mu$'] <= 4)],
            '$\\rho$',
            '$\\mu$',
            'Power',
            'Method', [
                (name_map['e-BH (round)'], 'e-BH'),
                (name_map['e-BH (dynamic)'], 'e-BH'),
                (name_map['e-BH (round, dynamic)'], 'e-BH'),
                (name_map['e-BH (UMI)'], 'e-BH'),
            ],
            height=height * .6,
            font_scale=font_scale * 1.5,
            cmap=LinearSegmentedColormap.from_list("bwrr",
                                                   colors=[(0, 'blue'),
                                                           (0.5, 'white'),
                                                           (0.55, (1., .5,
                                                                   .5)),
                                                           (1., 'red')]),
            gridspec_kws=gridspec_kws)
        save_figure(ebh_base_fg,
                    os.path.join(result_dir,
                                 f'ebh_base_heatmap_neg={neg}_power'),
                    dpi=500)
        ebh_base_fdr_diff_fg = make_diff_heatmap(
            neg_df[(neg_df['$\\mu$'] >= 1) & (neg_df['$\\mu$'] <= 4)],
            '$\\rho$',
            '$\\mu$',
            'FDR',
            'Method', [
                (name_map['e-BH (round)'], 'e-BH'),
                (name_map['e-BH (dynamic)'], 'e-BH'),
                (name_map['e-BH (round, dynamic)'], 'e-BH'),
                (name_map['e-BH (UMI)'], 'e-BH'),
            ],
            height=height * .6,
            font_scale=font_scale * 1.5,
            gridspec_kws=gridspec_kws)
        save_figure(ebh_base_fdr_diff_fg,
                    os.path.join(result_dir,
                                 f'ebh_base_heatmap_neg={neg}_fdr_diff'),
                    dpi=500)

        ebh_base_fdr_fg = make_heatmap(
            neg_df[(neg_df['$\\mu$'] >= 1) & (neg_df['$\\mu$'] <= 4)],
            '$\\rho$',
            '$\\mu$',
            'FDR',
            'Method', [
                'e-BH', name_map['e-BH (round)'], name_map['e-BH (dynamic)'],
                name_map['e-BH (round, dynamic)'], name_map['e-BH (UMI)']
            ],
            map_kwargs={
                'center': 0.00,
                'cmap': 'bwr',
                'xticklabels': 3,
                'yticklabels': 3,
                'cbar_kws': {
                    'shrink': 0.8
                },
                'square': True
            },
            height=height * .6,
            font_scale=font_scale * 1.5,
            gridspec_kws=gridspec_kws)
        save_figure(ebh_base_fdr_fg,
                    os.path.join(result_dir,
                                 f'ebh_base_heatmap_neg={neg}_fdr'),
                    dpi=500)

        ebh_abl_fg = make_diff_heatmap(
            neg_df[(neg_df['$\\mu$'] >= 1) & (neg_df['$\\mu$'] <= 4)],
            '$\\rho$',
            '$\\mu$',
            'Power',
            'Method',
            [(name_map['e-BH (round, dynamic)'], name_map['e-BH (round)']),
             (name_map['e-BH (round, dynamic)'], name_map['e-BH (dynamic)'])],
            height=height * .75,
            font_scale=font_scale,
            gridspec_kws=gridspec_kws)
        save_figure(ebh_abl_fg,
                    os.path.join(result_dir,
                                 f'ebh_ablation_heatmap_neg={neg}_power'),
                    dpi=300)

        ebh_abl_bad_fg = make_diff_heatmap(neg_df[(neg_df['$\\mu$'] >= 1)
                                                  & (neg_df['$\\mu$'] <= 4)],
                                           '$\\rho$',
                                           '$\\mu$',
                                           'Power',
                                           'Method', [('e-BH', 'BY+$U/X$')],
                                           height=height,
                                           font_scale=font_scale,
                                           gridspec_kws=gridspec_kws)
        save_figure(ebh_abl_bad_fg,
                    os.path.join(result_dir,
                                 f'ebh_ablation_umi_heatmap_neg={neg}_power'),
                    dpi=300)
