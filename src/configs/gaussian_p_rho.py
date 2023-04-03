from itertools import product
import os

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
    BY_rand_spec = {
        'method': 'eBH',
        'kwargs': {
            'correction': 'BY',
            'umi': 'single',
            **shared_kwargs
        },
        'pvalue': pvalue_spec
    }
    return [BH_spec, BY_spec, BY_rand_spec]


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


@register_experiment('gaussian_p_rho')
def gaussian_p_rho_exp(processes: int,
                       out_dir: str,
                       result_dir: str,
                       save_out: bool = True) -> None:

    trials = 500
    K = 100
    rhos = np.arange(0, 1., .1)
    null_mean = 0
    var = 1
    mus = np.linspace(1., 4., 10)
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
            'cov_mode': 'toeplitz' if neg is not True else 'uniform',
            'neg': neg,
            'seed': start_seed + idx
        }
    } for idx, (mu, p, rho,
                neg) in enumerate(product(mus, ps, rhos, [True, False]))]
    alg_names = ['BH', 'BY', 'BY (dynamic)']

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

    p_alg_names = ['BH', 'BY', 'BY (dynamic)']

    df_path = f'{out_dir}/res.csv'
    if not os.path.exists(df_path):
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

        df.to_csv(df_path)
    else:
        df = pd.read_csv(df_path)
    name_map = {
        'BH': 'BH',
        'BY': 'BY',
        'BY (dynamic)': 'U-BY',
    }
    df['Method'] = df['Method'].map(name_map)
    height = 2.4
    font_scale = 10
    gridspec_kws = {'wspace': 0.2}
    for neg in [True, False]:
        neg_df = df[df['Neg'] == neg]
        by_heatmap_fg = make_diff_heatmap(
            neg_df,
            '$\\rho$',
            '$\\mu$',
            'Power',
            'Method', [
                (name_map['BY (dynamic)'], name_map['BY']),
            ],
            height=height,
            font_scale=font_scale,
            gridspec_kws=gridspec_kws)
        save_figure(by_heatmap_fg,
                    os.path.join(result_dir, f'by_heatmap_neg={neg}'),
                    dpi=300)
        by_heatmap_fg_fdr_diff = make_diff_heatmap(
            neg_df,
            '$\\rho$',
            '$\\mu$',
            'FDR',
            'Method', [
                (name_map['BY (dynamic)'], name_map['BY']),
            ],
            height=height,
            font_scale=font_scale,
            gridspec_kws=gridspec_kws)
        save_figure(by_heatmap_fg_fdr_diff,
                    os.path.join(result_dir, f'by_heatmap_neg={neg}_fdr_diff'),
                    dpi=300)

        by_heatmap_fg_fdr = make_heatmap(
            neg_df,
            '$\\rho$',
            '$\\mu$',
            'FDR',
            'Method', [name_map['BY'], name_map['BY (dynamic)']],
            map_kwargs={
                'center': 0.,
                'cmap': 'bwr',
                'xticklabels': 3,
                'yticklabels': 3,
                'cbar_kws': {
                    'shrink': 0.8
                },
                'square': True
            },
            height=height,
            font_scale=font_scale,
            gridspec_kws=gridspec_kws)
        save_figure(by_heatmap_fg_fdr,
                    os.path.join(result_dir, f'by_heatmap_neg={neg}_fdr'),
                    dpi=300)
