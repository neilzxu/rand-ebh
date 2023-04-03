from itertools import product
import os

import matplotlib.pyplot as plt
import numpy as np  # NOQA
import pandas as pd
import seaborn as sns

from exp import load_results, register_experiment, run_exp, setup_df
import metrics
from utils import make_diff_heatmap, make_line_plot, save_figure


def make_datapoints(results):
    def get_dps(result):
        bool_alternates = np.floor(
            result['alternates']).astype(int).astype(bool)
        fdps = metrics.fdp(bool_alternates, result['rejsets'])[:, -1]
        tdps = metrics.tdp(bool_alternates, result['rejsets'])[:, -1]

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
            'dynamic': 'ind',
            **shared_kwargs
        },
        'pvalue': evalue_spec
    }
    eBH_round_dynamic_spec = {
        'method': 'eBH',
        'kwargs': {
            'stoch_round': True,
            'dynamic': True,
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
            'dynamic': True,
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
        BH_spec, BY_spec, BY_dynamic_spec, BY_umi_spec
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


@register_experiment('gaussian')
def gaussian_exp(processes: int,
                 out_dir: str,
                 result_dir: str,
                 save_out: bool = True) -> None:
    trials = 200
    K = 100
    rhos = [0.1]
    null_mean = 0
    var = 1
    mus = np.arange(0, 10., 1)
    ps = np.arange(0, 1., .1)
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
            'seed': start_seed + idx
        }
    } for idx, (mu, p, rho) in enumerate(product(mus, ps, rhos))]
    alg_names = [
        'e-BH', 'e-BH (round)', 'e-BH (dynamic)', 'e-BH (round, dynamic)',
        'BH', 'BY', 'BY (dynamic)', 'BY (UMI)'
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
        'e-BH (round)': 'R$_1$-e-BH',
        'e-BH (dynamic)': 'R$_2$-e-BH',
        'e-BH (round, dynamic)': 'R$_{12}$-e-BH',
        'BH': 'BH',
        'BY': 'BY',
        'BY (dynamic)': 'R$_2$-BY',
        'BY (UMI)': 'BY (UMI)'
    }
    alg_renames = [name_map[name] for name in alg_names]
    df['Method'] = df['Method'].map(name_map)
    mu_choice = 2
    rho_choice = 0.1

    sns.set_theme()
    sns.set_context('paper')

    p_comp_df = pd.melt(df[(df['$\\mu$'] == mu_choice)
                           & (df['$\\rho$'] == rho_choice)],
                        id_vars=['$p$', 'Method'],
                        value_vars=['FDR', 'Power'])
    fg = make_line_plot(p_comp_df, '$p$', alg_renames, alpha=alpha, rows=2)
    save_figure(fg.figure,
                os.path.join(result_dir, f'p_comp_mu={mu_choice:.0E}'))

    p_choice = 0.2
    mu_comp_df = pd.melt(df[(df['$p$'] == p_choice)
                            & (df['$\\rho$'] == rho_choice)],
                         id_vars=['$\\mu$', 'Method'],
                         value_vars=['FDR', 'Power'])
    fg = make_line_plot(mu_comp_df, '$\\mu$', alg_renames, alpha=alpha, rows=2)
    save_figure(fg, os.path.join(result_dir, f'mu_comp_p={p_choice:.0E}'))
