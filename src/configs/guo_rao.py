from itertools import product
import os

import matplotlib.pyplot as plt
import numpy as np  # NOQA
import pandas as pd
import seaborn as sns

from exp import load_results, register_experiment, run_exp, setup_df
import metrics
from utils import make_diff_heatmap, make_line_plot, save_figure, make_heatmap, gr_tight_fdr, H_K


def make_datapoints(results):
    def get_dps(result):
        bool_alternates = np.floor(
            result['alternates']).astype(int).astype(bool)
        fdps = metrics.fdp(bool_alternates, result['rejsets'])[:, -1]
        tdps = metrics.tdp(bool_alternates, result['rejsets'])[:, -1]

        threshs = [alg.alpha_G_inv for alg in result['instances']]
        evalues = [alg.e_values for alg in result['instances']]
        return fdps, tdps, threshs, result['pvalues'], evalues

    datapoints = [
        {
            'Method': result['name'],
            '$\\pi_0$': result['data_spec']['kwargs']['pi_0'],
            '$K$': result['data_spec']['kwargs']['K'],
            'FDR': fdp,
            'Power': tdp,
            'alpha_levels': np.array(alpha),
            'pvalues': pvalues,
            'evalues': evalues,
        } for result in results
        for fdp, tdp, alpha, pvalues, evalues in zip(*get_dps(result))
    ]
    return datapoints


alg_names = ['BH', 'BY', 'BY (dynamic)']
name_map = {
    'BH': 'BH',
    'BY': 'BY',
    'BY (debug)': 'BY (debug)',
    'BY (dynamic)': 'U-BY',
}


def build_alg_specs(alpha: float, trials: int, start_seed: float):
    pvalue_spec = {'method': 'identity', 'kwargs': {}}

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
    BY_debug_spec = {
        'method': 'BY',
        'kwargs': {
            'alpha': alpha
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

    res = [BH_spec, BY_spec, BY_rand_spec]
    assert len(res) == len(alg_names)
    return res


def plot_w_ref_line(*, x_ref, y_ref, ref_name, ref_alpha, ax, **kwargs):
    sns.lineplot(ax=ax, **kwargs)
    ax.plot(x_ref, y_ref, linestyle='dashed', color='black', label=ref_name)
    ax.annotate(ref_name,
                xy=(x_ref[-2], y_ref[-2]),
                xytext=(1.1 * x_ref[-1], 0.5 * y_ref[-2]),
                arrowprops=dict(facecolor='black',
                                shrink=0.05,
                                width=3,
                                headwidth=5,
                                headlength=5))
    ax.axhline(xmin=0.0,
               xmax=1.,
               y=ref_alpha,
               linestyle='dashed',
               color='gray')
    ax.text(x=x_ref[1],
            y=1.05 * ref_alpha,
            s=f'$\\alpha={ref_alpha}$',
            color='gray')


@register_experiment('guo_rao')
def guo_rao(processes: int,
            out_dir: str,
            result_dir: str,
            save_out: bool = True) -> None:

    trials = 1000
    Ks = [int(x) for x in np.ceil(np.geomspace(start=10, stop=10000, num=9))]
    pis = np.arange(0.1, 1.1, 0.1)
    alpha = 0.05
    start_seed = 322

    data_specs = [{
        'method': 'guo_rao_2008',
        'kwargs': {
            'K': K,
            'alpha': alpha,
            'pi_0': pi_0,
            'trials': trials,
            'seed': start_seed + idx
        }
    } for idx, (K, pi_0) in enumerate(product(Ks, pis))]

    all_names = alg_names * len(data_specs)
    all_flat_args = []
    for data_idx, data_spec in enumerate(data_specs):
        alg_specs = build_alg_specs(alpha,
                                    trials=trials,
                                    start_seed=2 * data_idx * trials +
                                    start_seed)

        all_flat_args += [(data_idx, alg_spec) for alg_spec in alg_specs]

    df_path = f'{out_dir}/res.csv'
    if not os.path.exists(df_path):

        all_results = run_exp(out_dir=out_dir,
                              data_specs=data_specs,
                              data_alg_specs=all_flat_args,
                              data_alg_names=all_names,
                              processes=processes,
                              save_result=save_out,
                              cache_data=True,
                              long_exp=False)
        if save_out:
            all_results = load_results(out_dir)
            df = setup_df(all_results, make_datapoints)
        else:
            df = setup_df(all_results, make_datapoints)

        df.to_csv(df_path)
    else:
        df = pd.read_csv(df_path)
    df['Method'] = df['Method'].map(name_map)
    height = 2.4
    font_scale = 10
    gridspec_kws = {'wspace': 0.2}
    by_heatmap_fg = make_diff_heatmap(
        df,
        '$\\pi_0$',
        '$K$',
        'Power',
        'Method', [
            (name_map['BY (dynamic)'], name_map['BY']),
        ],
        height=height,
        font_scale=font_scale,
        gridspec_kws=gridspec_kws)
    save_figure(by_heatmap_fg,
                os.path.join(result_dir, 'guo_rao_power_diff'),
                dpi=300)

    by_heatmap_fg_fdr = make_heatmap(
        df,
        '$\\pi_0$',
        '$K$',
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
            'square': True,
            'annot': True
        },
        height=height,
        font_scale=font_scale,
        gridspec_kws=gridspec_kws)
    save_figure(by_heatmap_fg_fdr,
                os.path.join(result_dir, 'guo_rao_fdr'),
                dpi=300)

    K = Ks[-1]
    sns.set_theme()
    fig, ax = plt.figure(figsize=(8, 4)), plt.gca()
    print(df['$K$'].unique())
    plot_w_ref_line(
        ax=ax,
        x_ref=pis,
        y_ref=[gr_tight_fdr(np.ceil(pi * K), alpha / H_K(K), K) for pi in pis],
        ref_name='Theoretical\nexact FDR',
        ref_alpha=alpha,
        data=df[(df['$K$'] == K)],
        x='$\\pi_0$',
        y='FDR',
        hue='Method',
        ci=95)
    ax.set_xlim((0.1, 1.))
    fig.tight_layout()
    save_figure(fig, os.path.join(result_dir, 'guo_rao_fdr_line'), dpi=300)
