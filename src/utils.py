import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def H_K(K: int) -> float:
    """Kth harmonic number."""
    return np.sum(1. / np.arange(1, K + 1))


def gr_tight_fdr(K_0: int, alpha: float, K: int):
    """Gets the tight FDR bound for the Guo and Rao (2008) example."""
    return (K_0 / K) * alpha * H_K(K_0)


class MidpointNormalize(mpl.colors.Normalize):
    """Normalise the colorbar."""
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        mpl.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        low_shift = (self.midpoint - self.vmin) / (self.vmax -
                                                   self.midpoint) * .5
        x, y = [self.vmin, self.midpoint, self.vmax], [.5 - low_shift, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))


def make_line_plot(df, x_val, alg_names, rows, alpha=None, **fg_kwargs):
    sns.set_theme()
    sns.set_context('poster')
    fg = sns.relplot(data=df,
                     x=x_val,
                     y='value',
                     col='variable',
                     hue='Method',
                     style='Method',
                     hue_order=alg_names,
                     legend='full',
                     kind='line',
                     height=6,
                     facet_kws={'sharey': False},
                     **fg_kwargs)
    for metric, ax in fg.axes_dict.items():
        ax.set_ylabel(metric)
        ax.set_title('')
        if metric == 'FDR' and alpha is not None:
            ax.axhline(alpha, color='black', linestyle='dashed')
    fg.tight_layout(rect=[0, 0.1, 1, 1], pad=1)
    lh = fg.axes[0, 0].get_legend_handles_labels()

    fg.figure.legend(*lh,
                     loc='center',
                     ncol=len(lh[0]) // rows,
                     bbox_to_anchor=(0, 0, 1, 0.1))
    return fg


def make_diff_heatmap(df,
                      x,
                      y,
                      metric,
                      method,
                      method_pairs,
                      center=0,
                      font_scale=5.5,
                      cmap='bwr',
                      map_kwargs={},
                      cbar_kws={'shrink': 0.8},
                      **fg_kwargs):
    """Heatmap of mean of metric (method 1) - mean of metric (method 2)"""

    with sns.plotting_context(font_scale=font_scale):
        method_names = {name for pair in method_pairs for name in pair}
        df = df[df[method].isin(method_names)]
        mean_df = df.loc[:, [x, y, method, metric]].groupby([x, y,
                                                             method]).mean()
        pivot_df = mean_df.reset_index().pivot(index=[x, y],
                                               columns=method,
                                               values=metric).reset_index()

        pair_names = []
        for name_1, name_2 in method_pairs:
            pair_name = f'{name_1} vs. {name_2}'
            pivot_df[pair_name] = pivot_df[name_1] - pivot_df[name_2]
            pair_names.append(pair_name)
        final_df = pivot_df.drop(method_names,
                                 axis=1).melt(id_vars=[x, y],
                                              value_vars=pair_names,
                                              var_name='Pair',
                                              value_name='Diff')

        vmin, vmax = final_df['Diff'].min(), final_df['Diff'].max()

        fg = sns.FacetGrid(data=final_df,
                           col='Pair',
                           col_order=pair_names,
                           sharex=False,
                           sharey=True,
                           **fg_kwargs)

        def draw_heatmap(*args, **kwargs):
            data = kwargs.pop('data')

            def float_str(s):
                return f"{float(s):.1f}"

            data = data.copy()
            data.loc[:, args[0]] = [float_str(s) for s in data[args[0]]]
            data.loc[:, args[1]] = [float_str(s) for s in data[args[1]]]
            d = data.pivot(index=args[1], columns=args[0], values=args[2])
            print(f'Heatmap diff value range: [{d.min()}, {d.max()}]')
            ax = plt.gca()
            cbar = True
            sns.heatmap(d, ax=ax, cbar=cbar, **kwargs)

        cbar_ax = fg.fig.add_axes([.92, .3, .02 / len(method_pairs),
                                   .4])  # <-- Create a colorbar axes

        fg.map_dataframe(draw_heatmap,
                         x,
                         y,
                         'Diff',
                         vmin=vmin,
                         vmax=vmax,
                         cmap=cmap,
                         center=center,
                         cbar_ax=cbar_ax,
                         xticklabels=3,
                         yticklabels=3,
                         cbar_kws=cbar_kws,
                         square=True,
                         **map_kwargs)

        if len(fg.axes.shape) == 2:
            y_ticklabels = fg.axes[0, 0].get_yticklabels()
        else:
            y_ticklabels = fg.axes[0].get_yticklabels()
        for idx, (pair, ax) in enumerate(fg.axes_dict.items()):
            # set aspect of all axis
            ax.set_title(pair)
            if idx == 0:
                ax.set_yticklabels(
                    [label.get_text() for label in y_ticklabels], rotation=0)
            ax.set_xticklabels(
                [f"{float(x.get_text()):.1f}" for x in ax.get_xticklabels()])
        fg.fig.subplots_adjust(right=.9)
        return fg


def make_heatmap(df,
                 x,
                 y,
                 metric,
                 method,
                 method_names,
                 font_scale=5.5,
                 map_kwargs={},
                 **fg_kwargs):
    """Heatmap of mean of metric (method 1) - mean of metric (method 2)"""

    df = df[df[method].isin(method_names)]
    final_df = df.loc[:,
                      [x, y, method, metric]].groupby([x, y, method
                                                       ]).mean().reset_index()
    with sns.plotting_context(font_scale=font_scale):
        fg = sns.FacetGrid(data=final_df,
                           col=method,
                           col_order=method_names,
                           sharex=False,
                           sharey=True,
                           **fg_kwargs)

        def draw_heatmap(*args, **kwargs):
            data = kwargs.pop('data')

            def float_str(s):
                return f"{float(s):.1f}"

            data = data.copy()
            data.loc[:, args[0]] = [float_str(s) for s in data[args[0]]]
            data.loc[:, args[1]] = [float_str(s) for s in data[args[1]]]
            d = data.pivot(index=args[1], columns=args[0], values=args[2])
            ax = plt.gca()
            cbar = True
            sns.heatmap(d, ax=ax, cbar=cbar, **kwargs)

        cbar_ax = fg.fig.add_axes([.92, .3, .02 / len(method_names),
                                   .4])  # <-- Create a colorbar axes
        vmin, vmax = final_df[metric].min(), final_df[metric].max()
        fg.map_dataframe(draw_heatmap,
                         x,
                         y,
                         metric,
                         cbar_ax=cbar_ax,
                         vmin=vmin,
                         vmax=vmax,
                         **map_kwargs)

        if len(fg.axes.shape) == 2:
            y_ticklabels = fg.axes[0, 0].get_yticklabels()
        else:
            y_ticklabels = fg.axes[0].get_yticklabels()
        for idx, (pair, ax) in enumerate(fg.axes_dict.items()):
            # set aspect of all axis
            ax.set_title(pair)
            if idx == 0:
                ax.set_yticklabels(
                    [label.get_text() for label in y_ticklabels], rotation=0)
            ax.set_xticklabels(
                [f"{float(x.get_text()):.1f}" for x in ax.get_xticklabels()])
        fg.fig.subplots_adjust(right=.9)
        return fg


def save_figure(fig, path, **kwargs):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    fig.savefig(path, **kwargs)


def build_registry():
    registry = {}

    def register_fn(name):
        def decorator(fn):
            registry[name] = fn
            return fn

        return decorator

    def get_dispatch(name):
        return registry[name]

    return registry, register_fn, get_dispatch
