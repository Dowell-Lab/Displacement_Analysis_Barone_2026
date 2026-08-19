import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests


def _split_subsets(meta_df, T_clust_genes, GC_clust_genes, genes_to_remove):
    '''
    * Takes:
    - meta_df         : DataFrame with a 'Gene' column and per-gene stats.
    - T_clust_genes    : iterable of gene names in the T-rich cluster.
    - GC_clust_genes   : iterable of gene names in the GC-rich cluster.
    - genes_to_remove  : iterable of gene names to exclude entirely.

    * Outputs:
    - gc_df, t_df : DataFrames restricted to each subset, each tagged with
                    a 'Subset' column ('GC-Rich' / 'Upstream T-rich').
    '''
    df = meta_df[~meta_df['Gene'].isin(genes_to_remove)].copy()

    gc_df = df[df['Gene'].isin(GC_clust_genes)].copy()
    gc_df['Subset'] = 'GC-Rich'

    t_df = df[df['Gene'].isin(T_clust_genes)].copy()
    t_df['Subset'] = 'Upstream T-rich'

    return gc_df, t_df


def _run_subsampled_mannwhit(gc_df, t_df, column_prefix, n_subsamples=10):
    '''
    * Takes:
    - gc_df         : DataFrame of the GC-Rich subset (not subsampled).
    - t_df          : DataFrame of the Upstream T-rich subset, subsampled
                       down to len(gc_df) rows on each iteration.
    - column_prefix : str, column name to test (e.g. 'mT-adj', 'sT').
    - n_subsamples  : int, number of random subsamples to draw from t_df.

    * Outputs:
    - p_values : list[float], one Mann-Whitney p-value per subsample.
    - worst_p  : float, the largest (least significant) p-value across
                 subsamples -- used as the conservative summary stat for
                 this (column, species) comparison.
    '''
    gc_vals = gc_df[column_prefix].dropna().values
    n_gc = len(gc_vals)

    t_col = t_df[column_prefix].dropna()

    p_values = []
    for i in range(n_subsamples):
        seed = 42 + i
        t_vals = t_col.sample(
            n=min(n_gc, len(t_col)),
            random_state=seed,
            replace=False
        ).values
        _, p = mannwhitneyu(gc_vals, t_vals, alternative='two-sided')
        p_values.append(p)

    worst_p = max(p_values)
    return p_values, worst_p


def _format_pval_priors(p):
    '''
    Format a p-value for display on plots.

    * Takes:
    - p : float, p-value.

    * Outputs:
    - str : "p < 0.0001", or "p = X.XXXX" / "p = X.XXX" depending on
            magnitude.
    '''
    if p < 0.0001:
        return "p < 0.0001"
    elif p < 0.001:
        return f"p = {p:.4f}"
    else:
        return f"p = {p:.3f}"


def _build_species_violin_df(gc_df, t_df, column_prefix, x_label, seed=42):
    '''
    * Takes:
    - gc_df, t_df   : DataFrames from _split_subsets.
    - column_prefix : str, column being plotted (e.g. 'mT-adj', 'sT').
    - x_label       : str, species/group label for the 'x' column.
    - seed          : int, random_state for the single display subsample.

    * Outputs:
    - violin_df : DataFrame combining gc_df and a size-matched t_df sample,
                  with an 'x' column and an ordered categorical 'Subset'
                  column.
    '''
    t_for_violin = t_df.sample(
        n=min(len(gc_df), len(t_df)),
        random_state=seed,
        replace=False
    ).copy()

    violin_df = pd.concat([gc_df, t_for_violin], ignore_index=True)
    violin_df['x'] = x_label
    violin_df['Subset'] = pd.Categorical(
        violin_df['Subset'],
        categories=['Upstream T-rich', 'GC-Rich'],
        ordered=True
    )
    return violin_df


def compute_corrected_pvalues(hg38_data, nhp_data_list, genes_to_remove,
                               columns, n_subsamples=10):
    '''
    * Takes:
    - hg38_data : dict with keys 'T_genes', 'GC_genes', 'df', 'label'
                       for the human reference species.
    - nhp_data_list : list of same-shaped dicts, one per non-human species.
    - genes_to_remove : iterable of gene names to exclude from all subsets.
    - columns : list[str], column names to test (e.g. ['mT-adj', 'sT']).
    - n_subsamples  : int, subsamples per (col, species) comparison.

    * Outputs:
    - result : {col: {species_label: corrected_p}}, where corrected_p is
               the worst BH-FDR corrected p-value (across that species'
               n_subsamples draws) within col's full correction family.
    '''
    all_species = [hg38_data] + nhp_data_list
    result = {}

    for col in columns:
        records = []
        for sp in all_species:
            gc_df, t_df = _split_subsets(
                sp['df'], sp['T_genes'], sp['GC_genes'], genes_to_remove
            )
            p_values, _ = _run_subsampled_mannwhit(gc_df, t_df, col, n_subsamples)
            for p in p_values:
                records.append({'label': sp['label'], 'raw_p': p})

        raw_ps = [r['raw_p'] for r in records]
        _, corrected_ps, _, _ = multipletests(raw_ps, method='fdr_bh')

        worst_corrected = {}
        for r, cp in zip(records, corrected_ps):
            label = r['label']
            worst_corrected[label] = max(worst_corrected.get(label, -np.inf), cp)

        result[col] = worst_corrected

    return result


def _draw_panel(ax, data, x_labels, p_vals_dict, show_legend, show_ylabel,
                column_prefix, param_label, subset_palette):
    '''
    * Takes:
    - ax             : matplotlib Axes to draw on.
    - data           : DataFrame with 'x', column_prefix, and 'Subset' columns.
    - x_labels       : list[str], ordered x-axis group labels to annotate.
    - p_vals_dict    : {label: corrected_p}, one entry per x_label.
    - show_legend    : bool, whether to draw the Subset legend on this panel.
    - show_ylabel    : bool, whether to draw a y-axis label on this panel.
    - column_prefix  : str, column being plotted -- controls y-tick/label
                        formatting for special-cased columns.
    - param_label    : str, human-readable parameter name (e.g. r'$\\mu_T$').

    * Outputs:
    - Modifies ax plot
    '''
    sns.violinplot(
        data=data,
        x='x',
        y=column_prefix,
        hue='Subset',
        split=True,
        palette=subset_palette,
        inner="quart",
        linewidth=1.2,
        dodge=False,
        ax=ax
    )

    # Pin tick locations before relabeling so labels stay matched to
    # positions even if a later ylim/autoscale call shifts the ticks.
    yticks = ax.get_yticks()
    ax.set_yticks(yticks)

    if column_prefix == "mT-adj":
        ax.set_yticklabels(
            ["A3E" if abs(y) < 1e-9 else f"{int(y):,}" for y in yticks],
            fontsize=12
        )
        if show_ylabel:
            ax.set_ylabel(f"|{param_label}-A3E|", fontsize=14)
    else:
        ax.set_yticklabels([f"{int(y):,}" for y in yticks], fontsize=12)
        if show_ylabel:
            ax.set_ylabel(param_label, fontsize=14)

    if not show_ylabel:
        ax.set_ylabel("")

    ax.set_xlabel("")
    ax.tick_params(axis='x', labelsize=12)

    if show_legend:
        handles, lbls = ax.get_legend_handles_labels()
        ax.legend(handles[:2], lbls[:2], fontsize=11, title='', loc='upper right')
    elif ax.get_legend():
        ax.get_legend().remove()

    y_lo, y_hi = ax.get_ylim()
    y_ann = y_hi - (y_hi - y_lo) * 0.03

    for i, label in enumerate(x_labels):
        p_corr = p_vals_dict.get(label)
        if p_corr is None:
            continue
        ann = f"{_format_pval_priors(p_corr)}\n(p-adj)"
        ax.text(
            i, y_ann, ann,
            ha='center', va='top',
            fontsize=8, style='italic', color='#333333'
        )


def violins_combined_subsample_mannwhit(
    hg38_data,
    nhp_data_list,
    genes_to_remove,
    column_prefix,
    subset_palette,
    savepath_base,
    corrected_pvals,
    n_subsamples=10
):
    '''
    * Takes:
    - hg38_data : dict with 'T_genes', 'GC_genes', 'df', 'label' for
                        the human reference species.
    - nhp_data_list  : list of same-shaped dicts, one per NHP species.
    - genes_to_remove : iterable of gene names to exclude from all subsets.
    - column_prefix  : str, column to plot (e.g. 'mT-adj', 'sT').
    - subset_palette : dict, {'Upstream T-rich': color, 'GC-Rich': color}.
    - savepath_base  : str, output path for the saved figure.
    - corrected_pvals : {species_label: corrected_p}, from
                         compute_corrected_pvalues()[column_prefix].
    - n_subsamples : int, unused for stats here (p-values come from
                        corrected_pvals) -- kept only to control how many
                        rows the display sample effectively represents.

    * Outputs:
    - Saves and displays the figure at savepath_base
    '''
    param_dict = {
        'mT-adj': r'$\mu_T$',
        'sT':     r'$\sigma_T$',
        'wT':     r'$w_T$',
    }
    param_label = param_dict.get(column_prefix, column_prefix)

    # Data prep (violin DataFrames only — p-values come from corrected_pvals,
    # not recomputed here).
    hg38_gc, hg38_t = _split_subsets(
        hg38_data['df'], hg38_data['T_genes'], hg38_data['GC_genes'], genes_to_remove
    )
    hg38_vdf = _build_species_violin_df(hg38_gc, hg38_t, column_prefix, hg38_data['label'])

    nhp_vdfs = []
    for sp in nhp_data_list:
        gc_df, t_df = _split_subsets(
            sp['df'], sp['T_genes'], sp['GC_genes'], genes_to_remove
        )
        vdf = _build_species_violin_df(gc_df, t_df, column_prefix, sp['label'])
        nhp_vdfs.append(vdf)

    nhp_df = pd.concat(nhp_vdfs, ignore_index=True)
    nhp_labels = [sp['label'] for sp in nhp_data_list]
    nhp_df['x'] = pd.Categorical(nhp_df['x'], categories=nhp_labels, ordered=True)

    n_nhp = len(nhp_data_list)
    fig, (ax_hg38, ax_nhp) = plt.subplots(
        1, 2,
        figsize=(2 + n_nhp * 1.8, 3.5),
        gridspec_kw={'width_ratios': [1, n_nhp]}
    )

    _draw_panel(ax_hg38, hg38_vdf,
                x_labels=[hg38_data['label']],
                p_vals_dict=corrected_pvals,
                show_legend=False,
                show_ylabel=True,
                column_prefix=column_prefix,
                param_label=param_label,
                subset_palette=subset_palette)

    _draw_panel(ax_nhp, nhp_df,
                x_labels=nhp_labels,
                p_vals_dict=corrected_pvals,
                show_legend=True,
                show_ylabel=False,
                column_prefix=column_prefix,
                param_label=param_label,
                subset_palette=subset_palette)

    plt.suptitle(param_label, fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(savepath_base, bbox_inches='tight')
    plt.show()
    plt.close()
    
    
