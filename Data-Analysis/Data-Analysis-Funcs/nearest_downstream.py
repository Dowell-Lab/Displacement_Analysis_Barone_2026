import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns

def _load_distances(filepath):
    """
    Read a bedtools closest output file and return an array of distances (bp).
    Assumes distance is in the last column and drops unmapped rows (-1).
    """
    df = pd.read_csv(filepath, sep='\t', header=None)
    distances = pd.to_numeric(df.iloc[:, -1], errors='coerce')
    distances = distances[distances >= 0].dropna()
    return distances.values


def _format_pval(p):
    if p < 0.0001:
        return "p-adj < 0.0001"
    elif p < 0.001:
        return f"p-adj = {p:.4f}"
    else:
        return f"p-adj = {p:.3f}"


def _run_subsampled_mwu_distances(gc_vals, t_vals, n_subsamples=10, seed_base=42):
    """
    Draw n_subsamples independent subsamples of t_vals (each matched in
    size to gc_vals, without replacement) and run a two-sample MWU test
    against gc_vals for every draw. The n_subsamples raw p-values are
    then BH-corrected AGAINST EACH OTHER (n_tests = n_subsamples).

    Returns
    -------
    draws              : list of dict {seed, t_sample, t_kb}
    raw_ps             : list of float, raw MWU p-value per draw
    corrected_ps       : list of float, BH-adjusted p-value per draw
    worst_corrected_p  : float, max of corrected_ps
    """
    n_gc = len(gc_vals)
    draws = []
    raw_ps = []

    for i in range(n_subsamples):
        seed = seed_base + i
        rng = np.random.default_rng(seed=seed)
        t_sample = rng.choice(t_vals, size=min(n_gc, len(t_vals)), replace=False)
        _, p = mannwhitneyu(gc_vals, t_sample, alternative='two-sided')
        draws.append({'seed': seed, 't_sample': t_sample, 't_kb': t_sample / 1000.0})
        raw_ps.append(p)

    _, corrected_ps, _, _ = multipletests(raw_ps, method='fdr_bh')
    worst_corrected_p = max(corrected_ps)

    return draws, raw_ps, corrected_ps, worst_corrected_p


def _collect_kde_stats(location, resdir, samp, n_subsamples=10, seed_base=42):
    """
    Returns
    -------
    dict with keys: samp, gc_kb, draws, raw_ps, corrected_ps, worst_corrected_p
    """
    gc_vals = _load_distances(f"{resdir}/GC_clust_{samp}{location}")
    t_vals  = _load_distances(f"{resdir}/T_clust_{samp}{location}")

    draws, raw_ps, corrected_ps, worst_corrected_p = _run_subsampled_mwu_distances(
        gc_vals, t_vals, n_subsamples=n_subsamples, seed_base=seed_base
    )

    return {
        'samp':               samp,
        'gc_kb':              gc_vals / 1000.0,
        'draws':              draws,
        'raw_ps':             raw_ps,
        'corrected_ps':       corrected_ps,
        'worst_corrected_p':  worst_corrected_p,
    }


def _plot_kde(gc_kb, t_kb, corrected_p, samp, outdir, subset_palette=None,
              title=None, p_label="(p-adj, BH)", filename_suffix="", save=True):
    if subset_palette is None:
        subset_palette = {'Upstream T-rich': '#c3c0c0', 'GC-Rich': '#e57a7a'}

    fig, ax = plt.subplots(figsize=(5, 3))
    print(f"T-rich (subsampled): {len(t_kb)}")
    print(f"GC-rich: {len(gc_kb)}")

    sns.kdeplot(t_kb, ax=ax, color=subset_palette['Upstream T-rich'],
                linewidth=2, fill=True, alpha=0.4, label='Upstream T-rich')
    sns.kdeplot(gc_kb, ax=ax, color=subset_palette['GC-Rich'],
                linewidth=2, fill=True, alpha=0.4, label='GC-Rich')

    ax.text(0.97, 0.95, f"{_format_pval(corrected_p)}\n{p_label}",
            transform=ax.transAxes, ha='right', va='top', fontsize=13, style='italic')

    ax.set_xlabel("Distance to nearest downstream gene (kb)", fontsize=13)
    ax.set_ylabel("Density", fontsize=13)
    ax.set_xlim(-10, 400)
    ax.set_title(title if title is not None else f"Nearest downstream gene - {samp}", fontsize=13)
    ax.tick_params(labelsize=13)
    ax.legend(fontsize=13, title='', loc='upper left')

    plt.tight_layout()
    if save:
        outpath = f"{outdir}/{samp}_nearest_ds_gene_kde{filename_suffix}.svg"
        plt.savefig(outpath, bbox_inches='tight')
        print(f"Saved: {outpath}")
    plt.show()
    plt.close()


def plot_kde_all_subsamples(stats, outdir, subset_palette=None):
    for draw, corr_p in zip(stats['draws'], stats['corrected_ps']):
        _plot_kde(
            gc_kb=stats['gc_kb'], t_kb=draw['t_kb'], corrected_p=corr_p,
            samp=stats['samp'], outdir=outdir, subset_palette=subset_palette,
            title=f"{stats['samp']} (seed={draw['seed']})",
            p_label="(p-adj, BH)", filename_suffix=f"_seed{draw['seed']}",
        )


def plot_kde_main(stats, outdir, subset_palette=None):
    first_draw = stats['draws'][0]
    _plot_kde(
        gc_kb=stats['gc_kb'], t_kb=first_draw['t_kb'], corrected_p=stats['worst_corrected_p'],
        samp=stats['samp'], outdir=outdir, subset_palette=subset_palette,
        title=f"Nearest downstream gene - {stats['samp']}",
        p_label="(p-adj, BH, worst of N subsamples)", filename_suffix="_main",
    )

