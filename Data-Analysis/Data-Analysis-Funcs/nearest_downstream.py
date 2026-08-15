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


def _format_pval(p, adjusted=True):
    """
    adjusted=True  -> "p-adj ..." (used for the per-draw grid, where FDR
                       genuinely changes the number).
    adjusted=False -> "p ..." (used for the single "worst of N" summary
                       plot: at the largest p-value in a BH/FDR-corrected
                       family, the correction leaves the value unchanged
                       -- rank n gets multiplier n/n = 1 -- so labelling
                       it "adjusted" would be misleading; it's just the
                       raw worst p-value, honestly reported as such).
    """
    label = "p-adj" if adjusted else "p"
    if p < 0.0001:
        return f"{label} < 0.0001"
    elif p < 0.001:
        return f"{label} = {p:.4f}"
    else:
        return f"{label} = {p:.3f}"


def _run_subsampled_mwu_distances(gc_vals, t_vals, n_subsamples=10, seed_base=42):
    """
    Draw n_subsamples independent subsamples of t_vals (each matched in
    size to gc_vals, without replacement) and run a two-sample MWU test
    against gc_vals for every draw. The n_subsamples raw p-values are
    then FDR (Benjamini-Hochberg) corrected AGAINST EACH OTHER (n_tests =
    n_subsamples), under the deliberately-careful assumption that the
    draws are n_subsamples independent tests -- despite actually being
    correlated resamples of the same underlying data.

    Note: BH/FDR's per-rank multiplier is n_subsamples / rank. The
    largest (worst) p-value in the family is always rank n_subsamples,
    so its multiplier is exactly 1 -- FDR provides NO adjustment at the
    max, by construction, for any n_subsamples. worst_corrected_p is
    therefore always mathematically identical to max(raw_ps). That's
    correct FDR math, not a bug -- see plot_kde_main, which reports this
    value labelled as a raw "p", not "p-adj", for that reason. The
    per-draw values used in the grid plots (plot_kde_all_subsamples) ARE
    genuinely FDR-adjusted for every rank except the single worst one.

    Returns
    -------
    draws              : list of dict {seed, t_sample, t_kb}
    raw_ps             : list of float, raw MWU p-value per draw
    corrected_ps       : list of float, FDR-adjusted p-value per draw
    worst_corrected_p  : float, max of corrected_ps (= max(raw_ps))
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
              title=None, p_label="(p-adj, FDR)", adjusted=True,
              filename_suffix="", save=True):
    if subset_palette is None:
        subset_palette = {'Upstream T-rich': '#c3c0c0', 'GC-Rich': '#e57a7a'}

    fig, ax = plt.subplots(figsize=(5, 3))
    print(f"T-rich (subsampled): {len(t_kb)}")
    print(f"GC-rich: {len(gc_kb)}")

    sns.kdeplot(t_kb, ax=ax, color=subset_palette['Upstream T-rich'],
                linewidth=2, fill=True, alpha=0.4, label='Upstream T-rich')
    sns.kdeplot(gc_kb, ax=ax, color=subset_palette['GC-Rich'],
                linewidth=2, fill=True, alpha=0.4, label='GC-Rich')

    ax.text(0.97, 0.95, f"{_format_pval(corrected_p, adjusted=adjusted)}\n{p_label}",
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
            p_label="(p-adj, FDR)", adjusted=True, filename_suffix=f"_seed{draw['seed']}",
        )


def plot_kde_main(stats, outdir, subset_palette=None):
    # worst_corrected_p == max(raw_ps): FDR does not adjust the largest
    # p-value in a family (see _run_subsampled_mwu_distances). Reported
    # here as a raw "p", not "p-adj", since that's what it actually is.
    first_draw = stats['draws'][0]
    _plot_kde(
        gc_kb=stats['gc_kb'], t_kb=first_draw['t_kb'], corrected_p=stats['worst_corrected_p'],
        samp=stats['samp'], outdir=outdir, subset_palette=subset_palette,
        title=f"Nearest downstream gene - {stats['samp']}",
        p_label="(worst of N subsamples)", adjusted=False, filename_suffix="_main",
    )

