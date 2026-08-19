import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns

def _load_distances(filepath):
    """
    * Takes:
    - Filepath
    
    * Outputs:
    - File containing distance to downstream gene
    
    """
    df = pd.read_csv(filepath, sep='\t', header=None)
    distances = pd.to_numeric(df.iloc[:, -1], errors='coerce')
    distances = distances[distances >= 0].dropna()
    return distances.values


def _format_pval_downstream(p, adjusted=True):
    """
    * Takes:
    - Pvalue
    
    * Outputs:
    - A properly formatted pvalue
    
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
    * Takes:
    - GC/Trich distances
    - Subsamples
    - Seeding
    
    * Outputs:
    - Mann Whitney U testing
    - Raw Pvalues
    
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

    return draws, raw_ps


def _collect_kde_stats(location, resdir, samp, n_subsamples=10, seed_base=42):
    """
    * Takes:
    - location : str, filename suffix identifying the distance file
    - resdir : str, directory containing the GC_clust_* and
                       T_clust_* bedtools-closest output files.
    - samp : str, species/sample label 
    - n_subsamples : int, number of random subsamples to draw from the
                       T-rich pool (default 10)
    - seeding

    * Outputs:
    - dict with keys:
        'samp'   : str, the species/sample label (echoes the samp input).
        'gc_kb'  : np.ndarray, GC-rich distances in kb (not subsampled --
                   the fixed reference group every T-rich draw is tested
                   against).
        'draws'  : list[dict], one entry per subsample draw, each with
                   keys 'seed', 't_sample' (bp), 't_kb' (kb).
        'raw_ps' : list[float], uncorrected Mann-Whitney p-value per
                   draw, aligned by index to 'draws'
    
    """
    gc_vals = _load_distances(f"{resdir}/GC_clust_{samp}{location}")
    t_vals  = _load_distances(f"{resdir}/T_clust_{samp}{location}")

    draws, raw_ps = _run_subsampled_mwu_distances(
        gc_vals, t_vals, n_subsamples=n_subsamples, seed_base=seed_base
    )

    return {
        'samp':   samp,
        'gc_kb':  gc_vals / 1000.0,
        'draws':  draws,
        'raw_ps': raw_ps,
    }


def _plot_kde(gc_kb, t_kb, corrected_p, samp, outdir, subset_palette=None,
              title=None, p_label="(p-adj, FDR)", adjusted=True,
              filename_suffix="", save=True):
    """
    * Takes:
    - GC/T-rich nearest downstream gene stats
    
    * Outputs:
    - Main plotting code for density plots
    
    """
    
    if subset_palette is None:
        subset_palette = {'Upstream T-rich': '#c3c0c0', 'GC-Rich': '#e57a7a'}

    fig, ax = plt.subplots(figsize=(4, 3))
    print(f"T-rich (subsampled): {len(t_kb)}")
    print(f"GC-rich: {len(gc_kb)}")

    sns.kdeplot(t_kb, ax=ax, color=subset_palette['Upstream T-rich'],
                linewidth=2, fill=True, alpha=0.4, label='Upstream T-rich')
    sns.kdeplot(gc_kb, ax=ax, color=subset_palette['GC-Rich'],
                linewidth=2, fill=True, alpha=0.4, label='GC-Rich')

    ax.text(0.97, 0.95, f"{_format_pval_downstream(corrected_p, adjusted=adjusted)}\n{p_label}",
            transform=ax.transAxes, ha='right', va='top', fontsize=13, style='italic')

    ax.set_xlabel("Distance to nearest downstream gene (kb)", fontsize=14)
    ax.set_ylabel("Density", fontsize=14)
    ax.set_xlim(-10, 400)
    ax.set_title(title if title is not None else f"Nearest downstream gene - {samp}", fontsize=14)
    ax.tick_params(labelsize=14)
    ax.legend(fontsize=14, title='', loc='upper left')

    plt.tight_layout()
    if save:
        outpath = f"{outdir}/{samp}_nearest_ds_gene_kde{filename_suffix}.svg"
        plt.savefig(outpath, bbox_inches='tight')
        print(f"Saved: {outpath}")
    plt.show()
    plt.close()
    
def compute_corrected_pvalues_kde(species_stats_list):
    """
    * Takes:
    - GC/T-rich nearest downstream gene stats
    
    * Outputs:
    - Runs p-value correction
    
    """
    
    records = []
    for stats in species_stats_list:
        for i, p in enumerate(stats['raw_ps']):
            records.append({'samp': stats['samp'], 'draw_idx': i, 'raw_p': p})

    raw_ps_all = [r['raw_p'] for r in records]
    _, corrected_ps_all, _, _ = multipletests(raw_ps_all, method='fdr_bh')

    by_samp = {stats['samp']: stats for stats in species_stats_list}
    for stats in species_stats_list:
        stats['corrected_ps'] = [None] * len(stats['raw_ps'])

    for r, cp in zip(records, corrected_ps_all):
        by_samp[r['samp']]['corrected_ps'][r['draw_idx']] = cp

    for stats in species_stats_list:
        worst_idx = int(np.argmax(stats['corrected_ps']))
        stats['worst_idx'] = worst_idx
        stats['worst_corrected_p'] = stats['corrected_ps'][worst_idx]

    return species_stats_list


def plot_kde_all_subsamples(stats, outdir, subset_palette=None):
    """
    * Takes:
    - General statistics
    - Out directory 
    
    * Outputs:
    - Density plot comparing distance downstream between the two subsets
    
    """
    for draw, corr_p in zip(stats['draws'], stats['corrected_ps']):
        _plot_kde(
            gc_kb=stats['gc_kb'], t_kb=draw['t_kb'], corrected_p=corr_p,
            samp=stats['samp'], outdir=outdir, subset_palette=subset_palette,
            title=f"{stats['samp']} (seed={draw['seed']})",
            p_label="(p-adj, FDR)", adjusted=True, filename_suffix=f"_seed{draw['seed']}",
        )
        

def plot_kde_main(stats, outdir, subset_palette=None):
    """
    * Takes:
    - General statistics
    - Out directory 
    
    * Outputs:
    - Density plot comparing distance downstream between the two subsets
    - Reports worst p-adj on first seed
    
    """
    # Correction is pooled across species, the worst draw's
    # p-value is FDR-adjusted 
    # reported as "p-adj"
    first_draw = stats['draws'][0]
    _plot_kde(
        gc_kb=stats['gc_kb'], t_kb=first_draw['t_kb'], corrected_p=stats['worst_corrected_p'],
        samp=stats['samp'], outdir=outdir, subset_palette=subset_palette,
        title=f"Nearest downstream gene - {stats['samp']}",
        p_label="(worst of N subsamples, FDR across species)", adjusted=True, filename_suffix="_main",
    )

