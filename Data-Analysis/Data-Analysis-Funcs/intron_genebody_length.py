import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns
import math

"""
Average intron length per gene (T-rich vs GC-rich), violin comparison.

Correction scheme (mirrors the KDE distance pipeline):
  - Draw n_subsamples independent subsamples of the (larger) T-rich set,
    matched in size to the GC-rich set.
  - Run one MWU test per draw -> n_subsamples raw p-values.
  - BH-correct those n_subsamples p-values AGAINST EACH OTHER (n_tests =
    n_subsamples), despite the draws being correlated resamples of the
    same underlying data rather than independent hypotheses -- done
    anyway as an extra-conservative guard.
  - Plot all n_subsamples draws as a single grid figure, each panel
    annotated with its own BH-corrected p-value.
  - Produce one final "main" plot using the FIRST seed's data, annotated
    with the WORST (max) BH-corrected p-value across all n_subsamples
    draws -- the ultra-conservative headline figure.
"""

# ──────────────────────────────────────────────────────────────────────
# 1. MANE exon BED -> average intron length per gene   (unchanged)
# ──────────────────────────────────────────────────────────────────────

def extract_gene_id(name_field):
    """
    Pull a gene identifier out of the BED `name` column.

    Confirmed format for this file: 'GENE|RefSeqTranscriptID', e.g.
    'OR4F5|NM_001005484.2'. Gene symbol is everything before the '|'.
    """
    name_field = str(name_field)
    if "|" in name_field:
        return name_field.split("|")[0]
    if "_exon" in name_field:
        return name_field.split("_exon")[0]
    if "exon" in name_field.lower():
        return name_field.lower().split("exon")[0].rstrip("_-.")
    return name_field  # assume it's already a gene symbol


def load_mane_exon_bed(bed_path, gene_col=None, n_preview=5):
    """
    Load the MANE exon BED file.

    Returns
    -------
    DataFrame with columns: chrom, start, end, name, score, strand, gene_id
    """
    df = pd.read_csv(bed_path, sep='\t', header=None)

    n_cols = df.shape[1]
    base_cols = ['chrom', 'start', 'end', 'name', 'score', 'strand']
    df.columns = base_cols[:n_cols] + [f'col{i}' for i in range(n_cols - len(base_cols))] \
        if n_cols > len(base_cols) else base_cols[:n_cols]

    print(f"Loaded {bed_path} — {len(df)} rows, {n_cols} columns")
    print("Preview:")
    print(df.head(n_preview))

    if gene_col is not None:
        df['gene_id'] = df.iloc[:, gene_col].astype(str)
    else:
        df['gene_id'] = df['name'].apply(extract_gene_id)

    print("\nExtracted gene_id preview:")
    print(df[['name', 'gene_id']].head(n_preview))
    print("... verify this looks like real gene symbols before proceeding.\n")

    return df


def compute_avg_intron_length_per_gene(exon_df):
    """
    For each gene, sort exons by start and compute gaps between
    consecutive exons (= introns). Returns average intron length per gene.
    Single-exon genes (no introns) are dropped.
    """
    records = []

    for gene_id, g in exon_df.groupby('gene_id'):
        g = g.sort_values('start')
        starts = g['start'].values
        ends = g['end'].values

        if len(g) < 2:
            continue

        intron_lengths = []
        for i in range(len(g) - 1):
            gap = starts[i + 1] - ends[i]
            if gap > 0:
                intron_lengths.append(gap)

        if len(intron_lengths) == 0:
            continue

        records.append({
            'gene_id': gene_id,
            'n_exons': len(g),
            'n_introns': len(intron_lengths),
            'avg_intron_length_bp': np.mean(intron_lengths),
        })

    result = pd.DataFrame.from_records(records)
    print(f"Computed avg intron length for {len(result)} multi-exon genes "
          f"(out of {exon_df['gene_id'].nunique()} total genes in BED).")
    return result


# ──────────────────────────────────────────────────────────────────────
# 2. Gene set membership (T-rich vs GC-rich)   (unchanged)
# ──────────────────────────────────────────────────────────────────────

def get_gene_sets(location, resdir, samp):
    columns = ['cluster_chr', 'cluster_start', 'cluster_end', 'cluster_name',
               'cluster_score', 'cluster_strand', 'gene_chr', 'gene_start',
               'gene_end', 'gene_name', 'gene_score', 'gene_strand', 'distance']

    t_df = pd.read_csv(f"{resdir}/T_clust_{samp}{location}", sep='\t',
                        header=None, names=columns)
    gc_df = pd.read_csv(f"{resdir}/GC_clust_{samp}{location}", sep='\t',
                         header=None, names=columns)

    t_genes = set(t_df['gene_name'].unique())
    gc_genes = set(gc_df['gene_name'].unique())

    print(f"T-rich unique genes: {len(t_genes)}")
    print(f"GC-rich unique genes: {len(gc_genes)}")

    overlap = t_genes & gc_genes
    if overlap:
        print(f"NOTE: {len(overlap)} genes appear in both sets "
              f"(a gene can be nearest to both a T-rich and GC-rich cluster).")

    return t_genes, gc_genes


# ──────────────────────────────────────────────────────────────────────
# 3. Stats -- BH correction across the n_subsamples draws themselves
# ──────────────────────────────────────────────────────────────────────

def _format_pval(p):
    if p == 0:
        return "p-adj < 1e-300"
    elif p < 0.0001:
        return f"p-adj = {p:.2e}"
    elif p < 0.001:
        return f"p-adj = {p:.6f}"
    else:
        return f"p-adj = {p:.3f}"


def _run_subsampled_mwu(gc_vals, t_vals, n_subsamples=10, seed_base=42):
    """
    Draw n_subsamples independent subsamples of t_vals (each matched in
    size to gc_vals, without replacement) and run a two-sample MWU test
    against gc_vals for every draw.

    The n_subsamples raw p-values are BH-corrected against EACH OTHER
    (n_tests = n_subsamples). The draws are correlated resamples of the
    same underlying hypothesis rather than truly independent tests, so
    this is a deliberately conservative choice, not a textbook use of
    BH -- done anyway per request, as an extra guard against overstating
    significance.

    Returns
    -------
    draws              : list of dict, one per subsample draw, each with
                          keys 'seed', 't_sample' (bp), 't_kb'
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


def collect_intron_stats(samp, t_genes, gc_genes, intron_df, n_subsamples=10, seed_base=42):
    """
    Match gene sets against the intron-length table and run the
    per-subsample, BH-corrected-across-subsamples MWU battery.

    Returns
    -------
    dict with keys: samp, gc_kb, draws, raw_ps, corrected_ps, worst_corrected_p
    """
    intron_lookup = intron_df.set_index('gene_id')['avg_intron_length_bp']

    t_vals = intron_lookup.reindex(list(t_genes)).dropna().values
    gc_vals = intron_lookup.reindex(list(gc_genes)).dropna().values

    print(f"{samp}: T-rich genes with intron data = {len(t_vals)}, "
          f"GC-rich genes with intron data = {len(gc_vals)}")

    draws, raw_ps, corrected_ps, worst_corrected_p = _run_subsampled_mwu(
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


# ──────────────────────────────────────────────────────────────────────
# 4. Plotting
# ──────────────────────────────────────────────────────────────────────

def _make_violin_df(t_kb, gc_kb, t_label='Upstream T-rich', gc_label='GC-Rich'):
    return pd.DataFrame({
        'value': np.concatenate([t_kb, gc_kb]),
        'group': [t_label] * len(t_kb) + [gc_label] * len(gc_kb),
    })


def _draw_violin_panel(ax, t_kb, gc_kb, corrected_p, subset_palette,
                        title=None, p_label="(p-adj, BH)", fs=9):
    """Draw one violin panel into a given Axes (shared by grid + main plot)."""
    df = _make_violin_df(t_kb, gc_kb)
    order = ['Upstream T-rich', 'GC-Rich']

    sns.violinplot(
        data=df, x='group', y='value', order=order,
        palette=subset_palette, ax=ax, cut=0, inner='quartile',
        linewidth=1.2, saturation=1
    )

    ax.text(
        0.97, 0.97,
        f"{_format_pval(corrected_p)}\n{p_label}",
        transform=ax.transAxes, ha='right', va='top',
        fontsize=fs, style='italic'
    )

    ax.set_xlabel("")
    ax.set_ylabel("Avg intron length (kb)", fontsize=fs)
    if title is not None:
        ax.set_title(title, fontsize=fs)
    ax.tick_params(labelsize=fs)


def plot_violin_grid(stats, outdir, subset_palette=None, ncols=5, save=True):
    """
    Single grid figure: one violin panel per subsample draw (n_subsamples
    total), each annotated with that draw's own BH-corrected p-value
    (correction applied across the n_subsamples draws, see
    _run_subsampled_mwu).
    """
    if subset_palette is None:
        subset_palette = {'Upstream T-rich': '#c3c0c0', 'GC-Rich': '#e57a7a'}

    draws = stats['draws']
    corrected_ps = stats['corrected_ps']
    n = len(draws)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3.5))
    axes = np.atleast_1d(axes).flatten()

    for i, (draw, corr_p) in enumerate(zip(draws, corrected_ps)):
        _draw_violin_panel(
            axes[i], draw['t_kb'], stats['gc_kb'], corr_p, subset_palette,
            title=f"seed={draw['seed']}", p_label="(p-adj, BH)", fs=9
        )

    for j in range(n, len(axes)):
        axes[j].axis('off')

    fig.suptitle(f"{stats['samp']}: T-rich vs GC-rich avg intron length "
                 f"— {n} subsample draws (BH-corrected)", fontsize=14, y=1.02)
    plt.tight_layout()

    if save:
        outpath = f"{outdir}/{stats['samp']}_avg_intron_length_subsample_grid.svg"
        plt.savefig(outpath, bbox_inches='tight')
        print(f"Saved: {outpath}")

    plt.show()
    plt.close()


def plot_violin_main(stats, outdir, subset_palette=None, save=True):
    """
    Single summary plot: uses the FIRST seed's subsample data, annotated
    with the WORST (max) BH-corrected p-value across all n_subsamples
    draws. This is the ultra-conservative headline figure.
    """
    if subset_palette is None:
        subset_palette = {'Upstream T-rich': '#c3c0c0', 'GC-Rich': '#e57a7a'}

    first_draw = stats['draws'][0]

    fig, ax = plt.subplots(figsize=(3, 3.5))
    print(f"T-rich (subsampled): {len(first_draw['t_kb'])}")
    print(f"GC-rich: {len(stats['gc_kb'])}")

    _draw_violin_panel(
        ax, first_draw['t_kb'], stats['gc_kb'], stats['worst_corrected_p'],
        subset_palette,
        title=f"Average intron length per gene – {stats['samp']}",
        p_label="", fs=13
    )
    ax.set_ylabel("Average intron \n length per gene (kb)", fontsize=14)

    plt.tight_layout()

    if save:
        outpath = f"{outdir}/{stats['samp']}_avg_intron_length_violin_main.svg"
        plt.savefig(outpath, bbox_inches='tight')
        print(f"Saved: {outpath}")

    plt.show()
    plt.close()
    
    
"""
Gene length (T-rich vs GC-rich), violin plot comparison.

Metric: gene's own genomic span (first exon start -> last exon end, kb),
computed from the MANE exon annotation BED.

Correction scheme (mirrors the intron-length / KDE distance pipelines):
  - Draw n_subsamples independent subsamples of the (larger) T-rich set,
    matched in size to the GC-rich set.
  - Run one MWU test per draw -> n_subsamples raw p-values.
  - BH-correct those n_subsamples p-values AGAINST EACH OTHER (n_tests =
    n_subsamples), despite the draws being correlated resamples of the
    same underlying data rather than independent hypotheses -- done
    anyway as an extra-conservative guard.
  - Plot all n_subsamples draws as a single grid figure, each panel
    annotated with its own BH-corrected p-value.
  - Produce one final "main" plot using the FIRST seed's data, annotated
    with the WORST (max) BH-corrected p-value across all n_subsamples
    draws -- the ultra-conservative headline figure.
"""


def compute_gene_length_per_gene(exon_df):
    """
    For each gene, take the full genomic span: min start across all its
    exons to max end across all its exons. That span (end - start) is
    the gene length. Unlike intron length, this is well-defined even
    for single-exon genes, so nothing gets dropped here.

    Returns
    -------
    DataFrame with columns: gene_id, n_exons, gene_length_bp
    """
    records = []

    for gene_id, g in exon_df.groupby('gene_id'):
        start = g['start'].min()
        end = g['end'].max()
        records.append({
            'gene_id': gene_id,
            'n_exons': len(g),
            'gene_length_bp': end - start,
        })

    result = pd.DataFrame.from_records(records)
    print(f"Computed gene length for {len(result)} genes.")
    return result

def collect_gene_length_stats(samp, t_genes, gc_genes, gene_length_df,
                               n_subsamples=10, seed_base=42):
    """
    Match gene sets against the gene-length table and run the
    per-subsample, BH-corrected-across-subsamples MWU battery.

    Returns
    -------
    dict with keys: samp, gc_kb, draws, raw_ps, corrected_ps, worst_corrected_p
    """
    length_lookup = gene_length_df.set_index('gene_id')['gene_length_bp']

    t_vals = length_lookup.reindex(list(t_genes)).dropna().values
    gc_vals = length_lookup.reindex(list(gc_genes)).dropna().values

    print(f"{samp}: T-rich genes with length data = {len(t_vals)}, "
          f"GC-rich genes with length data = {len(gc_vals)}")

    draws, raw_ps, corrected_ps, worst_corrected_p = _run_subsampled_mwu(
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


def plot_gene_length_grid(stats, outdir, subset_palette=None, ncols=5, save=True):
    """
    Single grid figure: one violin panel per subsample draw (n_subsamples
    total), each annotated with that draw's own BH-corrected p-value
    (correction applied across the n_subsamples draws, see
    _run_subsampled_mwu).
    """
    if subset_palette is None:
        subset_palette = {'Upstream T-rich': '#c3c0c0', 'GC-Rich': '#e57a7a'}

    draws = stats['draws']
    corrected_ps = stats['corrected_ps']
    n = len(draws)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3.5))
    axes = np.atleast_1d(axes).flatten()

    for i, (draw, corr_p) in enumerate(zip(draws, corrected_ps)):
        _draw_violin_panel(
            axes[i], draw['t_kb'], stats['gc_kb'], corr_p, subset_palette,
            title=f"seed={draw['seed']}", p_label="(p-adj, BH)", fs=9
        )
        axes[i].set_ylabel("Gene length (kb)", fontsize=14)   # was: ax.set_ylabel(...)


    for j in range(n, len(axes)):
        axes[j].axis('off')

    fig.suptitle(f"{stats['samp']}: T-rich vs GC-rich gene length "
                 f"— {n} subsample draws (BH-corrected)", fontsize=14, y=1.02)
    plt.tight_layout()

    if save:
        outpath = f"{outdir}/{stats['samp']}_gene_length_subsample_grid.svg"
        plt.savefig(outpath, bbox_inches='tight')
        print(f"Saved: {outpath}")

    plt.show()
    plt.close()

def plot_gene_length_main(stats, outdir, subset_palette=None, save=True):
    """
    Single summary plot: uses the FIRST seed's subsample data, annotated
    with the WORST (max) BH-corrected p-value across all n_subsamples
    draws. This is the ultra-conservative headline figure.
    """
    if subset_palette is None:
        subset_palette = {'Upstream T-rich': '#c3c0c0', 'GC-Rich': '#e57a7a'}

    first_draw = stats['draws'][0]

    fig, ax = plt.subplots(figsize=(3, 3.5))
    print(f"T-rich (subsampled): {len(first_draw['t_kb'])}")
    print(f"GC-rich: {len(stats['gc_kb'])}")

    _draw_violin_panel(
        ax, first_draw['t_kb'], stats['gc_kb'], stats['worst_corrected_p'],
        subset_palette,
        title=f"Gene length – {stats['samp']}",
        p_label="", fs=13
    )
    ax.set_ylabel("Average gene \n length (kb)", fontsize=14)

    plt.tight_layout()

    if save:
        outpath = f"{outdir}/{stats['samp']}_gene_length_violin_main.svg"
        plt.savefig(outpath, bbox_inches='tight')
        print(f"Saved: {outpath}")

    plt.show()
    plt.close()


