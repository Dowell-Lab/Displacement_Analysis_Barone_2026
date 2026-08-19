import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns
import math

###################################################################################
############################### Avg intron length #################################

def extract_gene_id(name_field):
    """
    * Takes: 
    - Gene|Isoform
    
    * Outputs: 
    - Gene
    
    """
    name_field = str(name_field)
    if "|" in name_field:
        return name_field.split("|")[0]
    if "_exon" in name_field:
        return name_field.split("_exon")[0]
    if "exon" in name_field.lower():
        return name_field.lower().split("exon")[0].rstrip("_-.")
    return name_field  


def load_mane_exon_bed(bed_path, gene_col=None, n_preview=5):
    """
    * Takes:
    - bed_path
    - gene_col 
    - n_preview : int, number of rows to show in the printed preview of loaded data 

    * Outputs:
    - df : DataFrame with columns 'chrom', 'start', 'end', 'name',
           'score', 'strand' 
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
    * Takes: 
    - Exon dataframe (MANE)
    
    * Outputs: 
    - Dataframe with average intron length per gene in hg38 MANE annotation
    
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


def get_gene_sets(resdir, samp):
    """
    * Takes:
    - location: suffix of file of interest
    - Bedtools resdir 
    - sample of interest (e.g., MANE consensus--hg38 all samples)

    * Outputs:
    - GC rich and T rich genes
    
    """
    
    columns = ['chr', 'start', 'end', 'gene_name',
               'score', 'strand']

    t_df = pd.read_csv(f"{resdir}/T_clust_{samp}.sorted.bed", sep='\t',
                        header=None, names=columns)
    gc_df = pd.read_csv(f"{resdir}/GC_clust_{samp}.sorted.bed", sep='\t',
                         header=None, names=columns)

    t_genes = set(t_df['gene_name'].unique())
    gc_genes = set(gc_df['gene_name'].unique())

    print(f"T-rich unique genes: {len(t_genes)}")
    print(f"GC-rich unique genes: {len(gc_genes)}")

    overlap = t_genes & gc_genes
    if overlap:
        print(f"NOTE: check input")

    return t_genes, gc_genes

def _format_pval_genebody(p, adjusted=True):
    """
    * Takes: 
    - Pvalues
    
    * Outputs: 
    - Formatted pvalues
    
    """
    label = "p-adj" if adjusted else "p"
    if p == 0:
        return f"{label} < 1e-300"
    elif p < 0.0001:
        return f"{label} = {p:.2e}"
    elif p < 0.001:
        return f"{label} = {p:.6f}"
    else:
        return f"{label} = {p:.3f}"


def _run_subsampled_mwu(gc_vals, t_vals, n_subsamples=10, seed_base=42):
    """
    * Takes: 
    - gc values 
    - t rich values
    
    * Outputs: 
    - Runs mann whitney u and corrects pvalues

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
    * Takes: 
    - intron-specific stats
    
    * Outputs: 
    - runs Runs mann whitney function

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

def _make_violin_df(t_kb, gc_kb, t_label='Upstream T-rich', gc_label='GC-Rich'):
    """
    * Takes: 
    -T-rich and GC-rich values to be plotted
    
    * Outputs: 
    - Violin plot

    """
    
    return pd.DataFrame({
        'value': np.concatenate([t_kb, gc_kb]),
        'group': [t_label] * len(t_kb) + [gc_label] * len(gc_kb),
    })


def _draw_violin_panel(ax, t_kb, gc_kb, corrected_p, subset_palette,
                        title=None, p_label="(p-adj, FDR)", adjusted=True, fs=9):
    """
    * Takes: 
    -T-rich and GC-rich values to be plotted
    
    * Outputs: 
    - Violin plot

    """
    df = _make_violin_df(t_kb, gc_kb)
    order = ['Upstream T-rich', 'GC-Rich']

    sns.violinplot(
        data=df, x='group', y='value', order=order,
        palette=subset_palette, ax=ax, cut=0, inner='quartile',
        linewidth=1.2, saturation=1
    )

    ax.text(
        0.97, 0.97,
        f"{_format_pval_genebody(corrected_p, adjusted=adjusted)}\n{p_label}",
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
    * Takes: 
    -T-rich and GC-rich values to be plotted
    
    * Outputs: 
    - Violin plot (grid)

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
            title=f"seed={draw['seed']}", p_label="(p-adj, FDR)", adjusted=True, fs=9
        )

    for j in range(n, len(axes)):
        axes[j].axis('off')

    fig.suptitle(f"{stats['samp']}: T-rich vs GC-rich avg intron length "
                 f"— {n} subsample draws (FDR-corrected)", fontsize=14, y=1.02)
    plt.tight_layout()

    if save:
        outpath = f"{outdir}/{stats['samp']}_avg_intron_length_subsample_grid.svg"
        plt.savefig(outpath, bbox_inches='tight')
        print(f"Saved: {outpath}")

    plt.show()
    plt.close()


def plot_violin_main(stats, outdir, subset_palette=None, save=True):
    """
    * Takes: GC-rich Trich subset stats
    
    * Outputs: 

    Single summary plot: uses the FIRST seed's subsample data, annotated
    with the WORST (max) p-value across all n_subsamples draws. This is
    the ultra-conservative headline figure. worst_corrected_p ==
    max(raw_ps) -- FDR gives the largest p-value in a family no
    adjustment (see _run_subsampled_mwu) -- so it's reported below as a
    raw "p", not "p-adj".
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
        p_label="(worst of N subsamples)", adjusted=False, fs=13
    )
    ax.set_ylabel("Average intron \n length per gene (kb)", fontsize=14)

    plt.tight_layout()

    if save:
        outpath = f"{outdir}/{stats['samp']}_avg_intron_length_violin_main.svg"
        plt.savefig(outpath, bbox_inches='tight')
        print(f"Saved: {outpath}")

    plt.show()
    plt.close() 

###################################################################################
############################### Gene length #######################################

def compute_gene_length_per_gene(exon_df):
    """
    * Takes: 
    - exon annotation file
    
    * Outputs: 
    - gene body lengths per gene (hg38 MANE)

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
    * Takes: 
    - T-rich and GC-rich statistics
    
    * Outputs: 
    - dict with keys: samp, gc_kb, draws, raw_ps, corrected_ps, worst_corrected_p
    
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
    * Takes:
    - stats
    - outdir
    - subset_palette
    - ncols --cols in panel gird
    - save (T/F)

    * Outputs:
    - Creates panels for violin plots 
    
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
            title=f"seed={draw['seed']}", p_label="(p-adj, FDR)", adjusted=True, fs=9
        )
        axes[i].set_ylabel("Gene length (kb)", fontsize=14)   # was: ax.set_ylabel(...)


    for j in range(n, len(axes)):
        axes[j].axis('off')

    fig.suptitle(f"{stats['samp']}: T-rich vs GC-rich gene length "
                 f"— {n} subsample draws (FDR-corrected)", fontsize=14, y=1.02)
    plt.tight_layout()

    if save:
        outpath = f"{outdir}/{stats['samp']}_gene_length_subsample_grid.svg"
        plt.savefig(outpath, bbox_inches='tight')
        print(f"Saved: {outpath}")

    plt.show()
    plt.close()

def plot_gene_length_main(stats, outdir, subset_palette=None, save=True):
    """
    * Takes: 
    - Gene length statistics
    
    * Outputs: 
    Single summary plot: uses the FIRST seed's subsample data, annotated
    with the WORST (max) p-value across all n_subsamples draws. This is
    the ultra-conservative headline figure. worst_corrected_p ==
    max(raw_ps) -- FDR gives the largest p-value in a family no
    adjustment (see _run_subsampled_mwu) -- so it's reported below as a
    raw "p", not "p-adj".
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
        p_label="(worst of N subsamples)", adjusted=False, fs=13
    )
    ax.set_ylabel("Average gene \n length (kb)", fontsize=14)

    plt.tight_layout()

    if save:
        outpath = f"{outdir}/{stats['samp']}_gene_length_violin_main.svg"
        plt.savefig(outpath, bbox_inches='tight')
        print(f"Saved: {outpath}")

    plt.show()
    plt.close()


