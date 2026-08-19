# Import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from Bio import AlignIO, SeqIO
import re
import os
from scipy.stats import linregress  
import seaborn as sns

def get_base_content_windows(fasta_file, 
                             window_size=100):
    """
    * Takes: 
    - Fasta file with base content information
    - Window size
    
    * Outputs:
    - Base content 
    - Length of sequence
    
    """
    
    seq_record = next(SeqIO.parse(fasta_file, "fasta"))
    seq = str(seq_record.seq).upper()
    windows = [seq[i:i+window_size] for i in range(0, len(seq), window_size)]
    base_content = {base: [w.count(base) / len(w) for w in windows] for base in 'ATGC'}
    return base_content, len(seq)

def smooth_data(data, window_size):
    """
    * Takes
    - Data to be smoothed (sliding scale)
    - Window size (moving avg window size)
    
    * Outputs
    - The smoothed data
    
    """
    smoothed_data = pd.Series(data).rolling(window=window_size, min_periods=1, center=True).mean().tolist()
    return smoothed_data

def add_downstream_gene(df):
    """
    * Takes:
    - df : pd.DataFrame, BED-style table with 6 columns in order
           [chrom, start, end, name, score, strand] 

    * Outputs:
    - pd.DataFrame, a copy of df with two new columns:
        - downstream_gene : name of the nearest 3'-downstream gene on
                             the same chromosome (any strand), or pd.NA
                             if none exists (e.g. last gene on a
                             chromosome, or malformed/missing strand).
        - downstream_dist : distance (bp) from the PAS to that
                             neighbor's boundary, always >= 0, or pd.NA
                             if no downstream neighbor was found.
    """
    
    df = df.copy()
    df.columns = ["chrom", "start", "end", "name", "score", "strand"]
    df["downstream_gene"] = pd.NA
    df["downstream_dist"] = pd.NA

    for chrom, grp in df.groupby("chrom", sort=False):

        # ── + strand: PAS = end; look for the gene with smallest start ≥ PAS ──
        sorted_start = grp.sort_values("start")
        starts       = sorted_start["start"].values
        orig_idx_s   = sorted_start.index.values        

        plus_idx = grp.index[grp["strand"] == "+"]
        for idx in plus_idx:
            pas = df.at[idx, "end"]
            pos = np.searchsorted(starts, pas, side="left")
            # skip self if it happens to sit at that boundary
            while pos < len(starts) and orig_idx_s[pos] == idx:
                pos += 1
            if pos < len(starts):
                nbr = orig_idx_s[pos]
                df.at[idx, "downstream_gene"] = df.at[nbr, "name"]
                df.at[idx, "downstream_dist"] = int(starts[pos] - pas)

        # ── - strand: PAS = start; look for the gene with largest end ≤ PAS ──
        sorted_end = grp.sort_values("end")
        ends       = sorted_end["end"].values
        orig_idx_e = sorted_end.index.values

        minus_idx = grp.index[grp["strand"] == "-"]
        for idx in minus_idx:
            pas = df.at[idx, "start"]
            pos = np.searchsorted(ends, pas, side="right") - 1
            while pos >= 0 and orig_idx_e[pos] == idx:
                pos -= 1
            if pos >= 0:
                nbr = orig_idx_e[pos]
                df.at[idx, "downstream_gene"] = df.at[nbr, "name"]
                df.at[idx, "downstream_dist"] = int(pas - ends[pos])

    return df

def load_gene_bed(bed_path):
    """
    * Takes:
    - bed_path : str, path to a 6-column BED file (no header) with
                 columns [chrom, start, end, Gene, score, strand].

    * Outputs:
    - pd.DataFrame with columns [Gene, chrom, PAS, strand], where PAS
      (poly-A site / transcription end site) is the end coordinate for
      '+' strand genes and the start coordinate for '-' strand genes.
    """
    
    ann = pd.read_csv(
        bed_path, sep='\t', header=None,
        names=['chrom', 'start', 'end', 'Gene', 'score', 'strand'],
        usecols=[0, 1, 2, 3, 4, 5]
    )
    # PAS = TES: end coord for + strand, start coord for - strand
    ann['PAS'] = ann.apply(
        lambda r: r['end'] if r['strand'] == '+' else r['start'], axis=1
    )
    return ann[['Gene', 'chrom', 'PAS', 'strand']]

def make_neighborhood_bed(liet_df, ann_df, genome_label, outdir, genes_with_same_neighbor, dist_df):
    """
    * Takes:
    - liet_df : pd.DataFrame, LIET fit results; must contain a 'Gene'
                column to merge on.
    - ann_df  : pd.DataFrame, output of load_gene_bed() for this
                genome; columns [Gene, chrom, PAS, strand].
    - genome_label : str, genome/assembly label used in the output
                filename (e.g. 'hg38', 'rheMac10').
    - outdir  : str, directory to write the combined output BED into.
    - genes_with_same_neighbor : list[str], currently unused -- was
                previously used to restrict output to genes with a
                conserved downstream neighbor across species; that
                filter is disabled below to keep the full gene list.
    - dist_df : pd.DataFrame, must contain 'name' and 'downstream_dist'
                columns (e.g. output of add_downstream_gene(), renamed
                to 'name'). Rows with NA downstream_dist will raise a
                TypeError in the current implementation and should be
                filtered out before calling this function.

    * Outputs:
    - pd.DataFrame, one row per gene, BED-style columns
      [chrom, start, end, name, score, strand], where the window
      spans from `buffer` bp inside the gene body (relative to PAS)
      out to `buffer` bp past the gene's actual downstream neighbor
      (PAS + downstream_dist), oriented per strand. Also writes this
      table to '{outdir}/{genome_label}_15kb_neighborhood.bed'
      (tab-separated, no header, sorted by chrom/start).
    """
    
    # 1. Merge LIET data with basic annotation (to get PAS and Strand)
    df = liet_df.merge(ann_df, on='Gene', how='inner')
    
    # 2. Rename 'Gene' to 'name' for consistency with your distance dataframe
    df = df.rename(columns={'Gene': 'name'})
    
    # 3. Merge with your distance dataframe (to get downstream_dist)
    # This df contains the 'downstream_dist' column
    df = df.merge(dist_df[['name', 'downstream_dist']], on='name', how='inner')
    
    # 4. Filter for only the conserved genes --> removed this to increase gene list
#     df = df[df["name"].isin(genes_with_same_neighbor)]
    
    rows = []
    for _, row in df.iterrows():
        pas = int(row['PAS'])
        ds_dist = int(row['downstream_dist'])
#         buffer = 7000
        buffer = 15000
        
        if row['strand'] == '+':
            # Start: 7kb before PAS (inside the gene body)
            # End: PAS + intergenic distance + 7kb (inside downstream gene)
            bed_start = pas - buffer
            bed_end   = pas + ds_dist + buffer
        else:
            # Start: PAS - intergenic distance - 7kb (inside downstream gene)
            # End: 7kb after PAS (inside the gene body)
            bed_start = pas - ds_dist - buffer
            bed_end   = pas + buffer

        rows.append([
            row['chrom'], 
            max(0, bed_start), 
            bed_end,
            row['name'], 
            0, 
            row['strand']
        ])

    # Create final BED DataFrame
    bed_df = pd.DataFrame(rows, columns=['chrom', 'start', 'end', 'name', 'score', 'strand'])
    bed_df = bed_df.sort_values(['chrom', 'start']).reset_index(drop=True)

    # Save to file
    outpath = os.path.join(outdir, f"{genome_label}_15kb_neighborhood.bed")
    bed_df.to_csv(outpath, sep='\t', header=False, index=False)
    
    print(f"Written: {outpath} ({len(bed_df)} regions)")
    return bed_df

def parse_and_stitch_maf(maf_path):
    '''
    * Takes:
    - maf_path : str, path to a MAF (Multiple Alignment Format) file
                 containing one or more alignment blocks.

    * Outputs:
    - genome_seqs : dict[str, str], one entry per genome encountered
                    across all blocks (keyed by the genome prefix
                    before the first '.' in each record's id), value
                    is that genome's full stitched sequence across all
                    blocks concatenated in block order. Genomes absent
                    from a given block are gap-filled ('-' * block_len)
                    for that block so all genomes remain equal length.
    - all_genomes : list[str], genome names in first-encountered order
                    across blocks.
                    
    '''
    blocks = list(AlignIO.parse(maf_path, "maf"))
    if not blocks:
        raise ValueError(f"No alignment blocks found in {maf_path}")

    all_genomes = []
    seen = set()
    for block in blocks:
        for rec in block:
            prefix = rec.id.split(".")[0]
            if prefix not in seen:
                all_genomes.append(prefix)
                seen.add(prefix)

    genome_seqs = {g: "" for g in all_genomes}
    for block in blocks:
        block_len = block.get_alignment_length()
        present = {rec.id.split(".")[0]: str(rec.seq) for rec in block}
        for genome in all_genomes:
            genome_seqs[genome] += present.get(genome, "-" * block_len)

    return genome_seqs, all_genomes

def genome_coord_to_aln_index(seq, target_coord):
    '''
    * Takes:
    - seq : str, a gapped alignment sequence (may contain '-').
    - target_coord : int, a 1-based ungapped genomic coordinate to
                     locate within seq.

    * Outputs:
    - int, the alignment-string index (0-based) of the base at which
      the ungapped base count first reaches target_coord, or None if
      target_coord < 1 or exceeds the number of ungapped bases in seq.
    '''
    
    if target_coord < 1: return None
    count = 0
    for i, base in enumerate(seq):
        if base != "-":
            count += 1
            if count >= target_coord:
                return i
    return None

def plot_msa_maf_mT(maf_path,
                    gene,
                    mt_df,
                    species_col_map,
                    species_st_col_map,
                    label_dict,
                    tcs_offset=7000,
                    species_order=None):
    '''
    * Takes:
    - maf_path : str, path to the MAF file for this gene's region.
    - gene : str, gene name; must exist in mt_df.index.
    - mt_df : pd.DataFrame, indexed by gene, containing per-species
              mT and sT columns (per species_col_map/species_st_col_map).
    - species_col_map : dict[str, str], maps genome/species name (as
              used in the MAF and label_dict) to its mT column name in
              mt_df.
    - species_st_col_map : dict[str, str], same but for the sT
              (standard deviation) column name.
    - label_dict : dict[str, str], maps genome name to a display label
              for the y-axis; also used to filter which genomes from
              the MAF are actually plotted (any genome not present in
              label_dict is dropped from `order`).
    - tcs_offset : int, bp offset from the start of each genome's
              extracted MAF region to the TCS/A3E reference point.
              Assumes this offset is identical across all species'
              extracted regions.
    - species_order : list[str] or None, explicit row order for the
              plot; defaults to detection order from the MAF if None.

    * Outputs:
    - Alignment plot
    
    '''

    # ── 1. Load and stitch MAF ────────────────────────────────────────────────
    genome_seqs, detected_order = parse_and_stitch_maf(maf_path)
    order = species_order if species_order is not None else detected_order
    order = [sp for sp in order if sp in label_dict]

    aln_len = len(next(iter(genome_seqs.values())))
    num_sp  = len(order)

    # ── 2. Build colour matrix ────────────────────────────────────────────────
    char_to_index = {"-": 0, "A": 1, "C": 2, "G": 3, "T": 4, "N": 5}
    colors = ["#979497", "#008000", "#008000", "#008000", "#008000", "#4472C4"]
    cmap = ListedColormap(colors)

    color_matrix = np.zeros((num_sp, aln_len), dtype=int)
    for i, genome in enumerate(order):
        seq = genome_seqs.get(genome, "-" * aln_len).upper()
        for j, base in enumerate(seq):
            color_matrix[i, j] = char_to_index.get(base, 6)

    # ── 3. Resolve Indices
    tcs_aln_indices  = {}
    mt_aln_indices   = {}
    st_range_indices = {}

    if gene not in mt_df.index:
        raise ValueError(f"STRICT STOP: Gene {gene} not found in mt_df.")

    row = mt_df.loc[gene]

    for genome in order:
        seq = genome_seqs.get(genome, "")
        if not seq:
            continue

        m_col = species_col_map.get(genome)
        s_col = species_st_col_map.get(genome)

        if m_col not in row.index or s_col not in row.index:
            raise ValueError(
                f"STRICT STOP: Species {genome} expects columns '{m_col}' and "
                f"'{s_col}', but they are missing from the dataframe."
            )

        mT_val = row[m_col]
        sT_val = row[s_col]

        if pd.isna(mT_val) or pd.isna(sT_val):
            raise ValueError(
                f"STRICT STOP: NaN detected for {genome} in gene {gene}. "
                f"Check your source data."
            )

        # Resolve mT (Mean)
        mt_ungapped = tcs_offset + int(round(mT_val))
        mt_idx = genome_coord_to_aln_index(seq, mt_ungapped)
        if mt_idx is not None:
            mt_aln_indices[genome] = mt_idx

        # Resolve sT Range (Mean +/- 2*sigma)
        low_ungapped  = tcs_offset + int(round(mT_val - 2 * sT_val))
        high_ungapped = tcs_offset + int(round(mT_val + 2 * sT_val))
        low_idx  = genome_coord_to_aln_index(seq, low_ungapped)
        high_idx = genome_coord_to_aln_index(seq, high_ungapped)

        # Clip to alignment edges if out of range
        final_low  = low_idx  if low_idx  is not None else 0
        final_high = high_idx if high_idx is not None else aln_len - 1
        st_range_indices[genome] = (final_low, final_high)

        # Resolve TCS index per species
        tcs_idx = genome_coord_to_aln_index(seq, tcs_offset)
        if tcs_idx is not None:
            tcs_aln_indices[genome] = tcs_idx

    # ── 4. Plotting ───────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 1 + num_sp * 0.8))
    ax.imshow(color_matrix, aspect='auto', cmap=cmap, vmin=0, vmax=6, interpolation='none')
    
    for i, genome in enumerate(order):

        # 2-sigma shaded band
        if genome in st_range_indices:
            low, high = st_range_indices[genome]
            ax.fill_betweenx([i - 0.42, i + 0.42], low, high,
                             color='black', alpha=0.3, edgecolor=None, zorder=3)

        # mT dashed line
        mid = mt_aln_indices.get(genome)
        if mid is not None:
            ax.vlines(mid, i - 0.45, i + 0.45, color='black',
                      linestyle='--', linewidth=2.5, zorder=4)

        # TCS per-species tick
        tcs_x = tcs_aln_indices.get(genome)
        if tcs_x is not None:
            ax.vlines(tcs_x, i - 0.45, i + 0.45, color='#D3D3D3',
                      linestyle=':', linewidth=2.5, zorder=5)

    # ── 5. Formatting ─────────────────────────────────────────────────────────
    ax.set_yticks(np.arange(num_sp))
    ax.set_yticklabels([label_dict.get(sp, sp) for sp in order], fontsize=18, fontweight='bold')
    ax.tick_params(axis='x', labelsize=16)
    ax.set_xlabel("Alignment Index (bp)", fontsize=18, labelpad=10)
    ax.set_title(f"{gene} Alignment", fontsize=22, pad=20, fontweight='bold')
    ax.set_xlim(0, aln_len)

    # ── 6. Legend ─────────────────────────────────────────────────────────────
    legend_handles = [
        Line2D([0], [0], color='black', linestyle='--', linewidth=2,   label=r'$\mu_T$'),
        Patch(facecolor='black', alpha=0.3, edgecolor=None,             label=r'$\pm 2\sigma_T$'),
        Line2D([0], [0], color='#D3D3D3', linestyle=':', linewidth=2.5, label='A3E'),
        Patch(facecolor="#008000", edgecolor='black',                   label='Aligned or Substitution'),
        Patch(facecolor="#4472C4", edgecolor='black',                   label='N / Ambiguous'),
        Patch(facecolor="#979497", edgecolor='black',                   label='Gap (-)'),
    ]
    ax.legend(handles=legend_handles, bbox_to_anchor=(1.02, 1),
              loc='upper left', fontsize=14, frameon=False)

    plt.tight_layout()
    plt.savefig(f"/Users/geba9152/3prime_end_reviews_genomeR/Figure3/{gene}-4-Species-hg38root.png", dpi=300)

    plt.show()

####################################################################################
######################### Base content near mT #####################################
def load_gene_bed_basecomp_alignments(bed_path):
    """
    * Takes: 
    - bed file path
    
    * Outputs: 
    - Annotation of 3' ends for fasta generation
    
    """
    ann = pd.read_csv(
        bed_path, sep='\t', header=None,
        names=['chrom', 'start', 'end', 'Gene', 'score', 'strand'],
        usecols=[0, 1, 2, 3, 4, 5]
    )
    # PAS = TES: end coord for + strand, start coord for - strand
    ann['PAS'] = ann.apply(
        lambda r: r['end'] if r['strand'] == '+' else r['start'], axis=1
    )
    return ann[['Gene', 'chrom', 'PAS', 'strand']]
 
    
#  make_neighborhood_bed that saves individual genes for fasta generation
def make_neighborhood_bed_basecomp_alignments(liet_df, ann_df, genome_label, outdir, 
                          utr_buffer, ds_buffer):
    """
    * Takes: 
    - LIET dataframe
    - Annotation dataframe
    - Genome
    - Outpath
    - Buffer for downstream of the 3' end
    - Buffer for upstream of the 3' end
    
    * Outputs: 
    - Bedfile for mT base composition plots
    - Tracking base comp near mT
    
    """
    # Create the genome-specific subdirectory (e.g., Base-Comp_Beds/hg38)
    genome_subdir = os.path.join(outdir, genome_label)
    os.makedirs(genome_subdir, exist_ok=True)

    df = liet_df.merge(ann_df, on='Gene', how='inner').rename(columns={'Gene': 'name'})

    count = 0
    for _, row in df.iterrows():
        pas = int(row['PAS'])
        gene_name = row['name']
        gene_name = str(row['name']).split("|")[0]
        print(gene_name)

        if row['strand'] == '+':
            bed_start = pas - utr_buffer
            bed_end   = pas + ds_buffer
        else:
            bed_start = pas - ds_buffer
            bed_end   = pas + utr_buffer

        # Data for this specific gene
        gene_row = [row['chrom'], max(0, bed_start), bed_end, gene_name, 0, row['strand']]
        
        # Define path for this specific gene's BED file
        gene_outfile = os.path.join(genome_subdir, f"{gene_name}.bed")
        
        # Write individual BED file (tab-separated, no header)
        with open(gene_outfile, 'w') as f:
            f.write("\t".join(map(str, gene_row)) + "\n")
        
        count += 1

    print(f"Written {count} individual gene BEDs to: {genome_subdir}")


def plot_base_content_all_species(xlim,
                                  ylim,
                                  gene,
                                  fasta_dir,
                                  mt_df,
                                  species_config,
                                  species_order,
                                  tcs_offset=7000,
                                  window_size=100,
                                  smoothing_window=3,
                                  output_dir="/Users/geba9152/Paper-Figures-RD4/Fig3/"):
    """
    * Inputs: 
    - FASTA paths: {fasta_dir}/{genome_label}/{gene}.fa
    - Xlimits and y limits
    - Gene of interest
    - fasta directory 
    - mT dataframe 
    - species config & order specifications
    - Offset value of the TCS (how far upstream the TCS is)
    - Window details
    
    * Outputs: 
    - Base content plot across species surrounding mT
    
    TCS is at tcs_offset bp into each FASTA; mT marked at tcs_offset + mT-adj.
    Shaded grey band shows ±2σ around mT using sT_col from species_config.
    Skips any species missing a FASTA; skips gene entirely if none found.
    Grey square badge = T-rich gene (per species); pink badge = GC-rich gene (per species).
    """
    from matplotlib.patches import Patch

    tcs_win = tcs_offset // window_size

    #  1. Load base content — path is now subdir/gene.fa
    species_data = {}
    for sp in species_order:
        cfg  = species_config[sp]
        path = f"{fasta_dir}/{cfg['genome_label']}/{gene}.fa"
        try:
            bc, _ = get_base_content_windows(path, window_size)
            species_data[sp] = bc
        except FileNotFoundError:
            print(f"  [{gene}] Skipping {cfg['label']}: FASTA not found at {path}")
            species_data[sp] = None

    if all(v is None for v in species_data.values()):
        print(f"Skipping {gene}: no FASTAs found for any species.")
        return

    #  2. Resolve mT window indices and ±2σ spans 
    mt_wins = {}
    st_wins = {}
    if gene in mt_df.index:
        row = mt_df.loc[gene]
        for sp in species_order:
            cfg    = species_config[sp]
            mt_col = cfg["mt_col"]
            st_col = cfg.get("st_col")

            if mt_col in row.index and pd.notna(row[mt_col]):
                mt_wins[sp] = (tcs_offset + int(round(row[mt_col]))) // window_size

            if st_col and st_col in row.index and pd.notna(row[st_col]):
                st_wins[sp] = row[st_col] / window_size

    #  3. Plot 
    n_sp = len(species_order)
    fig, axs = plt.subplots(n_sp, 1, figsize=(6, 1.1 * n_sp), sharex=True)
#     fig, axs = plt.subplots(n_sp, 1, figsize=(10, 1.5 * n_sp), sharex=True)

#     fig.subplots_adjust(hspace=0)

    for i, sp in enumerate(species_order):
        ax  = axs[i]
        cfg = species_config[sp]
        bc  = species_data[sp]

        if bc is None:
            ax.set_title(f"{cfg['label']} — no data", fontsize=12)
            continue

        a_sm = smooth_data(bc['A'], smoothing_window)
        t_sm = smooth_data(bc['T'], smoothing_window)
        c_sm = smooth_data(bc['C'], smoothing_window)
        g_sm = smooth_data(bc['G'], smoothing_window)
        x    = np.arange(len(a_sm)) - tcs_win   # relative to TCS

        ax.plot(x, a_sm, label='A', color=cfg['color_A'], linewidth=1.8)
        ax.plot(x, t_sm, label='T', color=cfg['color_T'], linewidth=1.8)
        ax.plot(x, c_sm, label='C', color=cfg['color_C'], linewidth=1.8)
        ax.plot(x, g_sm, label='G', color=cfg['color_G'], linewidth=1.8)

        ax.axvline(0, color="grey", linestyle="dashed", linewidth=1, label="A3E")

        if sp in mt_wins:
            mt_rel = mt_wins[sp] - tcs_win

            # ── ±2σ shaded band (drawn first so mT line sits on top) ──────
            if sp in st_wins:
                sigma_win = st_wins[sp]
                ax.axvspan(mt_rel - 2 * sigma_win,
                           mt_rel + 2 * sigma_win,
                           color="grey", alpha=0.2,
                           label=r"$\pm2\sigma_T$")

            ax.axvline(mt_rel, color="black", linestyle="dashed",
                       linewidth=1.2, label=r"$\mu_T$")

        ax.set_title(cfg['label'], fontsize=12, fontweight='bold')
        ax.set_ylim(ylim)
        ax.tick_params(axis='y', labelsize=16)

        # ── Gene-category badge (per species) ─────────────────────────────
        gc_genes = cfg.get('gc_genes', [])
        t_genes  = cfg.get('t_genes',  [])
        if gene in gc_genes:
            badge_color = '#e57a7aff'
        elif gene in t_genes:
            badge_color = 'grey'
        else:
            badge_color = None

        if badge_color:
            ax.add_patch(
                plt.Rectangle((0.88, 0.72), 0.10, 0.22,
                               transform=ax.transAxes,
                               facecolor=badge_color, edgecolor='black',
                               linewidth=0.8, clip_on=False, zorder=5)
            )

        if i == 0:
            ax.text(0.01, 0.95, gene, transform=ax.transAxes,
                    fontsize=16, va='top', ha='left', fontweight='bold')
            ax.set_ylabel("Base \n Content", fontsize=16)

            # ── Legend: line handles + category badge handles ──────────────
            line_handles, line_labels = ax.get_legend_handles_labels()
            badge_handles = [
                Patch(facecolor='#e57a7aff', edgecolor='black', label='GC-rich'),
                Patch(facecolor='grey',       edgecolor='black', label='T-rich'),
            ]
            
#             ax.legend(handles=line_handles + badge_handles,
#                       labels=line_labels + ['GC-rich', 'T-rich'],
#                       loc='upper right', fontsize=10, framealpha=1, edgecolor='black')

            ax.legend(handles=line_handles + badge_handles,
                      labels=line_labels + ['GC-rich', 'T-rich'],
                      loc='upper left',
                      bbox_to_anchor=(1.01, 1),
                      borderaxespad=0,
                      fontsize=16, framealpha=1, edgecolor='black')

    # ── 4. X-axis relative to TCS (kb scaling) ────────────────────────────────
    max_win   = max(len(species_data[sp]['A'])
                    for sp in species_order if species_data[sp] is not None)
    
    x_min     = -tcs_win  # This is -70 (for 7000 bp offset)
    x_max     = max_win - tcs_win
    
    tick_step_bp = 5000 
    tick_step_win = tick_step_bp // window_size
    
    # Create a range of ticks that covers your data
    xticks = np.arange(((x_min // tick_step_win) * tick_step_win), x_max, tick_step_win)
    
    # Generate labels: 0 becomes A3E, others become +/- X.X kb
    xlabels = []
    for t in xticks:
        bp_val = t * window_size
        if bp_val == 0:
            xlabels.append("A3E")
        else:
            kb_val = bp_val / 1000
            xlabels.append(f"{kb_val:+.1f}k") # e.g., -7.0k, +2.0k

    axs[-1].set_xticks(xticks)
    axs[-1].set_xticklabels(xlabels, fontsize=16, rotation=45)
    axs[-1].set_xlabel("Position relative to A3E (kb)", fontsize=16)
    
    if xlim is not None:
        axs[-1].set_xlim(xlim)  
    else:
        axs[-1].set_xlim(x_min, x_max)
    
    fig.subplots_adjust(hspace=0.5, top=0.95, bottom=0.15, left=0.15, right=0.85)
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/Base-Comp-4sp-{gene}.svg")
    plt.show()
    plt.close()
    
# With mouse plotting 
def plot_base_content_all_species_with_mouse(xlim,
                                  ylim,
                                  gene,
                                  fasta_dir,
                                  mt_df,
                                  species_config,
                                  species_order,
                                  tcs_offset=7000,
                                  window_size=100,
                                  smoothing_window=3,
                                  output_dir="/Users/geba9152/Paper-Figures-RD4/Fig3/"):
    """
    * Inputs: 
    - FASTA paths: {fasta_dir}/{genome_label}/{gene}.fa
    - Xlimits and y limits
    - Gene of interest
    - fasta directory 
    - mT dataframe 
    - species config & order specifications
    - Offset value of the TCS (how far upstream the TCS is)
    - Window details
    
    * Outputs: 
    - Base content plot across species surrounding mT
    
    TCS is at tcs_offset bp into each FASTA; mT marked at tcs_offset + mT-adj.
    Shaded grey band shows ±2σ around mT using sT_col from species_config.
    Skips any species missing a FASTA; skips gene entirely if none found.
    Grey square badge = T-rich gene (per species); pink badge = GC-rich gene (per species).
    
    ** Same as above but with mouse 
    """
    from matplotlib.patches import Patch

    tcs_win = tcs_offset // window_size

    # ── 1. Load base content — path is now subdir/gene.fa ─────────────────────
    species_data = {}
    for sp in species_order:
        cfg  = species_config[sp]
        path = f"{fasta_dir}/{cfg['genome_label']}/{gene}.fa"
        try:
            bc, _ = get_base_content_windows(path, window_size)
            species_data[sp] = bc
        except FileNotFoundError:
            print(f"  [{gene}] Skipping {cfg['label']}: FASTA not found at {path}")
            species_data[sp] = None

    if all(v is None for v in species_data.values()):
        print(f"Skipping {gene}: no FASTAs found for any species.")
        return

    # ── 2. Resolve mT window indices and ±2σ spans ────────────────────────────
    mt_wins = {}
    st_wins = {}
    if gene in mt_df.index:
        row = mt_df.loc[gene]
        for sp in species_order:
            cfg    = species_config[sp]
            mt_col = cfg["mt_col"]
            st_col = cfg.get("st_col")

            if mt_col in row.index and pd.notna(row[mt_col]):
                mt_wins[sp] = (tcs_offset + int(round(row[mt_col]))) // window_size

            if st_col and st_col in row.index and pd.notna(row[st_col]):
                st_wins[sp] = row[st_col] / window_size

    # ── 3. Plot ────────────────────────────────────────────────────────────────
    n_sp = len(species_order)
#     fig, axs = plt.subplots(n_sp, 1, figsize=(6, 1.5 * n_sp), sharex=True)
    fig, axs = plt.subplots(n_sp, 1, figsize=(10, 1.5 * n_sp), sharex=True)

#     fig.subplots_adjust(hspace=0)

    for i, sp in enumerate(species_order):
        ax  = axs[i]
        cfg = species_config[sp]
        bc  = species_data[sp]

        if bc is None:
            ax.set_title(f"{cfg['label']} — no data", fontsize=12)
            continue

        a_sm = smooth_data(bc['A'], smoothing_window)
        t_sm = smooth_data(bc['T'], smoothing_window)
        c_sm = smooth_data(bc['C'], smoothing_window)
        g_sm = smooth_data(bc['G'], smoothing_window)
        x    = np.arange(len(a_sm)) - tcs_win   # relative to TCS

        ax.plot(x, a_sm, label='A', color=cfg['color_A'], linewidth=1.8)
        ax.plot(x, t_sm, label='T', color=cfg['color_T'], linewidth=1.8)
        ax.plot(x, c_sm, label='C', color=cfg['color_C'], linewidth=1.8)
        ax.plot(x, g_sm, label='G', color=cfg['color_G'], linewidth=1.8)

        ax.axvline(0, color="grey", linestyle="dashed", linewidth=1, label="A3E")

        if sp in mt_wins:
            mt_rel = mt_wins[sp] - tcs_win

            # ── ±2σ shaded band (drawn first so mT line sits on top) ──────
            if sp in st_wins:
                sigma_win = st_wins[sp]
                ax.axvspan(mt_rel - 2 * sigma_win,
                           mt_rel + 2 * sigma_win,
                           color="grey", alpha=0.2,
                           label=r"$\pm2\sigma_T$")

            ax.axvline(mt_rel, color="black", linestyle="dashed",
                       linewidth=1.2, label=r"$\mu_T$")

        ax.set_title(cfg['label'], fontsize=12, fontweight='bold')
        ax.set_ylim(ylim)
        ax.tick_params(axis='y', labelsize=11)

        # ── Gene-category badge (per species) ─────────────────────────────
        gc_genes = cfg.get('gc_genes', [])
        t_genes  = cfg.get('t_genes',  [])
        if gene in gc_genes:
            badge_color = '#e57a7aff'
        elif gene in t_genes:
            badge_color = 'grey'
        else:
            badge_color = None

        if badge_color:
            ax.add_patch(
                plt.Rectangle((0.88, 0.72), 0.10, 0.22,
                               transform=ax.transAxes,
                               facecolor=badge_color, edgecolor='black',
                               linewidth=0.8, clip_on=False, zorder=5)
            )

        if i == 0:
            ax.text(0.01, 0.95, gene, transform=ax.transAxes,
                    fontsize=13, va='top', ha='left', fontweight='bold')
            ax.set_ylabel("Base Content", fontsize=11)

            # ── Legend: line handles + category badge handles ──────────────
            line_handles, line_labels = ax.get_legend_handles_labels()
            badge_handles = [
                Patch(facecolor='#e57a7aff', edgecolor='black', label='GC-rich'),
                Patch(facecolor='grey',       edgecolor='black', label='T-rich'),
            ]
            
#             ax.legend(handles=line_handles + badge_handles,
#                       labels=line_labels + ['GC-rich', 'T-rich'],
#                       loc='upper right', fontsize=10, framealpha=1, edgecolor='black')

            ax.legend(handles=line_handles + badge_handles,
                      labels=line_labels + ['GC-rich', 'T-rich'],
                      loc='upper left',
                      bbox_to_anchor=(1.01, 1),
                      borderaxespad=0,
                      fontsize=10, framealpha=1, edgecolor='black')

    # ── 4. X-axis relative to TCS (kb scaling) ────────────────────────────────
    max_win   = max(len(species_data[sp]['A'])
                    for sp in species_order if species_data[sp] is not None)
    
    x_min     = -tcs_win  # This is -70 (for 7000 bp offset)
    x_max     = max_win - tcs_win
    
    tick_step_bp = 5000 
    tick_step_win = tick_step_bp // window_size
    
    # Create a range of ticks that covers your data
    xticks = np.arange(((x_min // tick_step_win) * tick_step_win), x_max, tick_step_win)
    
    # Generate labels: 0 becomes A3E, others become +/- X.X kb
    xlabels = []
    for t in xticks:
        bp_val = t * window_size
        if bp_val == 0:
            xlabels.append("A3E")
        else:
            kb_val = bp_val / 1000
            xlabels.append(f"{kb_val:+.1f}k") # e.g., -7.0k, +2.0k

    axs[-1].set_xticks(xticks)
    axs[-1].set_xticklabels(xlabels, fontsize=11, rotation=45)
    axs[-1].set_xlabel("Position relative to A3E (kb)", fontsize=12)
    
    if xlim is not None:
        axs[-1].set_xlim(xlim)  
    else:
        axs[-1].set_xlim(x_min, x_max)
    
    fig.subplots_adjust(hspace=0.3, top=0.95, bottom=0.15, left=0.15, right=0.85)
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/Base-Comp-4sp-{gene}.svg")
    plt.show()
    plt.close()
    
    
############################################################################################################
######## Plotting genes above/below for alignment purposes (figuring out which MSAs to look at) ############
    

def mT_above_below_MSA(
    merged_df,
    sample1,
    sample2,
    outpath,
    label_dict,
    twosig,
    xlimylim,
    label_gene=None
):
    
    '''
    * Takes: 
    - dataframe with priors
    - Samples
    - Outpath 
    - Labels
    - Xlim & ylim
    - Genes to label
    
    * Outputs: 
    - Scatter with desired genes labels and information on genes above, below, and outside in general 2sigma
    '''
    
    # Linear regression (x=sample1, y=sample2, consistent with scatter)
    slope, intercept, r_value, p_value, std_err = linregress(
        merged_df[f"mT_adj_{sample1}"],
        merged_df[f"mT_adj_{sample2}"]
    )
    r_squared = r_value ** 2

    # Residuals: y - x, so positive = above the 1:1 line (sample2 > sample1)
    residuals = merged_df[f"mT_adj_{sample2}"] - merged_df[f"mT_adj_{sample1}"]

    genes_above  = merged_df.loc[residuals >  twosig, 'Gene'].tolist()
    genes_within = merged_df.loc[(residuals >= -twosig) & (residuals <= twosig), 'Gene'].tolist()
    genes_below  = merged_df.loc[residuals < -twosig, 'Gene'].tolist()

    # Plot
    plt.figure(figsize=(6, 6))
    ax = plt.gca()

    # x_vals drawn from sample1 (x-axis) min/max only
    x_vals = np.linspace(
        merged_df[f"mT_adj_{sample1}"].min(),
        merged_df[f"mT_adj_{sample1}"].max(),
        500
    )
    intsigma = int(twosig)
    plt.fill_between(
        x_vals,
        x_vals - twosig,
        x_vals + twosig,
        color="lightgrey",
        alpha=0.4,
        label=f"±2$\sigma$ ({intsigma})")
    plt.plot(x_vals, x_vals, linestyle="--", color="red", linewidth=1, label="1:1 line")

    lab1 = label_dict.get(sample1)
    lab2 = label_dict.get(sample2)

    sns.scatterplot(
        data=merged_df,
        x=f"mT_adj_{sample1}",
        y=f"mT_adj_{sample2}",
        color="#e3af6e",
        alpha=0.6,
        s=60,
        edgecolor=None
    )

    if label_gene is not None:
        gene_list = [label_gene] if isinstance(label_gene, str) else label_gene
        for gene in gene_list:
            if gene in merged_df['Gene'].values:
                gene_row = merged_df[merged_df['Gene'] == gene].iloc[0]
                gene_x = gene_row[f"mT_adj_{sample1}"]
                gene_y = gene_row[f"mT_adj_{sample2}"]
                plt.annotate(
                    gene,
                    xy=(gene_x, gene_y),
                    xytext=(gene_x + 500, gene_y + 500),
                    fontsize=12,
                    color='black',
                    fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.5)
                )
            else:
                print(f"Warning: Gene '{gene}' not found in the dataframe")

    plt.xlabel(f"{lab1} |A3E-$\mu_T$|", fontsize=14)
    plt.ylabel(f"{lab2} |A3E-$\mu_T$|", fontsize=14)
    plt.legend(loc="upper left", fontsize=11)
    plt.tight_layout()
    plt.xlim(xlimylim)
    plt.ylim(xlimylim)

    xticks = ax.get_xticks()
    yticks = ax.get_yticks()
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels(["A3E" if x == 0 else f"{int(x):,}" for x in xticks], fontsize=14, rotation=45)
    ax.set_yticklabels(["A3E" if y == 0 else f"{int(y):,}" for y in yticks], fontsize=14)

    plt.tight_layout()
    plt.savefig(outpath)
    plt.show()
    plt.close()

    return genes_above, genes_within, genes_below









































