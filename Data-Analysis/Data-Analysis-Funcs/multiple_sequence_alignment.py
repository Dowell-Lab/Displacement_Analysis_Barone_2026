# Import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from Bio import AlignIO, SeqIO
import re

def plot_msa_mT(msa_path, 
                gene="", 
                label_dict= None):
    
    '''
    * Takes: 
    - Multiple sequence alignment path
    - Gene name
    - Label dictionary
    
    * Outputs:
    - Multiple sequence alignment code
    
    '''
    
    # load alignment 
    alignment = AlignIO.read(msa_path, "clustal")
    print("Alignment record IDs:", [rec.id for rec in alignment])

    # color matrix
    char_to_index = {"-": 0, "A": 1, "C": 2, "G": 3, "T": 4}
    colors = [
        "#979497",  # gap - grey
        "#708eea",  # A -  blue
        "#f5ea5f",  # C - yellow
        "#f75a4c",  # G -  red
        "#c991d7",  # T - purple
        "#e6e0e7"   # n - light grey
    ]
    cmap = ListedColormap(colors)

    aln_len = alignment.get_alignment_length()
    num_sp = len(alignment)
    color_matrix = np.zeros((num_sp, aln_len), dtype=int)

    for i, rec in enumerate(alignment):
        seq = str(rec.seq).upper()
        for j, base in enumerate(seq):
            color_matrix[i, j] = char_to_index.get(base, 5)

    # convert genome coords to alignment index 
    def genome_coord_to_aln_index(seq, target_coord):
        count = 0
        for i, base in enumerate(seq):
            if base != "-":
                count += 1
                if count == target_coord:
                    return i
        return None

    mt2st_dict = {}
    for rec in alignment:
        # 3kb to end --> where mT is 
        seq_length = len(str(rec.seq).replace('-', '')) 
        mt_position = seq_length - 3000
        mt2st_dict[rec.id] = mt_position


    mt2st_indices = {}
    if mt2st_dict:
        for rec in alignment:
            species = rec.id
            seq = str(rec.seq)
            mt2st_pos = mt2st_dict.get(species)
            if mt2st_pos is not None:
                mt2st_idx = genome_coord_to_aln_index(seq, round(mt2st_pos)) 
                mt2st_indices[species] = mt2st_idx
                print(f"{species}: mT+2sT at {mt2st_pos:.2f} → aln_idx = {mt2st_idx}")

    # plot 
    fig, ax = plt.subplots(figsize=(7, 2 + num_sp * 0.5))
    ax.imshow(color_matrix, aspect='auto', cmap=cmap, interpolation='none')

    for i, rec in enumerate(alignment):
        sp = rec.id   
        mt_x = mt2st_indices.get(sp)
        if mt_x is not None:
            ax.vlines(mt_x, i - 0.4, i + 0.4, color='black', linestyle='--', linewidth=2, label='$\mu_T$')
            
    legend_lines = [
        Line2D([0], [0], color='black', linestyle='--', linewidth=2, label=r'$\mu_T$')    ]
    
    base_legend = [
        Patch(facecolor="#708eea", edgecolor='black', label='A'),
        Patch(facecolor="#f5ea5f", edgecolor='black', label='C'),
        Patch(facecolor="#f75a4c", edgecolor='black', label='G'),
        Patch(facecolor="#c991d7", edgecolor='black', label='T'),
        Patch(facecolor="#979497", edgecolor='black', label='Gap (-)')
    ]
    ax.legend(handles=legend_lines + base_legend, bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0., fontsize=12)

    ax.set_yticks(np.arange(num_sp))
    ax.set_yticklabels([label_dict[rec.id] for rec in alignment], fontsize=15)
    
    ax.tick_params(axis='x', labelsize=15) 
    
    # x-axis ticks and labels
    current_ticks = ax.get_xticks()
    new_labels = [str(int(tick)) for tick in current_ticks]

    # label TCS
    target_idx = np.argmin(np.abs(current_ticks - 5000))
    if current_ticks[target_idx] > 0 and current_ticks[target_idx] <= aln_len:
        new_labels[target_idx] = "TCS"  

    ax.set_xticks(current_ticks) 
    ax.set_xticklabels(new_labels)  
    
    ax.set_xlim(4200,19000)
    
    ax.set_xlabel("Alignment (bases)", fontsize=15) 
    ax.set_title(f"{gene} Sequence Alignment", fontsize=17)

    plt.tight_layout()
    plt.savefig(f"/Users/geba9152/Paper-Figures-RD4/Fig3/{gene}-alignment.png", dpi=300, bbox_inches='tight')
    plt.show()

################
## Base comp change plot functions 
def get_gene_name(filename):
    """
    * Takes: 
    - file name
    
    * Outputs:
    - Gene name
    
    """
    match = re.search(r'-(\w+)\.fa$', filename)
    return match.group(1) if match else None

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

def plot_base_content_single_gene_AT(gene,
                      original_human_dir,
                      original_rhesus_dir,
                      window_size=100,
                      mT_human=None,
                      mT_rhesus=None,
                      smoothing_window=3):  
    
    """
    * Takes
    - Gene
    - Window size 
    - mT human
    - mT rhesus
    - Insertions
    - Deletions
    - Smoothing window
    
    * Outputs
    - The smoothed data
    """
    
    # fasta files
    original_human_path = f"{original_human_dir}/Meta-BSA-hg38-{gene}.fa"
    original_rhesus_path = f"{original_rhesus_dir}/Meta-BSA-rheMac10-{gene}.fa"
    
    # base content
    try:
        human_base_content, _ = get_base_content_windows(original_human_path, window_size)
        rhesus_base_content, _ = get_base_content_windows(original_rhesus_path, window_size)
    except FileNotFoundError:
        print(f"no fasta file found")
        return
    
    # Apply smoothing
    human_a_smoothed = smooth_data(human_base_content['A'], smoothing_window)
    human_t_smoothed = smooth_data(human_base_content['T'], smoothing_window)
    rhesus_a_smoothed = smooth_data(rhesus_base_content['A'], smoothing_window)
    rhesus_t_smoothed = smooth_data(rhesus_base_content['T'], smoothing_window)
    fig, axs = plt.subplots(2, 1, figsize=(4.5, 3.5), sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0.5)
    
    # Human 
    axs[0].plot(human_a_smoothed, label='A', color='blue', linewidth=2) 
    axs[0].plot(human_t_smoothed, label='T', color='purple', linewidth=2) 
    axs[0].axvline(0, color="grey", linestyle="dashed", linewidth=1)
    if mT_human is not None:
        axs[0].axvline(int(mT_human // window_size), color="black", linestyle="dashed", linewidth=1, label="$\mu_T$")
    axs[0].set_title("Human", fontsize=13)
    axs[0].set_ylabel("Base Content", fontsize=13)
    axs[0].set_ylim(0, 0.6)
    axs[0].legend(loc='upper right', fontsize=12, framealpha=1, edgecolor='black')
    axs[0].text(0.01, 0.95, gene, transform=axs[0].transAxes, fontsize=14, va='top', ha='left')
    
    # Rhesus 
    axs[1].plot(rhesus_a_smoothed, color='blue', linewidth=2)  
    axs[1].plot(rhesus_t_smoothed, color='purple', linewidth=2)  
    axs[1].axvline(0, color="grey", linestyle="dashed", linewidth=1)
    if mT_rhesus is not None:
        axs[1].axvline(int(mT_rhesus // window_size), color="black", linestyle="dashed", linewidth=1)
    axs[1].set_title("Rhesus", fontsize=13)
    axs[1].set_ylim(0.2, 0.35)
    axs[1].set_xlim(0, 16000 // window_size)
    
    max_len = max(len(human_base_content['A']), len(rhesus_base_content['A']))
    xticks = np.arange(0, max_len, 2)
    xlabels = [("TCS" if i == 0 else str(i * window_size)) for i in xticks]
    axs[1].set_xticks(xticks)
    axs[1].set_xticklabels(xlabels, fontsize=13, rotation=45)
    axs[1].set_xlabel("Position (bp)", fontsize=13)
    axs[0].tick_params(axis='y', labelsize=13)
    axs[1].tick_params(axis='y', labelsize=13)
    
    handles, labels = axs[1].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    axs[1].legend(unique.values(), unique.keys(), loc='upper right', fontsize=12, framealpha=1, edgecolor='black')
    plt.tight_layout()
    plt.savefig("/Users/geba9152/Paper-Figures-RD4/Fig3/Base-Comp-UNG.svg")
    plt.show()
    plt.close()
