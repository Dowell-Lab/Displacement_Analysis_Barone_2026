# Import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import math

# Function for counting bases --> base comp analyses
def count_bases(seqs, window): 
    '''calculate per position base composition across multiple 
    sequences of even length. 
    
    Parameters
    ----------
    seqs : list or array
        sequences of even lengths
        
    outdir : str
        path for out directory
    
    sample_name : str
        name of sample
        
    window : int
        length of sequences 
        
    Returns
    -------
    anew,tnew,cnew,gnew,nnew : list of lists
        normalized base counts for every position across all
        sequences
        
    ADAPTED FROM RU'S CODE
    
    '''
    
    sequence_length = int(2*int(window)+1)
    
    ##initialize lists with len of sequences
    a = [0]*sequence_length
    t = [0]*sequence_length
    c = [0]*sequence_length
    g = [0]*sequence_length
    n = [0]*sequence_length

    ##for each positions count the occurance of each base
    ##across all sequences in the the input list
    for i in range(sequence_length):
        ##initialize counters
        count_a = 0
        count_t = 0
        count_c = 0
        count_g = 0
        count_n = 0
        for j in seqs:
            if j[i] == "a" or j[i] == "A":
                count_a = count_a + 1
                a[i] = count_a 
            elif j[i] == "t" or j[i] == "T":
                count_t = count_t + 1
                t[i] = count_t
            elif j[i] == "g" or j[i] == "G":
                count_g = count_g + 1 
                g[i] = count_g
            elif j[i] == "c" or j[i] == "C":
                count_c = count_c + 1   
                c[i] = count_c
            elif j[i] == "n" or j[i] == "N":
                count_n = count_n + 1   
                n[i] = count_n 
                
    ##evenly distribute Ns across all bases
    nnew = [x / 4 for x in n]

    anew = [ai + bi for ai,bi in zip(a,nnew)]
    tnew = [ai + bi for ai,bi in zip(t,nnew)]
    gnew = [ai + bi for ai,bi in zip(g,nnew)]
    cnew = [ai + bi for ai,bi in zip(c,nnew)]

    ##get the base frequencies of all bases
    anew = [x / len(seqs) for x in anew]
    tnew = [x / len(seqs) for x in tnew]
    cnew = [x / len(seqs) for x in cnew]
    gnew = [x / len(seqs) for x in gnew]
    
    base_df = pd.DataFrame({'A': anew,
                            'T': tnew,
                            'G': cnew,
                            'C': gnew})
    
    return anew, tnew, cnew, gnew, nnew

# Plots base composition in a dynamic grid (adjustable to how many fastas given) at A3E
def count_and_plot_base_composition_dynamic_grid_A3E(fadir_dict, 
                                                 outdir):
    """
    * Takes: 
    - A dictionary with labels as keys and fasta file paths as values
    - Output directory
    
    * Outputs: 
    - Grid with base composition across quantiles
    
    """
    if not fadir_dict:
        print("Error-no fasta dir")
        return

    num_files = len(fadir_dict)
    ncols = 2
    nrows = math.ceil(num_files / ncols)

    # Determine figure size based on the number of rows
    figsize_height = 4 * nrows 
    figsize_width = 8
    
    fig, axs = plt.subplots(nrows, ncols, figsize=(figsize_width, figsize_height),
                            sharex=True, sharey=False, squeeze=False)
    
    # The `squeeze=False`--> makes it so its always a 2D array
    
    for idx, (label, fasta_path) in enumerate(fadir_dict.items()):
        # calc row and column index for the current subplot
        row = idx // ncols
        col = idx % ncols
        ax = axs[row, col]

        # load seqs 
        sequences = []
        try:
            with open(fasta_path) as fasta:
                sequences = [line.strip() for line in fasta if not line.startswith('>')]
        except FileNotFoundError:
            print(f"File not found: {fasta_path}")
            continue
        
        if not sequences:
            print(f"No sequences found in {fasta_path}")
            continue

        window = len(sequences[0])
        half_window = (window // 2) - 1
        positions = np.arange(-half_window, half_window + 1)
        
        try:
            counts = count_bases(sequences, half_window)
        except NameError:
            print("count_bases function needs to be ran")
            continue

        # Smooth data (Savitzky-Golay filter)
        a1 = savgol_filter(counts[0], 61, 3)
        t1 = savgol_filter(counts[1], 61, 3)
        c1 = savgol_filter(counts[2], 61, 3)
        g1 = savgol_filter(counts[3], 61, 3)

        ax.plot(positions, a1, color='blue', alpha=0.75, label='A')
        ax.plot(positions, t1, color='purple', alpha=0.75, label='T')
        ax.plot(positions, c1, color='orange', alpha=0.75, label='C')
        ax.plot(positions, g1, color='red', alpha=0.75, label='G')
        ax.set_title(label, fontsize=18)
        ax.set_xlim(-4000, 4000)
        
        if col == 0:
            ax.set_ylabel('Base Content', fontsize=18)
        
        if row == nrows - 1:
            ax.set_xlabel('Distance (bp)', fontsize=18)

        xticks = ax.get_xticks()
        xtick_labels = [r'A3E' if int(x) == 0 else str(int(x)) for x in xticks]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xtick_labels, fontsize=18)
        ax.tick_params(axis='y', labelsize=18)
        ax.tick_params(axis='x', labelsize=18, labelrotation=30)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))

        if idx == num_files - 1:
            ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=13)
    
    for i in range(num_files, nrows * ncols):
        fig.delaxes(axs.flatten()[i])
    
    plt.tight_layout()
    plt.savefig(outdir)
    plt.show()
    plt.close()

# Plots base composition in a dynamic grid (adjustable to how many fastas given)
def count_and_plot_base_composition_dynamic_grid(fadir_dict, 
                                                 outdir):
    """
    * Takes: 
    - A dictionary with labels as keys and fasta file paths as values
    - Output directory
    
    * Outputs: 
    - Grid with base composition across quantiles
    
    """
    if not fadir_dict:
        print("Error-no fasta dir")
        return

    num_files = len(fadir_dict)
    ncols = 2
    nrows = math.ceil(num_files / ncols)

    # Determine figure size based on the number of rows
    figsize_height = 4 * nrows 
    figsize_width = 8
    
    fig, axs = plt.subplots(nrows, ncols, figsize=(figsize_width, figsize_height),
                            sharex=True, sharey=False, squeeze=False)
    
    # The `squeeze=False`--> makes it so its always a 2D array
    
    for idx, (label, fasta_path) in enumerate(fadir_dict.items()):
        # calc row and column index for the current subplot
        row = idx // ncols
        col = idx % ncols
        ax = axs[row, col]

        # load seqs 
        sequences = []
        try:
            with open(fasta_path) as fasta:
                sequences = [line.strip() for line in fasta if not line.startswith('>')]
        except FileNotFoundError:
            print(f"File not found: {fasta_path}")
            continue
        
        if not sequences:
            print(f"No sequences found in {fasta_path}")
            continue

        window = len(sequences[0])
        half_window = (window // 2) - 1
        positions = np.arange(-half_window, half_window + 1)
        
        try:
            counts = count_bases(sequences, half_window)
        except NameError:
            print("count_bases function needs to be ran")
            continue

        # Smooth data (Savitzky-Golay filter)
        a1 = savgol_filter(counts[0], 61, 3)
        t1 = savgol_filter(counts[1], 61, 3)
        c1 = savgol_filter(counts[2], 61, 3)
        g1 = savgol_filter(counts[3], 61, 3)

        ax.plot(positions, a1, color='blue', alpha=0.75, label='A')
        ax.plot(positions, t1, color='purple', alpha=0.75, label='T')
        ax.plot(positions, c1, color='orange', alpha=0.75, label='C')
        ax.plot(positions, g1, color='red', alpha=0.75, label='G')
        ax.set_title(label, fontsize=18)
        ax.set_xlim(-4000, 4000)
        
        if col == 0:
            ax.set_ylabel('Base Content', fontsize=18)
        
        if row == nrows - 1:
            ax.set_xlabel('Distance (bp)', fontsize=18)

        xticks = ax.get_xticks()
        xtick_labels = [r'$\mu_T$' if int(x) == 0 else str(int(x)) for x in xticks]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xtick_labels, fontsize=18)
        ax.tick_params(axis='y', labelsize=18)
        ax.tick_params(axis='x', labelsize=18, labelrotation=30)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))

        if idx == num_files - 1:
            ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=13)
    
    for i in range(num_files, nrows * ncols):
        fig.delaxes(axs.flatten()[i])
    
    plt.tight_layout()
    plt.savefig(outdir)
    plt.show()
    plt.close()

# Base composition plot for a 2 x 1 grid
def count_and_plot_base_composition_2x1_grid(fadir_dict, 
                                             outdir):
    """
    * Takes: 
    - A dictionary with labels as keys and fasta file paths as values
    - Outdir

    * Outputs: 
    - Base comp plot (2x1 grid)

    """

    if len(fadir_dict) != 2:
        raise ValueError("2 fasta files for a 1x2 grid")
    
#     fig, axs = plt.subplots(2, 1, figsize=(5.5, 8), sharex=True, sharey=True)
    fig, axs = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=False)

    for idx, (label, fasta_path) in enumerate(fadir_dict.items()):
        # load sequences
        sequences = []
        with open(fasta_path) as fasta:
            for line in fasta:
                line = line.strip()
                if not line.startswith(">"):
                    sequences.append(line)

        if not sequences:
            print(f"No sequences found in {fasta_path}. Skipping.")
            continue

        window = len(sequences[0])
        half_window = (window // 2) - 1
        positions = np.arange(-half_window, half_window + 1)

        # count bases
        counts = count_bases(sequences, half_window)

        # smooth
        a1 = savgol_filter(counts[0], 61, 3)
        t1 = savgol_filter(counts[1], 61, 3)
        c1 = savgol_filter(counts[2], 61, 3)
        g1 = savgol_filter(counts[3], 61, 3)

        # plot
        ax = axs[idx]
        ax.plot(positions, a1, color='blue', alpha=0.75, label='A')
        ax.plot(positions, t1, color='purple', alpha=0.75, label='T')
        ax.plot(positions, c1, color='orange', alpha=0.75, label='C')
        ax.plot(positions, g1, color='red', alpha=0.75, label='G')

        ax.set_title(label, fontsize=18)
        ax.set_xlim(-4000, 4000)

        # Labels
        if idx == 0:
            ax.set_ylabel("Base Content", fontsize=18)
            
        if idx == 0:
            ax.set_xlabel("Distance (bp)", fontsize=18)

#         # annotate ±1 kb
#         ax.axvline(-1000, color='gray', linestyle='--')
#         ax.axvline(1000, color='gray', linestyle='--')
        
#         if idx == 0:
#             ax.annotate("-1kb", xy=(-1500, 0.92), xycoords=('data', 'axes fraction'),
#                         fontsize=14, ha='center', va='bottom', color='#878481')
#             ax.annotate("+1kb", xy=(1500, 0.92), xycoords=('data', 'axes fraction'),
#                         fontsize=14, ha='center', va='bottom', color='#878481')

        xticks = ax.get_xticks()
        xtick_labels = ['$\mu_T$' if int(x) == 0 else str(int(x)) for x in xticks]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xtick_labels, fontsize=18)
        ax.tick_params(axis='y', labelsize=18)
        ax.tick_params(axis='x', labelsize=18, labelrotation=30)

        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))

        if idx == 1:
            ax.legend(
                loc='upper left',
                bbox_to_anchor=(1.02, 1),
                fontsize=13,
            )
            
    plt.tight_layout()
    plt.savefig(outdir)
    plt.show()
    plt.close()

# 9x1 base comp plot --> cross quantiles
def count_and_plot_base_composition_9x1_grid(fadir_dict, 
                                             outdir):
    
    """
    * Takes: 
    - A dictionary with labels as keys and fasta file paths as values
    - Outdir

    * Outputs: 
    - Base comp plot (9x1 grid)

    """

    fig, axs = plt.subplots(9, 1, figsize=(4.5, 25), sharex=True, sharey=False)

    for idx, (label, fasta_path) in enumerate(fadir_dict.items()):
        # load sequences
        sequences = []
        with open(fasta_path) as fasta:
            for line in fasta:
                line = line.strip()
                if not line.startswith(">"):
                    sequences.append(line)

        if not sequences:
            print(f"No sequences found in {fasta_path}. Skipping.")
            continue

        window = len(sequences[0])
        half_window = (window // 2) - 1
        positions = np.arange(-half_window, half_window + 1)

        # count bases
        counts = count_bases(sequences, half_window)

        # smooth
        a1 = savgol_filter(counts[0], 61, 3)
        t1 = savgol_filter(counts[1], 61, 3)
        c1 = savgol_filter(counts[2], 61, 3)
        g1 = savgol_filter(counts[3], 61, 3)

        # plot
        ax = axs[idx]
        ax.plot(positions, a1, color='blue', alpha=0.75, label='A')
        ax.plot(positions, t1, color='purple', alpha=0.75, label='T')
        ax.plot(positions, c1, color='orange', alpha=0.75, label='C')
        ax.plot(positions, g1, color='red', alpha=0.75, label='G')

        ax.set_title(label, fontsize=18)
        ax.set_xlim(-4000, 4000)

        # labels
        if idx == 0:
            ax.set_ylabel("Base Content", fontsize=18)
            
        if idx == 1:
            ax.set_xlabel("Distance (bp)", fontsize=18)

        # annotate ±1 kb
        ax.axvline(-1000, color='gray', linestyle='--')
        ax.axvline(1000, color='gray', linestyle='--')
        
        if idx == 0:
            ax.annotate("-1kb", xy=(-1500, 0.92), xycoords=('data', 'axes fraction'),
                        fontsize=14, ha='center', va='bottom', color='#878481')
            ax.annotate("+1kb", xy=(1500, 0.92), xycoords=('data', 'axes fraction'),
                        fontsize=14, ha='center', va='bottom', color='#878481')

        xticks = ax.get_xticks()
        xtick_labels = ['$\mu_T$' if int(x) == 0 else str(int(x)) for x in xticks]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xtick_labels, fontsize=18)
        ax.tick_params(axis='y', labelsize=18)
        ax.tick_params(axis='x', labelsize=18, labelrotation=30)

        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))

        if idx == 0:
            ax.legend(loc='upper right', fontsize=13)

    plt.tight_layout()
    plt.savefig(outdir)
    plt.show()
    plt.close()
