# Import packages
import pandas as pd
import matplotlib.pyplot as plt
import re
import os
import numpy as np

# Performs Qcut analysis--> Clustering genes based on GC content
def gc_qcut_pub(fasta_file, 
                bed_file, 
                genes_to_remove, 
                sample, 
                output_filefig, 
                num_quantiles=4, 
                output_file=None):
    '''
    * Takes: 
    - Fasta file with base content information near mT (+/- 1kb)
    - Bed file with coordinate information
    - Genes to remove (testing purposes)
    - Sample name
    - Output file path (figure)
    - Default number of quantiles
    - Output file path (dataframe)

    * Outputs: 
    - Barplot with QCut analysis--> breaks up genes based on GC content near mT. 
    - Dataframe with gene, GC content, quantile information

    '''
    
    # parse fasta and calculate GC%
    fasta_dict = {}
    with open(fasta_file, 'r') as f:
        chrom = None
        start = None
        seq_lines = []

        for line in f:
            if line.startswith('>'):
                if chrom is not None:
                    key = (chrom, start)
                    fasta_dict[key] = ''.join(seq_lines).upper()

                header = line.strip()
                match = re.search(r'(chr[\w]+):(\d+)-\d+', header)
                if match:
                    chrom = match.group(1)
                    start = int(match.group(2))
                    seq_lines = []
                else:
                    print(f"Warning: Could not parse FASTA header: {header}")
                    chrom = None
            else:
                seq_lines.append(line.strip())
                
        if chrom is not None:
            key = (chrom, start)
            fasta_dict[key] = ''.join(seq_lines).upper()
            
    gene_gc = {}
    bed_df = pd.read_csv(bed_file, sep='\t', header=None, names=['chrom', 'start', 'end', 'name', '.', 'strand'])
    
    for _, row in bed_df.iterrows():
        key = (row['chrom'], row['start'])

        if key in fasta_dict:
            seq = fasta_dict[key]
            if len(seq) == 0:
                print(f"Warning: Empty sequence for {key}")
                continue
            gc_count = seq.count('G') + seq.count('C')
            gc_percentage = (gc_count / len(seq)) * 100
            gene_gc[key] = gc_percentage
        else:
            print(f"{key} not found in FASTA sequences")
        
    # process BED file for output and merge GC%
    output_rows = []
    for _, row in bed_df.iterrows():
        chrom = row['chrom']
        start = row['start']
        end = row['end']
        gene_name = row['name']
        strand = row['strand']

        gc_key = (chrom, start)
        gc_percent = gene_gc.get(gc_key, None)

        if gc_percent is not None:
            output_rows.append([chrom, start, end, gene_name, strand, round(gc_percent, 2)])
        else:
            print(f"{gene_name} not found in data")
            
    gc_content_df = pd.DataFrame(output_rows, columns=["chrom", "start", "end", "Gene", "strand", "GC%"])
    gc_content_df = gc_content_df[~gc_content_df["Gene"].isin(genes_to_remove)]
   
    # apply qcut and create the plot
    if not gc_content_df.empty:
        gc_content_df['GC%-QCut'] = pd.qcut(gc_content_df['GC%'], q=num_quantiles, labels=[f'{i+1}' for i in range(num_quantiles)], duplicates='drop')

        # calculate the mean GC% for each quantile
        plot_data = gc_content_df.groupby('GC%-QCut')['GC%'].mean()

        plt.figure(figsize=(10, 6))
        ax = plot_data.plot(kind='bar', color="#c3c0c0") 
        
        # customize bar colors based on quantiles (pink = GC, grey = T-rich)
        colors = []
        for i in range(len(plot_data)):
            if i < 8:  # quantiles 1-8
                colors.append('#c3c0c0') 
            else:      
                colors.append('#e57a7a') 
        
        ax.patches[i].set_facecolor(colors[i])

        plt.title(f'GC% Near Disassociation Distribution ({sample})', fontsize =15)
        plt.suptitle('') 
        plt.xlabel('GC% Quantile', fontsize =15)
        plt.ylabel('GC% ±1kb of $\mu_T$', fontsize =15)
        plt.grid(axis='y')
        plt.xticks(fontsize=14) 
        plt.yticks(fontsize=14) 
        plt.savefig(output_filefig)
        plt.show(output_filefig)
        
        # calculate and print the difference between the last two quantiles
        if len(plot_data) >= 2:
            last_two_quantiles = plot_data.tail(2)
            difference = last_two_quantiles.iloc[1] - last_two_quantiles.iloc[0]
            print(f"difference in mean GC% between the last two quantiles (Q{len(plot_data)-1} and Q{len(plot_data)}): {difference:.2f}")

    else:
        print("no valid data to plot.")

    # save to file if output_file is provided
    if output_file:
        gc_content_df.to_csv(output_file, sep='\t', header=True, index=False)
        print(f"Saved GC% summary BED file: {output_file}")

    return gc_content_df

# Save qcut analysis information --> runs the function above (gc_qcut_pub)
def save_clusters(fasta_file, 
                  bed_file, 
                  outdir, 
                  genes_to_remove, 
                  species_name, 
                  output_filefig, 
                  num_quantiles=9):
    '''
    * Takes: 
    - Fasta file with base content information near mT (+/- 1kb)
    - Bedfile
    - Output directory
    - Genes to remove (testing purposes)
    - Species name
    - Output file (figure)

    * Outputs: 
    - Dataframe with gene, gc content near mT (1kb in paper--in bedfile)
    - Dataframes with T-rich/GC-rich genes (quantiles 1-8 and quantile 9)

    '''
    
    df = gc_qcut_pub(fasta_file, bed_file, genes_to_remove, species_name, output_filefig, num_quantiles=num_quantiles, output_file=None)

    # GC & Trich clusts
    GC_clust = df[df["GC%-QCut"] == f'{num_quantiles}']
    T_clust = df[df["GC%-QCut"].isin([f'{i}' for i in range(1, num_quantiles)])]

    T_clust_genes = T_clust['Gene'].to_list()
    GC_clust_genes = GC_clust['Gene'].to_list()

    # save new bedfiles
    bed_df = pd.read_csv(bed_file, sep='\t', header=None, names=['chrom', 'start', 'end', 'name', '.', 'strand'])

    GC_clust_bed_df = bed_df[bed_df['name'].isin(GC_clust_genes)]
    T_clust_bed_df = bed_df[bed_df['name'].isin(T_clust_genes)]

    # save
    name = os.path.basename(bed_file).split(".bed")[0]
    GC_clust_bed_df.to_csv(f"{outdir}{name}-{species_name}-GC-cluster.bed", sep="\t", header=None, index=None)
    T_clust_bed_df.to_csv(f"{outdir}{name}-{species_name}-T-cluster.bed", sep="\t", header=None, index=None)

    return df, T_clust_genes, GC_clust_genes
