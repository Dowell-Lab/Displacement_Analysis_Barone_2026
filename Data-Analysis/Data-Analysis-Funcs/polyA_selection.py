import pandas as pd 
import os
    
# Function to get top PAS per gene based on column 5 'score'
def get_top_pas(polyAdb, sample_col, shared_genes, output_name):
    '''
    * Takes: 
    - PolyA database
    - Samples column
    - Shared genes between samples
    - Output
    
    * Outputs: 
    - PAS with highest score in each sample (per cell types)
    '''
    
    df = polyAdb.copy()
    
    shared_genes = shared_genes[0].str.split("|").str[0].to_list()
    
    df = df[df['gene_name'].isin(shared_genes)]
    
    df = df[['gene_name', 'chrom', 'name', 'strand', sample_col]]
        
    # Keep the PAS with the highest score per gene
    df = df.sort_values(sample_col, ascending=False)
    df = df.groupby('gene_name').first().reset_index()
    
    # name parse
    df["PAS_adjusted_start"] = df["name"].str.split(":").str[1].astype(int)
    df["PAS_adjusted_stop"] = df["PAS_adjusted_start"].astype(int) + 1   
    
    df['chrom'] = df['chrom'].astype(str).str.replace('^', 'chr', regex=True)
    df['.'] = '.'
    df_bed = df[['chrom', 'PAS_adjusted_start', 'PAS_adjusted_stop', 'gene_name', '.', 'strand']]
    
    # Sort by chromosome and start
    df_bed = df_bed.sort_values(['chrom', 'PAS_adjusted_start'])
    
    # Save to file
    out_path = f"/Users/geba9152/3prime_end_reviews_genomeR/polyA-site-analysis/{output_name}.bed"
    df_bed.to_csv(out_path, sep='\t', header=False, index=False)
    
    return df_bed


def make_annotation_file_and_pad_PAS(ann, celltype, sample_df):
    '''
    * Takes: 
    - Cell type/sample of interest
    - Annotation file for this sample
    - 3Seq informed PAS sites
    
    * Outputs: 
    - 3Seq informed LIET ready annotation & pad files
    '''
    
    ############## Annotation ##############
    ########################################
    
    # read in ann cell type
    ann_ct = f"{ann}{celltype}/{celltype}-5p-UTR.liet.ann"
    ann_ct = pd.read_csv(ann_ct, sep = "\t", header = None)

    ann_ct['gene_name'] = ann_ct[3].str.split("|").str[0]

    # Merge with hela_average to get the adjusted values
    ann_ct = ann_ct.merge(sample_df[['gene_name', 'PAS_adjusted_start']], on='gene_name', how='left')
    ann_ct = ann_ct.dropna()
    
    ann_ct['PAS_adjusted_start'] = ann_ct['PAS_adjusted_start'].astype(int)
        
    # if gene is positive, replace [2] with PAS_adjusted start in hela_average
    ann_ct['shift'] = ann_ct.apply(
        lambda x: x['PAS_adjusted_start'] - x[2] if x[5] == '+' else x['PAS_adjusted_start'] - x[1], 
        axis=1)
    
    shift_tracker = ann_ct[['gene_name', 'shift']].copy()
    
    ann_ct[1] = ann_ct.apply(lambda x: x['PAS_adjusted_start'] if x[5] == '-' else x[1], axis=1)
    ann_ct[2] = ann_ct.apply(lambda x: x['PAS_adjusted_start'] if x[5] == '+' else x[2], axis=1)

    ann_ct = ann_ct[[0,1,2,3,4,5]]
    ann_ct.to_csv(f"{ann}{celltype}/{celltype}-3SeqPAS.liet.ann", sep = '\t', header = None, index = None)

    ############## Pad #####################
    ########################################
    
    # read in pad cell type
    pad_ct = f"{ann}{celltype}/{celltype}-5p-UTR.pad"
    pad_ct = pd.read_csv(pad_ct, sep = "\t", header = None)
    
    pad_ct['threep'] = pad_ct[1].str.split(",").str[1]
    pad_ct['fivep'] = pad_ct[1].str.split(",").str[0]

    pad_ct['threep'] = pad_ct['threep'].astype(int)
    pad_ct['fivep'] = pad_ct['fivep'].astype(int)

    pad_ct['gene_name'] = pad_ct[0].str.split("|").str[0]
    
    # merge with shift tracker
    pad_ct = shift_tracker.merge(pad_ct, on = "gene_name")
    
    # adjust 3p pad
    pad_ct['threep_adj'] = pad_ct['threep'] + pad_ct['shift']
    pad_ct['pad'] = pad_ct['fivep'].astype(str).str.cat(pad_ct['threep_adj'].astype(str), sep=',')
    
    # save
    pad_ct = pad_ct[["gene_name","pad"]]
    pad_ct.to_csv(f"{ann}{celltype}/{celltype}-3SeqPAS.pad", sep = '\t', header = None, index = None)