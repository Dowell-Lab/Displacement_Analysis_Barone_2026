# Import packages
import pandas as pd
import sys
import os
import glob
sys.path.append('/Users/geba9152/fall_2024_3prime/')
from load_in_liet import pull_out_params_meta_samples
import argparse

# 1. Remove genes with high minimum residuals
def parse_residuals_from_logs(log_files, directory):
    """
    * Takes: 
    - log files from LIET
    - directory
    
    * Outputs:
    - Residuals (parses file)
    
    """
    residuals_by_log = {}
    
    for log_file in log_files:
        log_name = os.path.basename(log_file).replace(".liet.log", "")
        genes, residuals = [], []

        with open(log_file, 'r') as f:
            current_gene = None
            for line in f:
                if line.startswith('>'):
                    current_gene = line.split(':')[0].split('>')[1]
                elif line.startswith('elbo_range:') and current_gene:
                    residual = float(line.split('(')[1].split(',')[0])
                    genes.append(current_gene)
                    residuals.append(residual)
                    current_gene = None

        if genes and residuals:
            residuals_by_log[log_name] = pd.DataFrame({
                'Gene': genes,
                'min_residual': residuals
            })

    return residuals_by_log

def scale_and_filter_residuals(metadata, directory, output_dir, experiment):
    """
    * Takes:
    - metadata (example shown in metadata-final-hg38.txt)
    - directory
    - output directory
    - experiment of interest
    
    * Outputs:
    - saves a filtered list (residuals)
   
    """
    experiment_results = {}

    # Read in metadata file/exp filter
    metadata_df = pd.read_csv(metadata, sep="\t")
    metadata_df = metadata_df.loc[metadata_df['Experiment'] == experiment]   
    
    # Set variables
    experiment_name = metadata_df['Experiment'].iloc[0]
    samples = metadata_df['Samples'].iloc[0].replace("'", "").split(",")  
    ann_file = metadata_df['Annotation'].iloc[0]
    residual_cutoff = metadata_df['Residual_Cutoff'].iloc[0]
        
    expdirectory = f"{directory}/{experiment_name}/"
    log_files = glob.glob(os.path.join(expdirectory, '*.log'))
    
    if not log_files:
        print(f"No log files--error")
    
    # Get residuals
    residuals = parse_residuals_from_logs(log_files, directory)

    # annotation file
    ann = pd.read_csv(ann_file, sep="\t", names=['chr', 'start', 'stop', 'Gene', 'len', 'strand'], header=None)
    all_cts = ann[['Gene']]
    lenoriginal = len(all_cts)

    for sample in samples:
        if sample in residuals:
            df = residuals[sample]
            merge = df.merge(ann[['Gene', 'len']], on="Gene")
            merge[f'scaled_min_residual_{sample}'] = merge['min_residual'] / merge['len']
            merge = merge[['Gene', f'scaled_min_residual_{sample}']]
            all_cts = all_cts.merge(merge, on="Gene", how="left")

    # ensure genes pass residual cutoff in all samples
    residual_columns = [col for col in all_cts.columns if col.startswith('scaled_min_residual_')]
    all_cts['passes_cutoff'] = all_cts[residual_columns].le(residual_cutoff).all(axis=1)
    all_cts = all_cts.loc[(all_cts['passes_cutoff'] == True)]
    postfilterlen = len(all_cts)

    # filter genes that pass in all samples
    filtered_results = all_cts[all_cts['passes_cutoff']].drop(columns=['passes_cutoff'])

    # Save results
    filtered_results = filtered_results['Gene']
    experiment_dir = os.path.join(output_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    filtered_results.to_csv(os.path.join(experiment_dir, 'residual_filtered_genes.txt'), index=False, header = False, sep = "\t")
    print(f"{experiment_name} filtered length {postfilterlen}, original length {lenoriginal}")

# 2. Remove genes with Tfit overlaps near mT
def parse_bedtools_intersect(metadata, intersectdir, outdir, experiment):
    """
    * Takes:
    - metadata
    - data with intersect information (post Tfit)
    - outdirectory
    - experiment of interest
    
    * Outputs:
    - saves a filtered list (Tfit)

    """
    # Read in metadata file
    metadata_df = pd.read_csv(metadata, sep="\t")
    metadata_df = metadata_df.loc[metadata_df['Experiment'] == experiment]   
    
    # Set variables
    experiment_name = metadata_df['Experiment'].iloc[0]
    samples = metadata_df['Samples'].iloc[0].replace("'", "").split(",")  
    ann_file = metadata_df['Annotation'].iloc[0]
    residual_cutoff = metadata_df['Residual_Cutoff'].iloc[0]
        
    # Read in pre-filtered genes per experiment
    residual_filtered_file = os.path.join(outdir, f"{experiment_name}/residual_filtered_genes.txt")

    if not os.path.exists(residual_filtered_file):
        print(f"Filtered file not found for {experiment_name}")

    # Start with only genes that have been prefiltered
    genes_to_keep = pd.read_csv(residual_filtered_file, sep="\t", header = None)
    genes_to_keep = genes_to_keep[0].to_list()
    starting_gene_len = len(genes_to_keep)

    # Initialize the intersection of genes across all samples
    genes_passing_all_samples = set(genes_to_keep)  

    # Loop through samples
    for samp in samples:
        
        # Read in intersect file
        intersect_file = os.path.join(intersectdir, f"{experiment_name}/{samp}_mT_bidir_intersect.bed")
        if not os.path.exists(intersect_file):
            print(f"Intersect file not found for sample {samp} {experiment_name}")
            continue

        df = pd.read_csv(intersect_file, sep="\t", header=None)

        # Keep only genes that passed residual filter
        df = df[df[3].isin(genes_to_keep)]

        # Filter rows where column 6 is not equal to 1 --> 1 means there was an intersection
        filtered_df = df[df[6] < 1]

        # Keep only the genes column (column 3)
        filtered_df = filtered_df[[3]]
        filteredlensamp = len(filtered_df)
        print(f"For {samp}, genes that pass tfit filter: {filteredlensamp} ")

        genes_passing_all_samples = genes_passing_all_samples.intersection(set(filtered_df[3].to_list()))

    final_gene_len = len(genes_passing_all_samples)
    print(f"{experiment_name}: Final genes after filtering across all samples: {final_gene_len}, original size {starting_gene_len}")

    output_file = os.path.join(outdir, f"{experiment_name}/tfit_residual_filtered_genes.txt")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    pd.DataFrame(list(genes_passing_all_samples), columns=["Gene"]).to_csv(output_file, sep="\t", index=False, header=False)

# 3. Remove genes with high wB,wB_a, remove genes with wT of 0.0, remove any gene where a weight parameter is > 95%
def weight_filter(metadata, res_dir, output_dir, experiment):
    """
    * Takes:
    - metadata
    - results directory
    - output directory
    - experiment of interest
    
    * Outputs:
    - saves a filtered list (weights)
    """

    # Read in metadata file
    metadata_df = pd.read_csv(metadata, sep="\t")
    
    # Search for experiment
    metadata_df = metadata_df.loc[metadata_df['Experiment'] == experiment]   
    
    # Set variables
    experiment_name = metadata_df['Experiment'].iloc[0]
    samples = metadata_df['Samples'].iloc[0].replace("'", "").split(",")  
    annotation_file = metadata_df['Annotation'].iloc[0]
    residual_cutoff = metadata_df['Residual_Cutoff'].iloc[0]
    wB_thresh = metadata_df['wB_Cutoff'].iloc[0]
    wB_a_thresh = metadata_df['wB_a_Cutoff'].iloc[0]

    print(f"Processing weight filtering for experiment: {experiment_name}")

    # Load residual-filtered genes for this experiment
    residual_filtered_file = os.path.join(output_dir, f"{experiment_name}/tfit_residual_filtered_genes.txt") 

    if not os.path.exists(residual_filtered_file):
        print(f"Residual-filtered file not found for {experiment_name}, skipping.")

    # Start with genes that have been prefiltered
    genes_to_keep = pd.read_csv(residual_filtered_file, sep = "\t", header = None)
    genes_to_keep = genes_to_keep[0].to_list()
    starting_gene_len = len(genes_to_keep)

    exp_res_dir = f"{res_dir}/{experiment_name}/"

    # Perform weight filtering
    res = pull_out_params_meta_samples(exp_res_dir, annotation_file)

    # Keep track of genes that have passed all filters
    genes_passing_all_samples = set(genes_to_keep)

    # Loop through samples
    for samp in samples:
        # Process weight data for each sample
        df = res[samp]

        # Break up weights into separate columns
        df["wL"] = df["w"].str[0].astype(float)
        df["wE"] = df["w"].str[1].astype(float)
        df["wT"] = df["w"].str[2].astype(float)
        df["wB"] = df["w"].str[3].astype(float)
        df["wL_a"] = df["w_a"].str[0].astype(float)
        df["wB_a"] = df["w_a"].str[1].astype(float)

        # Keep genes that passed residual step and weight thresholds
        # Adding in a step where any gene with a weight prior that is over a 0.95 is tossed
        regthresh = 0.95
        wTthresh = 0.0

        filtered_genes = df[
            (df['Gene'].isin(genes_passing_all_samples)) &
            (df["wB"] <= wB_thresh) &
            (df["wB_a"] <= wB_a_thresh) &
            (df["wL"] <= regthresh) &
            (df["wE"] <= regthresh) &
            (df["wT"] <= regthresh) &
            (df["wB"] <= regthresh) &
            (df["wT"] > wTthresh)
            ]['Gene']

        print(f"Sample {samp} ({experiment_name}) - Genes after weight filtering: {len(filtered_genes)}")

        # Update genes_passing_all_samples to retain only those passing for this sample
        genes_passing_all_samples.intersection_update(filtered_genes)

    # Convert the final set of genes to a list for output
    final_gene_list = list(genes_passing_all_samples)

    # Save the final list of genes
    output_file = os.path.join(output_dir, f"{experiment_name}/tfit_residual_weight_filtered_genes.txt")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    pd.DataFrame(final_gene_list, columns=["Gene"]).to_csv(output_file, index=False, sep="\t", header=None)

    print(f"Final list of genes passing weight thresholds for {experiment_name}: {len(final_gene_list)}, starting size {starting_gene_len}")

# 4. Remove genes with antisense + keep genes with sense coverage near mT
def antisense_sense_coverage_filter(metadata_df, res_dir, coverage_dir, outdir, experiment):
    """
    * Takes:
    - metadata
    - results directory
    - coverage directory
    - output directory
    - experiment of interest
    
    * Outputs:
    - saves a filtered list (antisense/sense coverage)
    """
    
    metadata_df = pd.read_csv(metadata_df, sep="\t")
    
    metadata_df = metadata_df.loc[metadata_df['Experiment'] == experiment]  

    for _, experiment in metadata_df.iterrows():
        experiment_name = experiment['Experiment']
        annotation_file = experiment['Annotation']
        samples = experiment['Samples'].split(',')
        coverage_thresh = experiment['Coverage_Antisense_Cutoff']
        
        # load residual-filtered genes for this experiment
        filtered_file = os.path.join(outdir, f"{experiment_name}/tfit_residual_weight_filtered_genes.txt")
        if not os.path.exists(filtered_file):
            print(f"Residual-Tfit-Weight-filtered file not found for {experiment_name}, skipping.")
            continue

        genes_to_keep = pd.read_csv(filtered_file, sep="\t", header=None, names=["Gene"])['Gene'].tolist()
        genes_passing_all_samples = set(genes_to_keep)
        starting_len = len(genes_passing_all_samples)

        # load annotation file once for all samples
        annotations = pd.read_csv(annotation_file, sep="\t", names=['chr', 'start', 'stop', 'Gene', 'length', 'strand'], header=None)
#         annotations = annotations[['Gene', 'strand']]
        annotations = annotations['Gene'].to_list()

        for samp in samples:
            print(f"Processing sample {samp} for experiment {experiment_name}")

            # load coverage data for both strands
            coverage_pos_file = os.path.join(coverage_dir, experiment_name, f"{samp}_tfit_prelim_cov_stranded_3000_window_pos.bed")
            coverage_neg_file = os.path.join(coverage_dir, experiment_name, f"{samp}_tfit_prelim_cov_stranded_3000_window_neg.bed")

            # Skip if coverage files are missing
            if not os.path.exists(coverage_pos_file) or not os.path.exists(coverage_neg_file):
                print(f"Error")
                continue

            # Read and combine coverage files
            coverage_pos = pd.read_csv(coverage_pos_file, sep="\t", header=None, names=['chr', 'start', 'stop', 'Gene', '.', 'strand', 'cov1', 'cov2', 'cov3', 'cov4'])
            coverage_pos['cov_strand'] = "+"
            
            coverage_neg = pd.read_csv(coverage_neg_file, sep="\t", header=None, names=['chr', 'start', 'stop', 'Gene', '.', 'strand', 'cov1', 'cov2', 'cov3', 'cov4'])
            coverage_neg['cov_strand'] = "-"
            
            coverage_data = pd.concat([coverage_pos, coverage_neg])
            
            df = coverage_data.sort_values('Gene')
            
            # Filter coverage data to only include relevant genes
            coverage_data = coverage_data[coverage_data['Gene'].isin(genes_passing_all_samples)]
            coverage_data = coverage_data.sort_values("Gene")

            # Keep only ANTISENSE coverage (where cov_strand != gene strand)
            antisense_coverage = coverage_data[coverage_data['cov_strand'] != coverage_data['strand']]

            # Keep SENSE coverage 
            sense_coverage = coverage_data[coverage_data['cov_strand'] == coverage_data['strand']]

            # Filter based on the coverage threshold (antisense coverage must be <= threshold)
            genes_meeting_antisense_threshold = antisense_coverage[antisense_coverage['cov4'] <= coverage_thresh]['Gene'].unique()

            genes_meeting_antisense_threshold = set(genes_meeting_antisense_threshold)
            print(f"Sample {samp} - Genes meeting antisense coverage threshold: {len(genes_meeting_antisense_threshold)}")
            
            # Lowest depth/quality experiment -- U2OS/Arsenic data
            # Tighter 3' end coverage cutoff to address the low quality of this sample
            if experiment_name in ['Meta-Nina-Arsenic', 'Nina-Arsenic', 'U2OS-SRRs', 'U2OS-Specific']:
                
                genes_meeting_sense_threshold = sense_coverage[sense_coverage['cov4'] >= 0.0075]['Gene'].unique()
                
            # Everything else is at a cutoff of 0.001
            else:
                
                genes_meeting_sense_threshold = sense_coverage[sense_coverage['cov4'] >= 0.001]['Gene'].unique()

            # Only keep genes that pass BOTH antisense and sense thresholds
            genes_meeting_threshold = genes_meeting_antisense_threshold.intersection(genes_meeting_sense_threshold)
            print(f"Sample {samp} - Genes meeting BOTH thresholds: {len(genes_meeting_threshold)}")

            # Update the set of genes passing across all samples
            genes_passing_all_samples.intersection_update(genes_meeting_threshold)
            print(f"Remaining genes after sample {samp}: {len(genes_passing_all_samples)}")

        # final list of genes for the experiment
        output_file = os.path.join(outdir, f"{experiment_name}/tfit_residual_weight_antisense_sense_coverage_filtered_genes.txt")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        pd.DataFrame(list(genes_passing_all_samples), columns=["Gene"]).to_csv(output_file, index=False, sep="\t", header=None)
        print(f"Final list of genes for {experiment_name}: {len(genes_passing_all_samples)}, starting len {starting_len}")

# dirs
coverage_dir = '/scratch/Users/geba9152/LIET-3end-analysis/LIET-Gene-Filtering/mT-stranded_cov/'
output_dir = "/scratch/Users/geba9152/LIET-3end-analysis/LIET-Gene-Filtering/filtered_genes/"

# 5. Final main function
def main(metadata_file, wd, inputdir, experiment):
    
    # Set dirs
    output_dir = os.path.join(wd, 'filtered_genes/')
    intersectdir = os.path.join(wd, 'bedtools-intersect/')
    coverage_dir = os.path.join(wd, 'mT-stranded_cov/')
    
    # 1. Residuals
    scale_and_filter_residuals(metadata_file, inputdir, output_dir, experiment)
    
    # 2. Intersect filter
    parse_bedtools_intersect(metadata_file, intersectdir, output_dir, experiment)
    
    # 3. Weight filter
    weight_filter(metadata_file, inputdir, output_dir, experiment)
    
    # 4. Antisense/sense coverage filter
    antisense_sense_coverage_filter(metadata_file, inputdir, coverage_dir, output_dir, experiment)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process gene filtering steps.")
    parser.add_argument("--metadata", type=str, required=True, help="Path to the metadata file.")
    parser.add_argument("--wd", type=str, required=True, help="Working directory path.")
    parser.add_argument("--inputdir", type=str, required=True, help="Input directory path.")
    parser.add_argument("--experiment", type=str, required=True, help="Experiment name.")
    args = parser.parse_args()

    main(args.metadata, args.wd, args.inputdir, args.experiment)
