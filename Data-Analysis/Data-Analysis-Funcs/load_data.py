# Import packages 
import pandas as pd
import sys
import matplotlib.pyplot as plt
from collections import defaultdict, OrderedDict
import glob
import os
import numpy as np
import re


# LIET library packages
sys.path.append('/Users/geba9152/LIET/liet/')
import rnap_lib_data_proc as dp
from liet_res_class import FitParse, fitparse_intersect

def pull_out_params_meta_samples(base_path, annotation):
    '''
    * Takes: 
    - Results path and annotation file (file LIET was ran with)

    * Returns: 
    - Dictionary of LIET results (per-sample)

    '''
    
    # Initialize dictionary to store the results
    results_dict = defaultdict(pd.DataFrame)
    
    lietpath = f"{base_path}/*.liet"


    # get list of files for each type
    lietfiles = glob.glob(lietpath)

    # process each group of files 
    for lietfile in lietfiles:
        celltype = os.path.basename(lietfile).split('.')[0]  # extract sample name from filename
       
        # call fitparse function to get output priors
        fit_parser = FitParse(lietfile)
        
        data = {
            "Gene": fit_parser.genes,
            "mL": fit_parser.mL,
            "mL_std": fit_parser.mL_std,
            "sL": fit_parser.sL,
            "sL_std": fit_parser.sL_std,
            "tI": fit_parser.tI,
            "tI_std": fit_parser.tI_std,
            "tI_a": fit_parser.tI_a,
            "tI_a_std": fit_parser.tI_a_std,
            "mT": fit_parser.mT,
            "mT_std": fit_parser.mT_std,
            "sT": fit_parser.sT,
            "sT_std": fit_parser.sT_std,
            "w": fit_parser.w,
            "w_std": fit_parser.w_std,
            "mL_a": fit_parser.mL_a,
            "mL_a_std": fit_parser.mL_a_std,
            "sL_a": fit_parser.sL_a,
            "sL_a_std": fit_parser.sL_a_std,
            "tI_a": fit_parser.tI_a,
            "tI_a_std": fit_parser.tI_a_std,
            "w_a": fit_parser.w_a,
            "w_a_std": fit_parser.w_a_std,
        }
        
        data_df = pd.DataFrame(data)
        original_len = len(data_df)
        
        print(f"Number of genes LIET fit ({celltype}): {original_len}")
                
        # read in annotation
        ann = pd.read_csv(annotation, sep = "\t", header = None)
        ann = ann[[3,4]]
        ann.columns = ['Gene','Length']
        
        df = ann.merge(data_df, on='Gene')
        merged_len = len(df)
              
        print(f"Number of genes LIET fit/merge with annotation ({celltype}): {merged_len}")
        
        df['mT-adj'] = df['mT'] - df['Length']
        df['raw-mT'] = df['mT'] 
                
        cellname = os.path.basename(lietfile).replace('.liet',"")
        
        results_dict[cellname] = pd.concat([results_dict[cellname], df], ignore_index=True)        

    return results_dict

# Function for pulling out priors -- hg38 (human)
def obtain_plotting_dict_hg38(metadata_df, respath, genelist, genes_to_remove):
    
    '''
    * Takes: 
    - Metadata table
    - Results path
    - Filtered gene list
    - Any genes we want to remove from analysis (for testing purposes only)

    * Returns: 
    - plotting_dict, which is a dictionary of dataframes with output prior information

    '''
    
    # Initalize plotting dictionary (dict of priors)
    plotting_dict = defaultdict(dict)  

    # Loop through metadata table
    for _, experiment in metadata_df.iterrows():
        experiment_name = experiment['Experiment']
        samples = experiment['Samples'].split(',')
        annotation_file = experiment['Annotation']
        colors = experiment['Colors'].split(',')

        # LIET results path
        exprespath = os.path.join(respath, experiment_name)

        # Pull out results
        res_dict = pull_out_params_meta_samples(exprespath, annotation_file)

        # Loop through samples
        for sample in samples:
            if sample not in res_dict:
                print(f"{sample} not found in results for experiment '{experiment_name}'.")
                continue

            sample_data = res_dict[sample]
            
            # read in filtered gene list (pipeline output)
            genes_to_keep_path = f"/scratch/Users/geba9152/LIET-3end-analysis/LIET-Gene-Filtering/filtered_genes/{experiment_name}/{genelist}"
            genes_to_keep = pd.read_csv(genes_to_keep_path, sep="\t", header=None)[0].tolist()
            print(len(genes_to_keep))
            
            # New filter post LIET run -- Make sure genes meet new coverage cutoff; in annotated lists
            # /scratch/Users/geba9152/LIET-3end-analysis/gene_curation_MANE/annotations/ (see section above for how I reran this)
            cov_filt_ann = pd.read_csv(annotation_file, sep="\t", header=None)[3].tolist()
            
            # Now intersect the list-- if the gene is in genes_to_keep_path, but not in cov_filt_ann; get rid of
            # This means that initally with the lower coverage cutoff the gene was okay, but now it is not
            genes_to_keep = [gene for gene in genes_to_keep if gene in cov_filt_ann]
            print(len(genes_to_keep))

            # Filter by gene list
            sample_data = sample_data[sample_data['Gene'].isin(genes_to_keep)]
            
            # filter out genes to remove (testing purposes; genes_to_remove is 0 for final analysis in each samp)
            sample_data = sample_data[~sample_data['Gene'].isin(genes_to_remove)]
            
            # Break up the weights into separate columns
            sample_data["wL"] = sample_data["w"].str[0].astype(float)
            sample_data["wE"] = sample_data["w"].str[1].astype(float)
            sample_data["wT"] = sample_data["w"].str[2].astype(float)
            sample_data["wB"] = sample_data["w"].str[3].astype(float)

            sample_data["wL_a"] = sample_data["w_a"].str[0].astype(float)
            sample_data["wB_a"] = sample_data["w_a"].str[1].astype(float)
            
            # Calculate pausing ratio a couple different ways
            sample_data["wT/wE"] = (sample_data["wT"] /  (sample_data["wE"] + 0.01))
            sample_data["wT/wE-Normalized"] = (sample_data["wT"] / sample_data["sT"]) / ((sample_data["wE"] + 0.01) / abs(sample_data["mL"] - sample_data["mT"]))
            sample_data["wT/wE-Length"] = (sample_data["wT"] / (sample_data["wE"] + 0.01)) * (abs(sample_data["mL"] - sample_data["mT"]))

            # Save to plotting dictionary
            plotting_dict[experiment_name][sample] = sample_data

    return plotting_dict

# Function for pulling out priors -- cross species
def obtain_plotting_dict_cross_species(metadata_df, respath, genelist, genes_to_remove):
    '''
    * Takes: 
    - Metadata table
    - Results path
    - Filtered gene list
    - Any genes we want to remove from analysis (for testing purposes only)

    * Returns: 
    - plotting_dict, which is a dictionary of dataframes with output prior information

    '''
    
    # Initalize plotting dictionary (dict of priors)
    plotting_dict = defaultdict(dict)  

    # Loop through metadata table
    for _, experiment in metadata_df.iterrows():
        experiment_name = experiment['Experiment']
        samples = experiment['Samples'].split(',')
        annotation_file = experiment['Annotation']
        colors = experiment['Colors'].split(',')

        # Experiment results path
        exprespath = os.path.join(respath, experiment_name)

        # Pull out results
        res_dict = pull_out_params_meta_samples(exprespath, annotation_file)

        for sample in samples:
            if sample not in res_dict:
                print(f"{sample} not found in results for experiment '{experiment_name}'.")
                continue

            sample_data = res_dict[sample]
            
            # read in filtered gene list
            genes_to_keep_path = f"/scratch/Users/geba9152/LIET-3end-analysis/LIET-Gene-Filtering-Cross-Species/filtered_genes/{experiment_name}/{genelist}"
            genes_to_keep = pd.read_csv(genes_to_keep_path, sep="\t", header=None)[0].tolist()
            print(len(genes_to_keep))
            
            # New as of 10.01.2025 -- Make sure genes meet new coverage cutoff; in annotated lists
            # /scratch/Users/geba9152/LIET-3end-analysis/gene_curation_MANE/annotations/ (see section above for how I reran this)
            cov_filt_ann = pd.read_csv(annotation_file, sep="\t", header=None)[3].tolist()
            
            # Now intersect the list-- incorperating additional coverage filtering
            genes_to_keep = [gene for gene in genes_to_keep if gene in cov_filt_ann]
            print(len(genes_to_keep))

            # Filter by gene list
            sample_data = sample_data[sample_data['Gene'].isin(genes_to_keep)]
            print(len(sample_data))
            
            # filter out genes to remove (testing purposes)
            sample_data = sample_data[~sample_data['Gene'].isin(genes_to_remove)]
            print(len(sample_data))

            # Save to plotting dictionary
            plotting_dict[experiment_name][sample] = sample_data

    return plotting_dict
