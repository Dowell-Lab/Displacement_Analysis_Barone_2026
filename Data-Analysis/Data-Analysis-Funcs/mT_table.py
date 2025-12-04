# Import
import pandas as pd
import os
import sys

sys.path.append('/Users/geba9152/3prime_end/Data-Analysis-Funcs/')
from load_data import pull_out_params_meta_samples
from load_data import obtain_plotting_dict_hg38
from load_data import obtain_plotting_dict_cross_species

def generate_mt_coordinate_table(metadata, 
                                 outpath, 
                                 filtereddir, 
                                 genes_to_remove, 
                                 res_path,
                                 col1='mT-adj'):
    '''
    * Takes:
    - Metadata table
    - Outdir
    - Filtered gene list directory
    - Genes to remove (for testing purposes)
    - Prior column (coordinate of prior--centered) 
    
    * Outputs: 
    - Single table with genes as rows and mT coordinates for each experiment-sample as columns
    '''
    
    all_mt_coords = {}
    all_genes = set()
    
    for index, row in metadata.iterrows():
        experiment = row['Experiment']
        samples = row['Samples'].replace("'", "").split(",")
        ann_path = row['Annotation']
        
        # load annotation
        ann = pd.read_csv(ann_path, sep="\t", header=None)
        ann.columns = ['chr', 'start', 'stop', 'Gene', "Length", "strand"]
        ann = ann[['chr', 'start', 'stop', 'strand', 'Gene']]
        
        # get sample results
        samples_res = pull_out_params_meta_samples(
            f"{res_path}/{experiment}/",
            ann_path
        )
        
        # load filtered gene list
        filtered_path = os.path.join(filtereddir, experiment, 'tfit_residual_weight_antisense_sense_coverage_filtered_genes.txt')
        filtered_genes = pd.read_csv(filtered_path, sep="\t", header=None)[0].tolist()
    
        # Coverage filter
        cov_filt_ann = pd.read_csv(ann_path, sep="\t", header=None)[3].tolist()
        print(ann_path)
        
        filtered_genes = [gene for gene in filtered_genes if gene in cov_filt_ann]
        filtered_genes = [gene for gene in filtered_genes if gene not in genes_to_remove]
        
        for samp in samples:
            try:
                df = samples_res[samp][['Gene', 'Length', col1]]
            except KeyError as e:
                print(f" error {samp}")
                continue
            
            # Merge with annotation
            bed_merged = df.merge(ann, on="Gene")
            
            # Calculate mT coordinates
            bed_merged['mT-coords'] = bed_merged.apply(
                lambda row: row['stop'] + row[col1] if row['strand'] == "+" else row['start'] - row[col1], 
                axis=1
            )
            
            bed_merged = bed_merged[bed_merged['Gene'].isin(filtered_genes)]
            
            # Store mT coordinates with experiment_sample key
            key = f"mT_{experiment}_{samp}"
            gene_mt_dict = bed_merged.set_index('Gene')['mT-coords'].to_dict()
            all_mt_coords[key] = gene_mt_dict
            
            all_genes.update(bed_merged['Gene'].tolist())
        
    final_df = pd.DataFrame(index=sorted(all_genes))
    final_df.index.name = 'Gene'
    
    # Add columns for each experiment-sample combination
    for key in sorted(all_mt_coords.keys()):
        final_df[key] = final_df.index.map(all_mt_coords[key])
    
    # Save the table
    final_df.to_csv(outpath, sep="\t", na_rep='NA')
    
    
    return final_df
