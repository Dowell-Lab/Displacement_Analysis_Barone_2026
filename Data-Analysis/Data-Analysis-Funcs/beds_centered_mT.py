# Import packages
import pandas as pd
import numpy as np
import os
from load_data import pull_out_params_meta_samples

# hg38
## Generate filtered mT beds cross species for scatter cluster analysis
def generate_and_filter_beds(metadata, 
                             window, 
                             base_outdir, 
                             filtereddir, 
                             genes_to_remove, 
                             col1='mT-adj'):
    '''
    * Takes:
    - Metadata table
    - Window 
    - Outdir
    - Filtered gene list directory
    - Genes to remove (for testing purposes)
    - Prior column (coordinate of prior--centered) 

    * Outputs: 
    - Bed file centered on prior

    '''
    
    half_window = window / 2

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
            f"/scratch/Users/geba9152/LIET-3end-analysis/gene_curation_MANE/results/{experiment}/",
            ann_path
        )
        
        print(f"Processing experiment: {experiment}")
        print(f"Attempting to read file in dir: {filtereddir}")

        # load filtered gene list
        filtered_path = os.path.join(filtereddir, experiment, 'tfit_residual_weight_antisense_sense_coverage_filtered_genes.txt')
        filtered_genes = pd.read_csv(filtered_path, sep="\t", header=None)[0].tolist()
    
        # New as of 10.01.2025 -- Make sure genes meet new coverage cutoff; in annotated lists
        # /scratch/Users/geba9152/LIET-3end-analysis/gene_curation_MANE/annotations/ (see section above for how I reran this)
        cov_filt_ann = pd.read_csv(ann_path, sep="\t", header=None)[3].tolist()
        
        print(ann_path)

        # Now intersect the list-- if the gene is in genes_to_keep_path, but not in cov_filt_ann; get rid of
        # This means that initally with the lower coverage cutoff the gene was okay, but now it is not
        filtered_genes = [gene for gene in filtered_genes if gene in cov_filt_ann]

        print(len(filtered_genes))

        # output directory
        outdir = os.path.join(base_outdir, experiment)
        os.makedirs(outdir, exist_ok=True)

        for samp in samples:
            print(f"Processing {experiment} - {samp}")

            try:
                df = samples_res[samp][['Gene', 'Length', col1]]
            except KeyError as e:
                print(f"Skipping sample '{samp}' due to missing columns: {e}")
                continue

            bed_merged = df.merge(ann, on="Gene")
            bed_merged['mT-coords'] = bed_merged.apply(
                lambda row: row['stop'] + row[col1] if row['strand'] == "+" else row['start'] - row[col1], axis=1
            )

            bed_merged[f'{col1}-start'] = bed_merged['mT-coords'] - half_window
            bed_merged[f'{col1}-end'] = bed_merged['mT-coords'] + half_window

            valid_mask = np.isfinite(bed_merged[f'{col1}-start']) & np.isfinite(bed_merged[f'{col1}-end'])
            bed_merged = bed_merged[valid_mask]
            bed_merged[[f'{col1}-start', f'{col1}-end']] = bed_merged[[f'{col1}-start', f'{col1}-end']].astype(int)

            # final bed 
            bed = bed_merged[['chr', f'{col1}-start', f'{col1}-end', 'Gene']]
            bed['.'] = '.'
            bed['strand'] = bed_merged['strand']
            bed = bed.sort_values(by=['chr', f'{col1}-start'])

            bed = bed[bed['Gene'].isin(filtered_genes) & ~bed['Gene'].isin(genes_to_remove)]
            bed = bed.drop_duplicates(subset=['Gene'], keep='first')

            # save
            outpath = os.path.join(outdir, f"{window}-window-{samp}-filtered.bed")
            os.makedirs(os.path.dirname(outpath), exist_ok=True)
            bed.to_csv(outpath, sep="\t", header=False, index=False)

# cross species
## Generate filtered mT beds cross species for scatter cluster analysis

def generate_and_filter_beds_non_hg38(metadata, 
                                      window, 
                                      base_outdir, 
                                      filtereddir, 
                                      genes_to_remove, 
                                      col1='mT-adj'):
    '''
    * Takes:
    - Metadata table
    - Window 
    - Outdir
    - Filtered gene list directory
    - Genes to remove (for testing purposes)
    - Prior column (coordinate of prior--centered) 

    * Outputs: 
    - Bed file centered on prior

    '''
    
    half_window = window / 2

    for index, row in metadata.iterrows():
        experiment = row['Experiment']
        samples = row['Samples'].replace("'", "").split(",")
        ann_path = row['Annotation']

        # annotation
        ann = pd.read_csv(ann_path, sep="\t", header=None)
        ann.columns = ['chr', 'start', 'stop', 'Gene', "Length", "strand"]
        ann = ann[['chr', 'start', 'stop', 'strand', 'Gene']]

        # sample results
        samples_res = pull_out_params_meta_samples(
            f"/scratch/Users/geba9152/LIET-3end-analysis/daniel-animal-liet/results/{experiment}/",
            ann_path
        )

        # filtered gene list
        filtered_path = os.path.join(filtereddir, experiment, 'tfit_residual_weight_antisense_sense_coverage_filtered_genes.txt')
        filtered_genes = pd.read_csv(filtered_path, sep="\t", header=None)[0].tolist()
    
        # New as of 10.01.2025 -- Make sure genes meet new coverage cutoff; in annotated lists
        # /scratch/Users/geba9152/LIET-3end-analysis/gene_curation_MANE/annotations/ (see section above for how I reran this)
        cov_filt_ann = pd.read_csv(ann_path, sep="\t", header=None)[3].tolist()
        
        print(ann_path)

        # Now intersect the list-- if the gene is in genes_to_keep_path, but not in cov_filt_ann; get rid of
        # This means that initally with the lower coverage cutoff the gene was okay, but now it is not
        filtered_genes = [gene for gene in filtered_genes if gene in cov_filt_ann]

        print(len(filtered_genes))

        outdir = os.path.join(base_outdir, experiment)
        os.makedirs(outdir, exist_ok=True)

        for samp in samples:
            print(f"Processing {experiment} - {samp}")

            try:
                df = samples_res[samp][['Gene', 'Length', col1]]
            except KeyError as e:
                print(f"Skipping sample '{samp}' due to missing columns: {e}")
                continue

            bed_merged = df.merge(ann, on="Gene")
            bed_merged['mT-coords'] = bed_merged.apply(
                lambda row: row['stop'] + row[col1] if row['strand'] == "+" else row['start'] - row[col1], axis=1
            )

            bed_merged[f'{col1}-start'] = bed_merged['mT-coords'] - half_window
            bed_merged[f'{col1}-end'] = bed_merged['mT-coords'] + half_window

            valid_mask = np.isfinite(bed_merged[f'{col1}-start']) & np.isfinite(bed_merged[f'{col1}-end'])
            bed_merged = bed_merged[valid_mask]
            bed_merged[[f'{col1}-start', f'{col1}-end']] = bed_merged[[f'{col1}-start', f'{col1}-end']].astype(int)

            # final bed 
            bed = bed_merged[['chr', f'{col1}-start', f'{col1}-end', 'Gene']]
            bed['.'] = '.'
            bed['strand'] = bed_merged['strand']
            bed = bed.sort_values(by=['chr', f'{col1}-start'])

            bed = bed[bed['Gene'].isin(filtered_genes) & ~bed['Gene'].isin(genes_to_remove)]
            bed = bed.drop_duplicates(subset=['Gene'], keep='first')

            # Save
            outpath = os.path.join(outdir, f"{window}-window-{samp}-filtered.bed")
            os.makedirs(os.path.dirname(outpath), exist_ok=True)
            bed.to_csv(outpath, sep="\t", header=False, index=False)


