# Import packages
import pandas as pd
import os
import sys
import argparse
sys.path.append('/Users/geba9152/fall_2024_3prime/')
from load_in_liet import pull_out_params_meta_samples

def bed_centered_on_prior(col1, 
                          ann_file, 
                          samples_res, 
                          samples, 
                          experiment_outdir, 
                          windowsize):
    '''
    * Takes:
    - col1 -- mT-adj (mT-TCS)
    - annotation file
    - sample results
    - samples
    - outdir
    - window size around mT
    
    * Outputs:
    - a file centered on mT for gene filtering 
    
    '''
    
    # Set half window
    half_window = windowsize // 2
    
    # Read in annotation file/obtain desired columns
    ann = pd.read_csv(ann_file, sep="\t", header=None, names=['chr', 'start', 'stop', 'Gene', "Length", "strand"])
    ann = ann[['chr', 'start', 'stop', 'strand', 'Gene']]

    # Loop through samples
    for sample in samples:
        
        # Obtain samp specific df/appropriate columns/merge with annotation
        df = samples_res[sample]
        df = df[['Gene', 'Length', 'mT-adj']]
        bed = df.merge(ann, on="Gene")
        
        # Find mT coordinates
        bed['mT-coords'] = bed.apply(lambda row: row['stop'] + row[col1] if row['strand'] == "+" else row['start'] - row[col1], axis=1)

        # Find start and stop windows
        bed['start'] = bed['mT-coords'] - half_window
        bed['stop'] = bed['mT-coords'] + half_window

        # Set start and stop as integers
        bed['start'] = bed['start'].astype(int)
        bed['stop'] = bed['stop'].astype(int)

        # Set a "." column
        bed['.'] = "."
    
        # Obtain apporopriate columns/sort/save
        bed = bed[['chr', 'start', 'stop', 'Gene', ".", 'strand']]
        bed = bed.drop_duplicates(subset=['Gene'])
        bed = bed.sort_values(['chr', 'start'])
        out_path = os.path.join(experiment_outdir, f"{windowsize}-window-{sample}.bed")
        bed.to_csv(out_path, sep="\t", index=False, header=False)

# main func
def main(metadata_file, 
         wd, 
         inputdir, 
         window, 
         experiment):
    
    # Read in metadata file
    metadata = pd.read_csv(metadata_file, sep="\t")
    
    # Search for experiment
    metadata = metadata.loc[metadata['Experiment'] == experiment]   
    
    # Set variables
    experiment = metadata['Experiment'].iloc[0]
    samples = metadata['Samples'].iloc[0].replace("'", "").split(",")  
    ann_file = metadata['Annotation'].iloc[0]
    print(f"{samples}")
    samples_res = pull_out_params_meta_samples(f"{inputdir}/{experiment}/", ann_file)
    print(f"{len(samples_res)}")
    
    # Set outdir/make outdir
    experiment_outdir = os.path.join(wd, 'tfit-in', experiment)
    os.makedirs(experiment_outdir, exist_ok=True)

    # centered on prior (mT-adj) function
    bed_centered_on_prior('mT-adj', ann_file, samples_res, samples, experiment_outdir, window)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create BED files for Tfit.")
    parser.add_argument("--metadata", required=True, help="Path to metadata file.")
    parser.add_argument("--wd", required=True, help="Working directory.")
    parser.add_argument("--window", type=int, required=True, help="Window sizes.")
    parser.add_argument("--inputdir", required=True, help="Input data directory.")
    parser.add_argument("--experiment", required=True, help="Experiment name.")
    args = parser.parse_args()
    main(args.metadata, args.wd, args.inputdir, args.window, args.experiment)
