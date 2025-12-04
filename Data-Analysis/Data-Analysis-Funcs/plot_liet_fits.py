# Import
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

sys.path.append('/Users/geba9152/LIET/liet/')
import rnap_lib_data_proc as dp
import rnap_lib_data_sim as ds
import rnap_lib_plotting as rp
from liet_res_class import FitParse, fitparse_intersect
from rnap_lib_data_sim import elongation_pdf_alt, termination_pdf
from rnap_lib_fitting_results import hist_generator, pdf_generator, results_loader

# plot fits
def plot_fits(workdir, 
              gene_list, 
              outdir, 
              samples, 
              ann, 
              experiment_name):
    """
    * Takes:
    - Directory of LIET run
    - Gene list
    - Out directory
    - Samples
    - Annotation
    - Experiment name
    
    * Outputs: 
    - LIET fits (model + data)
    """
    ann = pd.read_csv(ann, sep="\t", header=None)
    ann.columns = ["chr", "start", "stop", "gene", "length", "strand"]

    # Loop through each sample
    for sample in samples:
        config_file = f"{workdir}/config/{sample}.liet.config"
        results_file = f"{workdir}/results/{experiment_name}/{sample}.liet"
        log_file = f"{workdir}/results/{experiment_name}/{sample}.liet.log"
         
        gene_results_samp = results_loader(gene_list, config=config_file, result=results_file, log=log_file)

        for _, row in ann.iterrows():
            gene_id = row['gene']
            strand = row.get('strand')  
            start = row.get('start')      
            stop = row.get('stop')       

            if gene_id in gene_results_samp:
                x_value = (stop - start) if strand == "+" else -(stop - start)

                fig, ax = plt.subplots(figsize=[6, 2])

                xvals, preads, nreads, strand, model_params = gene_results_samp[gene_id]

                # Call plotting function
                rp.LIET_ax(ax, **model_params, strand=strand, xvals=xvals, data=(preads, nreads), nbins=1000)
                
                plt.axvline(x=x_value, color='k', linewidth=1, ymin=0.2, ymax=0.3, label = "TCS")
                
#                 plt.ylim([-0.00004, 0.00015])
                plt.ylim([-0.00004, 0.0001])

                # Replace 0 x-tick with 'TSS' and add 'TCS' label at x_value
                ticks = ax.get_xticks()
                tick_labels = []
                
                for t in ticks:
                    if t == 0:
                        tick_labels.append('TSS')
                    elif t == x_value:
                        tick_labels.append('TCS')
                    else:
                        tick_labels.append(f'{int(t)}')

                ax.set_xticks(ticks)
                ax.set_xticklabels(tick_labels, fontsize=12)
                
                #plt.xlim([-3000, 30000])
                
                ax.tick_params(axis='y', labelsize=12) 
                ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
                
                plt.ylabel('Probability/ \n Frequency', fontsize=14)
                plt.xlabel(f'Genomic Coordinates relative to {gene_id} TSS (bp)', fontsize=14)

                output_dir = f'{outdir}{sample}'
                os.makedirs(output_dir, exist_ok=True)
                
                plt.legend(fontsize = 14)
                
                plt.savefig(f'{output_dir}/{sample}{gene_id}-fit.svg')
                
                plt.show()

                plt.close()
