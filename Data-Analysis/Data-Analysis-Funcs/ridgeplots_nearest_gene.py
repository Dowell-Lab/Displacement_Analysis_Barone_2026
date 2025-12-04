# Import
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Used bedtools closest to calculate distance to nearest downstream gene (https://bedtools.readthedocs.io/en/latest/content/tools/closest.html)
# Function below reads output
def read_bedtools_output(file_path):
    '''
    * Takes: Bedtools closest output
    
    * Ouputs: Reads dataframe (bedtools closest output)
    '''
    
    columns = ['cluster_chr', 'cluster_start', 'cluster_end', 'cluster_name', 'cluster_score', 'cluster_strand',
               'gene_chr', 'gene_start', 'gene_end', 'gene_name', 'gene_score', 'gene_strand', 'distance']
    
    return pd.read_csv(file_path, sep='\t', header=None, names=columns)

def make_kde_nearest_ds_gene_keep_outliers_plot(location, 
                                                resdir, 
                                                samp, 
                                                outdir):
    '''
    * Takes: 
    - Distance location (distance to nearest downstream gene, upstream gene, etc...)
    - Results directory
    - Sample
    - Out directory
    
    * Ouputs: 
    - Denisty plot of distances to nearest (downstream/upstream/etc..) gene 
    
    '''    
    
    # labels columns for me
    t_rich = read_bedtools_output(f"{resdir}/T_clust_{samp}{location}.txt")    
    gc_rich = read_bedtools_output(f"{resdir}/GC_clust_{samp}{location}.txt")
    
    # distances
    t_distances = t_rich['distance'].abs()
    gc_distances = gc_rich['distance'].abs()
        
    print(f"Upstream T-rich clusters: {len(t_distances)}")
    print(f"GC-rich clusters: {len(gc_distances)}")
    
    t_color = '#817f7fff'    
    gc_color = '#e57a7aff'   
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 4))
    
    # 1st-- equal sample sizes (subsample T-rich)
    min_n = min(len(t_distances), len(gc_distances))
    
    # trying out different random states
#    t_subsample = t_distances.sample(n=min_n, random_state=42)
    t_subsample = t_distances.sample(n=min_n, random_state=30)
    
    sns.kdeplot(data=t_subsample, color=t_color, label=f'T-rich (n={min_n})', ax=ax1, linewidth=2, fill=True)
    sns.kdeplot(data=gc_distances, color=gc_color, label=f'GC-rich (n={len(gc_distances)})', ax=ax1, linewidth=2, fill=True)

    ax1.grid(False)
    ax1.legend(fontsize=14)
    ax1.tick_params(axis='both', which='major', labelsize=14)
    ax1.legend(fontsize=20)
    
    ax2.grid(False)
    ax2.tick_params(axis='both', which='major', labelsize=14)

    if location == ".nearest_downstream_genes_pergene" or location == '-Consensus.nearest_downstream_genes_pergene':
    
        ax1.set_xlabel('Distance to Nearest \n Downstream Gene (bp)', fontsize = "15")
        ax1.set_ylabel('Density', fontsize = "15")
        ax1.set_title('Distance to Nearest Downstream Genes\n(Equal Sample Sizes)', fontsize = "15")

    if location == ".nearest_any_direction_genes_pergene" or location == "-Consensus.nearest_any_direction_genes_pergene":

        ax1.set_xlabel('Distance to Nearest Gene (bp)', fontsize = "15")
        ax1.set_ylabel('Density', fontsize = "15")
        ax1.set_title('Distance to Nearest Genes\n(Equal Sample Sizes)', fontsize = "15")

    if location == ".nearest_upstream_genes_pergene" or location == "-Consensus.nearest_upstream_genes_pergene":

        ax1.set_xlabel('Distance to Nearest \n Upstream Gene (bp)', fontsize = "15")
        ax1.set_ylabel('Density', fontsize = "15")
        ax1.set_title('Distance to Nearest Upstream Genes\n(Equal Sample Sizes)', fontsize = "15")
    
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2nd --don't subsample
    sns.kdeplot(data=t_distances, color=t_color, label=f'Upstream T-rich (n={len(t_distances)})', ax=ax2, linewidth=2, fill=True)
    sns.kdeplot(data=gc_distances, color=gc_color, label=f'GC-rich (n={len(gc_distances)})', ax=ax2, linewidth=2, fill=True)
    
    if location == ".nearest_downstream_genes" or location == '-Consensus.nearest_downstream_genes':
        
        ax2.set_xlabel('Distance to Nearest \n Downstream Gene (bp)', fontsize = "15")
        ax2.set_ylabel('Density', fontsize = "15")
        ax2.set_title('Distance to Nearest Downstream Genes\n(Full Datasets)', fontsize = "15")

    if location == ".nearest_any_direction_genes" or location == "-Consensus.nearest_any_direction_genes":
    
        ax2.set_xlabel('Distance to Nearest Gene (bp)', fontsize = "15")
        ax2.set_ylabel('Density', fontsize = "15")
        ax2.set_title('Distance to Nearest Genes\n(Full Datasets)', fontsize = "15")

    if location == ".nearest_upstream_genes" or location == "-Consensus.nearest_upstream_genes":

        ax2.set_xlabel('Distance to Nearest \n Upstream Gene (bp)', fontsize = "15")
        ax2.set_ylabel('Density', fontsize = "15")
        ax2.set_title('Distance to Nearest Upstream Genes\n(Full Datasets)', fontsize = "15")
    
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0,400000)
    ax1.set_xlim(0,400000)
    
    plt.tight_layout()
    
    plt.savefig(f'{outdir}/{samp}{location}_downstream_distance_kde_plots.svg', dpi=300, bbox_inches='tight')
    plt.show()
    

