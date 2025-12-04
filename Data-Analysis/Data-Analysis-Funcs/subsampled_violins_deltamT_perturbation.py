# Import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, mannwhitneyu
from statsmodels.sandbox.stats.multicomp import multipletests

# violins subsample - 2x2 grid version
def compare_change_mT_clusters_violin_grid(plotting_dict_exp,
                                           experiment_name,
                                           sample1,
                                           sample2,
                                           genes_to_remove,
                                           outpath,
                                           label,
                                           label1,
                                           label2,
                                           cluster1_genes,
                                           cluster2_genes,
                                           color_perturbed,
                                           color_lab):
    '''
    * Takes: 
    - plotting_dict_exp: dictionary with LIET output information per sample
    - Experiment name
    - Sample 1 name
    - Sample 2 name
    - Genes to remove (for testing purposes)
    - Outpath
    - Label
    - Label for sample 1
    - Label for sample 2
    - Cluster 1 genes
    - Cluster 2 genes 
    - color_perturbed: GC-rich subset coloring
    - Color for label (correlates to perturbation)

    * Outputs:
    - 2x2 grid of violin plots comparing changes in mT values between subsets
    - Each subplot uses a different subsample of the T-rich set (n = GC-rich set size)
    - Statistical tests for each subsample
    '''    

    # Extract and merge data
    df1 = plotting_dict_exp[experiment_name][sample1][["Gene", "mT-adj"]].rename(columns={"mT-adj": f"mT_adj_{sample1}"})
    df2 = plotting_dict_exp[experiment_name][sample2][["Gene", "mT-adj"]].rename(columns={"mT-adj": f"mT_adj_{sample2}"})
    merged_df = pd.merge(df1, df2, on="Gene")
    merged_df = merged_df[~merged_df["Gene"].isin(genes_to_remove)]

    merged_df["Cluster"] = merged_df["Gene"].apply(
        lambda g: "1st Cluster" if g in cluster1_genes else ("2nd Cluster" if g in cluster2_genes else "Other")
    )

    merged_df = merged_df[merged_df["Cluster"] != "Other"]
    
    # change in mT--> measure of how much "run-on"
    merged_df["delta_mT"] = merged_df[f"mT_adj_{sample2}"] - merged_df[f"mT_adj_{sample1}"]

    df_c1 = merged_df[merged_df["Cluster"] == "1st Cluster"]
    df_c2 = merged_df[merged_df["Cluster"] == "2nd Cluster"]

    n_cluster2 = len(df_c2)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    all_stats = []

    # Create 4 different subsamples --> ensuring this trend is reproducible despite differences in n
    # each iteration there is a new random state--> changes the subset of T-rich that is tested
    for i in range(4):
        # sample from cluster 1 (T-rich) --> gc is much smaller
        df_c1_sample = df_c1.sample(n=n_cluster2, replace=False, random_state=i)
        
        ngenes = len(df_c1_sample)
        
        violin_df = pd.concat([
            df_c1_sample.assign(Cluster="Upstream T-rich Subset"),
            df_c2.assign(Cluster="GC-Rich Subset")
        ])
        violin_df["Comparison"] = "GC-Rich vs Upstream T-rich"
        violin_df["delta_mT"] = violin_df[f"mT_adj_{sample2}"] - violin_df[f"mT_adj_{sample1}"]

        # Perform statistical tests
        shapiro_pval_c1 = shapiro(df_c1_sample["delta_mT"]).pvalue
        shapiro_pval_c2 = shapiro(df_c2["delta_mT"]).pvalue
        
        mw_stat, mw_p = mannwhitneyu(
            df_c1_sample["delta_mT"],
            df_c2["delta_mT"],
            alternative='two-sided'
        )
        
        all_stats.append({
            'subsample': i + 1,
            'shapiro_c1': shapiro_pval_c1,
            'shapiro_c2': shapiro_pval_c2,
            'mw_pvalue': mw_p
        })

        ax = axes[i]
        sns.violinplot(
            data=violin_df,
            x="Comparison",
            y="delta_mT",
            hue="Cluster",
            split=True,
            palette={"Upstream T-rich Subset": "#c3c0c0", "GC-Rich Subset": color_perturbed},
            inner="quartile",
            linewidth=1.2,
            ax=ax
        )
        
        ax.set_title(f"Subsample {i+1}\nMann-Whitney p = {mw_p:.4f}", fontsize=14, color=color_lab)
        ax.set_ylabel(f"∆$\mu_T$", fontsize=14)
        ax.set_xlabel("")
        ax.tick_params(axis='both', labelsize=12)
        
        if i == 0:
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles[:2], labels[:2], fontsize=11)
        else:
            ax.get_legend().remove()

    fig.suptitle(f"$\mu_T$ {label2} - $\mu_T$ {label1} \n (subsampled n={ngenes})", 
                 fontsize=16, fontweight='bold', color=color_lab, y=0.995)

    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.show()

    # Print statistics summary
    print(f"\n{'='*60}")
    print(f"Statistical Summary for 4 T-rich Subsamples vs GC-Rich Set")
    print(f"{'='*60}")
    
    for stat in all_stats:
        print(f"\n--- Subsample {stat['subsample']} ---")
        print(f"Shapiro-Wilk p-value (T-rich, delta_mT): {stat['shapiro_c1']:.4f}")
        print(f"Shapiro-Wilk p-value (GC-rich, delta_mT): {stat['shapiro_c2']:.4f}")
        print(f"Mann-Whitney U test p-value: {stat['mw_pvalue']:.4f}")
    
    # Multiple testing correction across all 4 subsamples
    all_pvals = [stat['mw_pvalue'] for stat in all_stats]
    fdr_rejected, fdr_pvals_corrected, _, _ = multipletests(all_pvals, alpha=0.05, method='fdr_bh')
    bonf_rejected, bonf_pvals_corrected, _, _ = multipletests(all_pvals, alpha=0.05, method='bonferroni')
    
    print(f"\n{'='*60}")
    print(f"Multiple Testing Correction Across 4 Subsamples")
    print(f"{'='*60}")
    print(f"FDR-corrected: {np.sum(fdr_rejected)} out of 4 tests significant at alpha=0.05")
    print(f"Bonferroni-corrected: {np.sum(bonf_rejected)} out of 4 tests significant at alpha=0.05")
    print(f"Mean FDR-corrected p-value: {np.mean(fdr_pvals_corrected):.4f}")
    print(f"Mean Bonferroni-corrected p-value: {np.mean(bonf_pvals_corrected):.4f}")
    
    return all_stats
