# Import packages
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import hypergeom

# Hypergeometric test (GC-rich vs T-rich) test below 2 sigma (1-sided)
def perturbation_comparison_scatter_mT_cluster_hypergeom_below_2sigma(two_std,
                                                     plotting_dict_exp,
                                                     experiment_name,
                                                     sample1,
                                                     sample2,
                                                     genes_to_remove,
                                                     outpath, label,
                                                     label1,
                                                     label2,
                                                     cluster1_genes,
                                                     cluster2_genes,
                                                     color_perturbed,
                                                     Gene_to_Lab,
                                                     color_lab,
                                                     title, out):
    
    '''
    * Takes:
    - 2 standard deviation (from replicates)
    - plotting_dict_exp: dictionary with LIET output information per sample
    - Experiment name
    - Sample 1 name
    - Sample 2 name
    - Genes to remove (for testing purposes)
    - Outpath --> not used in current function, need to get rid of
    - Label
    - Label for sample 1
    - Label for sample 2
    - Cluster 1 genes
    - Cluster 2 genes 
    - color_perturbed: GC-rich subset coloring (confusing name, appologies) 
    - Gene to label
    - Color for label (correlates to perturbation)
    - Title for plot
    - Outpath --> saving outfile

    * Outputs: 
    - Bar plot for GC-rich and non GC-rich genes; % genes below 2 sigma
    - Hypergeometric pvalues plotted directly on the plots
    '''
    
    # extract and merge data
    df1 = plotting_dict_exp[experiment_name][sample1][["Gene", "mT-adj"]].rename(columns={"mT-adj": f"mT_adj_{sample1}"})
    df2 = plotting_dict_exp[experiment_name][sample2][["Gene", "mT-adj"]].rename(columns={"mT-adj": f"mT_adj_{sample2}"})
    merged_df = pd.merge(df1, df2, on="Gene")
    merged_df = merged_df[~merged_df["Gene"].isin(genes_to_remove)]

    # assign clusters
    merged_df["Cluster"] = merged_df["Gene"].apply(
        lambda g: "1st Cluster" if g in cluster1_genes else ("2nd Cluster" if g in cluster2_genes else "Other")
    )
    merged_df = merged_df[merged_df["Cluster"] != "Other"]
    
    delta = two_std
    merged_df["residual"] = merged_df[f"mT_adj_{sample2}"] - merged_df[f"mT_adj_{sample1}"]
    
    # genes below 2sigma (residual < -delta)
    # ** labeling is confusing here **
    merged_df["Above_2sigma"] = merged_df["residual"] < -delta
    
    # get gene sets
    gc_rich_genes_full = merged_df[merged_df["Cluster"] == "2nd Cluster"]
    t_rich_genes_full = merged_df[merged_df["Cluster"] == "1st Cluster"]
    
    # stats
    total_genes = len(merged_df)
    total_above_2sigma = len(merged_df[merged_df["Above_2sigma"] == True])
    
    # GC-rich subset analysis
    total_gc_rich = len(gc_rich_genes_full)
    gc_rich_above = len(gc_rich_genes_full[gc_rich_genes_full["Above_2sigma"] == True])
    
    # T-rich subset analysis
    total_t_rich = len(t_rich_genes_full)
    t_rich_above = len(t_rich_genes_full[t_rich_genes_full["Above_2sigma"] == True])
    
    # hypergeometric tests  --> survival function
    if gc_rich_above > 0:
        gc_rich_pval = hypergeom.sf(gc_rich_above - 1, total_genes, total_above_2sigma, total_gc_rich)
    else:
        gc_rich_pval = 1.0
        
    if t_rich_above > 0:
        t_rich_pval = hypergeom.sf(t_rich_above - 1, total_genes, total_above_2sigma, total_t_rich)
    else:
        t_rich_pval = 1.0    
    
    # percentages for plotting
    gc_rich_pct_above = (gc_rich_above / total_gc_rich) * 100
    t_rich_pct_above = (t_rich_above / total_t_rich) * 100

    # create bar plot
    fig, ax = plt.subplots(figsize=(3, 4))

    gc_text = f"p-val: {gc_rich_pval:.3}"  
    t_text = f"p-val: {t_rich_pval:.3}"

    # annotate on the plot
    ax.text(0, t_rich_pct_above + 1, t_text, ha='center', va='bottom',
            fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    ax.text(1, gc_rich_pct_above + 1, gc_text, ha='center', va='bottom', 
            fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

    groups = ['Upstream T-rich', 'GC-rich']
    percentages = [t_rich_pct_above, gc_rich_pct_above]
    colors = ['grey','#e57a7a']  

    bars = ax.bar(groups, percentages, color=colors, alpha=0.7, edgecolor='black', linewidth=1)

    ax.set_title(title, color = color_lab)
    ax.set_ylabel("% Genes Below 2$\sigma$ Band")
    ax.set_xlabel("")

    max_pct = max(percentages)
    ax.set_ylim(0,90)

    plt.tight_layout()
    plt.savefig(out)
    
    plt.show()

# Hypergeometric test (GC-rich vs T-rich) above two sigma (1-sided)
def perturbation_comparison_scatter_mT_cluster_hypergeom_above_2sigma(two_std,
                                                     plotting_dict_exp,
                                                     experiment_name,
                                                     sample1,
                                                     sample2,
                                                     genes_to_remove,
                                                     outpath, label,
                                                     label1,
                                                     label2,
                                                     cluster1_genes,
                                                     cluster2_genes,
                                                     color_perturbed,
                                                     Gene_to_Lab,
                                                     color_lab,
                                                     title, 
                                                     out):
    
    '''
    * Takes:
    - 2 standard deviation (from replicates)
    - plotting_dict_exp: dictionary with LIET output information per sample
    - Experiment name
    - Sample 1 name
    - Sample 2 name
    - Genes to remove (for testing purposes)
    - Outpath --> not used in current function, need to get rid of
    - Label
    - Label for sample 1
    - Label for sample 2
    - Cluster 1 genes
    - Cluster 2 genes 
    - color_perturbed: GC-rich subset coloring (confusing name, appologies) 
    - Gene to label
    - Color for label (correlates to perturbation)
    - Title for plot
    - Outpath --> saving outfile

    * Outputs: 
    - Bar plot for GC-rich and non GC-rich genes; % genes above 2 sigma
    - Hypergeometric pvalues plotted directly on the plots
    '''
    
    # extract and merge data
    df1 = plotting_dict_exp[experiment_name][sample1][["Gene", "mT-adj"]].rename(columns={"mT-adj": f"mT_adj_{sample1}"})
    df2 = plotting_dict_exp[experiment_name][sample2][["Gene", "mT-adj"]].rename(columns={"mT-adj": f"mT_adj_{sample2}"})
    merged_df = pd.merge(df1, df2, on="Gene")
    merged_df = merged_df[~merged_df["Gene"].isin(genes_to_remove)]

    # assign clusters
    merged_df["Cluster"] = merged_df["Gene"].apply(
        lambda g: "1st Cluster" if g in cluster1_genes else ("2nd Cluster" if g in cluster2_genes else "Other")
    )
    merged_df = merged_df[merged_df["Cluster"] != "Other"]
    
    delta = two_std
    merged_df["residual"] = merged_df[f"mT_adj_{sample2}"] - merged_df[f"mT_adj_{sample1}"]
    
    # look for genes above 2 sigma (residual > +delta)
    merged_df["Above_2sigma"] = merged_df["residual"] > delta
    
    gc_rich_genes_full = merged_df[merged_df["Cluster"] == "2nd Cluster"]
    t_rich_genes_full = merged_df[merged_df["Cluster"] == "1st Cluster"]
    
    # stats on the full dataset
    total_genes = len(merged_df)
    total_above_2sigma = len(merged_df[merged_df["Above_2sigma"] == True])
    
    # GC-rich subset 
    total_gc_rich = len(gc_rich_genes_full)
    gc_rich_above = len(gc_rich_genes_full[gc_rich_genes_full["Above_2sigma"] == True])
    
    # T-rich subset 
    total_t_rich = len(t_rich_genes_full)
    t_rich_above = len(t_rich_genes_full[t_rich_genes_full["Above_2sigma"] == True])
    
    # hypergeometric tests  --> survival function
    if gc_rich_above > 0:
        gc_rich_pval = hypergeom.sf(gc_rich_above - 1, total_genes, total_above_2sigma, total_gc_rich)
    else:
        gc_rich_pval = 1.0
        
    if t_rich_above > 0:
        t_rich_pval = hypergeom.sf(t_rich_above - 1, total_genes, total_above_2sigma, total_t_rich)
    else:
        t_rich_pval = 1.0    
    
    # Calc percentages for plotting
    gc_rich_pct_above = (gc_rich_above / total_gc_rich) * 100
    t_rich_pct_above = (t_rich_above / total_t_rich) * 100

    # bar plot
    fig, ax = plt.subplots(figsize=(3, 4))

    gc_text = f"p-val: {gc_rich_pval:.2}"  
    t_text = f"p-val: {t_rich_pval:.2}"

    # Annotate 
    ax.text(0, t_rich_pct_above + 1, t_text, ha='center', va='bottom',
            fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    ax.text(1, gc_rich_pct_above + 1, gc_text, ha='center', va='bottom', 
            fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

    groups = ['Upstream T-rich', 'GC-rich']
    percentages = [t_rich_pct_above, gc_rich_pct_above]
    colors = ['grey','#e57a7a']  

    bars = ax.bar(groups, percentages, color=colors, alpha=0.7, edgecolor='black', linewidth=1)

    ax.set_title(title, color = color_lab)
    ax.set_ylabel("% Genes Above 2$\sigma$ Band")
    ax.set_xlabel("")

    max_pct = max(percentages)
    ax.set_ylim(0, 40)

    plt.tight_layout()
    plt.savefig(out)
    
    plt.show()

# Hypergeometric test (GC-rich vs T-rich) above outside sigma (2-sided)
def perturbation_comparison_scatter_mT_cluster_hypergeom_outside_2sigma(two_std,
                                                     plotting_dict_exp,
                                                     experiment_name,
                                                     sample1,
                                                     sample2,
                                                     genes_to_remove,
                                                     outpath, label,
                                                     label1,
                                                     label2,
                                                     cluster1_genes,
                                                     cluster2_genes,
                                                     color_perturbed,
                                                     Gene_to_Lab,
                                                     color_lab,
                                                     title, 
                                                     out):
    
    '''
    * Takes:
    - 2 standard deviation (from replicates)
    - plotting_dict_exp: dictionary with LIET output information per sample
    - Experiment name
    - Sample 1 name
    - Sample 2 name
    - Genes to remove (for testing purposes)
    - Outpath --> not used in current function, need to get rid of
    - Label
    - Label for sample 1
    - Label for sample 2
    - Cluster 1 genes
    - Cluster 2 genes 
    - color_perturbed: GC-rich subset coloring (confusing name, appologies) 
    - Gene to label
    - Color for label (correlates to perturbation)
    - Title for plot
    - Outpath --> saving outfile

    * Outputs: 
    - Bar plot for GC-rich and non GC-rich genes; % genes below 2 sigma
    - Hypergeometric pvalues plotted directly on the plots
    '''
    
    # extract and merge data
    df1 = plotting_dict_exp[experiment_name][sample1][["Gene", "mT-adj"]].rename(columns={"mT-adj": f"mT_adj_{sample1}"})
    df2 = plotting_dict_exp[experiment_name][sample2][["Gene", "mT-adj"]].rename(columns={"mT-adj": f"mT_adj_{sample2}"})
    merged_df = pd.merge(df1, df2, on="Gene")
    merged_df = merged_df[~merged_df["Gene"].isin(genes_to_remove)]

    # assign clusters
    merged_df["Cluster"] = merged_df["Gene"].apply(
        lambda g: "1st Cluster" if g in cluster1_genes else ("2nd Cluster" if g in cluster2_genes else "Other")
    )
    merged_df = merged_df[merged_df["Cluster"] != "Other"]
    
    delta = two_std
    merged_df["residual"] = merged_df[f"mT_adj_{sample2}"] - merged_df[f"mT_adj_{sample1}"]
    
    # look for genes above OR below 2 sigma (|residual| > +delta)
    # ** again, labeling is confusing here **
    merged_df["Above_2sigma"] = merged_df["residual"].abs() > delta
    
    gc_rich_genes_full = merged_df[merged_df["Cluster"] == "2nd Cluster"]
    t_rich_genes_full = merged_df[merged_df["Cluster"] == "1st Cluster"]
    
    # Calculate stats
    total_genes = len(merged_df)
    total_above_2sigma = len(merged_df[merged_df["Above_2sigma"] == True])
    
    # GC-rich subset 
    total_gc_rich = len(gc_rich_genes_full)
    gc_rich_above = len(gc_rich_genes_full[gc_rich_genes_full["Above_2sigma"] == True])
    
    # T-rich subset 
    total_t_rich = len(t_rich_genes_full)
    t_rich_above = len(t_rich_genes_full[t_rich_genes_full["Above_2sigma"] == True])
    
    # hypergeometric tests  --> survival function
    if gc_rich_above > 0:
        gc_rich_pval = hypergeom.sf(gc_rich_above - 1, total_genes, total_above_2sigma, total_gc_rich)
    else:
        gc_rich_pval = 1.0
        
    if t_rich_above > 0:
        t_rich_pval = hypergeom.sf(t_rich_above - 1, total_genes, total_above_2sigma, total_t_rich)
    else:
        t_rich_pval = 1.0    
    
    # Calculate percentages for plotting
    gc_rich_pct_above = (gc_rich_above / total_gc_rich) * 100
    t_rich_pct_above = (t_rich_above / total_t_rich) * 100

    # bar plot
    fig, ax = plt.subplots(figsize=(3, 4))

    gc_text = f"p-val: {gc_rich_pval:.3}"  
    t_text = f"p-val: {t_rich_pval:.3}"

    # Annotate 
    ax.text(0, t_rich_pct_above, t_text, ha='center', va='bottom',
            fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    ax.text(1, gc_rich_pct_above, gc_text, ha='center', va='bottom', 
            fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

    groups = ['Upstream T-rich', 'GC-rich']
    percentages = [t_rich_pct_above, gc_rich_pct_above]
    colors = ['grey','#e57a7a']  

    bars = ax.bar(groups, percentages, color=colors, alpha=0.7, edgecolor='black', linewidth=1)

    ax.set_title(title, color=color_lab)
    ax.set_ylabel("% Genes Outside 2$\sigma$ Band")
    ax.set_xlabel("")
    
    ax.set_ylim(0,40)

    max_pct = max(percentages)
    
    plt.tight_layout()
    plt.savefig(out)
    
    plt.show()
