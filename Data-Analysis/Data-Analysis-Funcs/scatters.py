import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.stats import linregress, shapiro, mannwhitneyu, binom
from statsmodels.stats.multitest import multipletests


############################################################################################
########################## Helper functions used in most scatters ##########################
############################################################################################

def _prepare_merged_df(plotting_dict_exp, experiment_name, sample1, sample2, genes_to_remove):
    '''
    * Takes: 
    - Plotting dictionary with prior values
    - Experiment name
    - Samples to plot
    - Genes to be removed
    
    * Outputs:
    - Dataframe with mT-adj values in two specific samples
    
    '''
    df1 = plotting_dict_exp[experiment_name][sample1][["Gene", "mT-adj"]].rename(
        columns={"mT-adj": f"mT_adj_{sample1}"}
    )
    df2 = plotting_dict_exp[experiment_name][sample2][["Gene", "mT-adj"]].rename(
        columns={"mT-adj": f"mT_adj_{sample2}"}
    )
    merged_df = pd.merge(df1, df2, on="Gene")
    merged_df = merged_df[~merged_df["Gene"].isin(genes_to_remove)]
    return merged_df
 
def _axis_range(merged_df, sample1, sample2, n=500):
    '''
    * Takes: 
    - merged_df (dataframe with mT-adj values in two specific samples)
    - Samples for plotting
    
    * Outputs:
    - Range of values for samples for plotting

    '''
    combined_min = min(merged_df[f"mT_adj_{sample1}"].min(), merged_df[f"mT_adj_{sample2}"].min())
    combined_max = max(merged_df[f"mT_adj_{sample1}"].max(), merged_df[f"mT_adj_{sample2}"].max())
    return np.linspace(combined_min, combined_max, n)
 
def _plot_band_and_diagonal(ax, x_vals, delta, band_label=None, band_color="darkgrey", band_alpha=0.5):
    '''
    * Takes: 
    - ax: plot
    - x_vals: range of values 
    - Delta: 2sigma value for band
    
    * Outputs:
    - Plots line with shaded delta
    
    '''
    if band_label is None:
        band_label = r"±2$\sigma$"
    ax.fill_between(x_vals, x_vals - delta, x_vals + delta, color=band_color, alpha=band_alpha, label=band_label)
    ax.plot(x_vals, x_vals, linestyle="--", color="red", linewidth=1, label="1:1 line")

def _annotate_gene(ax, df, gene, sample1, sample2, offset=500):
    '''
    * Takes: 
    - ax: plot
    - Gene to label
    - Two samples for plotting
    
    * Outputs:
    - Annotates gene on plot 
    
    '''
    if not gene:
        return
    
    label_gene = df[df["Gene"] == gene].copy()
    
    if label_gene.empty:
        return
    
    label_gene["Gene"] = label_gene["Gene"].str.split("|").str[0]
    
    for _, row in label_gene.iterrows():
        ax.annotate(
            row["Gene"],
            xy=(row[f"mT_adj_{sample1}"], row[f"mT_adj_{sample2}"]),
            xytext=(row[f"mT_adj_{sample1}"] + offset, row[f"mT_adj_{sample2}"] + offset),
            fontsize=12,
            fontweight="bold",
            color="black",
            arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
        )

def _format_a3e_ticks(ax, fontsize=14, xrotation=20):
    '''
    * Takes: 
    - ax: plot
    
    * Outputs:
    - Set ticks, replace 0 with the A3E
    
    '''
    x_ticks = ax.get_xticks()
    y_ticks = ax.get_yticks()
    ax.set_xticklabels(["A3E" if x == 0 else f"{int(x):,}" for x in x_ticks], fontsize=fontsize, rotation=xrotation)
    ax.set_yticklabels(["A3E" if y == 0 else f"{int(y):,}" for y in y_ticks], fontsize=fontsize)
 
 
def _add_stats_box(ax, text, xy=(0.02, 0.98), fontsize=12, va="top"):
    '''
    * Takes: 
    - ax: plot
    - Text for stats box
    
    * Outputs:
    - Annotates stats on plot 
    
    '''
    ax.text(
        xy[0], xy[1], text,
        transform=ax.transAxes,
        fontsize=fontsize,
        verticalalignment=va,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="black"),
    )

def _binomial_pval(count_hit, total, expected_prob):
    '''
    * Takes:
    - count_hit: number of genes flagged (e.g. genes above/below/outside
      the 2-sigma band)
    - total: total number of genes tested
    - expected_prob: null-hypothesis probability of a gene falling in the
      flagged region under a normal distribution of residuals
      (e.g. 0.025 for one-sided above/below, 0.05 for two-sided outside)

    * Outputs:
    - pval: one-sided p-value, P(X >= count_hit), from the binomial
      survival function
    
    '''
    return binom.sf(count_hit - 1, total, expected_prob)
 
def _finish_and_save(ax, xlimylim, outpath, legend_loc="lower right", legend_fontsize=12):
    '''
    * Takes: 
    - Plot
    - xlim/ylims
    - Output path

    
    * Outputs:
    - Saves output plot
    
    '''
    ax.set_xlim(xlimylim)
    ax.set_ylim(xlimylim)
    _format_a3e_ticks(ax)
    ax.legend(fontsize=legend_fontsize, loc=legend_loc)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.show()
 
 ##################################################################
 ################## Scatters ######################################
##################################################################

# Comparison between two samples, no clustering or stat testing
# Comparing mT (mT-A3E) values between two samples
# Looks at % of genes outside of two sigma (2-sided)
def perturbation_comparison_twosided_no_clust_scatter_mT(two_std, 
                                       plotting_dict_exp,
                                       experiment_name,
                                       sample1,
                                       sample2,
                                       genes_to_remove,
                                       outpath,
                                       label1,
                                       label2,
                                       dot_color,
                                       Gene_to_Lab,
                                       color_lab,
                                       title,
                                       xlimylim):
    '''
    * Takes: 
    - Two standard deviations from the 1:1 line in control replicates (helps to determine what falls outside of "normal")
    - plotting_dict_exp: plotting dictionary from load_data functions, contains sample + prior information
    - Experiment name
    - Sample 1 for comparison
    - Sample 2 for comparison
    - List of genes to remove (for testing purposes)
    - Outfile path
    - Label 1 (goes with sample 1)
    - Label 2 (goes with sample 2)
    - Color of dots on scatter
    - Gene to label if desired
    - Color of the plot title
    - Title of plot
    - xlim and ylim

    * Outputs:
    - Figure comparing mT values between samples (2-sided, both directions)

    '''
    
    merged_df = _prepare_merged_df(plotting_dict_exp, experiment_name, sample1, sample2, genes_to_remove)
    delta = two_std
    merged_df["residual"] = merged_df[f"mT_adj_{sample2}"] - merged_df[f"mT_adj_{sample1}"]
    merged_df["Outside_Band"] = merged_df["residual"].abs() > delta
 
    total_genes = len(merged_df)
    genes_outside_band = merged_df["Outside_Band"].sum()
    pct_outside_band = (genes_outside_band / total_genes) * 100
 
    fig, ax = plt.subplots(figsize=(6, 6))
    x_vals = _axis_range(merged_df, sample1, sample2)
    _plot_band_and_diagonal(ax, x_vals, delta)
 
    sns.scatterplot(
        data=merged_df, x=f"mT_adj_{sample1}", y=f"mT_adj_{sample2}",
        color=dot_color, alpha=0.6, s=60, edgecolor="black", linewidth=0.5, ax=ax
    )
 
    ax.set_xlabel(f"{label1} |$\\mu_T$-A3E|", fontsize=16)
    ax.set_ylabel(f"{label2} |$\\mu_T$-A3E|", fontsize=16)
    ax.set_title(f"{title}", fontsize=17, color=color_lab)
 
    _annotate_gene(ax, merged_df, Gene_to_Lab, sample1, sample2)
 
    stats_text = f"Total genes: {total_genes:,}\nGenes Outside 2$\\sigma$: {genes_outside_band} ({pct_outside_band:.1f}%)"
    _add_stats_box(ax, stats_text)
 
    _finish_and_save(ax, xlimylim, outpath)
    
    
# Comparison between two samples, no clustering or stat testing
# Comparing mT (mT-A3E) values between two samples
# Looks at % of genes above of two sigma (2-sided)
def perturbation_comparison_above_no_clust_scatter_mT(two_std, 
                                       plotting_dict_exp,
                                       experiment_name,
                                       sample1,
                                       sample2,
                                       genes_to_remove,
                                       outpath,
                                       label1,
                                       label2,
                                       dot_color,
                                       Gene_to_Lab,
                                       color_lab,
                                       title,
                                       xlimylim):
    '''
    * Takes: 
    - Two standard deviations from the 1:1 line in control replicates (helps to determine what falls outside of "normal")
    - plotting_dict_exp: plotting dictionary from load_data functions, contains sample + prior information
    - Experiment name
    - Sample 1 for comparison
    - Sample 2 for comparison
    - List of genes to remove (for testing purposes)
    - Outfile path
    - Label 1 (goes with sample 1)
    - Label 2 (goes with sample 2)
    - Color of dots on scatter
    - Gene to label if desired
    - Color of the plot title
    - Title of plot
    - x lim and y lim

    * Outputs:
    - Figure comparing mT values between samples (1-sided above)

    '''
    merged_df = _prepare_merged_df(plotting_dict_exp, experiment_name, sample1, sample2, genes_to_remove)
    delta = two_std
    merged_df["residual"] = merged_df[f"mT_adj_{sample2}"] - merged_df[f"mT_adj_{sample1}"]
    merged_df["Outside_Band"] = merged_df["residual"] > delta
 
    total_genes = len(merged_df)
    genes_outside_band = merged_df["Outside_Band"].sum()
    pct_outside_band = (genes_outside_band / total_genes) * 100
 
    fig, ax = plt.subplots(figsize=(6, 6))
    x_vals = _axis_range(merged_df, sample1, sample2)
    _plot_band_and_diagonal(ax, x_vals, delta)
 
    sns.scatterplot(
        data=merged_df, x=f"mT_adj_{sample1}", y=f"mT_adj_{sample2}",
        color=dot_color, alpha=0.6, s=60, edgecolor="black", linewidth=0.5, ax=ax
    )
 
    ax.set_xlabel(f"{label1} |$\\mu_T$-A3E|", fontsize=16)
    ax.set_ylabel(f"{label2} |$\\mu_T$-A3E|", fontsize=16)
    ax.set_title(f"{title}", fontsize=17, color=color_lab)
 
    _annotate_gene(ax, merged_df, Gene_to_Lab, sample1, sample2)
 
    stats_text = f"Total genes: {total_genes:,}\nGenes > 2$\\sigma$: {genes_outside_band} ({pct_outside_band:.1f}%)"
    _add_stats_box(ax, stats_text)
 
    _finish_and_save(ax, xlimylim, outpath)
    

    
# Comparing mT between cell types, Figure 1
def mT_cross_ct_comparison_singluar_comparison(
    merged_df,
    sample1,
    sample2,
    outpath,
    label_dict,
    xlimylim
):
    '''
    * Takes: 
    - merged_df: dataframe containing mT-A3E information on two samples
    - Sample 1 name
    - Sample 2 name
    - Outpath
    - Label dictionary specifying labels
    - x lim and y lim

    * Outputs:
    - Figure comparing mT values between samples

    '''
    fig, ax = plt.subplots(figsize=(4, 4))
    x_vals = _axis_range(merged_df, sample1, sample2)
 
    residuals = merged_df[f"mT_adj_{sample1}"] - merged_df[f"mT_adj_{sample2}"]
    sigma_calculated = 2 * np.std(residuals)
    sigma_int = int(sigma_calculated)
 
    _plot_band_and_diagonal(ax, x_vals, sigma_calculated, band_label=f"±2$\\sigma$ ({sigma_int})", band_color="lightgrey", band_alpha=0.4)
 
    lab1 = label_dict.get(sample1)
    lab2 = label_dict.get(sample2)
 
    sns.scatterplot(
        data=merged_df, x=f"mT_adj_{sample1}", y=f"mT_adj_{sample2}",
        color="#e3af6e", alpha=0.6, s=60, edgecolor=None, ax=ax
    )
 
    ax.set_xlabel(f"{lab1} |$\\mu_T$-A3E|", fontsize=14)
    ax.set_ylabel(f"{lab2} |$\\mu_T$-A3E|", fontsize=14)
 
    _finish_and_save(ax, xlimylim, outpath, legend_loc="upper left", legend_fontsize=11)
    _format_a3e_ticks(ax, xrotation=45)
    plt.close()
    
    
 # Comparing mT between replicates, Figure 1
def replicate_comparison_scatter_mT_fig1(plotting_dict_exp, 
                                         experiment_name, 
                                         sample1, 
                                         sample2, 
                                         genes_to_remove,
                                         outpath, 
                                         label,
                                         xlimylim):
    '''
    * Takes: 
    - plotting_dict_exp: dictionary with LIET output information per sample
    - Experiment name
    - Sample 1 name
    - Sample 2 name
    - Genes to remove (for testing purposes)
    - Outpath
    - Label
    - xlim and ylim

    * Outputs:
    - Figure comparing mT values between samples
    - 2stdev from 1:1 line

    '''
    merged_df = _prepare_merged_df(plotting_dict_exp, experiment_name, sample1, sample2, genes_to_remove)
    merged_df["residual"] = merged_df[f"mT_adj_{sample1}"] - merged_df[f"mT_adj_{sample2}"]
    two_std = 2 * np.std(merged_df["residual"])
    print(f"±2 standard deviations from the 1:1 line: {two_std:.2f}")
 
    fig, ax = plt.subplots(figsize=(4, 4))
    x_vals = _axis_range(merged_df, sample1, sample2)
    std_int = int(two_std)
    _plot_band_and_diagonal(ax, x_vals, two_std, band_label=f"±2$\\sigma$ ({std_int})", band_color="lightgrey", band_alpha=0.4)
 
    sns.scatterplot(
        data=merged_df, x=f"mT_adj_{sample1}", y=f"mT_adj_{sample2}",
        color="#e3af6e", alpha=0.6, s=60, edgecolor=None, ax=ax
    )
 
    ax.set_xlabel(f"{label} Rep1 |$\\mu_T$-A3E|", fontsize=14)
    ax.set_ylabel(f"{label} Rep2 |$\\mu_T$-A3E|", fontsize=14)
 
    _finish_and_save(ax, xlimylim, outpath, legend_loc="upper left", legend_fontsize=11)
    _format_a3e_ticks(ax, xrotation=45)
    plt.close()
 
    return two_std

# Replicate comparison (perturbation analysis, Figures 4)
# No coloring on subsets 
def replicate_comparison_no_clust_scatter_mT(plotting_dict_exp, 
                                             experiment_name, 
                                             sample1, 
                                             sample2, 
                                             genes_to_remove, 
                                             outpath, 
                                             label, 
                                             color_perturbed):
    '''
    * Takes: 
    - plotting_dict_exp: dictionary with LIET output information per sample
    - Experiment name
    - Sample 1 name
    - Sample 2 name
    - Genes to remove (for testing purposes)
    - Outpath
    - Label
    - Color for label (correlates to perturbation)

    * Outputs:
    - Figure comparing mT values between samples
    - 2stdev from 1:1 line 
    - Genes above 2stdev

    '''
    merged_df = _prepare_merged_df(plotting_dict_exp, experiment_name, sample1, sample2, genes_to_remove)
    merged_df["residual"] = merged_df[f"mT_adj_{sample2}"] - merged_df[f"mT_adj_{sample1}"]
    two_std = 2 * np.std(merged_df["residual"])
    twosigma_int = int(two_std)
    print(f"±2 standard deviations from the 1:1 line: {two_std:.2f}")
 
    slope, intercept, r_value, p_value, std_err = linregress(merged_df[f"mT_adj_{sample1}"], merged_df[f"mT_adj_{sample2}"])
    r_squared = r_value ** 2
 
    fig, ax = plt.subplots(figsize=(6, 6))
    x_vals = _axis_range(merged_df, sample1, sample2)
    _plot_band_and_diagonal(ax, x_vals, two_std, band_label=f"±2$\\sigma$ ({twosigma_int})", band_color="lightgrey", band_alpha=0.4)
 
    merged_df["Outside_Band"] = merged_df["residual"] > two_std
    n_genes_outside = int(merged_df["Outside_Band"].sum())
    totalgenes = len(merged_df)
    print(n_genes_outside)
    print(totalgenes)
    genes_outside_band = merged_df.loc[merged_df["Outside_Band"], "Gene"].tolist()
 
    ax.plot(x_vals, intercept + slope * x_vals, color="black", linewidth=1.5, label=f"Linear regression \n R² = {r_squared:.2f}")
 
    sns.scatterplot(
        data=merged_df, x=f"mT_adj_{sample1}", y=f"mT_adj_{sample2}",
        color=color_perturbed, alpha=0.6, s=60, edgecolor=None, ax=ax
    )
 
    ax.set_xlabel(f"{label} Rep1 A3E-$\\mu_T$", fontsize=14)
    ax.set_ylabel(f"{label} Rep2 A3E-$\\mu_T$", fontsize=14)
    ax.legend(loc="upper left", fontsize=11)
    _format_a3e_ticks(ax, xrotation=45)
    plt.tight_layout()
    plt.show()
    plt.close()
 
    return two_std, genes_outside_band

# Replicate comparison for perturbation analysis-- coloring on subsets
def replicate_comparison_scatter_mT(plotting_dict_exp,
                                    experiment_name,
                                    sample1,
                                    sample2,
                                    genes_to_remove,
                                    outpath,
                                    label,
                                    cluster1_genes, 
                                    cluster2_genes, 
                                    color_perturbed):
    
    '''
    * Takes: 
    - plotting_dict_exp: dictionary with LIET output information per sample
    - Experiment name
    - Sample 1 name
    - Sample 2 name
    - Genes to remove (for testing purposes)
    - Outpath
    - Label
    - Color for label (correlates to perturbation)
    - Cluster 1 genes
    - Cluster 2 genes
    - color_perturbed: GC-rich subset coloring (confusing name, appologies) 

    * Outputs:
    - Figure comparing mT values between samples
    - 2stdev from 1:1 line 
    - Genes above 2stdev
    
    '''
    
    merged_df = _prepare_merged_df(plotting_dict_exp, experiment_name, sample1, sample2, genes_to_remove)
 
    merged_df["Cluster"] = merged_df["Gene"].apply(
        lambda g: "1st Cluster" if g in cluster1_genes else ("2nd Cluster" if g in cluster2_genes else "Other")
    )
    merged_df = merged_df[merged_df["Cluster"] != "Other"]
 
    merged_df["residual"] = merged_df[f"mT_adj_{sample1}"] - merged_df[f"mT_adj_{sample2}"]
    two_std = 2 * np.std(merged_df["residual"])
    print(f"±2 standard deviations from the 1:1 line: {two_std:.2f}")
 
    slope, intercept, r_value, p_value, std_err = linregress(merged_df[f"mT_adj_{sample1}"], merged_df[f"mT_adj_{sample2}"])
    r_squared = r_value ** 2
 
    fig, ax = plt.subplots(figsize=(6, 6))
    x_vals = _axis_range(merged_df, sample1, sample2)
    _plot_band_and_diagonal(ax, x_vals, two_std, band_label=r"±2$\sigma_T$ region", band_color="lightgrey", band_alpha=0.4)
 
    merged_df["Outside_Band"] = merged_df["residual"].abs() > two_std
    genes_outside_band = merged_df.loc[merged_df["Outside_Band"], "Gene"].tolist()
 
    ax.plot(x_vals, intercept + slope * x_vals, color="black", linewidth=1.5, label=f"Linear regression \n R² = {r_squared:.2f}")
 
    df_c1 = merged_df[merged_df["Cluster"] == "1st Cluster"]
    df_c2 = merged_df[merged_df["Cluster"] == "2nd Cluster"]
 
    sns.scatterplot(data=df_c1, x=f"mT_adj_{sample1}", y=f"mT_adj_{sample2}", color="#c3c0c0", alpha=0.4, s=60, label="Non GC-Rich Subset", edgecolor=None, ax=ax)
    sns.scatterplot(data=df_c2, x=f"mT_adj_{sample1}", y=f"mT_adj_{sample2}", color=color_perturbed, alpha=0.6, s=60, label="GC-Rich Subset", edgecolor=None, ax=ax)
 
    ax.set_xlabel(f"{label} Rep1 A3E-$\\mu_T$", fontsize=14)
    ax.set_ylabel(f"{label} Rep2 A3E-$\\mu_T$", fontsize=14)
    ax.legend(loc="upper left", fontsize=11)
    _format_a3e_ticks(ax, xrotation=45)
    plt.tight_layout()
    plt.show()
    plt.close()
 
    return two_std, genes_outside_band


# Perturbation scatters & subsampled violin plots; meta-samples
# Figure 5 --> color on GC-rich vs non GC-rich subset
def perturbation_comparison_scatter_mT_cluster_stats(two_std, 
                                                     plotting_dict_exp,
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
                                                     Gene_to_Lab,
                                                     color_lab,
                                                     title,
                                                     xlimylim):
    '''
    * Takes: 
    - 2 standard deviation (from replicates)
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
    - color_perturbed: GC-rich subset coloring (confusing name, appologies) 
    - Gene to label
    - Color for label (correlates to perturbation)
    - Title for plot
    - x lim and y lim

    * Outputs:
    - Scatter plot comparing mT values between samples; colored on GC-rich genes
    - Violin plot comparing changes in mT values between subsets (subsampled so n = the same between subsets, 1st iteration)
    - 2stdev from 1:1 line 
    - genes_outside_band_gc_rich: GC-rich genes above 2stdev 

    '''   
    merged_df = _prepare_merged_df(plotting_dict_exp, experiment_name, sample1, sample2, genes_to_remove)
 
    merged_df["Cluster"] = merged_df["Gene"].apply(
        lambda g: "1st Cluster" if g in cluster1_genes else ("2nd Cluster" if g in cluster2_genes else "Other")
    )
    merged_df = merged_df[merged_df["Cluster"] != "Other"]
 
    merged_df["delta_mT"] = merged_df[f"mT_adj_{sample2}"] - merged_df[f"mT_adj_{sample1}"]
 
    delta = two_std
    merged_df["residual"] = merged_df[f"mT_adj_{sample1}"] - merged_df[f"mT_adj_{sample2}"]
    merged_df["Outside_Band"] = merged_df["residual"].abs() > delta
 
    genes_outside_band_gc_rich = merged_df.loc[
        (merged_df["Cluster"] == "2nd Cluster") & merged_df["Outside_Band"], "Gene"
    ].tolist()
 
    df_c1 = merged_df[merged_df["Cluster"] == "1st Cluster"]
    df_c2 = merged_df[merged_df["Cluster"] == "2nd Cluster"]
 
    n_iter = 100
    n_cluster2 = len(df_c2)
 
    df_c1_sample_normality = df_c1.sample(n=n_cluster2, random_state=42)
    shapiro_pval_c1_delta = shapiro(df_c1_sample_normality["delta_mT"]).pvalue
    shapiro_pval_c2_delta = shapiro(df_c2["delta_mT"]).pvalue
    print(f"Shapiro-Wilk normality p-value (1st Cluster, delta_mT): {shapiro_pval_c1_delta:.4f}")
    print(f"Shapiro-Wilk normality p-value (2nd Cluster, delta_mT): {shapiro_pval_c2_delta:.4f}")
 
    mw_pvals_delta = []
    violin_df = None
    for i in range(n_iter):
        df_c1_sample = df_c1.sample(n=n_cluster2, replace=False, random_state=i)
        if i == 0:
            violin_df = pd.concat([
                df_c1_sample.assign(Cluster="1st Cluster"),
                df_c2.assign(Cluster="2nd Cluster"),
            ])
            violin_df["Comparison"] = "GC-Rich vs Upstream T-rich"
 
        _, mw_p_delta = mannwhitneyu(df_c1_sample["delta_mT"], df_c2["delta_mT"], alternative="two-sided")
        mw_pvals_delta.append(mw_p_delta)
 
    # Combined plot: scatter (left) + violin (right)
    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 2], wspace=0.3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
 
    x_vals = _axis_range(merged_df, sample1, sample2)
    _plot_band_and_diagonal(ax1, x_vals, delta)
 
    sns.scatterplot(data=df_c1, x=f"mT_adj_{sample1}", y=f"mT_adj_{sample2}", color="#c3c0c0", alpha=0.4, s=60, edgecolor="black", linewidth=0.5, label="Upstream T-rich Subset", ax=ax1)
    sns.scatterplot(data=df_c2, x=f"mT_adj_{sample1}", y=f"mT_adj_{sample2}", color=color_perturbed, alpha=0.6, s=60, edgecolor="black", linewidth=0.5, label="GC-Rich Subset", ax=ax1)
 
    ax1.set_xlabel(f"{label1} |$\\mu_T$-A3E|", fontsize=16)
    ax1.set_ylabel(f"{label2} |$\\mu_T$-A3E|", fontsize=16)
    ax1.set_title(f"{title}", fontsize=17, color=color_lab)
 
    _annotate_gene(ax1, df_c1, Gene_to_Lab, sample1, sample2)
    _annotate_gene(ax1, df_c2, Gene_to_Lab, sample1, sample2)
 
    ax1.legend(fontsize=12, loc="lower right")
    ax1.set_xlim(xlimylim)
    ax1.set_ylim(xlimylim)
    _format_a3e_ticks(ax1, xrotation=20)
 
    violin_df["Cluster"] = violin_df["Cluster"].replace({
        "1st Cluster": "Upstream T-rich Subset",
        "2nd Cluster": "GC-Rich Subset",
    })
    violin_df["delta_mT"] = violin_df[f"mT_adj_{sample2}"] - violin_df[f"mT_adj_{sample1}"]
 
    sns.violinplot(
        data=violin_df, x="Comparison", y="delta_mT", hue="Cluster", split=True,
        palette={"Upstream T-rich Subset": "#c3c0c0", "GC-Rich Subset": color_perturbed},
        inner="quartile", linewidth=1.2, ax=ax2,
    )
    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(handles[:2], labels[:2], fontsize=12)
    ax2.set_title(f"$\\mu_T$ {label2} - $\\mu_T$ {label1}", fontsize=16, color=color_lab)
    ax2.set_ylabel("∆$\\mu_T$", fontsize=16)
    ax2.tick_params(axis="x", labelsize=14)
    ax2.tick_params(axis="y", labelsize=14)
    ax2.set_xlabel("")
 
    plt.tight_layout()
    plt.savefig(outpath)
    plt.show()
 
    print(f"\nAverage Mann–Whitney U test p-value for delta_mT over {n_iter} runs: {np.mean(mw_pvals_delta):.4f}")
    fdr_rejected_delta, fdr_pvals_corrected_delta, _, _ = multipletests(mw_pvals_delta, alpha=0.05, method="fdr_bh")
    bonf_rejected_delta, bonf_pvals_corrected_delta, _, _ = multipletests(mw_pvals_delta, alpha=0.05, method="bonferroni")
 
    print("\n--- Multiple Comparison Correction Results (delta_mT) ---")
    print(f"FDR-corrected: {np.sum(fdr_rejected_delta)} out of {n_iter} tests were significant at alpha=0.05")
    print(f"Bonferroni-corrected: {np.sum(bonf_rejected_delta)} out of {n_iter} tests were significant at alpha=0.05")
    print(f"Mean FDR-corrected p-value: {np.mean(fdr_pvals_corrected_delta):.4f}")
    print(f"Mean Bonferroni-corrected p-value: {np.mean(bonf_pvals_corrected_delta):.4f}")
 
    return genes_outside_band_gc_rich

# Cross species scatter plot
def mT_cross_species(
    merged_df,
    sample1,
    sample2,
    outpath,
    label_dict,
    twosig,
    xlimylim,
    label_gene=None
):
    '''
    * Takes: 
    - merged_df: dataframe containing mT-A3E information on two samples
    - Sample 1 name
    - Sample 2 name
    - Outpath
    - Label dictionary specifying labels
    - Two sigma value (calculated from replicates)
    - xlim and ylim
    - Optional gene to label

    * Outputs:
    - Figure comparing mT values between samples

    '''   
    slope, intercept, r_value, p_value, std_err = linregress(
        merged_df[f"mT_adj_{sample2}"], merged_df[f"mT_adj_{sample1}"]
    )
 
    fig, ax = plt.subplots(figsize=(4, 4))
    x_vals = _axis_range(merged_df, sample1, sample2)
    intsigma = int(twosig)
    _plot_band_and_diagonal(ax, x_vals, twosig, band_label=f"±2$\\sigma$ ({intsigma})", band_color="lightgrey", band_alpha=0.4)
 
    lab1 = label_dict.get(sample1)
    lab2 = label_dict.get(sample2)
 
    sns.scatterplot(data=merged_df, x=f"mT_adj_{sample1}", y=f"mT_adj_{sample2}", color="#e3af6e", alpha=0.6, s=60, edgecolor=None, ax=ax)
 
    if label_gene is not None:
        if label_gene in merged_df["Gene"].values:
            _annotate_gene(ax, merged_df, label_gene, sample1, sample2)
        else:
            print(f"Warning: Gene '{label_gene}' not found in the dataframe")
 
    ax.set_xlabel(f"{lab1} \n |A3E-$\\mu_T$|", fontsize=14)
    ax.set_ylabel(f"{lab2} \n |A3E-$\\mu_T$|", fontsize=14)
 
    _finish_and_save(ax, xlimylim, outpath, legend_loc="upper left", legend_fontsize=11)
    _format_a3e_ticks(ax, xrotation=45)
    plt.close()
    
    
# Binomial scatters core code
def _binomial_scatter_core(
    two_std, plotting_dict_exp, experiment_name, sample1, sample2, genes_to_remove,
    label1, label2, color_perturbed, Gene_to_Lab, color_lab, title, out, xlimylim,
    direction,  # "above", "below", or "outside"
):
    """
    - Two standard deviations from the 1:1 line in control replicates (helps to determine what falls outside of "normal")
    - plotting_dict_exp: plotting dictionary from load_data functions, contains sample + prior information
    - Experiment name
    - Sample 1 for comparison
    - Sample 2 for comparison
    - List of genes to remove (for testing purposes)
    - Outfile path --> not used right now
    - Label 1 (goes with sample 1)
    - Label 2 (goes with sample 2)
    - Color of dots on scatter
    - Gene to label if desired
    - Color of the plot title
    - Title of plot
    - Outfile path
    - xlim and ylim
    - Direction of binomial

    * Outputs:
    - Performs 1-sided (above 2sigma) binomial test
    
    """
    merged_df = _prepare_merged_df(plotting_dict_exp, experiment_name, sample1, sample2, genes_to_remove)
    delta = two_std
    merged_df["residual"] = merged_df[f"mT_adj_{sample2}"] - merged_df[f"mT_adj_{sample1}"]
 
    if direction == "above":
        merged_df["Flagged"] = merged_df["residual"] > delta
        expected_prob = 0.025
        legend_label = "Genes > 2$\\sigma$"
        pval_label = "p-val (1-sided)"
        dot_color = "grey"
    elif direction == "below":
        merged_df["Flagged"] = merged_df["residual"] < -delta
        expected_prob = 0.025
        legend_label = "Genes < 2$\\sigma$"
        pval_label = "p-val (1-sided)"
        dot_color = "grey"
    elif direction == "outside":
        merged_df["Flagged"] = merged_df["residual"].abs() > delta
        expected_prob = 0.05
        legend_label = "Genes outside 2$\\sigma$"
        pval_label = "p-val (2-sided)"
        dot_color = color_perturbed
    else:
        raise ValueError("direction must be 'above', 'below', or 'outside'")
 
    total_genes = len(merged_df)
    n_flagged = int(merged_df["Flagged"].sum())
    pct_flagged = (n_flagged / total_genes) * 100
    pval = _binomial_pval(n_flagged, total_genes, expected_prob)
 
    fig, ax = plt.subplots(figsize=(6, 6))
    x_vals = _axis_range(merged_df, sample1, sample2)
    _plot_band_and_diagonal(ax, x_vals, delta)
 
    ax.scatter(
        merged_df[f"mT_adj_{sample1}"], merged_df[f"mT_adj_{sample2}"],
        color=dot_color, alpha=0.8 if direction == "outside" else 0.4,
        s=60, edgecolor="black", linewidth=0.5,
    )
 
    ax.set_xlabel(f"{label1} |$\\mu_T$-A3E|", fontsize=16)
    ax.set_ylabel(f"{label2} |$\\mu_T$-A3E|", fontsize=16)
    ax.set_title(f"{title}", fontsize=17, color=color_lab)
 
    _annotate_gene(ax, merged_df, Gene_to_Lab, sample1, sample2)
 
    stats_text = f"Total genes: {total_genes:,}\n{legend_label}: {n_flagged} ({pct_flagged:.1f}%)\n{pval_label}: {pval:.2e}"
    stats_xy = (0.5, 0.88) if direction in ("above", "below") else (0.02, 0.98)
    _add_stats_box(ax, stats_text, xy=stats_xy)
 
    _finish_and_save(ax, xlimylim, out)
 
    flagged_genes = merged_df.loc[merged_df["Flagged"], "Gene"].tolist()
    return {
        "genes_above_2sigma": flagged_genes,  # name kept for backwards compatibility
        "total_genes": total_genes,
        "count_above_2sigma": n_flagged,
        "pct_above_2sigma": pct_flagged,
        "pval": pval,
    }

# Scatter comparing mT (mT-A3E) values between two samples
# Looks at % of genes above two sigma (1-sided)
def perturbation_comparison_scatter_mT_all_genes_above_2sigma(two_std,
                                                     plotting_dict_exp,
                                                     experiment_name,
                                                     sample1,
                                                     sample2,
                                                     genes_to_remove,
                                                     outpath, 
                                                     label,
                                                     label1,
                                                     label2,
                                                     color_perturbed,
                                                     Gene_to_Lab,
                                                     color_lab,
                                                     title, 
                                                     out,
                                                     xlimylim):
    
    '''
    * Takes: 
    - Two standard deviations from the 1:1 line in control replicates (helps to determine what falls outside of "normal")
    - plotting_dict_exp: plotting dictionary from load_data functions, contains sample + prior information
    - Experiment name
    - Sample 1 for comparison
    - Sample 2 for comparison
    - List of genes to remove (for testing purposes)
    - Outfile path --> not used right now
    - Label 1 (goes with sample 1)
    - Label 2 (goes with sample 2)
    - Color of dots on scatter
    - Gene to label if desired
    - Color of the plot title
    - Title of plot
    - Outfile path
    - xlim and ylim

    * Outputs:
    - Figure comparing mT values between samples (1-sided, above 2sigma)
    - Performs 1-sided (above 2sigma) binomial test

    '''
    return _binomial_scatter_core(
        two_std, plotting_dict_exp, experiment_name, sample1, sample2, genes_to_remove,
        label1, label2, color_perturbed, Gene_to_Lab, color_lab, title, out, xlimylim, direction="above"
    )

# Scatter comparing mT (mT-A3E) values between two samples
# Looks at % of genes below two sigma (1-sided)
def perturbation_comparison_scatter_mT_all_genes_below_2sigma(two_std,
                                                     plotting_dict_exp,
                                                     experiment_name,
                                                     sample1,
                                                     sample2,
                                                     genes_to_remove,
                                                     outpath, 
                                                     label,
                                                     label1,
                                                     label2,
                                                     color_perturbed,
                                                     Gene_to_Lab,
                                                     color_lab,
                                                     title, 
                                                     out,
                                                     xlimylim):
    
    
    '''
    * Takes: 
    - Two standard deviations from the 1:1 line in control replicates (helps to determine what falls outside of "normal")
    - plotting_dict_exp: plotting dictionary from load_data functions, contains sample + prior information
    - Experiment name
    - Sample 1 for comparison
    - Sample 2 for comparison
    - List of genes to remove (for testing purposes)
    - Outfile path --> not used right now
    - Label 1 (goes with sample 1)
    - Label 2 (goes with sample 2)
    - Color of dots on scatter
    - Gene to label if desired
    - Color of the plot title
    - Title of plot
    - Outfile path
    - x lim and y lim

    * Outputs:
    - Figure comparing mT values between samples (1-sided, below 2sigma)
    - Performs 1-sided (below 2sigma) binomial test

    '''
    return _binomial_scatter_core(
        two_std, plotting_dict_exp, experiment_name, sample1, sample2, genes_to_remove,
        label1, label2, color_perturbed, Gene_to_Lab, color_lab, title, out, xlimylim, direction="below"
    )

# Scatter comparing mT (mT-A3E) values between two samples
# Looks at % of genes outside two sigma (2-sided)
def perturbation_comparison_scatter_mT_all_genes_outside_2sigma(two_std,
                                                     plotting_dict_exp,
                                                     experiment_name,
                                                     sample1,
                                                     sample2,
                                                     genes_to_remove,
                                                     outpath, 
                                                     label,
                                                     label1,
                                                     label2,
                                                     color_perturbed,
                                                     Gene_to_Lab,
                                                     color_lab,
                                                     title, 
                                                     out,
                                                     xlimylim):
    '''
    * Takes: 
    - Two standard deviations from the 1:1 line in control replicates (helps to determine what falls outside of "normal")
    - plotting_dict_exp: plotting dictionary from load_data functions, contains sample + prior information
    - Experiment name
    - Sample 1 for comparison
    - Sample 2 for comparison
    - List of genes to remove (for testing purposes)
    - Outfile path --> not used right now
    - Label 1 (goes with sample 1)
    - Label 2 (goes with sample 2)
    - Color of dots on scatter
    - Gene to label if desired
    - Color of the plot title
    - Title of plot
    - Outfile path
    -xlim and y lim

    * Outputs:
    - Figure comparing mT values between samples (2-sided, outside 2sigma)
    - Performs 2-sided (outside 2sigma) binomial test
    
    '''
    return _binomial_scatter_core(
        two_std, plotting_dict_exp, experiment_name, sample1, sample2, genes_to_remove,
        label1, label2, color_perturbed, Gene_to_Lab, color_lab, title, out, xlimylim, direction="outside"
    )
    
    

##################################################################################################################
########################## Scatter grids GC/T distance from identity line in each pairwise comparison ############
##################################################################################################################
def mT_scatter_on_GC_T_species_grid(plotting_dict_exp,
                                     experiment_name1,
                                     experiment_name2,
                                     sample1,
                                     sample2,
                                     genes_to_remove,
                                     label1,
                                     label2,
                                     xlimylim,
                                     stable_GC,
                                     stable_T,
                                     GC_to_T,
                                     T_to_GC,
                                    species,
                                    outdir,
                                    bound=2000):
    """
    Generates a 2×2 grid of scatter plots, one per evolutionary group,
    with all other genes shown as grey background points.
    Annotates each panel with gene counts inside/outside the 2σ bound.
    
    * Takes: 
    - Plotting dict
    - Experiment names (species here)
    - Samples
    - Genes for removal (testing purposes only)
    - Labels
    - X and Y lims
    - GC and T rich genes that do not change between species for comparison
    - Switches (GC --> T and vice versa)
    
    * Outputs: 
    - Grid colored on GC vs T-rich subsets across species
    
    """

    # ── 1. Prepare & Merge ──────────────────────────────────────────────────
    col1 = f"mT_adj_{sample1}"
    col2 = f"mT_adj_{sample2}"

    df1 = (plotting_dict_exp[experiment_name1][sample1][["Gene", "mT-adj"]]
           .rename(columns={"mT-adj": col1}))
    df2 = (plotting_dict_exp[experiment_name2][sample2][["Gene", "mT-adj"]]
           .rename(columns={"mT-adj": col2}))

    merged_df = pd.merge(df1, df2, on="Gene")
    merged_df = merged_df[~merged_df["Gene"].isin(genes_to_remove)].copy()

    # ── 2. Assign Evolutionary Group ────────────────────────────────────────
    def assign_group(gene):
        if gene in stable_GC:  return "GC in both species"
        elif gene in stable_T: return "T in both species"
        elif gene in GC_to_T:  return f"GC(human)→T({species})"
        elif gene in T_to_GC:  return f"T(human)→GC({species})"
        else:                   return "Other"

    merged_df["cluster_group"] = merged_df["Gene"].apply(assign_group)

    # ── 3. 2σ Status ────────────────────────────────────────────────────────
    merged_df["residual"] = merged_df[col1] - merged_df[col2]
    merged_df["Status"]   = np.where(
        merged_df["residual"].abs() > bound, "Outside 2σ", "Within 2σ"
    )

    # ── 4. Palette & Plot Order ──────────────────────────────────────────────
    palette = {
        "GC in both species":   "#CF5D5D",
        "T in both species":    "#9E9E9E",
        f"GC(human)→T({species})": "#F4A261",
        f"T(human)→GC({species})": "#2A9D8F",
    }
    plot_order = [
        "GC in both species",
        "T in both species",
        f"GC(human)→T({species})",
        f"T(human)→GC({species})",
    ]

    # ── 5. 2×2 Grid ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(8, 6))
    gs  = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.4)
    limit   = xlimylim[1]
    x_range = np.linspace(xlimylim[0], limit, 200)

    for idx, group in enumerate(plot_order):
        ax     = fig.add_subplot(gs[idx // 2, idx % 2])
        subset = merged_df[merged_df["cluster_group"] == group]

        # ── 1:1 line & shaded band ──
        ax.plot([xlimylim[0], limit], [xlimylim[0], limit],
                linestyle="--", color="black", alpha=0.4, zorder=1)
        ax.fill_between(x_range, x_range - bound, x_range + bound,
                        color="gray", alpha=0.15, zorder=0)

        # ── Background: all other genes ──
        bg = merged_df[merged_df["cluster_group"] != group]
        ax.scatter(bg[col2], bg[col1],
                   color="#CCCCCC", alpha=0.2, s=20,
                   edgecolors="none", zorder=1)

        # ── Focal group ──
        ax.scatter(subset[col2], subset[col1],
                   color=palette[group], alpha=0.75, s=40,
                   edgecolors="white", linewidth=0.4, zorder=3)

        ax.set_xlim(xlimylim)
        ax.set_ylim(xlimylim)
        ax.set_xlabel(f"{label2} |$\mu_T$-A3E|", fontsize=10)
        ax.set_ylabel(f"{label1} |$\mu_T$-A3E|", fontsize=10)
        ax.set_title(f"{group}  (n={len(subset)})",
                     fontsize=10, color=palette[group], fontweight="bold")

        # ── Count annotation ──
        n_outside = (subset["Status"] == "Outside 2σ").sum()
        n_within  = (subset["Status"] == "Within 2σ").sum()
        annot = (
            f"Outside 2σ: {n_outside}\n"
            f"Within 2σ:  {n_within}"
        )
        ax.text(0.04, 0.97, annot,
                transform=ax.transAxes,
                fontsize=8.5, va="top", ha="left",
                family="monospace",
                bbox=dict(boxstyle="round,pad=0.4", fc="white",
                          ec=palette[group], alpha=0.85, linewidth=1.2))

    fig.suptitle(f"|$\mu_T$-A3E| {label1} vs {label2}  (2σ bound = {bound:,})",
                 fontsize=13, fontweight="bold", y=1.01)

    plt.savefig(
        f"{outdir}/mT_scatter_grid_{species}.svg",
        bbox_inches="tight"
    )
    plt.show()
    return merged_df

#####################################################################
########################## Scatter colored on GC content ############
#####################################################################

def scatter_mT_gc_colored(gc_df,
                           plotting_dict_exp,
                           experiment_name,
                           sample1,
                           sample2,
                           genes_to_remove,
                           outpath,
                           label1,
                           label2,
                           title,
                           xlimylim,
                           two_std=None,
                           n_bins=4,
                           cmap="magma"):
    '''
    * Takes: 
    - Dataframe with GC content
    - Plotting dictionary
    - Experiment
    - Samples
    - Genes for removal
    - Outpath
    - Labels
    - Title
    - X and Y limits
    
    * Output: 
    - Scatter colored on GC content
    
    '''
    # --- 1. Pull mT values ---
    df1 = (plotting_dict_exp[experiment_name][sample1][["Gene", "mT-adj"]]
           .rename(columns={"mT-adj": f"mT_adj_{sample1}"}))
    df2 = (plotting_dict_exp[experiment_name][sample2][["Gene", "mT-adj"]]
           .rename(columns={"mT-adj": f"mT_adj_{sample2}"}))

    merged_df = pd.merge(df1, df2, on="Gene")
    merged_df = merged_df[~merged_df["Gene"].isin(genes_to_remove)]

    # --- 2. Merge GC% ---
    gc_subset = gc_df[["Gene", "GC%"]].drop_duplicates()
    merged_df = pd.merge(merged_df, gc_subset, on="Gene", how="inner")
    print(f"Genes with both mT and GC% data: {len(merged_df)}")

    # --- 3. Bin GC% into discrete, equal-width bins ---
    gc_min, gc_max = merged_df["GC%"].min(), merged_df["GC%"].max()
    bin_edges = np.linspace(gc_min, gc_max, n_bins + 1)

    merged_df["GC%_bin"] = pd.cut(
        merged_df["GC%"],
        bins=bin_edges,
        labels=False,           # 0-indexed integers
        include_lowest=True
    ).astype(int)

    # Each bin gets a fixed color drawn evenly from magma
    # Avoid the very dark ends (0.05–0.90) for better contrast on white bg
    cmap_obj = plt.get_cmap(cmap)
    palette = [cmap_obj(v) for v in np.linspace(0.10, 0.90, n_bins)]

    # Build tick labels: "X% – Y%" for each bin
    bin_labels = [f"{bin_edges[i]:.1f}% – {bin_edges[i+1]:.1f}%" 
                  for i in range(n_bins)]

    # --- 4. Sort so the darkest points don't bury the lightest ---
    # Plot low-GC (dark magma) first, high-GC (light magma) on top
    merged_df = merged_df.sort_values("GC%_bin", ascending=True)

    # --- 5. Plot ---
    fig, ax = plt.subplots(figsize=(5, 4))

    # Optional 2σ band
    if two_std is not None:
        x_vals = np.linspace(xlimylim[0], xlimylim[1], 500)
        ax.fill_between(x_vals, x_vals - two_std, x_vals + two_std,
                        color="lightgrey", alpha=0.45, zorder=0, label=f"±2σ")
        ax.plot(x_vals, x_vals, linestyle="--", color="red",
                linewidth=1, zorder=1, label="1:1 line")

    # Plot each bin as its own scatter layer so legend is clean
    for bin_idx in range(n_bins):
        subset = merged_df[merged_df["GC%_bin"] == bin_idx]
        ax.scatter(
            subset[f"mT_adj_{sample1}"],
            subset[f"mT_adj_{sample2}"],
            color=palette[bin_idx],
            label=bin_labels[bin_idx],
            alpha=0.75,
            s=55,
            edgecolor="black",
            linewidth=0.3,
            zorder=2
        )

    # --- 6. Legend for bins ---
    handles, labels_leg = ax.get_legend_handles_labels()
    # Separate 2σ / 1:1 handles from bin handles
    band_handles = [h for h, l in zip(handles, labels_leg) if l in ["±2σ", "1:1 line"]]
    band_labels  = [l for l in labels_leg if l in ["±2σ", "1:1 line"]]
    bin_handles  = [h for h, l in zip(handles, labels_leg) if l not in ["±2σ", "1:1 line"]]
    bin_labels_  = [l for l in labels_leg if l not in ["±2σ", "1:1 line"]]
    
    if band_handles:
        ax.legend(band_handles, band_labels, fontsize=10, loc="lower right")

    # GC% bins in a separate legend box
    legend_bins = ax.legend(bin_handles, bin_labels_,
                            title="GC% ±1kb of $\mu_T$",
                            title_fontsize=11,
                            fontsize=10,
                            loc="upper left")
    
    ax.add_artist(legend_bins)
    
    # --- 7. Axes formatting ---
    ax.set_xlabel(f"{label1} |$\mu_T$-A3E|", fontsize=15)
    ax.set_ylabel(f"{label2} |$\mu_T$-A3E|", fontsize=15)
    ax.set_title(title, fontsize=16)

    # Set limits FIRST so get_xticks reflects the correct range
    ax.set_xlim(xlimylim)
    ax.set_ylim(xlimylim)

    # Generate evenly spaced ticks within the range
    tick_min, tick_max = xlimylim
    step = 5000  # adjust if needed
    ticks = [t for t in range(
                int(np.ceil(tick_min / step)) * step,
                int(np.floor(tick_max / step)) * step + step,
                step)
             if tick_min <= t <= tick_max]

    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(
        ["A3E" if t == 0 else f"{int(t):,}" for t in ticks],
        fontsize=13, rotation=20)
    ax.set_yticklabels(
        ["A3E" if t == 0 else f"{int(t):,}" for t in ticks],
        fontsize=13)

    # Re-enforce after tick setting
    ax.set_xlim(xlimylim)
    ax.set_ylim(xlimylim)

    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.show()

 
 























