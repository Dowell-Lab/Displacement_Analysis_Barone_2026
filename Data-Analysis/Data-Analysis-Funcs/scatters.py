# Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress, shapiro, mannwhitneyu, binom
from statsmodels.stats.multitest import multipletests
import matplotlib.gridspec as gridspec

# Comparison between two samples, no clustering or stat testing

# Scatter comparing mT (mT-TCS) values between two samples
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
    
    # extract and merge data
    df1 = plotting_dict_exp[experiment_name][sample1][["Gene", "mT-adj"]].rename(columns={"mT-adj": f"mT_adj_{sample1}"})
    df2 = plotting_dict_exp[experiment_name][sample2][["Gene", "mT-adj"]].rename(columns={"mT-adj": f"mT_adj_{sample2}"})
    merged_df = pd.merge(df1, df2, on="Gene")
    merged_df = merged_df[~merged_df["Gene"].isin(genes_to_remove)]

    # calculate residual and determine # genes outside band
    delta = two_std
    merged_df["residual"] = merged_df[f"mT_adj_{ sample2}"] - merged_df[f"mT_adj_{sample1}"]
    
    # two sided analysis --> looking for mT values above or below band
    merged_df["Outside_Band"] = merged_df["residual"].abs() > delta
    
    # calculate statistics on how many genes fall outside of the band
    total_genes = len(merged_df)
    genes_outside_band = merged_df["Outside_Band"].sum()
    pct_outside_band = (genes_outside_band / total_genes) * 100

    # create figure
    fig, ax = plt.subplots(figsize=(6, 6))

    # plot 2-sigma band and 1:1 line
    x_vals = np.linspace(merged_df[f"mT_adj_{sample2}"].min(), merged_df[f"mT_adj_{sample2}"].max(), 500)
    ax.fill_between(x_vals, x_vals - delta, x_vals + delta, color="darkgrey", alpha=0.5, label=f"±2$\sigma$")
    ax.plot(x_vals, x_vals, linestyle="--", color="red", linewidth=1, label="1:1 line")

    # scatter plot
    sns.scatterplot(
        data=merged_df, 
        x=f"mT_adj_{sample1}", 
        y=f"mT_adj_{sample2}",
        color=dot_color, 
        alpha=0.6, 
        s=60, 
        edgecolor="black", 
        linewidth=0.5, 
        ax=ax
    )

    # labels/title
    ax.set_xlabel(f"{label1} $\mu_T$-TCS", fontsize=16)
    ax.set_ylabel(f"{label2} $\mu_T$-TCS", fontsize=16)
    ax.set_title(f"{title}", fontsize=17, color=color_lab)
    
    # annotate gene if provided
    if Gene_to_Lab:
        label_gene = merged_df[merged_df["Gene"] == Gene_to_Lab].copy()
        if not label_gene.empty:
            label_gene["Gene"] = label_gene["Gene"].str.split("|").str[0]
            
            for i, row in label_gene.iterrows():
                ax.annotate(
                    row['Gene'],
                    xy=(row[f'mT_adj_{sample1}'], row[f'mT_adj_{sample2}']),
                    xytext=(row[f'mT_adj_{sample1}'] + 500, row[f'mT_adj_{sample2}'] + 500),
                    fontsize=12,
                    fontweight='bold',
                    color='black',
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.5)
                )

    # add stats text box
    stats_text = f"Total genes: {total_genes:,}\nGenes Outside 2$\sigma$: {genes_outside_band} ({pct_outside_band:.1f}%)"
    ax.text(0.02, 0.98, stats_text, 
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black'))

    # set axis limits
    ax.set_xlim(xlimylim)
    ax.set_ylim(xlimylim)

    # format tick labels
    x_ticks = ax.get_xticks()
    y_ticks = ax.get_yticks()
    ax.set_xticklabels(["TCS" if x == 0 else f"{int(x):,}" for x in x_ticks], fontsize=14, rotation=20)
    ax.set_yticklabels(["TCS" if y == 0 else f"{int(y):,}" for y in y_ticks], fontsize=14)

    ax.legend(fontsize=12, loc='lower right')

    plt.tight_layout()
    plt.savefig(outpath)
    plt.show()

# Scatter comparing mT (mT-TCS) values between two samples
# Looks at % of genes above of two sigma (1-sided)
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
    
    # extract and merge data
    df1 = plotting_dict_exp[experiment_name][sample1][["Gene", "mT-adj"]].rename(columns={"mT-adj": f"mT_adj_{sample1}"})
    df2 = plotting_dict_exp[experiment_name][sample2][["Gene", "mT-adj"]].rename(columns={"mT-adj": f"mT_adj_{sample2}"})
    merged_df = pd.merge(df1, df2, on="Gene")
    merged_df = merged_df[~merged_df["Gene"].isin(genes_to_remove)]

    # calculate residual and determine genes outside band
    delta = two_std
    merged_df["residual"] = merged_df[f"mT_adj_{sample2}"] - merged_df[f"mT_adj_{sample1}"]
    
    # looking at only genes above 2 sigma in this analysis
    merged_df["Outside_Band"] = merged_df["residual"] > delta
    
    # calculate statistics
    total_genes = len(merged_df)
    genes_outside_band = merged_df["Outside_Band"].sum()
    pct_outside_band = (genes_outside_band / total_genes) * 100

    # create figure
    fig, ax = plt.subplots(figsize=(6, 6))

    # plot 2-sigma band/1:1 line
    x_vals = np.linspace(merged_df[f"mT_adj_{sample2}"].min(), merged_df[f"mT_adj_{sample2}"].max(), 500)
    ax.fill_between(x_vals, x_vals - delta, x_vals + delta, color="darkgrey", alpha=0.5, label=f"±2$\sigma$")
    ax.plot(x_vals, x_vals, linestyle="--", color="red", linewidth=1, label="1:1 line")

    # scatter plot
    sns.scatterplot(
        data=merged_df, 
        x=f"mT_adj_{sample1}", 
        y=f"mT_adj_{sample2}",
        color=dot_color, 
        alpha=0.6, 
        s=60, 
        edgecolor="black", 
        linewidth=0.5, 
        ax=ax
    )

    # labels/title
    ax.set_xlabel(f"{label1} $\mu_T$-TCS", fontsize=16)
    ax.set_ylabel(f"{label2} $\mu_T$-TCS", fontsize=16)
    ax.set_title(f"{title}", fontsize=17, color=color_lab)
    
    # annotate specific gene if provided
    if Gene_to_Lab:
        label_gene = merged_df[merged_df["Gene"] == Gene_to_Lab].copy()
        if not label_gene.empty:
            label_gene["Gene"] = label_gene["Gene"].str.split("|").str[0]
            
            for i, row in label_gene.iterrows():
                ax.annotate(
                    row['Gene'],
                    xy=(row[f'mT_adj_{sample1}'], row[f'mT_adj_{sample2}']),
                    xytext=(row[f'mT_adj_{sample1}'] + 500, row[f'mT_adj_{sample2}'] + 500),
                    fontsize=12,
                    fontweight='bold',
                    color='black',
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.5)
                )

    # stats text box
    stats_text = f"Total genes: {total_genes:,}\nGenes > 2$\sigma$: {genes_outside_band} ({pct_outside_band:.1f}%)"
    ax.text(0.02, 0.98, stats_text, 
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black'))

    # set axis limits
    ax.set_xlim(xlimylim)
    ax.set_ylim(xlimylim)

    # format tick labels
    x_ticks = ax.get_xticks()
    y_ticks = ax.get_yticks()
    ax.set_xticklabels(["TCS" if x == 0 else f"{int(x):,}" for x in x_ticks], fontsize=14, rotation=20)
    ax.set_yticklabels(["TCS" if y == 0 else f"{int(y):,}" for y in y_ticks], fontsize=14)

    ax.legend(fontsize=12, loc='lower right')

    plt.tight_layout()
    plt.savefig(outpath)
    plt.show()

# Comparisons in Figure 1

# Comparison between celltypes mT-TCS vs. mT-TCS
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
    - merged_df: dataframe containing mT-TCS information on two samples
    - Sample 1 name
    - Sample 2 name
    - Outpath
    - Label dictionary specifying labels
    - x lim and y lim

    * Outputs:
    - Figure comparing mT values between samples

    '''

    # linear regression
    slope, intercept, r_value, p_value, std_err = linregress(
        merged_df[f"mT_adj_{sample2}"],
        merged_df[f"mT_adj_{sample1}"]
    )
    r_squared = r_value ** 2

    # llot
    plt.figure(figsize=(6, 6))
    ax = plt.gca()

    x_vals = np.linspace(
        merged_df[f"mT_adj_{sample2}"].min(),
        merged_df[f"mT_adj_{sample1}"].max(),
        500
    )

    # calculate residuals from 1:1 line and 2-sigma
    residuals = merged_df[f"mT_adj_{sample1}"] - merged_df[f"mT_adj_{sample2}"]
    sigma_calculated = 2 * np.std(residuals)
    print(sigma_calculated)
    
    sigma_int = int(sigma_calculated)
    
    plt.fill_between(
    x_vals,
    x_vals - sigma_calculated,
    x_vals + sigma_calculated,
    color="lightgrey",
    alpha=0.4,
    label=f"±2$\sigma$ ({sigma_int})")
    
    # identity and regression lines
    plt.plot(x_vals, x_vals, linestyle="--", color="red", linewidth=1, label="1:1 line")
#     plt.plot(x_vals, intercept + slope * x_vals, color="black", linewidth=1.5,
#              label=f"Linear regression\nR² = {r_squared:.2f}")
    
    lab1 = label_dict.get(sample1)
    lab2 = label_dict.get(sample2)

    # scatter 
    sns.scatterplot(
        data=merged_df,
        x=f"mT_adj_{sample1}",
        y=f"mT_adj_{sample2}",
        color="#e3af6e",
        alpha=0.6,
        s=60,
        edgecolor=None
    )

    # labels and ticks
    plt.xlabel(f"{lab1} $\mu_T$-TCS", fontsize=14)
    plt.ylabel(f"{lab2} $\mu_T$-TCS", fontsize=14)
    plt.legend(loc="upper left", fontsize=11)
    plt.tight_layout()

    plt.xlim(xlimylim)
    plt.ylim(xlimylim)

    xticks = ax.get_xticks()
    yticks = ax.get_yticks()
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels(["TCS" if x == 0 else f"{int(x):,}" for x in xticks], fontsize=14, rotation=45)
    ax.set_yticklabels(["TCS" if y == 0 else f"{int(y):,}" for y in yticks], fontsize=14)

    plt.tight_layout()
    plt.savefig(outpath)
    plt.show()
    plt.close()

# Comparison between replicates 
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
    # extract and merge data
    df1 = plotting_dict_exp[experiment_name][sample1][["Gene", "mT-adj"]].rename(columns={"mT-adj": f"mT_adj_{sample1}"})
    df2 = plotting_dict_exp[experiment_name][sample2][["Gene", "mT-adj"]].rename(columns={"mT-adj": f"mT_adj_{sample2}"})
    merged_df = pd.merge(df1, df2, on="Gene")
    merged_df = merged_df[~merged_df["Gene"].isin(genes_to_remove)]

    # calculate residuals from the 1:1 line
    merged_df["residual"] = merged_df[f"mT_adj_{sample1}"] - merged_df[f"mT_adj_{sample2}"]
    std_dev = np.std(merged_df["residual"])
    two_std = 2 * std_dev
    print(f"±2 standard deviations from the 1:1 line: {two_std:.2f}")

    # linear regression
    slope, intercept, r_value, p_value, std_err = linregress(merged_df[f"mT_adj_{sample2}"], merged_df[f"mT_adj_{sample1}"])
    r_squared = r_value ** 2

    # prepare plot
    plt.figure(figsize=(6, 6))
    ax = plt.gca()

    # shaded ±2SD region around 1:1
    std_int = int(two_std)
    x_vals = np.linspace(merged_df[f"mT_adj_{sample2}"].min(), merged_df[f"mT_adj_{sample1}"].max(), 500)
    plt.fill_between(x_vals, x_vals - two_std, x_vals + two_std, color="lightgrey", alpha=0.4, label=f"±2$\sigma$ ({std_int})")

    # 1:1 line
    plt.plot(x_vals, x_vals, linestyle="--", color="red", linewidth=1, label="1:1 line")

    # Regression line
#     plt.plot(x_vals, intercept + slope * x_vals, color="black", linewidth=1.5, label=f"Linear regression\nR² = {r_squared:.2f}")

    # scatter plot 
    sns.scatterplot(
        data=merged_df,
        x=f"mT_adj_{sample1}",
        y=f"mT_adj_{sample2}",
        color="#e3af6e",  # orange --> disassoc peak in LIET diagram is orange
        alpha=0.6,
        s=60,
        edgecolor=None
    )

    # formatting
    plt.xlabel(f"{label} Rep1 $\mu_T$-TCS", fontsize=14)
    plt.ylabel(f"{label} Rep2 $\mu_T$-TCS", fontsize=14)
    plt.legend(loc="upper left", fontsize=11)
    plt.tight_layout()

    plt.xlim(xlimylim)
    plt.ylim(xlimylim)

    # replace 0 tick with TCS on x-axis and y-axis
    xticks = ax.get_xticks()
    yticks = ax.get_yticks()

    xtick_labels = ["TCS" if x == 0 else f"{int(x):,}" for x in xticks]
    ytick_labels = ["TCS" if y == 0 else f"{int(y):,}" for y in yticks]

    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels, fontsize=14, rotation=45)

    ax.set_yticks(yticks)
    ax.set_yticklabels(ytick_labels, fontsize=14)
    
    plt.tight_layout()

    plt.savefig(outpath)
    plt.show()
    plt.close()

    return two_std

# Replicate comparison (perturbation analysis)

# replicate comparison --> no coloring on clusters
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
    
    # extract and merge data
    df1 = plotting_dict_exp[experiment_name][sample1][["Gene", "mT-adj"]].rename(columns={"mT-adj": f"mT_adj_{sample1}"})
    df2 = plotting_dict_exp[experiment_name][sample2][["Gene", "mT-adj"]].rename(columns={"mT-adj": f"mT_adj_{sample2}"})
    merged_df = pd.merge(df1, df2, on="Gene")
    merged_df = merged_df[~merged_df["Gene"].isin(genes_to_remove)]
    
    # calculate residuals from the 1:1 line
    merged_df["residual"] = merged_df[f"mT_adj_{sample1}"] - merged_df[f"mT_adj_{sample2}"]
    std_dev = np.std(merged_df["residual"])
    two_std = 2 * std_dev
    twosigma_int = int(two_std)
    print(f"±2 standard deviations from the 1:1 line: {two_std:.2f}")
    
    # linear regression
    slope, intercept, r_value, p_value, std_err = linregress(merged_df[f"mT_adj_{sample2}"], merged_df[f"mT_adj_{sample1}"])
    r_squared = r_value ** 2
    
    plt.figure(figsize=(6, 6))
    ax = plt.gca()
    
    # shaded ±2SD region around 1:1 line
    x_vals = np.linspace(merged_df[f"mT_adj_{sample2}"].min(), merged_df[f"mT_adj_{sample1}"].max(), 500)
    plt.fill_between(x_vals, x_vals - two_std, x_vals + two_std, color="lightgrey", alpha=0.4, label=f"±2$\sigma$ ({twosigma_int})")
    
    # looking above 2sigma
    merged_df["Outside_Band"] = merged_df["residual"] > two_std
    n_genes_outside = len(merged_df[merged_df['Outside_Band'] == True])
    print(n_genes_outside)
    totalgenes = len(merged_df)
    print(totalgenes)
    
    percent_genes_outside = (n_genes_outside/totalgenes) * 100
        
    # filter for genes outside the 2SD band
    genes_outside_band = merged_df[merged_df["Outside_Band"] == True]["Gene"].tolist()
    
    # 1:1 line
    plt.plot(x_vals, x_vals, linestyle="--", color="red", linewidth=1, label="1:1 line")
    
    # regression line
    plt.plot(x_vals, intercept + slope * x_vals, color="black", linewidth=1.5, label=f"Linear regression \n R² = {r_squared:.2f}")
    
    # plot 
    sns.scatterplot(
        data=merged_df,
        x=f"mT_adj_{sample1}",
        y=f"mT_adj_{sample2}",
        color=color_perturbed,
        alpha=0.6,
        s=60,
        edgecolor=None
    )
    
    # formatting
    plt.xlabel(f"{label} Rep1 TCS-$\mu_T$", fontsize=14)
    plt.ylabel(f"{label} Rep2 TCS-$\mu_T$", fontsize=14)
    plt.legend(loc="upper left", fontsize=11)
    plt.tight_layout()
    
    # Replace 0 tick with TCS on x-axis and y-axis
    xticks = ax.get_xticks()
    yticks = ax.get_yticks()
    
    # Format tick labels, replacing 0 with "TCS"
    xtick_labels = ["TCS" if x == 0 else f"{int(x):,}" for x in xticks]
    ytick_labels = ["TCS" if y == 0 else f"{int(y):,}" for y in yticks]
    
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels, fontsize=14, rotation=45)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ytick_labels, fontsize=14)
    
    plt.show()
    plt.close()
    
    return two_std, genes_outside_band

# replicate comparison-- coloring on clusters
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

    # calculate residuals from the 1:1 line
    merged_df["residual"] = merged_df[f"mT_adj_{sample1}"] - merged_df[f"mT_adj_{sample2}"]
    std_dev = np.std(merged_df["residual"])
    two_std = 2 * std_dev
    three_std = 3 * std_dev
    print(f"±2 standard deviations from the 1:1 line: {two_std:.2f}")

    # linear regression
    slope, intercept, r_value, p_value, std_err = linregress(merged_df[f"mT_adj_{sample2}"], merged_df[f"mT_adj_{sample1}"])
    r_squared = r_value ** 2

    plt.figure(figsize=(6, 6))
    ax = plt.gca()

    # shaded ±2SD region around 1:1 line
    x_vals = np.linspace(merged_df[f"mT_adj_{sample2}"].min(), merged_df[f"mT_adj_{sample1}"].max(), 500)
    plt.fill_between(x_vals, x_vals - two_std, x_vals + two_std, color="lightgrey", alpha=0.4, label=f"±2$\sigma_T$ region")
    
    merged_df["Outside_Band"] = merged_df["residual"].abs() > three_std
    
    # filter for genes in the "GC-Rich Subset" (2nd Cluster) and outside the 2SD band
    genes_outside_band = merged_df[(merged_df["Outside_Band"] == True)]["Gene"].tolist()

    # 1:1 line
    plt.plot(x_vals, x_vals, linestyle="--", color="red", linewidth=1, label="1:1 line")

    # regression line
    plt.plot(x_vals, intercept + slope * x_vals, color="black", linewidth=1.5, label=f"Linear regression \n R² = {r_squared:.2f}")

    # plot points: cluster 1 first, then cluster 2
    df_c1 = merged_df[merged_df["Cluster"] == "1st Cluster"]
    df_c2 = merged_df[merged_df["Cluster"] == "2nd Cluster"]

    sns.scatterplot(
        data=df_c1,
        x=f"mT_adj_{sample1}",
        y=f"mT_adj_{sample2}",
        color="#c3c0c0",
        alpha=0.4,
        s=60,
        label="Non GC-Rich Subset",
        edgecolor=None
    )
    sns.scatterplot(
        data=df_c2,
        x=f"mT_adj_{sample1}",
        y=f"mT_adj_{sample2}",
        color=color_perturbed,
        alpha=0.6,
        s=60,
        label="GC-Rich Subset",
        edgecolor=None
    )

    # formatting
    plt.xlabel(f"{label} Rep1 TCS-$\mu_T$", fontsize = 14)
    plt.ylabel(f"{label} Rep2 TCS-$\mu_T$", fontsize = 14)
    plt.legend(loc="upper left", fontsize = 11)
    plt.tight_layout()

#     plt.xlim(-1000,20000)
#     plt.ylim(-1000,20000)

    # replace 0 tick with "TCS" on x-axis and y-axis
    xticks = ax.get_xticks()
    yticks = ax.get_yticks()

    xtick_labels = ["TCS" if x == 0 else f"{int(x):,}" for x in xticks]
    ytick_labels = ["TCS" if y == 0 else f"{int(y):,}" for y in yticks]

    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels, fontsize = 14,rotation=45)

    ax.set_yticks(yticks)
    ax.set_yticklabels(ytick_labels, fontsize = 14)

#     plt.savefig(outpath)
    plt.show()
    plt.close()
    
    return two_std, genes_outside_band

# Perturbation scatters & subsampled violin plots

# For figure 5 --> color on GC-rich vs non GC-rich subset
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
    
    # Change in mT after perturbation
    merged_df["delta_mT"] = merged_df[f"mT_adj_{sample2}"] - merged_df[f"mT_adj_{sample1}"]

    delta = two_std
    
    merged_df["residual"] = merged_df[f"mT_adj_{sample1}"] - merged_df[f"mT_adj_{sample2}"]
    
    # above the 2SD band
    merged_df["Outside_Band"] = merged_df["residual"].abs() > delta
    
    # filter for genes in the "GC-Rich Subset" (2nd Cluster) and above the 2SD band
    genes_outside_band_gc_rich = merged_df[
        (merged_df["Cluster"] == "2nd Cluster") & (merged_df["Outside_Band"] == True)
    ]["Gene"].tolist()

    df_c1 = merged_df[merged_df["Cluster"] == "1st Cluster"]
    df_c2 = merged_df[merged_df["Cluster"] == "2nd Cluster"]

    # Violin plot
    # Iterate 100 times where n is the same btwn clusters
    n_iter = 100
    n_cluster2 = len(df_c2)

    # test normality (for delta_mT)
    df_c1_sample_normality = df_c1.sample(n=n_cluster2, random_state=42)
    # Changed to delta_mT
    shapiro_pval_c1_delta = shapiro(df_c1_sample_normality["delta_mT"]).pvalue
    # Changed to delta_mT
    shapiro_pval_c2_delta = shapiro(df_c2["delta_mT"]).pvalue
    print(f"Shapiro-Wilk normality p-value (1st Cluster, delta_mT): {shapiro_pval_c1_delta:.4f}")
    print(f"Shapiro-Wilk normality p-value (2nd Cluster, delta_mT): {shapiro_pval_c2_delta:.4f}")

    mw_pvals_delta = []  # list to store p-values for delta_mT

    for i in range(n_iter):
        df_c1_sample = df_c1.sample(n=n_cluster2, replace=False, random_state=i)

        if i == 0:
            violin_df = pd.concat([
                df_c1_sample.assign(Cluster="1st Cluster"),
                df_c2.assign(Cluster="2nd Cluster")
            ])
            violin_df["Comparison"] = "GC-Rich vs Upstream T-rich"

        # Perform Mann-Whitney U test on delta_mT
        mw_stat_delta, mw_p_delta = mannwhitneyu(
            df_c1_sample["delta_mT"],  # Use delta_mT for comparison
            df_c2["delta_mT"],  # Use delta_mT for comparison
            alternative='two-sided'
        )
        mw_pvals_delta.append(mw_p_delta)

    # Combined Plot
    fig = plt.figure(figsize=(12, 6))  
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 2], wspace=0.3)

    # SCATTER PLOT (wider, shorter)
    ax1 = fig.add_subplot(gs[0, 0])

    # VIOLIN PLOT (taller, narrower)
    ax2 = fig.add_subplot(gs[0, 1])

    # SCATTER PLOT
    x_vals = np.linspace(merged_df[f"mT_adj_{sample2}"].min(), merged_df[f"mT_adj_{sample2}"].max(), 500)
    ax1.fill_between(x_vals, x_vals - delta, x_vals + delta, color="darkgrey", alpha=0.5, label=f"±2$\sigma$")
    ax1.plot(x_vals, x_vals, linestyle="--", color="red", linewidth=1, label="1:1 line")

    sns.scatterplot(
        data=df_c1, x=f"mT_adj_{sample1}", y=f"mT_adj_{sample2}",
        color="#c3c0c0", alpha=0.4, s=60, edgecolor="black", linewidth=0.5, label="Upstream T-rich Subset", ax=ax1
    )
    
    sns.scatterplot(
        data=df_c2, x=f"mT_adj_{sample1}", y=f"mT_adj_{sample2}",
        color=color_perturbed, alpha=0.6, s=60, edgecolor="black", linewidth=0.5, label="GC-Rich Subset", ax=ax1
    )

    ax1.set_xlabel(f"{label1} $\mu_T$-TCS", fontsize = 16)
    ax1.set_ylabel(f"{label2} $\mu_T$-TCS", fontsize = 16)
    ax1.set_title(f"{title}", fontsize = 17, color = color_lab)
    
    label_gene_cf1 = df_c1[df_c1["Gene"] == Gene_to_Lab]
    label_gene_cf1["Gene"] = label_gene_cf1["Gene"].str.split("|").str[0]
    
    label_gene_cf2 = df_c2[df_c2["Gene"] == Gene_to_Lab]
    label_gene_cf2["Gene"] = label_gene_cf2["Gene"].str.split("|").str[0]

    for i, row in label_gene_cf1.iterrows():
        ax1.annotate(
            row['Gene'],
            xy=(row[f'mT_adj_{sample1}'], row[f'mT_adj_{sample2}']),
            xytext=(row[f'mT_adj_{sample1}'] + 500, row[f'mT_adj_{sample2}'] + 500),
            fontsize=12,
            fontweight='bold',
            color='black',
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5)
        )

    for i, row in label_gene_cf2.iterrows():
        ax1.annotate(
            row['Gene'],
            xy=(row[f'mT_adj_{sample1}'], row[f'mT_adj_{sample2}']),
            xytext=(row[f'mT_adj_{sample1}'] + 500, row[f'mT_adj_{sample2}'] + 500),
            fontsize=12,
            fontweight='bold',
            color='black',
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5)
        )

    # ax1.set_title(f"Scatter Plot of mT-adj Values:\n{sample1} vs {sample2}")
    ax1.legend(fontsize = 12, loc='lower right')
#     ax1.legend(fontsize = 12, loc='upper left')


    # Set the x and y axis limits
    ax1.set_xlim(xlimylim)
    ax1.set_ylim(xlimylim)

    # Replace 0 ticks with 'TCS' in scatter plot
    x_ticks = ax1.get_xticks()
    y_ticks = ax1.get_yticks()
    ax1.set_xticklabels(["TCS" if x == 0 else f"{int(x):,}" for x in x_ticks], fontsize = 14, rotation = 20)
    ax1.set_yticklabels(["TCS" if y == 0 else f"{int(y):,}" for y in y_ticks], fontsize = 14)

    violin_df["Cluster"] = violin_df["Cluster"].replace({
        "1st Cluster": "Upstream T-rich Subset",
        "2nd Cluster": "GC-Rich Subset"
    })
    
    violin_df["delta_mT"] = violin_df[f"mT_adj_{sample2}"] - violin_df[f"mT_adj_{sample1}"]

    # VIOLIN PLOT --> with subsampled (first iteration data)
    sns.violinplot(
        data=violin_df,
        x="Comparison",
        y=f"delta_mT",
        hue="Cluster",
        split=True,
        palette={"Upstream T-rich Subset": "#c3c0c0", "GC-Rich Subset": color_perturbed},
        inner="quartile",
        linewidth=1.2,
        ax=ax2
    )
    handles, labels = ax2.get_legend_handles_labels()
    
    ax2.legend(handles[:2], labels[:2], fontsize = 12)
    # ax2.set_title("TCS-$\mu_T$ \n(1st Subsample)")

    ax2.set_title(f"$\mu_T$ {label2} - $\mu_T$ {label1}", fontsize = 16, color = color_lab)

    ax2.set_ylabel(f"∆$\mu_T$", fontsize = 16)
    
    # Make tick label bigger
    ax2.tick_params(axis='x', labelsize=14)
    ax2.tick_params(axis='y', labelsize=14)
    ax2.set_xlabel("")

    plt.tight_layout()
    plt.savefig(outpath)

    plt.show()

    print(f"\nAverage Mann–Whitney U test p-value for delta_mT over {n_iter} runs: {np.mean(mw_pvals_delta):.4f}")
    fdr_rejected_delta, fdr_pvals_corrected_delta, _, _ = multipletests(mw_pvals_delta, alpha=0.05, method='fdr_bh')
    bonf_rejected_delta, bonf_pvals_corrected_delta, _, _ = multipletests(mw_pvals_delta, alpha=0.05, method='bonferroni')

    print(f"\n--- Multiple Comparison Correction Results (delta_mT) ---")
    print(f"FDR-corrected: {np.sum(fdr_rejected_delta)} out of {n_iter} tests were significant at alpha=0.05")
    print(f"Bonferroni-corrected: {np.sum(bonf_rejected_delta)} out of {n_iter} tests were significant at alpha=0.05")
    print(f"Mean FDR-corrected p-value: {np.mean(fdr_pvals_corrected_delta):.4f}")
    print(f"Mean Bonferroni-corrected p-value: {np.mean(bonf_pvals_corrected_delta):.4f}")

    return genes_outside_band_gc_rich

# Cross species scatters
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
    - merged_df: dataframe containing mT-TCS information on two samples
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
    
    # Linear regression
    slope, intercept, r_value, p_value, std_err = linregress(
        merged_df[f"mT_adj_{sample2}"],
        merged_df[f"mT_adj_{sample1}"]
    )
    r_squared = r_value ** 2
    # Plot
    plt.figure(figsize=(6, 6))
    ax = plt.gca()
    # Range and shading
    x_vals = np.linspace(
        merged_df[f"mT_adj_{sample2}"].min(),
        merged_df[f"mT_adj_{sample1}"].max(),
        500
    )
    
    # Calculate residuals from 1:1 line and 2-sigma
    residuals = merged_df[f"mT_adj_{sample1}"] - merged_df[f"mT_adj_{sample2}"]
#     sigma_calculated = 2 * np.std(residuals)

    intsigma = int(twosig)
    
    plt.fill_between(
    x_vals,
    x_vals - twosig,
    x_vals + twosig,
    color="lightgrey",
    alpha=0.4,
    label=f"±2$\sigma$ ({intsigma})")
    
    # Identity and regression lines
    
    plt.plot(x_vals, x_vals, linestyle="--", color="red", linewidth=1, label="1:1 line")
#     plt.plot(x_vals, intercept + slope * x_vals, color="black", linewidth=1.5,
#              label=f"Linear regression\nR² = {r_squared:.2f}")
    
    lab1 = label_dict.get(sample1)
    lab2 = label_dict.get(sample2)
    # Scatter points
    sns.scatterplot(
        data=merged_df,
        x=f"mT_adj_{sample1}",
        y=f"mT_adj_{sample2}",
        color="#e3af6e",
        alpha=0.6,
        s=60,
        edgecolor=None
    )
        
    if label_gene is not None:
        if label_gene in merged_df['Gene'].values:
                        
            gene_row = merged_df[merged_df['Gene'] == label_gene].iloc[0]
            gene_x = gene_row[f"mT_adj_{sample1}"]
            gene_y = gene_row[f"mT_adj_{sample2}"]

            plt.annotate(
                label_gene,
                xy=(gene_x, gene_y),
                xytext=(gene_x + 500, gene_y + 500),
                fontsize=12,
                color='black',
                fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5)
            )
        else:
            print(f"Warning: Gene '{label_gene}' not found in the dataframe")
    
    # Labels and ticks
    plt.xlabel(f"{lab1} TCS-$\mu_T$", fontsize=14)
    plt.ylabel(f"{lab2} TCS-$\mu_T$", fontsize=14)
    plt.legend(loc="upper left", fontsize=11)
    plt.tight_layout()
    plt.xlim(xlimylim)
    plt.ylim(xlimylim)
    xticks = ax.get_xticks()
    yticks = ax.get_yticks()
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels(["TCS" if x == 0 else f"{int(x):,}" for x in xticks], fontsize=14, rotation=45)
    ax.set_yticklabels(["TCS" if y == 0 else f"{int(y):,}" for y in yticks], fontsize=14)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.show()
    plt.close()

# Binomial scatters

# Scatter comparing mT (mT-TCS) values between two samples
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
    
    # extract and merge data
    df1 = plotting_dict_exp[experiment_name][sample1][["Gene", "mT-adj"]].rename(columns={"mT-adj": f"mT_adj_{sample1}"})
    df2 = plotting_dict_exp[experiment_name][sample2][["Gene", "mT-adj"]].rename(columns={"mT-adj": f"mT_adj_{sample2}"})
    merged_df = pd.merge(df1, df2, on="Gene")
    merged_df = merged_df[~merged_df["Gene"].isin(genes_to_remove)]
    
    delta = two_std
    merged_df["residual"] = merged_df[f"mT_adj_{sample2}"] - merged_df[f"mT_adj_{sample1}"]
    
    # genes above 2sigma (residual > +delta)
    merged_df["Above_2sigma"] = merged_df["residual"] > delta
    
    # stats on all genes
    total_genes = len(merged_df)
    genes_above_2sigma = len(merged_df[merged_df["Above_2sigma"] == True])
    pct_above_2sigma = (genes_above_2sigma / total_genes) * 100
    
    # Null hypothesis: under normal distribution, ~2.5% of genes expected above 2sigma in either dir (5% total)
    expected_prob = 0.025  # ~2.5% expected above
    pval = binom.sf(genes_above_2sigma - 1, total_genes, expected_prob)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    x_vals = np.linspace(merged_df[f"mT_adj_{sample1}"].min(), merged_df[f"mT_adj_{sample1}"].max(), 500)
    ax.fill_between(x_vals, x_vals - delta, x_vals + delta, color="darkgrey", alpha=0.5, label=f"±2$\sigma$")
    ax.plot(x_vals, x_vals, linestyle="--", color="red", linewidth=1, label="1:1 line")
    
    below_2sigma = merged_df[merged_df["Above_2sigma"] == False]
    above_2sigma = merged_df[merged_df["Above_2sigma"] == True]
    
    ax.scatter(merged_df[f"mT_adj_{sample1}"], merged_df[f"mT_adj_{sample2}"],
              color='grey', alpha=0.4, s=60, edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel(f"{label1} $\mu_T$-TCS", fontsize=16)
    ax.set_ylabel(f"{label2} $\mu_T$-TCS", fontsize=16)
    ax.set_title(f"{title}", fontsize=17, color=color_lab)
    
    if Gene_to_Lab:
        label_gene = merged_df[merged_df["Gene"] == Gene_to_Lab].copy()
        if not label_gene.empty:
            label_gene["Gene"] = label_gene["Gene"].str.split("|").str[0]
            
            for i, row in label_gene.iterrows():
                ax.annotate(
                    row['Gene'],
                    xy=(row[f'mT_adj_{sample1}'], row[f'mT_adj_{sample2}']),
                    xytext=(row[f'mT_adj_{sample1}'] + 500, row[f'mT_adj_{sample2}'] + 500),
                    fontsize=12,
                    fontweight='bold',
                    color='black',
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.5)
                )
    
    # stats box -- binomial
    pval_text = f"p-val (1-sided): {pval:.2e}"
    stats_text = f"Total genes: {total_genes:,}\nGenes > 2$\sigma$: {genes_above_2sigma} ({pct_above_2sigma:.1f}%)\n{pval_text}"
    ax.text(0.5, 0.88, stats_text, 
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black'))
    
    # Set axis limits
    ax.set_xlim(xlimylim)
    ax.set_ylim(xlimylim)

    # tick labels
    x_ticks = ax.get_xticks()
    y_ticks = ax.get_yticks()
    ax.set_xticklabels(["TCS" if x == 0 else f"{int(x):,}" for x in x_ticks], fontsize=14, rotation=20)
    ax.set_yticklabels(["TCS" if y == 0 else f"{int(y):,}" for y in y_ticks], fontsize=14)
    
    ax.legend(fontsize=12, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(out)
    plt.show()
    
    genes_above_2sigma_list = merged_df[merged_df["Above_2sigma"]]["Gene"].tolist()
    
    return {
        'genes_above_2sigma': genes_above_2sigma_list,
        'total_genes': total_genes,
        'count_above_2sigma': genes_above_2sigma,
        'pct_above_2sigma': pct_above_2sigma,
        'pval': pval
    }

# Scatter comparing mT (mT-TCS) values between two samples
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
    
    # extract and merge data
    df1 = plotting_dict_exp[experiment_name][sample1][["Gene", "mT-adj"]].rename(columns={"mT-adj": f"mT_adj_{sample1}"})
    df2 = plotting_dict_exp[experiment_name][sample2][["Gene", "mT-adj"]].rename(columns={"mT-adj": f"mT_adj_{sample2}"})
    merged_df = pd.merge(df1, df2, on="Gene")
    merged_df = merged_df[~merged_df["Gene"].isin(genes_to_remove)]
    
    delta = two_std
    merged_df["residual"] = merged_df[f"mT_adj_{sample2}"] - merged_df[f"mT_adj_{sample1}"]
    
    # genes below 2sigma (residual < -delta)
    # ** column name confusing--> assessing below 2sigma **
    merged_df["Above_2sigma"] = merged_df["residual"] < -delta
    
    # statistics on all genes
    total_genes = len(merged_df)
    genes_above_2sigma = len(merged_df[merged_df["Above_2sigma"] == True])
    pct_above_2sigma = (genes_above_2sigma / total_genes) * 100
    
    expected_prob = 0.025  # ~2.5% expected below 
    pval = binom.sf(genes_above_2sigma - 1, total_genes, expected_prob)
    
    # scatter plot
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # plot 2-sigma band and 1:1 line
    x_vals = np.linspace(merged_df[f"mT_adj_{sample1}"].min(), merged_df[f"mT_adj_{sample1}"].max(), 500)
    ax.fill_between(x_vals, x_vals - delta, x_vals + delta, color="darkgrey", alpha=0.5, label=f"±2$\sigma$")
    ax.plot(x_vals, x_vals, linestyle="--", color="red", linewidth=1, label="1:1 line")
    
    below_2sigma = merged_df[merged_df["Above_2sigma"] == False]
    above_2sigma = merged_df[merged_df["Above_2sigma"] == True]
    
    ax.scatter(merged_df[f"mT_adj_{sample1}"], merged_df[f"mT_adj_{sample2}"],
              color='grey', alpha=0.4, s=60, edgecolor='black', linewidth=0.5)
    
    # labels and title
    ax.set_xlabel(f"{label1} $\mu_T$-TCS", fontsize=16)
    ax.set_ylabel(f"{label2} $\mu_T$-TCS", fontsize=16)
    ax.set_title(f"{title}", fontsize=17, color=color_lab)
    
    if Gene_to_Lab:
        label_gene = merged_df[merged_df["Gene"] == Gene_to_Lab].copy()
        if not label_gene.empty:
            label_gene["Gene"] = label_gene["Gene"].str.split("|").str[0]
            
            for i, row in label_gene.iterrows():
                ax.annotate(
                    row['Gene'],
                    xy=(row[f'mT_adj_{sample1}'], row[f'mT_adj_{sample2}']),
                    xytext=(row[f'mT_adj_{sample1}'] + 500, row[f'mT_adj_{sample2}'] + 500),
                    fontsize=12,
                    fontweight='bold',
                    color='black',
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.5)
                )
    
    # stats text box
    pval_text = f"p-val (1-sided): {pval:.2e}"
    stats_text = f"Total genes: {total_genes:,}\nGenes < 2$\sigma$: {genes_above_2sigma} ({pct_above_2sigma:.1f}%)\n{pval_text}"
    ax.text(0.5, 0.88, stats_text, 
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black'))
    
    ax.set_xlim(xlimylim)
    ax.set_ylim(xlimylim)

    # tick labels
    x_ticks = ax.get_xticks()
    y_ticks = ax.get_yticks()
    ax.set_xticklabels(["TCS" if x == 0 else f"{int(x):,}" for x in x_ticks], fontsize=14, rotation=20)
    ax.set_yticklabels(["TCS" if y == 0 else f"{int(y):,}" for y in y_ticks], fontsize=14)
    
    ax.legend(fontsize=12, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(out)
    plt.show()
    
    genes_above_2sigma_list = merged_df[merged_df["Above_2sigma"]]["Gene"].tolist()
    
    return {
        'genes_above_2sigma': genes_above_2sigma_list,
        'total_genes': total_genes,
        'count_above_2sigma': genes_above_2sigma,
        'pct_above_2sigma': pct_above_2sigma,
        'pval': pval
    }

# Scatter comparing mT (mT-TCS) values between two samples
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

    
    # extract and merge data
    df1 = plotting_dict_exp[experiment_name][sample1][["Gene", "mT-adj"]].rename(columns={"mT-adj": f"mT_adj_{sample1}"})
    df2 = plotting_dict_exp[experiment_name][sample2][["Gene", "mT-adj"]].rename(columns={"mT-adj": f"mT_adj_{sample2}"})
    merged_df = pd.merge(df1, df2, on="Gene")
    merged_df = merged_df[~merged_df["Gene"].isin(genes_to_remove)]
    
    delta = two_std
    merged_df["residual"] = merged_df[f"mT_adj_{sample2}"] - merged_df[f"mT_adj_{sample1}"]
    
    # genes outside 2sigma (|residual| > +delta)
    merged_df["Above_2sigma"] = merged_df["residual"].abs() > delta  
    
    # stats on all genes
    total_genes = len(merged_df)
    genes_above_2sigma = len(merged_df[merged_df["Above_2sigma"] == True])
    pct_above_2sigma = (genes_above_2sigma / total_genes) * 100
    
    expected_prob = 0.05  # ~5% expected outside 2sigma 
    pval = binom.sf(genes_above_2sigma - 1, total_genes, expected_prob)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # 2-sigma band and 1:1 line
    x_vals = np.linspace(merged_df[f"mT_adj_{sample1}"].min(), merged_df[f"mT_adj_{sample1}"].max(), 500)
    ax.fill_between(x_vals, x_vals - delta, x_vals + delta, color="darkgrey", alpha=0.5, label=f"±2$\sigma$")
    ax.plot(x_vals, x_vals, linestyle="--", color="red", linewidth=1, label="1:1 line")
    
    below_2sigma = merged_df[merged_df["Above_2sigma"] == False]
    above_2sigma = merged_df[merged_df["Above_2sigma"] == True]
    
    ax.scatter(merged_df[f"mT_adj_{sample1}"], merged_df[f"mT_adj_{sample2}"],
              color=color_perturbed, alpha=0.8, s=60, edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel(f"{label1} $\mu_T$-TCS", fontsize=16)
    ax.set_ylabel(f"{label2} $\mu_T$-TCS", fontsize=16)
    ax.set_title(f"{title}", fontsize=17, color=color_lab)
    
    if Gene_to_Lab:
        label_gene = merged_df[merged_df["Gene"] == Gene_to_Lab].copy()
        if not label_gene.empty:
            label_gene["Gene"] = label_gene["Gene"].str.split("|").str[0]
            
            for i, row in label_gene.iterrows():
                ax.annotate(
                    row['Gene'],
                    xy=(row[f'mT_adj_{sample1}'], row[f'mT_adj_{sample2}']),
                    xytext=(row[f'mT_adj_{sample1}'] + 500, row[f'mT_adj_{sample2}'] + 500),
                    fontsize=12,
                    fontweight='bold',
                    color='black',
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.5)
                )
    
    # stats
    pval_text = f"p-val (2-sided): {pval:.2e}"
    stats_text = f"Total genes: {total_genes:,}\nGenes outside 2$\sigma$: {genes_above_2sigma} ({pct_above_2sigma:.1f}%)\n{pval_text}"
    ax.text(0.02, 0.98, stats_text, 
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black'))
    
    # Set axis limits
    ax.set_xlim(xlimylim)
    ax.set_ylim(xlimylim)

    x_ticks = ax.get_xticks()
    y_ticks = ax.get_yticks()
    ax.set_xticklabels(["TCS" if x == 0 else f"{int(x):,}" for x in x_ticks], fontsize=14, rotation=20)
    ax.set_yticklabels(["TCS" if y == 0 else f"{int(y):,}" for y in y_ticks], fontsize=14)
    
    ax.legend(fontsize=12, loc='lower right')
    
    
    plt.tight_layout()
    plt.savefig(out)
    plt.show()
    
    genes_above_2sigma_list = merged_df[merged_df["Above_2sigma"]]["Gene"].tolist()
    
    return {
        'genes_above_2sigma': genes_above_2sigma_list,
        'total_genes': total_genes,
        'count_above_2sigma': genes_above_2sigma,
        'pct_above_2sigma': pct_above_2sigma,
        'pval': pval
    }
