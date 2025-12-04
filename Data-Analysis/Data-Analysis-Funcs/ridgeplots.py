# Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp
import matplotlib.gridspec as gridspec

# qcut ridgeplot 
def plot_kde_of_priors_perturb_overlay(df, 
                                       plotting_dict, 
                                       samples_to_compare, 
                                       sample_category, 
                                       label_dict, 
                                       figsize, 
                                       outdir = None):
    
    '''
    * Takes: 
    - GC content quantile dataframe
    - Dictionary containing LIET prior output information per-sample
    - sample_category: sample
    - Dictionary of samples for label mapping
    - Size of figure 
    - Optional outdirectory

    * Outputs:
    - Ridgeplot (mT distributions between quantiles)

    '''

    # Prepare data for plotting
    merged_dfs = []
    for samp in samples_to_compare:
        df_prior = plotting_dict[sample_category][samp]
        df_prior = df_prior[["Gene", 'mT-adj', "sT"]]
        df_temp = df.merge(df_prior, on="Gene")
        df_temp['Sample_Type'] = samp
        df_temp['Sample_Label'] = df_temp['Sample_Type'].map(label_dict)
        print(len(df_temp))
        merged_dfs.append(df_temp)
    combined_df = pd.concat(merged_dfs, ignore_index=True)

    # Calculate the mean mT-adj for each quantile and sample type
    mean_mt_adj = combined_df.groupby(['GC%-QCut', 'Sample_Type'])['mT-adj'].mean().reset_index()

    # Prepare for plotting
    gc_qcut_order = sorted(combined_df['GC%-QCut'].unique())

    # --- Plot for 'mT-adj' (Overlayed KDEs) --- #
    fig_mt, axes_mt = plt.subplots(figsize=(figsize), nrows=len(gc_qcut_order), sharex=True)
#     fig_mt, axes_mt = plt.subplots(figsize=(10, 9), nrows=len(gc_qcut_order), sharex=True)

    # colors_per_sample = sns.color_palette("viridis", n_colors=len(samples_to_compare))
    colors_per_sample =["#c3c0c0",'#e57a7a']
    colors_per_sample_line = ["#696564", "#A83011"]

    for i, gc_qcut in enumerate(gc_qcut_order):
        for j, samp_label in enumerate(samples_to_compare):
            subset = combined_df[(combined_df['GC%-QCut'] == gc_qcut) & (combined_df['Sample_Type'] == samp_label)]
            sns.kdeplot(
                subset['mT-adj'],
                ax=axes_mt[i],
                fill=True,
                color=colors_per_sample[j],
                alpha=0.6, # Adjust alpha for better visibility
                linewidth=1.5,
                label = f'{label_dict[samp_label]}' # Add label for legend
            )

            # Add vertical line for the mean mT-adj
            current_mean = mean_mt_adj[(mean_mt_adj['GC%-QCut'] == gc_qcut) & (mean_mt_adj['Sample_Type'] == samp_label)]['mT-adj'].values[0]
            axes_mt[i].axvline(x=current_mean, color=colors_per_sample_line[j], linestyle='--', linewidth=1, ymin=0.3, ymax=1)
            
            # Add a black "avg mT" label to the dashed line
            axes_mt[i].text(current_mean, 0.95, "Avg. \n $\mu_T$", color='black', 
                             fontsize=8, ha='center', va='bottom', transform=axes_mt[i].get_xaxis_transform()) # Places text near the top of the line


        axes_mt[i].set_ylabel(f'Quantile {gc_qcut}', rotation=0, ha='right', fontsize =13)
        axes_mt[i].set_yticks([])
        axes_mt[i].spines[['left', 'right', 'top']].set_visible(False)
        axes_mt[i].set_xlim(-5000,20000)

        # Get current x-axis ticks and labels
        ticks = axes_mt[i].get_xticks()
        labels = [str(tick) for tick in ticks]

        try:
            zero_index = np.where(np.isclose(ticks, 0))[0][0]
            labels[zero_index] = 'TCS'
        except IndexError:
            new_ticks = sorted(list(ticks) + [0])
            new_labels = [str(tick) for tick in new_ticks]
            zero_index = new_ticks.index(0)
            new_labels[zero_index] = 'TCS'
            axes_mt[i].set_xticks(new_ticks)
            labels = new_labels

        axes_mt[i].set_xticklabels(labels, fontsize=13, rotation = 20)

    axes_mt[-1].set_xlabel('$\mu_T$-TCS', fontsize = 15)
    # fig_mt.suptitle(f'Quantile Distributions (Density Plot) of $\mu_T$-PAS \n (Subsampled)', y=1.02, fontsize = 15)
    fig_mt.suptitle(f'Quantile Distributions (Density Plot) of $\mu_T$-TCS', y=1.02, fontsize = 15)

    # Add a legend to the last subplot or a shared legend if desired
    axes_mt[0].legend(fontsize=12, title_fontsize=13, bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout(rect=[0, 0, 0.95, 1])
    
    if outdir: 
        plt.savefig(f"{outdir}")

    plt.show()

# Fig 1 ridge 
# mT across celltypes
def disassociation_ridgeplots_per_gene(merged_df, 
                                       genes_to_include, 
                                       celltypes, 
                                       label_mapping, 
                                       file_suffix, 
                                       celltype_color_map, 
                                       outdir):
    
    '''
    * Takes: 
    - merged_df: dataframe containing mT-TCS information on two samples
    - Genes for plotting
    - Celltypes surveying
    - file_suffix --> Not currently being used
    - celltype_color_map: Colors for ridgeplot
    - Outdirectory for daving

    * Outputs:
    - Figure comparing mT values between samples

    '''

    xmin = -5000
    xmax = 25000  
    
    fig = plt.figure(figsize=(12, 10))  
    gs = gridspec.GridSpec(len(celltypes) * len(genes_to_include) + (len(genes_to_include) - 1), 1, hspace=-0.5)
    ax_objs = []
    index = 0

    for g_idx, gene in enumerate(genes_to_include):
        gene_df = merged_df[merged_df["Gene"] == gene]
        
        print(gene_df)
    
        if gene_df.empty:
            continue

        for celltype in celltypes:
            ax = fig.add_subplot(gs[index, 0])
            ax_objs.append(ax)

            # Safely fetch mT and sT from merged_df
            mT_col = f"{celltype}_mT_adj"
            sT_col = f"{celltype}_sT"

            if mT_col not in gene_df.columns or sT_col not in gene_df.columns:
                index += 1
                continue

            try:
                mT = gene_df[mT_col].values[0]
                sT = gene_df[sT_col].values[0]
            except IndexError:
                index += 1
                continue

            color = celltype_color_map.get(celltype, '#FF3232')

            x = np.linspace(xmin, xmax, xmax - xmin + 1)
            y = sp.stats.norm.pdf(x, loc=mT, scale=sT)

            ax.plot(x, y, label=f"{celltype} - {gene}", color=color)
            ax.fill_between(x, y, 0, alpha=0.7, color=color)

            y_max = max(y) * 3
            ax.set_ylim([0, y_max])

            ax.vlines(x=mT, ymin=0, ymax=y_max * 0.1, color='black', linewidth=1, linestyle='--')
            ax.axvline(x=0, color='grey', linewidth=0.6)

            tick_positions = [-5000, 0, 5000, 10000, 15000]
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(['-5000', 'TCS', '5000', '10000', '15000'], fontsize=30)

            ax.set_xlim([xmin, xmax])
            ax.set_yticks([])
            ax.set_ylabel(f'{celltype}\n{gene}', rotation=0, labelpad=50)

            if index == len(celltypes) * len(genes_to_include) + (len(genes_to_include) - 2):
                ax.set_xlabel('Genomic Coordinates Relative to the TCS (bases)', labelpad=10, fontsize=30)
            else:
                ax.set_xticks([])

            index += 1

        if g_idx < len(genes_to_include) - 1:
            index += 1

    for i, ax in enumerate(ax_objs):
        ax.patch.set_alpha(0)
        ax.set_yticklabels([])
        ax.set_ylabel('')
        if i != len(ax_objs) - 1:
            ax.set_xticklabels([])
        for s in ["top", "right", "left", "bottom"]:
            ax.spines[s].set_visible(False)

        condition = label_mapping.get(celltypes[i % len(celltypes)], celltypes[i % len(celltypes)]).replace(" ", "\n")
        gene_name_only = genes_to_include[i // len(celltypes)].split('|')[0]
        condition = condition.replace("\n", " ")
        ax.text(1.12, 0.25, f"{condition}", fontsize=25, ha="center", va='center', transform=ax.transAxes)
        if i == 0:
            ax.text(0.1, 0.40, f"{gene_name_only}", fontsize=30, ha="center", va='center', transform=ax.transAxes)

    # Create a proxy line for μ_T (mean)
    mu_line = plt.Line2D([], [], color='black', linewidth=2, label=r"$\mu_T$", linestyle='--')

    # Add one global legend to the full figure (top right)
    fig.legend(
        handles=[mu_line],
        loc='upper right',
        bbox_to_anchor=(0.9, 0.85),  # (x, y) coords in figure space
        fontsize=30)
    
    plt.tight_layout()
    plt.savefig(f"{outdir}/{gene}-disassociation-modeled-peak.svg")
    plt.show()
    plt.close()
    
