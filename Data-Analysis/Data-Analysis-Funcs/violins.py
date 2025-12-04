# Import
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def violins_across_samples_subset(T_clust_genes, 
                                  GC_clust_genes, 
                                  meta_df, 
                                  genes_to_remove, 
                                  column_prefix, 
                                  subset_palette, 
                                  label, 
                                  savepath_base):
    
    """
    * Takes:
    - T cluster gene list
    - GC cluster gene list
    - Metadata table
    - Genes to remove (testing purposes)
    - LIET prior of interest (mT-adj)
    - subset_palette: colors for two subsets
    - Label
    - Outdir

    * Outputs: 
    - Split violin comparing Non-GC Rich and GC Rich subsets on the same violin

    """

    param_dict = {
        'mT-adj': '$\mu_T$',
        'sT': '$\sigma_T$'
    }

    df = meta_df.copy()
    df = df[~df['Gene'].isin(genes_to_remove)].copy()
    
    GC = df[df["Gene"].isin(GC_clust_genes)]
    GC["Subset"] = "GC-Rich"
    
    T = df[df["Gene"].isin(T_clust_genes)]
    T["Subset"] = "Upstream T-rich"

    df = pd.concat([GC, T])
    
    # balance group sizes  -- same size for each subset 
    gc_rich_df = df[df['Subset'] == 'GC-Rich']
    non_gc_df = df[df['Subset'] == 'Upstream T-rich'].sample(n=len(gc_rich_df), random_state=42)
    
    violin_df = pd.concat([gc_rich_df, non_gc_df], ignore_index=True)

    # dummy x for a single violin
    violin_df['x'] = 'All'
    
    # order rows
    violin_df['Subset'] = pd.Categorical(
    violin_df['Subset'],
    categories=['Upstream T-rich', 'GC-Rich'],
    ordered=True)
    
    fig, ax = plt.subplots(figsize=(3, 4))

    sns.violinplot(
        data=violin_df,
        x='x',                
        y=column_prefix,
        hue='Subset',
        split=True,
        palette=subset_palette,
        inner="quart",
        linewidth=1.2,
        dodge=False,
        ax=ax
    )

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2], labels[:2], fontsize=12, title='')

    violin_y_ticks = ax.get_yticks()
    
    if column_prefix == "mT-adj":
        ax.set_yticklabels(["TCS" if y == 0 else f"{int(y):,}" for y in violin_y_ticks], fontsize=14)
        ax.set_title(f"{param_dict.get(column_prefix, column_prefix)}", fontsize=16)
        ax.set_ylabel(f"{param_dict.get(column_prefix, column_prefix)}-TCS {label}", fontsize=16)
        ax.set_xlabel("Genes", fontsize=16)
        ax.tick_params(axis='x', labelsize=0)
        
    else:
        yticks = ax.get_yticks()
        ax.set_yticklabels([f"{int(y):,}" for y in yticks], fontsize=14)
        ax.set_title(f"{param_dict.get(column_prefix, column_prefix)}", fontsize=16)
        ax.set_ylabel(f"{param_dict.get(column_prefix, column_prefix)} {label}", fontsize=16)
        ax.set_xlabel("Genes", fontsize=16)
        ax.tick_params(axis='x', labelsize=0)  

    plt.tight_layout()
    plt.savefig(f"{savepath_base}")
    plt.show()
    plt.close()
