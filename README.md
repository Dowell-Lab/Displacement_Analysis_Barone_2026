# Disassociation Meta-Analysis

This repository contains data processing scripts from Barone et al. (2025). New sequencing data mentioned in this paper can be found at the following GEO accession numbers:

| Description | Accession Numbers |
|-------------|-------------------|
| Arsenic PRO-seq control experiments | GSM9321236, GSM9321237 |
| Arsenic PRO-seq perturbed experiments | GSM9321238, GSM9321239 |
| Human PRO-seq experiments | GSM6716757, GSM6716758 |
| Gibbon monkey PRO-seq experiments | GSM6716755, GSM6716756 |
| Rhesus monkey PRO-seq experiments | GSM6716759, GSM6716760 |
| Squirrel monkey PRO-seq experiments | GSM6716761, GSM6716762 |
| ATAC-seq experiment | GSM5269423 |

## Directory Contents

### `3p_Bedgraph_Generation/`

Nextflow pipeline for generating bedgraphs with the 3'-most position of reads.

**Files:**
- **`main.nf`**: Main Nextflow pipeline
- **`bedgraph_generator_nf.sbatch`**: Example SBATCH submission script
- **`bedgraph_gen_samples_BATCH0.csv`**: Example input CSV file

---

### `Cross-Species-liftOver/`

Scripts for cross-species analyses.

**Files:**
- **`busco.sbatch`**: Example script for running Benchmarking Universal Single-Copy Orthologs (BUSCO) [1]
- **`Hal-liftover.sbatch`**: Example script for utilizing halLiftover to lift over annotations from one species to another using Progressive Cactus multiple-genome aligner [2,3]. Installation: [HAL toolkit](https://github.com/ComparativeGenomicsToolkit/hal)
- **`LIET-Input-File-Generation-Cross-Species.ipynb`**: Jupyter Notebook for generating LIET [4] input files (cross-species)

---

### `Data-Analysis/`

Data analysis and visualization code.

**Files:**
- **`Figure-Analysis.ipynb`**: Jupyter Notebook containing all figure generation code and analyses

---

### `Gene-Curation/`

Scripts for gene filtering and curation workflow.

**Files:**
- **`LIET-Input-File-Generation-Human.ipynb`**: Jupyter Notebook for generating LIET [4] input files (human experiments)
- **`annotation-isolation-filter.sbatch`**: Script for identifying annotation-isolated genes (pre-emptive step before running LIET)
- **`create_bedfiles.py`**: Python script that creates BED files centered on $\mu_T$
- **`gene-filtering.py`**: Python script that performs the majority of gene curation (filters genes post-LIET run)
- **`pipeline.sbatch`**: SBATCH script that runs the gene curation pipeline
- **`tfit_model.sh`** & **`tfit-config/`**: Scripts for running Tfit on constrained regions ($\mu_T$ regions)

> **Note:** Gene filtering in the cross-species analyses follows the same procedure as the human analysis, but omits the annotation isolation step. Additionally, each subdirectory contains a `packages.txt` file with required software versions and dependencies.

References:
1. Manni, M., Berkeley, M.R., Seppey, M., Simao, F.A., Zdobnov, E.M.: Busco update: Novel and streamlined workflows along with broader and deeper phylogenetic coverage for scoring of eukaryotic, prokaryotic, and viral genomes. Molecular Biology and Evolution 38(10), 4647–4654 (2021) https://doi.org/10.1093/molbev/msab199
2. Zoonomia Consortium: A comparative genomics multitool for scientific discovery and conservation. Nature 587, 240–245 (2020) https://doi.org/10.1038/s41586-020-2876-6
3. Kuderna, L.F.K., Ulirsch, J.C., Rashid, S., et al.: Identification of constrained sequence elements across 239 primate genomes. Nature 625, 735–742 (2024) https://doi.org/10.1038/s41586-023-06798-8
4. Stanley, J.T., Barone, G.E.F., Townsend, H.A., Sigauke, R.F., Allen, M.A., Dowell, R.D.: LIET model: capturing the kinetics of RNA polymerase from loading to termination. Nucleic Acids Res 53(7), 246 (2025) https://doi.org/10.1093/nar/gkaf246
