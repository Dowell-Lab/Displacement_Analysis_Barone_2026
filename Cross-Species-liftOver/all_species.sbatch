#!/bin/bash
#SBATCH --job-name=hal-termination-align
#SBATCH --mail-type=ALL
#SBATCH --mail-user=geba9152@colorado.edu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20GB
#SBATCH --time=06:00:00
#SBATCH --partition=short
#SBATCH --output=/scratch/Users/geba9152/LIET-3end-analysis/alignments-revisions/eofiles/%x.%j.out
#SBATCH --error=/scratch/Users/geba9152/LIET-3end-analysis/alignments-revisions/eofiles/%x.%j.err

echo "Job: $SLURM_JOB_NAME with ID $SLURM_JOB_ID"
echo "Running on host $(hostname)"
echo "Job started at $(date +"%T %a %d %b %Y")"
echo "Directory is $(pwd)"
echo "Using $SLURM_NTASKS processors, across $SLURM_NNODES nodes, with $SLURM_JOB_CPUS_PER_NODE cpus per node"

# ── Modules ───────────────────────────────────────────────────────────────────
module load bedtools/2.25.0
module load hdf5/1.10.5

# ── Paths ─────────────────────────────────────────────────────────────────────
halexe="/Users/geba9152/repos/hal/bin/"
hal_447="/scratch/Shares/public/zoonomia/447-mammalian-2022v1.hal"
hal_241="/scratch/Shares/public/zoonomia/241-mammalian-2020v2.hal"

# ─── Configuration ──────────────────────────────────────────────────────────
# The directory where your Python script saved the BED files
BED_DIR="/scratch/Users/geba9152/LIET-3end-analysis/alignments-revisions/15kbupPAS_15kbintoTSS/"
HAL_FILE="/scratch/Shares/public/zoonomia/241-mammalian-2020v2.hal"  
HAL_EXE="/Users/geba9152/repos/hal/bin/"

# Define the reference (the master list of genes you want to align)
#REF_LABEL="hg38"
#REF_LABEL="saiBol1"
REF_LABEL="nomLeu3"
#REF_LABEL="rheMac8"
REF_BED="${BED_DIR}/${REF_LABEL}_15kb_neighborhood.bed"

# Species mapping: Map your BED labels to HAL genome names
# Verify these names with: halStats --species "$HAL_FILE"
declare -A HAL_NAMES
HAL_NAMES["hg38"]="Homo_sapiens"
HAL_NAMES["rheMac8"]="Macaca_mulatta"
HAL_NAMES["nomLeu3"]="Nomascus_leucogenys"
HAL_NAMES["saiBol1"]="Saimiri_boliviensis"

REF_HAL_NAME=${HAL_NAMES[$REF_LABEL]}
#TARGET_SPECIES=("rheMac8" "nomLeu3" "saiBol1")
TARGET_SPECIES=("rheMac8" "saiBol1" "hg38")
#TARGET_SPECIES=("nomLeu3" "hg38" "saiBol1")

# Output Setup
#WORK_DIR="/scratch/Users/geba9152/LIET-3end-analysis/alignments-revisions/hal_alignment_output_noreq_same_gene_ds_15kb/"
#WORK_DIR="/scratch/Users/geba9152/LIET-3end-analysis/alignments-revisions/hal_alignment_output_noreq_same_gene_ds_15kb_rhesus_ref/"
WORK_DIR="/scratch/Users/geba9152/LIET-3end-analysis/alignments-revisions/hal_alignment_output_noreq_same_gene_ds_15kb_gibbon_ref/"
mkdir -p "${WORK_DIR}"
mkdir -p "${WORK_DIR}/tmp_source"
mkdir -p "${WORK_DIR}/maf"
mkdir -p "${WORK_DIR}/beds"

# ─── Per-Gene Loop ──────────────────────────────────────────────────────────
# We iterate over every gene in your conserved human list
for gene in $(cut -f4 "$REF_BED"); do

    echo "--- Processing Gene: $gene ---"

    # Extract the human window for this specific gene
    grep -w "$gene" "$REF_BED" > "${WORK_DIR}/tmp_source/${gene}.bed"

    if [ ! -s "${WORK_DIR}/tmp_source/${gene}.bed" ]; then
        continue
    fi

    # ── 1. LiftOver to target species ───────────────────────────────────────
    # This ensures we find the exact orthologous neighborhood in the other genomes
    for sp_label in "${TARGET_SPECIES[@]}"; do
        target_hal=${HAL_NAMES[$sp_label]}
        mkdir -p "${WORK_DIR}/beds/${target_hal}"

        ${HAL_EXE}halLiftover \
            "$HAL_FILE" \
            "$REF_HAL_NAME" \
            "${WORK_DIR}/tmp_source/${gene}.bed" \
            "$target_hal" \
            "${WORK_DIR}/beds/${target_hal}/${gene}.lifted.bed"

        # If liftover creates multiple fragments, we clean them up
        if [ -s "${WORK_DIR}/beds/${target_hal}/${gene}.lifted.bed" ]; then
            sort -k1,1 -k2,2n "${WORK_DIR}/beds/${target_hal}/${gene}.lifted.bed" | \
            bedtools merge -d 5000 -i stdin | \
            awk -v gene="$gene" -v OFS='\t' '{print $1,$2,$3,gene,"0","."}' \
            > "${WORK_DIR}/beds/${target_hal}/${gene}.ready.bed"
        fi
    done

    # ── 2. Extract Multiple Alignment (MAF) ─────────────────────────────────
    # This pulls the actual sequences for all species based on the human window
    # We use a comma-separated list of HAL names for --targetGenomes
    TARGET_GENOMES=$(IFS=,; echo "Macaca_mulatta,Nomascus_leucogenys,Saimiri_boliviensis")

    ${HAL_EXE}hal2maf \
        "$HAL_FILE" \
        "${WORK_DIR}/maf/${gene}.maf" \
        --refGenome "$REF_HAL_NAME" \
        --refTargets "${WORK_DIR}/tmp_source/${gene}.bed" \
        --targetGenomes "${TARGET_GENOMES},${REF_HAL_NAME}" \
        --noAncestors

done

# ─── Final Cleanup ──────────────────────────────────────────────────────────
# Combine all per-gene MAFs into one large project file
echo "Combining MAF files..."
merged_maf="${WORK_DIR}/all_conserved_genes.maf"
first=true
for maf in "${WORK_DIR}/maf/"*.maf; do
    if $first; then
        cat "$maf" > "$merged_maf"
        first=false
    else
        grep -v "^##maf" "$maf" >> "$merged_maf"
    fi
done

echo "Done. Final alignment: $merged_maf"
