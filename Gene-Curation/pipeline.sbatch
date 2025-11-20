#!/bin/bash 
#SBATCH --job-name=job_name
#SBATCH -p long
#SBATCH --mail-type=FAIL 
#SBATCH --mail-user=email@xxx.edu
#SBATCH --nodes=1 
#SBATCH --ntasks=32     
#SBATCH --mem=40GB 
#SBATCH --time=20:00:00 
#SBATCH --output=/path-to-eofiles/%x_%j.out
#SBATCH --error=/path-to-eofiless/%x_%j.err

# Display job context #
printf "Run on: $(hostname)"
printf "\nRun from: $(pwd)"
printf "\nScript: $0\n"
date

# Metadata #
printf 'Run on: %s\n' 'fiji-3.colorado.edu'
printf "\nYou've requested $SLURM_CPUS_ON_NODE core(s).\n"

# Load modules/environment #
# Activate conda environment with specified packages in packages.txt file
module purge
source /Users/geba9152/miniconda3/bin/activate
conda activate pymc_5v2

module load samtools/1.8
module load bedtools/2.28.0
module load igvtools/2.3.75

# Set variables #
experiment="Eric-HS"
metadata='/path-to-metadata/metadata-final-hg38.txt'
wd='/path-to-wd/'
genome='/path-to-hg38-genome/hg38/hg38.fa'
chrom_sizes='/path-to-hg38-chrom-sizes/hg38/hg38.chrom.sizes'
res_dir='/path-to-liet-res-dir/'
tfit_path='/path-to-where-tfit-is-installed/Tfit/src/Tfit'
tfit_config="${wd}/tfit-config/tfit_config_file_K1.txt"
tfit_prelim_dir="path-to-tfit-input/tfit-in/${experiment}/"
COMBINED_BG_DIR="path-to-bedgraph-combined-strands/${experiment}/"
THREEP_BG_DIR="path-to-bedgraph-single-strands/${experiment}/"
BGS=(${COMBINED_BG_DIR}/*.bedGraph)

# intersect variables #
overlaps="/scratch/Users/geba9152/LIET-3end-analysis/LIET-Gene-Filtering/bedtools-intersect/${experiment}/"
mkdir -p ${overlaps}

# 1. Get mT centered 3kb windows
python create_bedfiles.py --metadata ${metadata} --wd ${wd} --inputdir ${res_dir} --window 3000 --experiment ${experiment}

# 2. Get mT centered 5kb windows
python create_bedfiles.py --metadata ${metadata} --wd ${wd} --inputdir ${res_dir} --window 5000 --experiment ${experiment}

# 3. Run Tfit and Antisense Coverage 
echo "=============File information============="
echo "Using Tfit prelim bedfile: ${tfit_prelim}"
echo "Using Tfit config file: ${tfit_config}"
echo "Using BGs: ${BGS[@]}"

echo "Only 3' ends: running Tfit only"

# For each BG
for i in "${!BGS[@]}"; do
    echo "============================================================="
    echo "================ BG: ${BGS[i]} ==============="
    # Get prefix
    prefix=$(basename "${BGS[i]}" .3p.combined.bedGraph)
    echo ${prefix}
    exp_prefix="${experiment}_${prefix}"
    echo ${exp_prefix}

    # Get preliminary regions for sample
    tfit_prelim=${tfit_prelim_dir}/5000-window-${exp_prefix}.bed
    antisense_cov_calc=${tfit_prelim_dir}/3000-window-${exp_prefix}.bed

    # Set vars
    tfit_outdir=${wd}/Tfit-out/${experiment}/
    mkdir -p ${tfit_outdir}

    antisense_cov_outdir=${wd}/mT-stranded_cov/${experiment}/
    mkdir -p ${antisense_cov_outdir}

    ###########################
    ###    Running Tfit    ###
    ##########################
    echo "====Running Tfit"
    echo ${tfit_path}
    echo ${tfit_config}
    echo "${BGS[i]}"
    echo ${tfit_prelim}
    echo ${tfit_outdir}

    ${wd}/tfit_model.sh -t ${tfit_path} -c ${tfit_config} -b "${BGS[i]}" -k "${tfit_prelim}" -p "${exp_prefix}" -n 32 
    mv ${wd}/Tfit-out/${exp_prefix}*.* ${tfit_outdir}

    ###########################
    ###    Calc Anti Cov    ### 
    ##########################

    # Calculate coverage for antisense cov filtering
    bedtools coverage -a ${antisense_cov_calc} -b ${THREEP_BG_DIR}/${prefix}.3p.pos.bedGraph > ${antisense_cov_outdir}/${exp_prefix}_tfit_prelim_cov_stranded_3000_window_pos.bed

    bedtools coverage -a ${antisense_cov_calc} -b ${THREEP_BG_DIR}/${prefix}.3p.neg.bedGraph > ${antisense_cov_outdir}/${exp_prefix}_tfit_prelim_cov_stranded_3000_window_neg.bed

done

# 4. Run bedtools intersect to intersect tfit results

tfit_outdir=${wd}/Tfit-out/${experiment}/

# Loop through files in the Tfit-out directory 
for file in "${tfit_outdir}"*_split_bidir_predictions.bed; do

    # Check if the file exists
    if [[ -e "$file" ]]; then
        # Get the base name of the file 
        base_name=$(basename "$file" "_split_bidir_predictions.bed")

        # Define output file path
        intesect_output_file="${overlaps}${base_name}_mT_bidir_intersect.bed"

        # Run bedtools intersect
        bedtools intersect -c -a "${tfit_prelim_dir}5000-window-${base_name}.bed" -b "$file" > "$intesect_output_file"

    else
        echo "Error check file structure"
        break
    fi

done

# 5. Perform the rest of filtering
python gene-filtering.py --metadata ${metadata} \
 --wd ${wd} \
 --inputdir ${res_dir} \
 --experiment ${experiment}
