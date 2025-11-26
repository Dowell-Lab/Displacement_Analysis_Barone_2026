#!/bin/bash
##SBATCH --job-name=job_name
##SBATCH --mail-type=ALL
##SBATCH --mail-user=email@xxx.edu
##SBATCH --nodes=1
##SBATCH --cpus-per-task=1
##SBATCH --mem=5GB
##SBATCH --time=01:00:00
##SBATCH --partition=short
##SBATCH --output=/path-to-eofiles/%x_%j.out
##SBATCH --error=/path-to-eofiles/%x_%j.err

# Load metadata #
echo Job: $SLURM_JOB_NAME with ID $SLURM_JOB_ID
echo Running on host `hostname`
echo Job started at `date +"%T %a %d %b %Y"`
echo Directory is `pwd`
echo Using $SLURM_NTASKS processors, across $SLURM_NNODES nodes, with $SLURM_JOB_CPUS_PER_NODE cpus per node

# Load modules #
module load bedtools/2.25.0
module load hdf5/1.10.5

# Paths/variables #
#species=${1} # eg: panTro6
#species_name=${2} #eg: Pan_troglodytes
species="rheMac10"
species_name="Macaca_mulatta"

# Uncomment out based on which lift over file using #
#hal_file="/scratch/Shares/public/zoonomia/241-mammalian-2020v2.ha"
hal_file="/scratch/Shares/public/zoonomia/447-mammalian-2022v1.hal"
inputBED="/scratch/Users/geba9152/LIET-3end-analysis/daniel-animal-liet/ortholog-output/MANE-annotation-overlaps-across-all-species/overlapping_orthologs_Hal_format.MANE.bed"
halexe="/Users/geba9152/repos/hal/bin/"
dir="/Shares/txpnevol/geba9152/447-mammalian-hal-liftOver/${species}/"
chromAlias="/Shares/down/public/omics_databases/zoonomia/genome_alias/"

# Create directories
mkdir -p tmp_species1
mkdir -p tmp_species2

# Loop through each gene in the BED file
for gene in `cut -f4 ${inputBED}`
do
        # Separates the initial BED file onto a single BED file per line.
	grep -w "$gene" "$inputBED" > "${dir}/tmp_species1/${gene}.bed"

	# Check if the temporary file is non-empty
    	if [ -s "${dir}/tmp_species1/${gene}.bed" ]; then
            # Perform HAL liftover
            #${halexe}halLiftover "$hal_file" Homo_sapiens "${dir}/tmp_species1/${gene}.bed" ${species_name} "${dir}/tmp_species2/${gene}.lifted.bed"

            # Sorts the individual output BED files.
            sort \
            -k 1,1 -k2,2n \
            "${dir}/tmp_species2/${gene}.lifted.bed" \
            > "${dir}/tmp_species2/${gene}.tmp2.bed"

            # Merges all the entries that are within 1000 bp of each other.
            mergeBed \
            -s \
            -d 5000 \
            -i ${dir}/tmp_species2/${gene}.tmp2.bed \
            > ${dir}/tmp_species2/${gene}.tmp3.bed


            # Adds 5th column with the length of each interval.
            awk -v OFS='\t' '{print $1, $2, $3, $4, $3-$2}' \
            ${dir}/tmp_species2/${gene}.tmp3.bed \
            > ${dir}/tmp_species2/${gene}.tmp4.bed


            # Sorts intervals by their length (5th column) in descending order.
            sort -t $'\t' -k5,5rn \
            ${dir}/tmp_species2/${gene}.tmp4.bed \
            > ${dir}/tmp_species2/${gene}.tmp5.bed

            # Keeps only the biggest merged region per BED file.
            head -1 ${dir}/tmp_species2/${gene}.tmp5.bed \
            > ${dir}/tmp_species2/${gene}.tmp6.bed


            # Adds the entry name as the last column.
            awk -v OFS='\t' '{print $1, $2, $3, $4}' \
            ${dir}/tmp_species2/${gene}.tmp6.bed \
            > ${dir}/tmp_species2/${gene}.ready.bed
            sed -i "s/$/\t${gene}/g" ${dir}/tmp_species2/${gene}.ready.bed
	fi

done


# Combines all individual merged BED files onto a single file.
printf '%s\0' ${dir}/tmp_species2/*.ready.bed | xargs -r0 cat -- > ${dir}/merged.hg38_${species}.bed

# Renames chromosome names.
for ID in `cut -f1 ${dir}/merged.hg38_${species}.bed | sort | uniq`
        do
        newID=`grep ${ID} ${chromAlias}/${species}.chromAlias.txt | cut -f1`
        sed -i "s/${ID}/${newID}/g" ${dir}/merged.hg38_${species}.bed
done










