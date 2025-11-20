#!/bin/bash
#SBATCH --job-name=job_name
#SBATCH --output=/path-to-eofiles/%x_%j.out
#SBATCH --error=/path-to-eofiless/%x_%j.err
#SBATCH --time=024:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=45gb
#SBATCH --partition long
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=email@xxx.edu

# Display job context #
printf "Run on: $(hostname)"
printf "\nRun from: $(pwd)"
printf "\nScript: $0\n"
date

# Metadata #
printf 'Run on: %s\n' 'fiji-3.colorado.edu'
printf "\nYou've requested $SLURM_CPUS_ON_NODE core(s).\n"

# Load modules/environment #
module purge
module load bedtools

# Variable assignment #
FIVE_PAD=3000
THREE_PAD=20000
BASE_DIR="path-to-bedgraphs--combined-stranded-file"
BED_DIR="path-to-annotation-file/transcript.MANE.refseq.bed"
OUTPUT_DIR="path-to-output-dir"
GENOME="path-to-genome/hg38/hg38.chrom.sizes"
WD="path-to-working-directory"
SLOP_OUT="${WD}padded-${FIVE_PAD}-5p-${THREE_PAD}-3p-transcript.MANE.bed"
CANONICAL_SLOP_OUT="${WD}padded-${FIVE_PAD}-5p-${THREE_PAD}-3p-transcript-canonical-chr.MANE.bed"
SORTED_SLOP_OUT="${WD}padded-${FIVE_PAD}-5p-${THREE_PAD}-3p-transcript.MANE.sorted.bed"
ANNOTATION_ISO_OUT="${WD}isolated-${FIVE_PAD}-5p-${THREE_PAD}-3p-transcript.MANE.bed"
SORTED_ANNOTATION_ISO_OUT="${WD}isolated-${FIVE_PAD}-5p-${THREE_PAD}-3p-transcript.MANE.sorted.bed"

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# Bedtools slop to get padded regions
bedtools slop -i ${BED_DIR} -g ${GENOME} -l ${FIVE_PAD} -r ${THREE_PAD} -s > ${SLOP_OUT}

# 2. Get rid of non-canonical chromosomes
grep -v -E '_fix|_alt' ${SLOP_OUT} > ${CANONICAL_SLOP_OUT}

# 3. sort slop output
bedtools sort -i ${CANONICAL_SLOP_OUT} > ${SORTED_SLOP_OUT}

# 4. bedtools merge
# Notes on how to perform this operation are here: https://stataiml.com/posts/81_filter_ovlp_regions_bedtools_bio/#:~:text=Here%2C%20you%20can%20use%20bedtools,only%20keep%20non%2Doverlapping%20regions.&text=In%20the%20above%20command%2C%20we,be%20applied%20to%20that%20column.

bedtools merge -i ${SORTED_SLOP_OUT} -c 2,4 -o count_distinct,collapse | awk '{OFS="\t"}{if ($4 == 1) print $1,$2, $3, $5}' > ${ANNOTATION_ISO_OUT}

# 4. sort merge output
bedtools sort -i ${ANNOTATION_ISO_OUT} > ${SORTED_ANNOTATION_ISO_OUT}


