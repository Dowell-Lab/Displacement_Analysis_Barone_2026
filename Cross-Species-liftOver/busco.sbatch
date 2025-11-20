#!/bin/bash
#SBATCH --job-name=busco
#SBATCH --output=/scratch/Users/geba9152/LIET-3end-analysis/daniel-animal-liet/eofiles/%x_%j.out
#SBATCH --error=/scratch/Users/geba9152/LIET-3end-analysis/daniel-animal-liet/eofiles/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=12
#SBATCH --mem=50gb
#SBATCH --partition highmem
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=geba9152@colorado.edu

# Load metadata #
echo Job: $SLURM_JOB_NAME with ID $SLURM_JOB_ID
echo Running on host `hostname`
echo Job started at `date +"%T %a %d %b %Y"`
echo Directory is `pwd`
echo Using $SLURM_NTASKS processors, across $SLURM_NNODES nodes, with $SLURM_JOB_CPUS_PER_NODE cpus per node

# Load modules #
module purge
source /Users/geba9152/miniconda3/bin/activate
conda activate busco

# Define the input and output directories #
SPECIES=${1}
GENOME="/Shares/txpnevol/genomes/${SPECIES}/${SPECIES}.fa"

# Run busco
busco -i ${GENOME} -l mammalia_odb10 -o busco-${SPECIES}-res -m genome
