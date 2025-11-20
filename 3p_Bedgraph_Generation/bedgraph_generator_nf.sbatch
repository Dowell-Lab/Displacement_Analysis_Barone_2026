#!/bin/bash        
#SBATCH --job-name=bedgraph_generator
#SBATCH --output=/scratch/Users/geba9152/LIET-3end-analysis/bedGraph_generation/eofiles/%x_%j.out                     
#SBATCH --error=/scratch/Users/geba9152/LIET-3end-analysis/bedGraph_generation/eofiles/%x_%j.err
#SBATCH --mail-user=geba9152@colorado.edu
#SBATCH --mail-type=FAIL
#SBATCH -p long
#SBATCH --time=90:00:00      
#SBATCH -N 1
#SBATCH -c 1         
#SBATCH --mem=2gb            

### Display job context
printf "Run on: $(hostname)"
printf "\nRun from: $(pwd)"
printf "\nScript: $0\n"
date

# Metadata
printf 'Run on: %s\n' 'fiji-3.colorado.edu'
printf "\nYou've requested $SLURM_CPUS_ON_NODE core(s).\n"

### Clear modules and load conda environment
module purge

# Required Modules
module load samtools/1.8
module load bedtools/2.28.0
module load igvtools/2.3.75

### Script filename
SCRIPTPATH='/scratch/Users/geba9152/LIET-3end-analysis/bedGraph_generation/'
SCRIPTNAME='bedgraph_generator_nf.sbatch'

printf "\n######################SCRIPT############################\n"
cat $SCRIPTPATH$SCRIPTNAME
printf "\n########################################################\n"

# Run the NF pipeline to generate bedGraphs and LIET config files (separate for single/paired ends)
echo 'Running pipeline'
NF_EXE='/scratch/Shares/dowell/dbnascent/pipeline_assets/nextflow'
NF_MAIN=$SCRIPTPATH'main.nf'

${NF_EXE} ${NF_MAIN}
echo 'Pipeline complete'

date
