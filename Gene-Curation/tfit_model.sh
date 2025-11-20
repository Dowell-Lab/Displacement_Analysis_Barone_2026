#!/bin/bash

## Copied from Bidirectional-Flow

##help message for for the script
usage() { echo "Usage: $0 [-t <tfit paths>] [-c <tfit config>] [-b <bedgraph file>] [-k <prelim bed file>] [-p <prefix>] [-n <threads>]" 1>&2; exit 1; }

while getopts ":t:c:b:k:p:n:" arg; do
    case "${arg}" in
        t)
            t=${OPTARG}
            ;;
        c)
            c=${OPTARG}
            ;;
	b)
	    b=${OPTARG}
	    ;;
        k)
            k=${OPTARG}
            ;;
	p)
	    p=${OPTARG}
	    ;;
	n)
	    n=${OPTARG}
	    ;;
	o)
	    o=${OPTARG}
	    ;;
        h | *) ##display help message
            usage
            ;;
    esac
done
shift $((OPTIND-1))

if [ -z "${t}" ] || [ -z "${c}" ] ; then
    usage
fi

##load modules to run tfit
module purge
module load openmpi/1.6.4
module load gcc/7.1.0
module load bedtools/2.28.0

##print and export parameters
o=/scratch/Users/geba9152/LIET-3end-analysis/LIET-Gene-Filtering/Tfit-out/
export "OMP_NUM_THREADS=1"
echo "Number of threads: ${n}"
echo "Run on           : ${SLURM_JOB_NODELIST}"

echo "Tfit path        : ${t}"
echo "Tfit config      : ${c}"
echo "Bedgraph path    : ${b}"
echo "Prelim bed file  : ${k}"
echo "Output prefix    : ${p}"
echo "Output directory : ${o}"

##run tfit!
mpirun ${t} model -config ${c} -ij ${b} -k ${k} -N ${p} -o ${o}


## get bedtools coverage of the part of interest (from lines 640 of main.nf)
# get the now header version
grep -v '#' ${o}/${p}-1_bidir_predictions.bed > ${o}/temp_noheader.bed
# sort the file
cat ${o}/temp_noheader.bed | bedtools sort > ${o}/${p}_split_bidir_predictions.bed
# get the coverage
bedtools coverage -a ${o}/${p}_split_bidir_predictions.bed -b ${b} > ${o}/${p}_split_bidir_cov.bed
rm ${o}/temp_noheader.bed

