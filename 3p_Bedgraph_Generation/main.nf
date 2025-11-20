#!/usr/bin/env nextflow

// Pipeline context
nextflow.enable.dsl=2

inputCsv = "/scratch/Users/geba9152/LIET-3end-analysis/bedGraph_generation/bedgraph_gen_samples_BATCH0.csv"
genomeRef = "/scratch/Shares/dowell/genomes/mm10/mm10.fa"
chromSizes = "/scratch/Shares/dowell/genomes/mm10/mm10.chrom.sizes"
outputDir = "/scratch/Users/geba9152/LIET-3end-analysis/bedGraph_generation/temp/"

// Define the process to convert CRAM to BAM
process convertToBam {
    cpus 16 
    executor 'slurm'
    queue 'long'
    memory '40 GB'

    input:
    tuple path(cram), val(single_paired), val(forward_reverse)

    output:
    tuple path("${cram.simpleName}_${single_paired}_${forward_reverse}.bam"), val(single_paired), val(forward_reverse)

    script:
    """
    # BAM generation for single-end samples
    if [ "$single_paired" == "SE" ]; then
        samtools view \
            -@ 16 -h --bam \
            -T $genomeRef \
            $cram > ${cram.simpleName}_${single_paired}_${forward_reverse}.bam
    # BAM generation for paired-end forward-stranded samples
    elif [ "$single_paired" == "PE" ] && [ "$forward_reverse" == "F" ]; then
        samtools view \
            -@ 16 -h --bam -f 128 \
            -T $genomeRef \
            $cram > ${cram.simpleName}_${single_paired}_${forward_reverse}.bam
    # BAM generation for paired-end reverse-stranded samples
    elif [ "$single_paired" == "PE" ] && [ "$forward_reverse" == "R" ]; then
        samtools view \
            -@ 16 -h --bam -f 64 \
            -T $genomeRef \
            $cram > ${cram.simpleName}_${single_paired}_${forward_reverse}.bam
    # Check contains of your input sample file
    else
        echo "Invalid combination of end and orientation"
    fi
    """
}
// NOTE: variable.baseName only removes the endmost extension. variable.simpleName removes all extensions.


// Define the process to create 3'end bedGraphs for fitting
process create3pBedGraphsAndTDF {
    executor 'slurm'
    queue 'long'
    memory '40 GB'
    
    publishDir "$outputDir", mode: 'copy', pattern: "*_3p.pos.bedGraph"
    publishDir "$outputDir", mode: 'copy', pattern: "*_3p.neg.bedGraph"
    publishDir "$outputDir", mode: 'copy', pattern: "*_3p.tdf"

    input:
    tuple path(bamFile), val(single_paired), val(forward_reverse)

    output:
    path "${bamFile.baseName}_3p.pos.bedGraph"
    path "${bamFile.baseName}_3p.neg.bedGraph"
    path "${bamFile.simpleName}_3p.tdf"

    script:
    """
    # bedGraph generation for single-end samples
    if [ "$single_paired" == "SE" ]; then
        genomeCoverageBed \
            -bg -3 \
            -strand + \
            -g $chromSizes \
            -ibam $bamFile \
            | sortBed > "${bamFile.baseName}_3p.pos.bedGraph"

        genomeCoverageBed \
            -bg -3 \
            -strand - \
            -g $chromSizes \
            -ibam $bamFile \
            | awk 'BEGIN{FS=OFS="\t"} {\$4=-\$4}1' \
            | sortBed > "${bamFile.baseName}_3p.neg.bedGraph"

#| awk 'BEGIN{FS=OFS="\t"} {\$4=-\$4}1' > "${bamFile.baseName}_3p.neg.bedGraph"

    # bedGraph generation for paired-end samples
    elif [ "$single_paired" == "PE" ]; then
       genomeCoverageBed \
            -bg -5 \
            -strand - \
            -g $chromSizes \
            -ibam $bamFile \
            | sortBed > "${bamFile.baseName}_3p.pos.bedGraph"

       genomeCoverageBed \
            -bg -5 \
            -strand + \
            -g $chromSizes \
            -ibam $bamFile \
            | awk 'BEGIN{FS=OFS="\t"} {\$4=-\$4}1' \
            | sortBed > "${bamFile.baseName}_3p.neg.bedGraph"

    # Check contents of your input sample file
    else
        echo "Invalid combination of end and orientation"
    fi
    
    # Combine positive and negative strand bedGraphs
    cat "${bamFile.baseName}_3p.pos.bedGraph" "${bamFile.baseName}_3p.neg.bedGraph" \
    | sortBed > "combined_${bamFile.simpleName}_3p.bedGraph"

    # Generate a single TDF file from the combined bedGraph
    igvtools toTDF "combined_${bamFile.simpleName}_3p.bedGraph" "${bamFile.simpleName}_3p.tdf" $chromSizes
    """
}



// Define the process to create full-read bedGraphs for fitting
process createFullBedGraphsAndTDF {
    executor 'slurm'
    queue 'long'
    memory '40 GB'

    publishDir "$outputDir", mode: 'copy', pattern: "*_full.pos.bedGraph"
    publishDir "$outputDir", mode: 'copy', pattern: "*_full.neg.bedGraph"
    publishDir "$outputDir", mode: 'copy', pattern: "*_full.tdf"

    input:
    tuple path(bamFile), val(single_paired), val(forward_reverse)

    output:
    path "${bamFile.baseName}_full.pos.bedGraph"
    path "${bamFile.baseName}_full.neg.bedGraph"
    path "${bamFile.simpleName}_full.tdf"

    script:
    """
    # bedGraph generation for single-end samples
    if [ "$single_paired" == "SE" ]; then
        genomeCoverageBed \
            -bg \
            -strand + \
            -g $chromSizes \
            -ibam $bamFile \
            | sortBed > "${bamFile.baseName}_full.pos.bedGraph"

        genomeCoverageBed \
            -bg \
            -strand - \
            -g $chromSizes \
            -ibam $bamFile \
            | awk 'BEGIN{FS=OFS="\t"} {\$4=-\$4}1' \
            | sortBed > "${bamFile.baseName}_full.neg.bedGraph"

    # bedGraph generation for paired-end samples
    elif [ "$single_paired" == "PE" ]; then
       genomeCoverageBed \
            -bg \
            -strand - \
            -g $chromSizes \
            -ibam $bamFile \
            | sortBed > "${bamFile.baseName}_full.pos.bedGraph"

       genomeCoverageBed \
            -bg \
            -strand + \
            -g $chromSizes \
            -ibam $bamFile \
            | awk 'BEGIN{FS=OFS="\t"} {\$4=-\$4}1' \
            | sortBed > "${bamFile.baseName}_full.neg.bedGraph"

    # Check contents of your input sample file
    else
        echo "Invalid combination of end and orientation"
    fi

    # Combine positive and negative strand bedGraphs
    cat "${bamFile.baseName}_full.pos.bedGraph" "${bamFile.baseName}_full.neg.bedGraph" \
    | sortBed > "combined_${bamFile.simpleName}_full.bedGraph"

    # Generate a single TDF file from the combined bedGraph
    igvtools toTDF "combined_${bamFile.simpleName}_full.bedGraph" "${bamFile.simpleName}_full.tdf" $chromSizes
    """
}




// Define the workflow
workflow {
    // Read the CSV file and convert it to a list of tuples
    // NOTE: CSV line mapping field names MUST match header line names if <header:true>
    samples = Channel \
        .fromPath(inputCsv) \
        .splitCsv(header:true) \
        .map { row-> tuple(row.cram, row.single_paired, row.forward_reverse) }


    // Use the defined processes with the list of tuples
    convertedBams = convertToBam(samples)


    // Use the defined process to create 3'end bedGraphs (for fitting)
    bedGraphs3p = create3pBedGraphsAndTDF(convertedBams)

    // Use the defined process to create full-read spanning bedGraphs (for visualization)
    bedGraphsFull = createFullBedGraphsAndTDF(convertedBams)

    // Generate both 3p and full-read TDF's from combined bedGraphs
    //generateCombinedTDF(bedGraphs3p)
    //generateCombinedFullTDF(bedGraphsFull)
}

