#!/bin/bash -i

alias calculate_concept_metrics='python /home/awesomerek/UniversityOfPittsburgh/ConceptExtractionPaper/gpt_o3_distant_labelling/calculate_concept_metrics.py'

set -x

calculate_concept_metrics ./concepts_bio/kpe/cs1502.txt ./concepts_bio/gold_all/gold_cscs1502_all_concepts.txt
calculate_concept_metrics ./concepts_bio/kpe/cs0441.txt ./concepts_bio/gold_all/gold_cscs0441_all_concepts.txt
calculate_concept_metrics ./concepts_bio/kpe/cs0007.txt ./concepts_bio/gold_all/gold_cscs0007_all_concepts.txt
calculate_concept_metrics ./concepts_bio/kpe/cs1567.txt ./concepts_bio/gold_all/gold_cscs1567_all_concepts.txt
calculate_concept_metrics ./concepts_bio/kpe/cs1622.txt ./concepts_bio/gold_all/gold_cscs1622_all_concepts.txt
calculate_concept_metrics ./concepts_bio/kpe/cs1550.txt ./concepts_bio/gold_all/gold_cscs1550_all_concepts.txt
calculate_concept_metrics ./concepts_bio/kpe/cs0447.txt ./concepts_bio/gold_all/gold_cscs0447_all_concepts.txt
calculate_concept_metrics ./concepts_bio/kpe/cs1541.txt ./concepts_bio/gold_all/gold_cscs1541_all_concepts.txt

