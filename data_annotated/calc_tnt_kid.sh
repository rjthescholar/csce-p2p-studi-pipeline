#!/bin/bash -i

alias calculate_concept_metrics='python /home/awesomerek/UniversityOfPittsburgh/ConceptExtractionPaper/gpt_o3_distant_labelling/calculate_concept_metrics.py'

set -x

echo "CS-1502 RES"
calculate_concept_metrics ./concepts_bio/tnt-kid/cs1502.txt ./concepts_bio/gold_all/gold_cscs1502_all_concepts.txt
echo "CS-0441 RES"
calculate_concept_metrics ./concepts_bio/tnt-kid/cs0441.txt ./concepts_bio/gold_all/gold_cscs0441_all_concepts.txt
echo "CS-0007 RES"
calculate_concept_metrics ./concepts_bio/tnt-kid/cs0007.txt ./concepts_bio/gold_all/gold_cscs0007_all_concepts.txt
echo "CS-1567 RES"
calculate_concept_metrics ./concepts_bio/tnt-kid/cs1567.txt ./concepts_bio/gold_all/gold_cscs1567_all_concepts.txt
echo "CS-1622 RES"
calculate_concept_metrics ./concepts_bio/tnt-kid/cs1622.txt ./concepts_bio/gold_all/gold_cscs1622_all_concepts.txt
echo "CS-1550 RES"
calculate_concept_metrics ./concepts_bio/tnt-kid/cs1550.txt ./concepts_bio/gold_all/gold_cscs1550_all_concepts.txt
echo "CS-0447 RES"
calculate_concept_metrics ./concepts_bio/tnt-kid/cs0447.txt ./concepts_bio/gold_all/gold_cscs0447_all_concepts.txt
echo "CS-1541 RES"
calculate_concept_metrics ./concepts_bio/tnt-kid/cs1541.txt ./concepts_bio/gold_all/gold_cscs1541_all_concepts.txt

