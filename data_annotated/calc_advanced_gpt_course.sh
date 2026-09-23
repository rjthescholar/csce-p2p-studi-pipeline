#!/bin/bash -i

alias calculate_concept_metrics='python /home/awesomerek/UniversityOfPittsburgh/ConceptExtractionPaper/gpt_o3_distant_labelling/calculate_concept_metrics.py'

set -x

calculate_concept_metrics ./new_data/concepts/advanced-gpt/outputs-1502/cs1502.txt ./new_data/concepts_bio/gold_all/gold_cscs1502_all_concepts.txt
calculate_concept_metrics ./new_data/concepts/advanced-gpt/outputs-0441/cs0441.txt ./new_data/concepts_bio/gold_all/gold_cscs0441_all_concepts.txt
calculate_concept_metrics ./new_data/concepts/advanced-gpt/outputs-0007/cs0007.txt ./new_data/concepts_bio/gold_all/gold_cscs0007_all_concepts.txt
calculate_concept_metrics ./new_data/concepts/advanced-gpt/outputs-1567/cs1567.txt ./new_data/concepts_bio/gold_all/gold_cscs1567_all_concepts.txt
calculate_concept_metrics ./new_data/concepts/advanced-gpt/outputs-1622/cs1622.txt ./new_data/concepts_bio/gold_all/gold_cscs1622_all_concepts.txt
calculate_concept_metrics ./new_data/concepts/advanced-gpt/outputs-1550/cs1550.txt ./new_data/concepts_bio/gold_all/gold_cscs1550_all_concepts.txt
calculate_concept_metrics ./new_data/concepts/advanced-gpt/outputs-0449/cs0449.txt ./new_data/concepts_bio/gold_all/gold_cscs0449_all_concepts.txt

