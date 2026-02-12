#!/bin/bash
set -evx
cd new_data/dataset
for dir in */; do
    if [ -d "$dir" ]; then
        mkdir ../dataset_conll/$dir -p
		mkdir ../dataset_bio/$dir -p
        mkdir ../concepts_bio/$dir -p
        mkdir ../tagged_concepts_bio/$dir -p
		cd "$dir"
		for course in */; do
            if [ -d "$course" ]; then
                mkdir ../../dataset_conll/$dir/$course -p
                mkdir ../../dataset_bio/$dir/$course -p
                mkdir ../../dataset_bio_uniform/$dir/$course -p
                mkdir ../../concepts_bio/$dir/$course -p
                mkdir ../../tagged_concepts_bio/$dir/$course -p
                # mkdir ../../dataset_bio/distant/$course -p
                cd "$course"
                for file in *.jsonl; do
                    #../../../../to_conll.py -f $file -o ../../../dataset_conll/$dir$course"${file%.*}".conll
                    #../../../../to_bio_json.py -f "../../../dataset_conll/$dir$course${file%.*}".conll -o ../../../dataset_bio/$dir$course"${file%.*}".json
                    ../../../../concept_scripts/extract_concepts.py -f "../../../dataset_bio/$dir$course${file%.*}".json -o ../../../tagged_concepts_bio/$dir$course"${file%.*}".txt
                    ../../../../concept_scripts/extract_concepts_tagged.py -f "../../../dataset_bio/$dir$course${file%.*}".json -o ../../../tagged_concepts_bio/$dir$course"${file%.*}".txt
                    #../../../../concept_scripts/distant_label.py -f "../../../dataset_bio/unlabeled/$course${file%.*}".json -o "../../../dataset_bio/distant/$dir$course${file%.*}".json -d ../../../concepts/distant/$course"${file%.*}".json
                done
            fi
            cd ..
        done
        cd ..
    fi
done
