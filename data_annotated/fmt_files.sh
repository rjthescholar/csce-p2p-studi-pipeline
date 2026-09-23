#!/bin/bash
set -evx
cd labeled_data
for dir in */; do
    if [ -d "$dir" ]; then
		mkdir ../labeled_data_bio/$dir -p
        mkdir ../concepts_bio/$dir -p
        mkdir ../tagged_concepts_bio/$dir -p
		cd "$dir"
		for course in */; do
            if [ -d "$course" ]; then
                mkdir ../../labeled_data_bio/$dir/$course -p
                mkdir ../../concepts_bio/$dir/$course -p
                mkdir ../../tagged_concepts_bio/$dir/$course -p
                # mkdir ../../labeled_data_bio/distant/$course -p
                cd "$course"
                for file in *.conll; do
                    ../../../to_bio_json.py -f "../../../labeled_data/$dir$course${file%.*}".conll -o ../../../labeled_data_bio/$dir$course"${file%.*}".json
                    ../../../concept_scripts/extract_concepts.py -f "../../../labeled_data_bio/$dir$course${file%.*}".json -o ../../../concepts_bio/$dir$course"${file%.*}".txt
                    ../../../concept_scripts/extract_concepts_tagged.py -f "../../../labeled_data_bio/$dir$course${file%.*}".json -o ../../../tagged_concepts_bio/$dir$course"${file%.*}".txt
                    #../../../../concept_scripts/distant_label.py -f "../../../labeled_data_bio/unlabeled/$course${file%.*}".json -o "../../../labeled_data_bio/distant/$dir$course${file%.*}".json -d ../../../concepts/distant/$course"${file%.*}".json
                done
            fi
            cd ..
        done
        cd ..
    fi
done
