#!/bin/bash
set -evx
cd data/dataset
for dir in */; do
    if [ -d "$dir" ]; then
        mkdir ../dataset_conll/$dir -p
		mkdir ../dataset_bio/$dir -p
        mkdir ../concepts_bio/$dir -p
		cd "$dir"
        for file in *.jsonl; do
			../../../to_conll.py -f $file -o ../../dataset_conll/$dir"${file%.*}".conll
			../../../to_bio_json.py -f "../../dataset_conll/$dir${file%.*}".conll -o ../../dataset_bio/$dir"${file%.*}".json
            ../../../concept_scripts/extract_concepts.py -f "../../dataset_bio/$dir${file%.*}".json -o ../../concepts_bio/$dir"${file%.*}".txt
        done
        cd ..
    fi
done