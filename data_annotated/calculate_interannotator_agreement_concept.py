#!/usr/bin/env python

import os
import nltk
from pathlib import Path
import json
import argparse
import csv
import sys
import numpy as np
from statsmodels.stats.inter_rater import fleiss_kappa, aggregate_raters
from irrCAC.raw import CAC
from irrCAC.benchmark import Benchmark
import pandas as pd
import inflect

def word_tokenize(tokens):
    return [token.replace("''", '"').replace("``", '"') for token in nltk.word_tokenize(tokens)]

def bio_to_num(string):
    if string == 'O':
        return 0
    if string == 'B-Concept':
        return 1
    if string == 'I-Concept':
        return 2
    return -1

if __name__=="__main__":
    inf_eng = inflect.engine()
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", help = "File Folder Input", type=Path)
    args = parser.parse_args(sys.argv[1:])
    files = Path(args.file).glob('**/*.txt')
    annotations = []
    for file in files:
        if file:
            with open(file, 'rb') as f:
                text = f.read()
                annotations.append(json.loads(text))
    data = []
    for n, annotator in enumerate(annotations):
        for i, lec in enumerate(annotator):
            concepts = {line.rstrip("\n").lower() for line in lec if line.strip()}
            concepts = {inf_eng.singular_noun(item.lower()) if (item[0:1].isalnum() and inf_eng.singular_noun(item.lower())) else item.lower() for item in concepts if len(item) > 0}
            data.append(dict())
            for concept in concepts:
                if concept not in data[i]:
                    data[i][concept] = [0 for annotator in annotations]
                data[i][concept][n] = 1
    print(data)
    cac_data = []
    cac_data_per_lec = []
    for i, _ in enumerate(annotator):
        cac_data_per_lec.append([])
        for concept in data[i]:
            cac_data_per_lec[i].append(data[i][concept])
            cac_data.append(data[i][concept])
    print(cac_data)
    irrcac_labels_combined = pd.DataFrame(cac_data)
    irrcac_labels_per_lec = [pd.DataFrame(x) for x in cac_data_per_lec]
    ratings_fl_per_lec = [CAC(x).fleiss()['est'] for x in irrcac_labels_per_lec]
    ratings_fl_combined = CAC(irrcac_labels_combined).fleiss()['est']
    print(ratings_fl_per_lec)
    print(ratings_fl_combined)


    
    
