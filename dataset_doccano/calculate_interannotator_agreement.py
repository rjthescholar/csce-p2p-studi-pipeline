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

def word_tokenize(tokens):
    return [token.replace("''", '"').replace("``", '"') for token in nltk.word_tokenize(tokens)]

def bio_to_num(string):
    if string == 'O':
        return 0
    if string == 'B':
        return 1
    if string == 'I':
        return 2
    return -1

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", help = "File Folder Input", type=Path)
    args = parser.parse_args(sys.argv[1:])
    files = Path(args.file).glob('**/*.jsonl')
    labels = []
    for file in files:
        if file:
            with open(file, 'rb') as f:
                text = f.read()
                labels.append(json.loads(text))
    print(labels)
    deck_indexed_labels = []
    flattened_labels = dict()
    for i in range(len(labels)):
        flattened_labels[i] = []
        for deck in labels[i]:
            flattened_labels[i].extend(deck['word_labels'])
    print(flattened_labels)
    irrcac_labels_combined = pd.DataFrame(flattened_labels)
    ratings_gwet_combined = CAC(irrcac_labels_combined).gwet()['est']
    ratings_fl_combined = CAC(irrcac_labels_combined).fleiss()['est']
    for ann in labels:
        for i in range(len(ann)):
            if len(deck_indexed_labels) == i:
                deck_indexed_labels.append([])
            deck_indexed_labels[i].append(ann[i]['word_labels'])
    irrcac_labels = [pd.DataFrame({i: deck[i] for i in range(len(deck))}, index=pd.Index([i+1 for i in range(len(deck[0]))], name="Units")) for deck in deck_indexed_labels]
    kappa_labels = [np.array(deck).transpose() for deck in deck_indexed_labels]
    print(kappa_labels[0])
    cac_4raters = CAC(irrcac_labels[0])
    print(CAC(irrcac_labels[0]).fleiss()['est']['coefficient_value'])
    ratings_fl = np.array([fleiss_kappa(aggregate_raters(deck)[0], method='fleiss') for deck in kappa_labels])
    ratings_fl_irr = np.array([CAC(deck).fleiss()['est']['coefficient_value'] for deck in irrcac_labels])
    ratings_rep_ac1 = [CAC(deck).gwet() for deck in irrcac_labels]
    ratings_ac1 = np.array([deck['est']['coefficient_value'] for deck in ratings_rep_ac1])
    print("Fleiss's Kappa (statsmodels)")
    print(ratings_fl)
    print(ratings_fl.mean())
    print(ratings_fl.std())
    print("Fleiss's Kappa (irrCAC)")
    print(ratings_fl_irr)
    print(ratings_fl_irr.mean())
    print(ratings_fl_irr.std())
    print("Gwet's AC1 (irrCAC)")
    print(ratings_ac1)
    print(ratings_ac1.mean())
    print(ratings_ac1.std())
    print("Fleiss's Kappa Combined")
    print(ratings_fl_combined['coefficient_value'])
    print(ratings_fl_combined['se'])
    print(pd.DataFrame(Benchmark(coeff=ratings_fl_combined['coefficient_value'],se=ratings_fl_combined['se']).landis_koch()))
    print("Gwet's AC1 Combined")
    print(ratings_gwet_combined['coefficient_value'])
    print(ratings_gwet_combined['se'])
    print(pd.DataFrame(Benchmark(coeff=ratings_gwet_combined['coefficient_value'],se=ratings_gwet_combined['se']).landis_koch()))
    
    
