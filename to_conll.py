#!/usr/bin/env python

import os
import nltk
import spacy
from pathlib import Path
import json
import argparse
import sys
from doccano_transformer.datasets import NERDataset
from doccano_transformer.utils import read_jsonl

nlp = spacy.blank("en")

def word_tokenize(text):
    tokenized = [token.text for token in nlp(text.replace("''", '"').replace("``", '"'))]
    tokenized = list(filter(lambda x: x != "\n", tokenized))
    print(tokenized)
    return tokenized

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", help = "File Input", type=Path)
    parser.add_argument("-o", "--out", help = "ConLL Output", type=Path)
    args = parser.parse_args(sys.argv[1:])

    dataset = read_jsonl(filepath=args.file, dataset=NERDataset, encoding='utf-8')
    with open(args.out, 'w') as f:
        for line in dataset.to_conll2003(tokenizer=word_tokenize):
            f.write(line['data'] + "\n")
    