#!/usr/bin/env python

import os
import nltk
from pathlib import Path
import json
import argparse
import sys
from doccano_transformer.datasets import NERDataset
from doccano_transformer.utils import read_jsonl

def word_tokenize(tokens):
    return [token.replace("''", '"').replace("``", '"') for token in nltk.word_tokenize(tokens)]

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", help = "File Input", type=Path)
    parser.add_argument("-o", "--out", help = "ConLL Output", type=Path)
    args = parser.parse_args(sys.argv[1:])

    dataset = read_jsonl(filepath=args.file, dataset=NERDataset, encoding='utf-8')
    with open(args.out, 'w') as f:
        for line in dataset.to_conll2003(tokenizer=word_tokenize):
            f.write(line['data'] + "\n")
    