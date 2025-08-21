#!/usr/bin/env python

import os
from nltk.tokenize import word_tokenize
from pathlib import Path
import json
import argparse
import sys

ROOT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))

def eval(concepts, gold_concepts):
	fn = gold_concepts - concepts
	tp = gold_concepts & concepts
	fp = concepts - gold_concepts
	print(f"TP: {len(tp)}, FP: {len(fp)}, FN: {len(fn)}")
	return (len(tp),len(fp),len(fn))

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-f", "--file", help = "File Input", type=Path)
	parser.add_argument("-g", "--gold", help = "Gold Input", type=Path)
	args = parser.parse_args(sys.argv[1:])

	print(args.file)
	print(args.gold)
	file_path = Path(args.file).glob('**/*.json')
	gold_path = Path(args.gold).glob('**/*.json')
	tp, fp, fn = 0, 0, 0
	for file in file_path:
		gold_path = Path(args.gold).glob('**/*.json')
		for gold_file in gold_path:
			if file.stem == gold_file.stem:
				print(file.stem)
				print(gold_file.stem)
				with open(file, 'rb') as f:
					text = f.read()
				concepts = set(json.loads(text))
				with open(gold_file, 'rb') as f:
					text = f.read()
				gold = set(json.loads(text))
				tp_c, fp_c, fn_c = eval(concepts, gold)
				tp += tp_c
				fp += fp_c
				fn += fn_c
    
	p = tp / (tp + fp) if tp + fp > 0 else 0
	r = tp / (tp + fn) if tp + fn > 0 else 0
	f1 = 2 * (p * r) / (p + r) if p + r > 0 else 0
	print(f"precision: {p}, recall: {r}, F1: {f1}")