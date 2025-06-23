#!/usr/bin/env python

import os
import nltk
from pathlib import Path
import json
import argparse
import csv
import sys

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
	parser = argparse.ArgumentParser()
	parser.add_argument("-f", "--file", help = "File Input", type=Path)
	parser.add_argument("-o", "--out", help = "BIO JSON Output", type=Path)
	args = parser.parse_args(sys.argv[1:])
	
	bio_list = []
	if args.file:
		with open(args.file, 'r') as f:
			text = f.read()
			for row in text.split('\n'):
				if row == '':
					bio_list.append({"sentence": [], 'word_labels': []})
					continue
				srow = row.split(' ')
				if srow[0] == '-DOCSTART-':
					bio_list.append({"sentence": [], 'word_labels': []})
					continue
				bio_list[-1]['sentence'].append(srow[0])
				bio_list[-1]['word_labels'].append(bio_to_num(srow[3]))
				
	
	if args.out:
		with open(args.out, 'w') as f2:
			f2.write(json.dumps(bio_list, indent=4))
	