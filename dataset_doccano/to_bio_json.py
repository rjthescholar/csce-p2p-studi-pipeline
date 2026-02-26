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

def bio_stripped(string):
	if string == 'O':
		return 'O'
	if string[0] == 'B':
		return 'B'
	if string[0] == 'I':
		return 'I'
	return string

if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-f", "--file", help = "File Input", type=Path)
	parser.add_argument("-o", "--out", help = "BIO JSON Output", type=Path)
	parser.add_argument("-n", "--nosplit", help = "For doccano outputs that have not been split", action='store_true')
	args = parser.parse_args(sys.argv[1:])
	
	bio_list = []
	if args.file:
		with open(args.file, 'r') as f:
			text = f.read()
			line_text=text.split('\n')
			if not args.nosplit:
				segment, course, lec = line_text[0].split('|')
			prev_row = 'x'
			for i, row in enumerate(line_text):
				if i == 0 and not args.nosplit:
					continue
				if row == '':
					continue
				srow = row.split(' ')
				if srow[0] == '-DOCSTART-':
					bio_list.append({"sentence": [], 'word_labels': []})
					continue
				if row == ' _ _ O' and not args.nosplit:
					bio_list.append({"sentence": [], 'word_labels': []})
					continue
				bio_list[-1]['sentence'].append(srow[0])
				bio_list[-1]['word_labels'].append(bio_stripped(srow[3]))
				prev_row = row
				
	
	if args.out:
		with open(args.out, 'w') as f2:
			if args.nosplit:
				f2.write(json.dumps(bio_list, indent=4))
			else:
				f2.write(json.dumps({'segment': segment, 'course': course, 'lec': lec, 'data': bio_list}, indent=4))
	
