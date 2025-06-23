#!/usr/bin/env python
import os
import sys
from pathlib import Path
import json
import argparse

def extract_concepts(dataset):
	concepts=set()
	concept=""
	extracting = False
	for processed in dataset:
		for i in range(len(processed['sentence'])):
			if processed['word_labels'][i] == 0 and extracting:
				concepts.add(concept.strip().lower() + "\n")
				concept = ""
				extracting = False
			if processed['word_labels'][i] == 1:
				if extracting:
					concepts.add(concept.strip().lower() + "\n")
					concept = ""
				extracting = True
			if extracting:
				concept += processed['sentence'][i].strip()
				if processed['sentence'][i] == "-":
					continue
				if i + 1 < len(processed['sentence']) and processed['sentence'][i+1] == "-":
					continue
				concept += " "
	return concepts

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", help = "File Input", type=Path)
parser.add_argument("-o", "--output", help = "File Output", type=Path)
args = parser.parse_args(sys.argv[1:])

if args.file:
	with open(args.file, 'rb') as f:
		text = f.read()
processed = json.loads(text)
concepts = extract_concepts(processed)
print(concepts)
if args.output:
	with open(args.output, 'w') as f2:
		f2.writelines(concepts)