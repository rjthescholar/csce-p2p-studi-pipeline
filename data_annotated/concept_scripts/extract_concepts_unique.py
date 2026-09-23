#!/usr/bin/env python
import os
import sys
from pathlib import Path
import json
import argparse

def set_default(obj):
    if isinstance(obj, set):
        return list(obj)
    raise TypeError

def extract_concepts(dataset):
	concepts=set()
	concept=""
	extracting = False
	punct = ",./?\"\':;-_=+\\|!"
	o_brac = "([{<"
	c_brac = ")]}>"
	for processed in dataset['data']:
		for i in range(len(processed['sentence'])):
			if processed['word_labels'][i] == 'O' and extracting:
				concepts.add(concept.strip().lower() + "\n")
				concept = ""
				extracting = False
			if processed['word_labels'][i] == 'B':
				if extracting:
					concepts.add(concept.strip().lower() + "\n")
					concept = ""
				extracting = True
			if extracting:
				concept += processed['sentence'][i].strip()
				if processed['sentence'][i] in punct+o_brac:
					continue
				if i + 1 < len(processed['sentence']) and (processed['sentence'][i+1] in punct+c_brac\
					or processed['sentence'][i+1][0] in '’\''):
					continue
				concept += " "
	return concepts

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", help = "File Input", type=Path)
parser.add_argument("-o", "--output", help = "File Output", type=Path)
parser.add_argument("-l", "--list", help = "Is JSONL",  action='store_true')
args = parser.parse_args(sys.argv[1:])
concepts = []
if args.file:
	with open(args.file, 'rb') as f:
		text = f.read()
processed = json.loads(text)
if args.list:
	for i in processed:
		concepts.append(extract_concepts(i))
else:
	concepts = extract_concepts(processed)

print(concepts)
if args.output:
	with open(args.output, 'w') as f2:
		if args.list:
			f2.write(json.dumps(concepts, default=set_default))
		else:
			f2.writelines(concepts)
