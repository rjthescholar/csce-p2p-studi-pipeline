#!/usr/bin/env python

import os
from nltk.tokenize import word_tokenize
from pathlib import Path
import json
import argparse
import sys

ROOT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))

def clean_concepts(concepts):
    return list(set([str.lower(concept) for concept in concepts]))

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-f", "--file", help = "File Input", type=Path)
	args = parser.parse_args(sys.argv[1:])

	if args.file:
		with open(args.file, 'rb') as f:
			text = f.read()
		processed = json.loads(text)
		print(processed)
		labeled = clean_concepts(processed)
		with open(args.file, 'w') as f:
			f.write(json.dumps(labeled, indent=4))