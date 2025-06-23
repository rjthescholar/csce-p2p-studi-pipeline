#!/usr/bin/env python

import os
import nltk
from pathlib import Path
import json
import argparse
import sys

def word_tokenize(tokens):
    return [token.replace("''", '"').replace("``", '"') for token in nltk.word_tokenize(tokens)]

if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-f", "--file", help = "File Input", type=Path)
	parser.add_argument("-o", "--out", help = "Out Folder", type=Path)
	args = parser.parse_args(sys.argv[1:])
	with open(args.file, mode='r') as file:	
		for line in file:
			json_line = json.loads(line)
			course = json_line["course"]
			lecture = json_line["lec"] + ".jsonl"
			outfile = args.out / "concepts" / course / lecture
			outfile.parent.mkdir(exist_ok=True, parents=True)
			with open(outfile, mode="w") as df:
				for label in json_line["label"]:
					df.write(json_line["text"][label[0]:label[1]] + "\n")
			outfile = args.out / "dataset" / course / lecture
			outfile.parent.mkdir(exist_ok=True, parents=True)
			with open(outfile, mode="w") as df:
				df.write(json.dumps(json_line))