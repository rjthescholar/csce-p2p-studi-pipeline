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
	parser.add_argument("-n", "--number", help = "Number Concepts", action='store_true')
	args = parser.parse_args(sys.argv[1:])
	with open(args.file, mode='r') as file:	
		for line in file:
			json_line = json.loads(line)
			segments = json_line["segment"]
			print(segments)
			course = json_line["course"]
			lecture = json_line["lec"] + ".jsonl"
			if isinstance(segments, str):
				segments = [segments]
			for segment in segments:
				outfile = args.out / "concepts" / segment / course / lecture
				outfile.parent.mkdir(exist_ok=True, parents=True)
				with open(outfile, mode="w") as df:
					for i, label in enumerate(json_line["label"]):
						p= (f"{i+1}" + ": " if args.number else "")
						df.write(p + json_line["text"][label[0]:label[1]] + "\n")
				outfile = args.out / "dataset" / segment / course / lecture
				outfile.parent.mkdir(exist_ok=True, parents=True)
				with open(outfile, mode="w") as df:
					df.write(json.dumps(json_line))
