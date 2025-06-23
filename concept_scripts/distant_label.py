#!/usr/bin/env python

import os
from nltk.tokenize import word_tokenize
from pathlib import Path
import json
import csv
import pke
import argparse
import sys

ROOT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
LABEL_ALL_KEYWORD = False

dirs = [os.path.join('final_slides_json_distant', 'CS-0441 Lecture Slides'),
  os.path.join('final_slides_json_distant', 'CS-0449 Lecture Slides'),
  os.path.join('final_slides_json_distant', 'CS-1541 Lecture Notes'),
  os.path.join('final_slides_json_distant', 'CS-1550 Lecture Slides'),
  os.path.join('final_slides_json_distant', 'CS-1567 Lecture Notes'),
  os.path.join('final_slides_json_distant', 'CS-1622 Lecture Slides')]

def get_concept_set(filename, filetype):
	concept_set = set()
	with open(filename, 'r') as f:
		if filetype == '.csv':		
			reader = csv.reader(f)
			for row in reader:
				concept_set.add(row[0])
		elif filetype == '.json':
			concepts = json.loads(f.read())
			print(concepts)
			concept_set = set(concepts)
		else:
			pass
	return concept_set

def get_keywords(text_in):
	pos = {'NOUN', "PROPN", "ADJ"}
	extractor = pke.unsupervised.YAKE()
	extractor.load_document(input=text_in, language='en', normalization=None)
	extractor.candidate_selection()
	extractor.candidate_weighting()
	keyphrases = extractor.get_n_best(n=15, stemming=False)
	print(keyphrases)
	return set([phrase[0] for phrase in keyphrases])


def label_json(json, text_in, concept_set_file, use_keyword_extraction=True, file_type='.csv'):
	concept_set = get_concept_set(concept_set_file, file_type)
	if use_keyword_extraction:
		keyword_set = get_keywords(text_in)
		concept_set = concept_set.union(keyword_set)
	for i in range(len(json)):
		json[i]['labels'] = ['O' for label in json[i]['labels']]
	for j, slides in enumerate(json):
		k = 0
		for i in range(len(slides['words'])):
			if k > 0:
				json[j]['labels'][i] = 'I'
				k -= 1
			candidates = [word_tokenize(item) for item in concept_set if item.startswith(slides['words'][i])]
			# print("Candidates", candidates)
			matches = [candidate for candidate in candidates if list(map(str.lower, candidate)) == list(map(str.lower, slides['words'][i:i+len(candidate)]))]
			if matches:
				print("Matches", matches)
				k = max(map(len, matches)) - 1
				json[j]['labels'][i] = 'B'
	return json			
				
 

if __name__ == "__main__":
	if LABEL_ALL_KEYWORD:
		print("Num unique concepts:", len(get_concept_set(os.path.join(ROOT_DIRECTORY,"all_fields_concepts.csv"))))
		for dir in dirs:
			files = Path(dir).glob('**/*.json')
			txdir = os.path.join(ROOT_DIRECTORY, "data_text", Path(dir).stem)
			for file in files:
				with open(file, 'rb') as f:
					nwdir = os.path.join(ROOT_DIRECTORY, "final_slides_json_distant", Path(dir).stem)
					print(nwdir)
					try:
						os.mkdir(nwdir)
					except Exception as e:
						pass
					print(file)
					file_text=""
					with open(os.path.join(txdir, Path(file).stem)+".txt", 'rt') as tf:
						file_text=tf.read()
					text = f.read()
					processed = json.loads(text)
					pair = label_json(processed, file_text, os.path.join(ROOT_DIRECTORY,"all_fields_concepts.csv"))
					with open(os.path.join(nwdir, Path(file).stem)+".json", 'w') as f2:
						f2.write(json.dumps(pair, indent=4))
	else:
		parser = argparse.ArgumentParser()
		parser.add_argument("-f", "--file", help = "File Input", type=Path)
		parser.add_argument("-o", "--out", help = "File Output", type=Path)
		parser.add_argument("-d", "--dist", help = "Distant Labels", type=Path)
		parser.add_argument("-k", "--keyword", help = "Use Keyword instead of files", action='store_true')
		args = parser.parse_args(sys.argv[1:])

		if args.file:
			with open(args.file, 'rb') as f:
				text = f.read()
			processed = json.loads(text)
			print(processed)
			if args.dist:
				labeled = label_json(processed, "", args.dist, file_type=Path(args.dist).suffix)
				with open(args.out, 'w') as f:
					f.write(json.dumps(labeled, indent=4))
