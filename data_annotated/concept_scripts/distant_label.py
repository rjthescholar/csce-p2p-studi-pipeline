#!/usr/bin/env python

import os
import spacy
from pathlib import Path
import json
import csv
#import pke
import argparse
import sys

ROOT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
LABEL_ALL_KEYWORD = False

def custom_tokenizer(nlp):
    prefix_re = spacy.util.compile_prefix_regex(nlp.Defaults.prefixes)
    suffix_re = spacy.util.compile_suffix_regex(nlp.Defaults.suffixes)
    custom_infixes = ['\.\.\.+', '(?<=[0-9])-(?=[0-9])', '[!&:,()//<>\[\]`\']']
    infix_re = spacy.util.compile_infix_regex(custom_infixes)

    tokenizer = spacy.tokenizer.Tokenizer(nlp.vocab,
                                            nlp.Defaults.tokenizer_exceptions,
                                            prefix_re.search,
                                            suffix_re.search,
                                            infix_re.finditer,
                                            token_match=None)
    return lambda text: tokenizer(text)

nlp = spacy.load("en_core_web_sm")
nlp.tokenizer = custom_tokenizer(nlp)

def word_tokenize(text):
    ignore_chars = ["\n", "\n\n"]
    tokenized = [token.text for token in nlp(text.replace("''", '"').replace("``", '"'))]
    tokenized = list(filter(lambda x: not x.isspace(), tokenized))
    print(tokenized)
    return tokenized

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
			for row in f.readlines():
				concept_set.add(row)
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
	# for i in range(len(json)):
	# 	json['data'][i]['word_labels'] = ['O' for label in json['data'][i]['word_labels']]
	for i in range(len(json['data'])): # was: for i in range(len(json)):
		json['data'][i]['word_labels'] = ['O' for label in json['data'][i]['word_labels']]
	for j, slides in enumerate(json['data']):
		k = 0
		for i in range(len(slides['sentence'])):
			if k > 0:
				json['data'][j]['word_labels'][i] = 'I'
				k -= 1
			candidates = [word_tokenize(item) for item in concept_set if item.startswith(slides['sentence'][i])]
			# print("Candidates", candidates)
			matches = [candidate for candidate in candidates if list(map(str.lower, candidate)) == list(map(str.lower, slides['sentence'][i:i+len(candidate)]))]
			if matches:
				print("Matches", matches)
				k = max(map(len, matches)) - 1
				json['data'][j]['word_labels'][i] = 'B'
	return json

def label_json_extended(json_in, text_in, concept_set_file, use_keyword_extraction=True, file_type=".csv", out_path=None,):
    """
    - If json_in is a dict: label and return dict (backward compatible)
    - If json_in is a file path: load, label, optionally write to out_path
    - If json_in is a directory: recursively label *.json, mirror outputs to out_path (required)
      Matching rules in directory mode:
        * If concept_set_file is a FILE -> use it for every input JSON
        * If concept_set_file is a DIR  -> for each input JSON, look for a concept file with same
          relative path but extension in {'.csv', '.json', '.txt'} under concept_set_file
    """

    SUPPORTED_CONCEPT_EXTS = (".csv", ".json", ".txt")

    def _reset_labels(doc):
        # ensure word_labels exists and is correct length
        for ex in doc.get("data", []):
            sent = ex.get("sentence", [])
            ex["word_labels"] = ["O"] * len(sent)

    def _label_doc(doc, concept_file_path: Path):
        concept_set = get_concept_set(str(concept_file_path), concept_file_path.suffix)

        if use_keyword_extraction:
            # !!This will only work if pke is imported above
            keyword_set = get_keywords(text_in)
            concept_set = concept_set.union(keyword_set)

        _reset_labels(doc)

        for j, slides in enumerate(doc.get("data", [])):
            k = 0
            sent = slides.get("sentence", [])
            labels = slides.get("word_labels", [])

            # safety: keep lengths aligned
            if len(labels) != len(sent):
                slides["word_labels"] = ["O"] * len(sent)
                labels = slides["word_labels"]

            for i in range(len(sent)):
                if k > 0:
                    labels[i] = "I"
                    k -= 1
                    continue

                candidates = [
                    word_tokenize(item)
                    for item in concept_set
                    if item.startswith(sent[i])
                ]
                matches = [
                    cand for cand in candidates
                    if list(map(str.lower, cand)) == list(map(str.lower, sent[i:i+len(cand)]))
                ]
                if matches:
                    k = max(map(len, matches)) - 1
                    labels[i] = "B"

        return doc

    def _find_concept_for_rel(concept_root: Path, rel_json_path: Path):
        # Try same relative path with different allowed extensions
        for ext in SUPPORTED_CONCEPT_EXTS:
            candidate = (concept_root / rel_json_path).with_suffix(ext)
            if candidate.exists():
                return candidate
        return None

    # --- Normalize inputs ---
    concept_path = Path(concept_set_file)

    # Case 1: in-memory dict (original behavior)
    if isinstance(json_in, dict):
        concept_file = concept_path
        if concept_file.is_dir():
            raise ValueError("concept_set_file is a directory, but json_in is a dict. Pass a concept FILE.")
        return _label_doc(json_in, concept_file)

    # Otherwise treat json_in as a path
    input_path = Path(json_in)

    # Case 2: single file
    if input_path.is_file():
        doc = json.loads(input_path.read_text())

        # concept can be a file or a directory in this mode:
        if concept_path.is_file():
            concept_file = concept_path
        elif concept_path.is_dir():
            # match by relative path *name only* in same directory structure,
            # using rel path against input file's parent is ambiguous, so we use stem-only match.
            # Prefer exact mirror if user passes a rel-style path.
            # Simple rule: concept_root/<same filename stem>.<ext> (in concept_root root)
            concept_file = None
            for ext in SUPPORTED_CONCEPT_EXTS:
                cand = (concept_path / input_path.name).with_suffix(ext)
                if cand.exists():
                    concept_file = cand
                    break
            if concept_file is None:
                raise FileNotFoundError(f"No concept file found in {concept_path} for {input_path.name}")
        else:
            raise FileNotFoundError(f"concept_set_file not found: {concept_path}")

        labeled = _label_doc(doc, concept_file)

        if out_path is not None:
            outp = Path(out_path)
            outp.parent.mkdir(parents=True, exist_ok=True)
            outp.write_text(json.dumps(labeled, indent=4))
        return labeled

    # Case 3: directory mode
    if input_path.is_dir():
        if out_path is None:
            raise ValueError("Directory mode requires out_path to be provided (output directory).")

        out_root = Path(out_path)
        out_root.mkdir(parents=True, exist_ok=True)

        use_single_concept = concept_path.is_file()
        if (not use_single_concept) and (not concept_path.is_dir()):
            raise FileNotFoundError(f"concept_set_file must be a file or directory: {concept_path}")

        for jf in input_path.rglob("*.json"):
            rel = jf.relative_to(input_path)
            out_file = out_root / rel
            out_file.parent.mkdir(parents=True, exist_ok=True)

            doc = json.loads(jf.read_text())

            if use_single_concept:
                concept_file = concept_path
            else:
                concept_file = _find_concept_for_rel(concept_path, rel)
                if concept_file is None:
                    print(f"[SKIP] No concept file for {jf} under {concept_path}")
                    continue

            labeled = _label_doc(doc, concept_file)
            out_file.write_text(json.dumps(labeled, indent=4))
            print(f"[OK] {jf} -> {out_file}")

        # In directory mode, we mainly write files; returning None is fine
        return None

    raise FileNotFoundError(f"Input path not found: {input_path}")

 

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
				labeled = label_json(processed, "", args.dist, file_type=Path(args.dist).suffix, use_keyword_extraction=False)
				with open(args.out, 'w') as f:
					f.write(json.dumps(labeled, indent=4))
