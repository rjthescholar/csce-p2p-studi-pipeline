#!/usr/bin/env python

import os
import nltk
import spacy
from pathlib import Path
import json
import argparse
import sys
import string
from doccano_transformer.datasets import NERDataset
from doccano_transformer.utils import read_jsonl

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

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", help = "File Input", type=Path)
    parser.add_argument("-o", "--out", help = "ConLL Output", type=Path)
    args = parser.parse_args(sys.argv[1:])
    with open(args.file, mode='r') as file:
        parsed_data = json.loads(file.read())
        segment=parsed_data['segment']
        course=parsed_data['course']
        lec=parsed_data['lec']
    dataset = read_jsonl(filepath=args.file, dataset=NERDataset, encoding='utf-8')
    with open(args.out, 'w') as f:
        f.write(f"{segment}|{course}|{lec}\n")
        for line in dataset.to_conll2003(tokenizer=word_tokenize):
            f.write(line['data'] + "\n")
    
