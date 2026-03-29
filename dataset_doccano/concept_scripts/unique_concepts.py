#!/usr/bin/env python
import os
import sys
from pathlib import Path
import json
import argparse
import inflect
import re

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", help = "File Input", type=Path)
parser.add_argument("-o", "--output", help = "File Output", type=Path)
args = parser.parse_args(sys.argv[1:])

with open(args.file, 'r') as f:
    inf_eng = inflect.engine()
    concepts = {line.lower() for line in f if line.strip()}
    concepts = {inf_eng.singular_noun(item.lower()) if (item[0:1].isalnum() and inf_eng.singular_noun(item.lower())) else item.lower() for item in concepts if len(item) > 0}
    concepts = {re.sub(r'\W+', ' ', concept) + '\n' for concept in concepts}
with open(args.output, 'w') as f:
    for concept in concepts:
        f.write(concept)
