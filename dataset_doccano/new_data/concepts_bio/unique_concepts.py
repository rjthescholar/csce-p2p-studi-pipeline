#!/usr/bin/env python
import os
import sys
from pathlib import Path
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", help = "File Input", type=Path)
parser.add_argument("-o", "--output", help = "File Output", type=Path)
args = parser.parse_args(sys.argv[1:])

with open(args.file, 'r') as f:
    concepts = {line.lower() for line in f if line.strip()}
with open(args.output, 'w') as f:
    for concept in concepts:
        f.write(concept)
