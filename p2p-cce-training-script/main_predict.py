#!/usr/bin/env python3

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import XLNetForTokenClassification
import sys
import argparse
from datetime import datetime


from helpers import *
from json_processing import *
from dataset import *
from eval import *
from xlnet_train import *
from self_train import *

EXPERIMENTS = 4

def flatten(dictionary):
	flattened = []
	for item in dictionary:
		flattened.extend(dictionary[item])
	return flattened

def double_flatten(dictionary):
	flattened = []
	for item in dictionary:
		for item2 in dictionary[item]:
			flattened.extend(dictionary[item][item2])
	return flattened

def without_keys(d, keys):
    return {x: d[x] for x in d if x not in keys}

if __name__ == "__main__":
	# Fix randomness, for predictability
	set_seed(0)
	# Get the current date and time
	current_datetime = datetime.now()

	# Format the date and time into a string, avoiding characters illegal in file names (e.g., colons)
	# Example format: YYYY-MM-DD_HH-MM-SS
	timestamp_str = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")

	self_training_type = 'p2p'

	parser = argparse.ArgumentParser()
	parser.add_argument("-t", "--test-path", help = "Test Path", type=Path)
	parser.add_argument("-m", "--model", help = "Models Path", type=Path)
	args = parser.parse_args(sys.argv[1:])

	# Prepare the test dataset.
	sentences, labels = get_data(args.test_path)
	test_dataset = pd.DataFrame({'sentence': sentences, 'word_labels': labels})
		
	#The test dataset, but separated by file.
	sentences, labels = get_fs_data(args.test_path)
	test_block_datasets = [pd.DataFrame(({'sentence': sentences[file], 'word_labels': labels[file]})) for file in sentences]
	
	testing_set = dataset(test_dataset, tokenizer, MAX_LEN)
	test_block_set = [dataset(data_set, tokenizer, MAX_LEN) for data_set in test_block_datasets]

	testing_loader = DataLoader(testing_set, **test_params)
	test_block_loader = [DataLoader(data_set, **test_params) for data_set in test_block_set]

	for data_set in test_block_set:
		gold_concepts = extract_concepts(data_set, gold=True)
		print(f"actual concepts for file: {gold_concepts}")

	do_training = True
	if do_training:
		model = XLNetForTokenClassification.from_pretrained(args.model,
													num_labels=len(id2label),
													id2label=id2label,
													label2id=label2id)
		model.to(device)
		optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

		print("=============== MODEL EVALUATION ================")
		_, labels, predictions, chunk_stats = valid(model, testing_loader)
		print(sklmet.classification_report(labels, predictions, target_names=["B", "I", "O"]))
		print(classification_report([labels], [predictions]))

		bio_stats = classification_report([labels], [predictions], output_dict=True)
		concept_stats = eval_concepts(model, test_block_loader)





