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
	parser.add_argument("-r", "--train-path", help = "Train Path", type=Path)
	parser.add_argument("-l", "--labeled-path", help = "Labeled Path", type=Path)
	parser.add_argument("-v", "--val-path", help = "Validation Path", type=Path, required=True)
	parser.add_argument("-s", "--self-label-1-path", help = "Self-Training 1 Path", type=Path, required=True)
	parser.add_argument("-e", "--self-label-2-path", help = "Self-Training 2 Path", type=Path, required=True)
	parser.add_argument("-d", "--distant-path", help = "Distant Labels Path", type=Path)
	parser.add_argument("-o", "--out", help = "Models Path", type=Path)
	args = parser.parse_args(sys.argv[1:])

	# Prepare the validation dataset.
	sentences, labels = get_data(args.val_path)
	validation_dataset = pd.DataFrame({'sentence': sentences, 'word_labels': labels})

	sentences, labels = get_fs_data(args.val_path)
	validation_block_datasets = [pd.DataFrame(({'sentence': sentences[file], 'word_labels': labels[file]})) for file in sentences]
	
	# Prepare the self-training datasets.
	sentences, labels = get_data(args.self_label_1_path)
	self_training_dataset_1 = pd.DataFrame({'sentence': sentences, 'word_labels': labels})
	
	sentences, labels = get_data(args.self_label_2_path)
	self_training_dataset_2 = pd.DataFrame({'sentence': sentences, 'word_labels': labels})

	self_training_dataset = pd.concat([self_training_dataset_1, self_training_dataset_2])
	self_training_dataset = self_training_dataset.reset_index(drop=True)

	self_training_set_1 = dataset(self_training_dataset_1, tokenizer, MAX_LEN)
	self_training_set_2 = dataset(self_training_dataset_2, tokenizer, MAX_LEN)
	self_training_set = dataset(self_training_dataset, tokenizer, MAX_LEN)
	validation_set = dataset(validation_dataset, tokenizer, MAX_LEN)
	val_block_set = [dataset(data_set, tokenizer, MAX_LEN) for data_set in validation_block_datasets]

	self_training_loader_1 = DataLoader(self_training_set_1, **train_params)
	self_training_loader_2 = DataLoader(self_training_set_2, **train_params)
	self_training_loader = DataLoader(self_training_set, **train_params)
	validation_loader = DataLoader(validation_set, **valid_params)
	val_block_loader = [DataLoader(data_set, **valid_params) for data_set in val_block_set]

	if not (args.labeled_path or (args.test_path and args.train_path)):
		parser.error("Neither labeled or test and train sets specified.")

	if args.labeled_path:
		sentences, labels = get_course_data(args.labeled_path)
		if args.distant_path:
			d_sentences, d_labels = get_course_data(args.distant_path)
			d_datas = [
				pd.DataFrame((
					{
					'sentence': double_flatten(without_keys(d_sentences, {course})),
	   				'word_labels': double_flatten(without_keys(d_labels, {course}))
					}
				))
			for course in sentences
			]
		course_list = [*sentences]
		print(course_list)
		test_block_datasetses = [
			[
				pd.DataFrame((
					{
					'sentence': sentences[course][file],
	   				'word_labels': labels[course][file]
					}
				))
				for file in sentences[course]
			]
			for course in sentences
		]

		test_datasets = [
				pd.DataFrame((
					{
					'sentence': flatten(sentences[course]),
	   				'word_labels': flatten(labels[course])
					}
				))
			for course in sentences
		]

		train_datasets = [
			pd.DataFrame((
					{
					'sentence': double_flatten(without_keys(sentences, {course})),
	   				'word_labels': double_flatten(without_keys(labels, {course}))
					}
				))
			for course in sentences
		]
	else:
		# Prepare the test dataset.
		sentences, labels = get_data(args.test_path)
		test_dataset = pd.DataFrame({'sentence': sentences, 'word_labels': labels})
		
		#The test dataset, but separated by file.
		sentences, labels = get_fs_data(args.test_path)
		test_block_datasets = [pd.DataFrame(({'sentence': sentences[file], 'word_labels': labels[file]})) for file in sentences]

		# Prepare the train dataset.
		sentences, labels = get_data(args.train_path)
		train_dataset = pd.DataFrame({'sentence': sentences, 'word_labels': labels})

		if args.distant_path:
			d_sentences, d_labels = get_data(args.distant_path)
			d_data = pd.DataFrame({'sentence': d_sentences, 'word_labels': d_labels})
			d_datas = [d_data] * EXPERIMENTS

		train_datasets = [train_dataset] * EXPERIMENTS
		test_block_datasetses = [test_block_datasets] * EXPERIMENTS
		test_datasets = [test_dataset] * EXPERIMENTS
		course_list = [i+1 for i in range(EXPERIMENTS)]
	stats = []
	p2p_stats = []
	val_stats = []
	p2p_val_stats = []
	for experiment_count in range(len(train_datasets)):
		train_dataset = train_datasets[experiment_count]
		test_block_datasets = test_block_datasetses[experiment_count]
		test_dataset = test_datasets[experiment_count]

		if args.distant_path:
			d_data = d_datas[experiment_count]
			train_dataset = pd.concat([train_dataset, d_data])
			train_dataset = train_dataset.reset_index(drop=True)

		training_set = dataset(train_dataset, tokenizer, MAX_LEN)
		testing_set = dataset(test_dataset, tokenizer, MAX_LEN)
		test_block_set = [dataset(data_set, tokenizer, MAX_LEN) for data_set in test_block_datasets]

		training_loader = DataLoader(training_set, **train_params)
		testing_loader = DataLoader(testing_set, **test_params)
		test_block_loader = [DataLoader(data_set, **test_params) for data_set in test_block_set]

		for data_set in test_block_set:
			gold_concepts = extract_concepts(data_set, gold=True)
			print(f"actual concepts for file: {gold_concepts}")

		do_training = True
		if do_training:
			print("<============= BEGINNING TRAINING ==================>\n")
			print(f"=============== XLNET MODEL TRAINING TRIAL {course_list[experiment_count]} ================")

			model = XLNetForTokenClassification.from_pretrained('xlnet-large-cased',
														num_labels=len(id2label),
														id2label=id2label,
														label2id=label2id)
			model.to(device)
			optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

			train_epochs(model, optimizer, training_loader, validation_loader, testing_loader, EPOCHS)
			print("=============== XLNET MODEL DEV EVAL ================")
			_, labels, predictions, chunk_stats = valid(model, validation_loader)
			print(sklmet.classification_report(labels, predictions, target_names=["B", "I", "O"]))
			print(classification_report([labels], [predictions]))

			bio_stats = classification_report([labels], [predictions], output_dict=True)
			concept_stats = eval_concepts(model, val_block_loader)

			val_stats.append({"bio": bio_stats, "chunk": chunk_stats, "concept": concept_stats})

			print("=============== XLNET MODEL EVALUATION ================")
			_, labels, predictions, chunk_stats = valid(model, testing_loader)
			print(sklmet.classification_report(labels, predictions, target_names=["B", "I", "O"]))
			print(classification_report([labels], [predictions]))

			bio_stats = classification_report([labels], [predictions], output_dict=True)
			concept_stats = eval_concepts(model, test_block_loader)

			stats.append({"bio": bio_stats, "chunk": chunk_stats, "concept": concept_stats})

			if args.out:
				save_path = args.out / timestamp_str / "base" / course_list[experiment_count]
				save_path.parent.mkdir(exist_ok=True, parents=True)
				model.save_pretrained(save_path)

			print("=============== P2P MODEL TRAINING ================")

			model_1 = copy.deepcopy(model)
			model_2 = copy.deepcopy(model)
			model_best = copy.deepcopy(model)
			model_1.to(device)
			model_2.to(device)
			model_best.to(device)
			optimizer_1 = torch.optim.Adam(params=model_1.parameters(), lr=LEARNING_RATE)
			optimizer_2 = torch.optim.Adam(params=model_2.parameters(), lr=LEARNING_RATE)

			if self_training_type == "p2p":
				model_best=p2p_self_train(model_1, model_2, model_best, optimizer_1, optimizer_2, self_training_loader_1, self_training_loader_2, validation_loader, rounds=ROUNDS, self_epochs=SELF_TRAIN_EPOCH, best_model_init=best_model_init)
			elif self_training_type == "ts":
				model_best=ts_self_train(teacher_model=model_1, student_model=model_2, teacher_optimizer=optimizer_1, student_optimizer=optimizer_2, self_training_loader=self_training_loader, validation_loader=validation_loader, rounds=ROUNDS, self_epochs=SELF_TRAIN_EPOCH)

			print("=============== P2P MODEL DEV EVAL ================")
			print("=============== MODEL 1 DEV EVALUATION ================")
			_, labels, predictions, _ = valid(model_1, validation_loader)
			print(sklmet.classification_report(labels, predictions, target_names=["B", "I", "O"]))
			print(classification_report([labels], [predictions]))
			eval_concepts(model_1, val_block_loader)
			print("=============== MODEL 2 DEV EVALUATION ================")
			_, labels, predictions, _ = valid(model_2, validation_loader)
			print(sklmet.classification_report(labels, predictions, target_names=["B", "I", "O"]))
			print(classification_report([labels], [predictions]))
			eval_concepts(model_2, val_block_loader)
			print("=============== FINAL MODEL DEV EVALUATION ================")
			_, labels, predictions, p2p_chunk_stats = valid(model_best, validation_loader)
			print(sklmet.classification_report(labels, predictions, target_names=["B", "I", "O"]))
			print(classification_report([labels], [predictions]))
			
			p2p_bio_stats = classification_report([labels], [predictions], output_dict=True)
			p2p_concept_stats = eval_concepts(model_best, val_block_loader)

			p2p_val_stats.append({"bio": p2p_bio_stats, "chunk": p2p_chunk_stats, "concept": p2p_concept_stats})

			print("=============== P2P MODEL EVALUATION ================")
			print("=============== MODEL 1 EVALUATION ================")
			_, labels, predictions, _ = valid(model_1, testing_loader)
			print(sklmet.classification_report(labels, predictions, target_names=["B", "I", "O"]))
			print(classification_report([labels], [predictions]))
			eval_concepts(model_1, test_block_loader)
			print("=============== MODEL 2 EVALUATION ================")
			_, labels, predictions, _ = valid(model_2, testing_loader)
			print(sklmet.classification_report(labels, predictions, target_names=["B", "I", "O"]))
			print(classification_report([labels], [predictions]))
			eval_concepts(model_2, test_block_loader)
			print("=============== FINAL MODEL EVALUATION ================")
			_, labels, predictions, p2p_chunk_stats = valid(model_best, testing_loader)
			print(sklmet.classification_report(labels, predictions, target_names=["B", "I", "O"]))
			print(classification_report([labels], [predictions]))
			
			p2p_bio_stats = classification_report([labels], [predictions], output_dict=True)
			p2p_concept_stats = eval_concepts(model_best, test_block_loader)

			p2p_stats.append({"bio": p2p_bio_stats, "chunk": p2p_chunk_stats, "concept": p2p_concept_stats})

			if args.out:
				save_path = args.out / timestamp_str / "p2p" / course_list[experiment_count]
				save_path.parent.mkdir(exist_ok=True, parents=True)
				model_best.save_pretrained(save_path)


	print("<============= FINAL EVALUATION ==================>")
	print(stats)
	print(p2p_stats)



