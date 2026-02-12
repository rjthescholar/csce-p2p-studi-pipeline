import json
import os
from pathlib import Path
import itertools

from helpers import tokenize_and_preserve_labels, tokenizer, MAX_LEN

# Input: data_path
# Output: All sentences and labels in the path.
def get_data(data_path, do_merge=False):
	sentences = []
	labels = []
	# for dir in Path(data_path).glob('*'):
	for filename in Path(data_path).glob('**/*.json'):
		path = filename
		if os.path.isfile(path):
			f = open(path)
			#print(path, f)
			file_data = json.load(f)
			tokens_scooped = 0
			sentence = []
			label = []
			for item in file_data['data']:
				tokenized_sentences, _ = tokenize_and_preserve_labels(item['sentence'], item['word_labels'], tokenizer)
				if tokens_scooped + len(tokenized_sentences) > MAX_LEN or \
					 (not do_merge and tokens_scooped > 0):
					sentences.append(sentence)
					labels.append(label)
					sentence = []
					label = []
					tokens_scooped = 0
				tokens_scooped += len(tokenized_sentences)
				sentence.extend(item['sentence'])
				label.extend(item['word_labels'])
				#print(labels)
			sentences.append(sentence)
			labels.append(label)
	return (sentences, labels)

# Input: data_path
# Output: All sentences and labels in the path, separated by course and file.
def get_course_data(data_path, do_merge=False):
	sentences = dict()
	labels = dict()
	# for dir in Path(data_path).glob('*'):
	print(Path(data_path).glob('**/*.json'))
	for filename in Path(data_path).glob('**/*.json'):
		path = filename
		print(filename)
		if os.path.isfile(path):
			f = open(path)
			print(path, f)
			file_data = json.load(f)
			course = file_data['course']
			lec = file_data['lec']
			if course not in sentences:
				sentences[course] = dict()
				labels[course] = dict()
			sentences[course][lec] = []
			labels[course][lec] = []
			tokens_scooped = 0
			sentence = []
			label = []
			for item in file_data['data']:
				tokenized_sentences, _ = tokenize_and_preserve_labels(item['sentence'], item['word_labels'], tokenizer)
				if tokens_scooped + len(tokenized_sentences) > MAX_LEN or\
					  (not do_merge and tokens_scooped > 0):
					sentences[course][lec].append(sentence)
					labels[course][lec].append(label)
					sentence = []
					label = []
					tokens_scooped = 0
				tokens_scooped += len(tokenized_sentences)
				sentence.extend(item['sentence'])
				label.extend(item['word_labels'])
				#print(label)
			sentences[course][lec].append(sentence)
			labels[course][lec].append(label)
	return (sentences, labels)

# Input: data_path
# Output: All sentences and labels in the path, separated by file.
def get_fs_data(data_path, do_merge=False):
	sentences = dict()
	labels = dict()
	# for dir in Path(data_path).glob('*'):
	print(Path(data_path).glob('**/*.json'))
	for filename in Path(data_path).glob('**/*.json'):
		path = filename
		print(filename)
		if os.path.isfile(path):
			f = open(path)
			#print(path, f)
			file_data = json.load(f)
			file_id = f"{file_data['course']}-{file_data['lec']}"
			sentences[file_id] = []
			labels[file_id] = []
			tokens_scooped = 0
			sentence = []
			label = []
			for item in file_data['data']:
				tokenized_sentences, _ = tokenize_and_preserve_labels(item['sentence'], item['word_labels'], tokenizer)
				if tokens_scooped + len(tokenized_sentences) > MAX_LEN or\
					 (not do_merge and tokens_scooped > 0):
					sentences[file_id].append(sentence)
					labels[file_id].append(label)
					sentence = []
					label = []
					tokens_scooped = 0
				sentence.extend(item['sentence'])
				label.extend(item['word_labels'])
				tokens_scooped += len(tokenized_sentences)
			sentences[file_id].append(sentence)
			labels[file_id].append(label)
	return sentences, labels
