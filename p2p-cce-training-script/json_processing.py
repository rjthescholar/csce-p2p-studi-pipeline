import json
import os
from pathlib import Path
import itertools

# Input: data_path
# Output: All sentences and labels in the path.
def get_data(data_path):
	sentences = []
	labels = []
	for dir in Path(data_path).glob('*'):
		for filename in Path(dir).glob('**/*.json'):
			path = filename
			if os.path.isfile(path):
				f = open(path)
				print(path, f)
				file_data = json.load(f)
				for item in file_data['data']:
					sentences.append(item['sentence'])
					labels.append(item['word_labels'])
	return (sentences, labels)

# Input: data_path
# Output: All sentences and labels in the path, separated by course and file.
def get_course_data(data_path):
	sentences = dict()
	labels = dict()
	for dir in Path(data_path).glob('*'):
		for filename in Path(dir).glob('**/*.json'):
			path = filename
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
				for item in file_data['data']:
					sentences[course][lec].append(item['sentence'])
					labels[course][lec].append(item['word_labels'])
	return (sentences, labels)

# Input: data_path
# Output: All sentences and labels in the path, separated by file.
def get_fs_data(data_path):
	sentences = dict()
	labels = dict()
	for dir in Path(data_path).glob('*'):
		for filename in Path(dir).glob('**/*.json'):
			path = filename
			if os.path.isfile(path):
				f = open(path)
				print(path, f)
				file_data = json.load(f)
				file_id = f"{file_data['course']}-{file_data['lec']}"
				sentences[file_id] = []
				labels[file_id] = []
				for item in file_data['data']:
					sentences[file_id].append(item['sentence'])
					labels[file_id].append(item['word_labels'])
	return sentences, labels
