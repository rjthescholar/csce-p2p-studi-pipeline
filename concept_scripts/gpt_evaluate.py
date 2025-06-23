import os
from pathlib import Path
from seqeval.metrics import classification_report
import itertools
import json

is_flat = False
window = True
WINDOW_SIZE=1

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
          if not is_flat:
            if not window:
              for item in file_data:
                sentences.append(item['words'])
                labels.append(item['labels'])
            else:
              for i in range(len(file_data)):
                window_words = []
                window_labels = []
                for item in file_data[i:i+WINDOW_SIZE]:
                  window_words.extend(item['words'])
                  window_labels.extend(item['labels'])
                sentences.append(window_words)
                labels.append(window_labels)
          else:
            sentences.append(file_data['words'])
            labels.append(file_data['labels'])
  return (sentences, labels)

def get_chunks(seq):
    """
    Adapted from BOND paper
    """
    chunks = []

    chunk_start = None
    for i, tok in enumerate(seq):
        if tok == "O" and chunk_start is not None:
            chunk = (chunk_start, i)
            chunks.append(chunk)
            chunk_start = None
        if tok == "B":
            if chunk_start is not None:
                chunk = (chunk_start, i)
                chunks.append(chunk)
            chunk_start=i
    if chunk_start is not None:
        chunk = (chunk_start, len(seq))
        chunks.append(chunk)
    return chunks

sentences, labels = get_data('final_slides_test')
sentences, predictions = get_data('final_slides_test_gpt')

labels = [label for sentence in labels for label in sentence]
predictions = [label for sentence in predictions for label in sentence]

label_chunks, pred_chunks = set(get_chunks(labels)), set(get_chunks(predictions))
print(label_chunks)
print(pred_chunks)
correct_preds = len(label_chunks & pred_chunks)
total_preds = len(pred_chunks)
total_correct = len(label_chunks)

p   = correct_preds / total_preds if correct_preds > 0 else 0
r   = correct_preds / total_correct if correct_preds > 0 else 0
new_F  = 2 * p * r / (p + r) if correct_preds > 0 else 0

print(f"Chunk Precision: {p}")
print(f"Chunk Recall: {r}")
print(f"Chunk F1: {new_F}")
print(classification_report([labels], [predictions]))