from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.dummy import DummyClassifier
import sklearn.metrics as sklmet
from seqeval.metrics import classification_report

from helpers import *

def get_predictions(model, loader, model_name=""):
  dataset = copy.deepcopy(loader.dataset)
  dataset.set_model(model, model_name)
  return dataset


def eval_dummy(train_dataset, test_dataset):
  dummy = DummyClassifier(strategy='most_frequent')
  train_real_dataset = {"words": [], "labels": []}
  test_real_dataset = {"words": [], "labels": []}
  for (sentence, labels) in zip(train_dataset["sentence"], train_dataset["word_labels"]):
      train_real_dataset["words"].extend(sentence)
      train_real_dataset["labels"].extend(labels)
  for (sentence, labels) in zip(test_dataset["sentence"], test_dataset["word_labels"]):
      test_real_dataset["words"].extend(sentence)
      test_real_dataset["labels"].extend(labels)
  dummy.fit(train_real_dataset["words"], train_real_dataset["labels"])

  _predictions = dummy.predict(test_real_dataset["words"])
  print("Dummy Accuracy:", dummy.score(_predictions.tolist(), test_real_dataset["labels"]))
  print(classification_report([test_real_dataset["labels"]], [_predictions.tolist()]))

def extract_concepts(dataset, gold=False):
  concepts=set()
  concept=[]
  extracting = False
  field = 'targets' if gold else 'predictions'
  for processed in dataset:
    for i in range(len(processed['ids'])):
      if id2label[processed[field][i].item()] == "O" and extracting:
        concepts.add(tokenizer.decode(concept))
        concept = []
        extracting = False
      if id2label[processed[field][i].item()] == "B":
        if extracting and tokenizer.convert_ids_to_tokens(processed['ids'][i].item()).startswith("▁"):
          concepts.add(tokenizer.decode(concept))
          concept = []
        extracting = True
      if extracting:
        concept.append(processed['ids'][i].item())
  return concepts

def eval_deck(concepts, gold_concepts):
	fn = gold_concepts - concepts
	tp = gold_concepts & concepts
	fp = concepts - gold_concepts
	print(f"TP: {len(tp)}, FP: {len(fp)}, FN: {len(fn)}")
	return (len(tp),len(fp),len(fn))

def eval_concepts(model, deck_loader_list):
  tp, fp, fn = 0, 0, 0
  for deck_loader in deck_loader_list:
    pred_ds = get_predictions(model, deck_loader)
    concepts = extract_concepts(pred_ds)
    gold_concepts = extract_concepts(pred_ds, gold=True)
    print(f"predicted concepts for file: {concepts}")
    print(f"actual concepts for file: {gold_concepts}")
    fn_concepts = gold_concepts - concepts
    tp_concepts = gold_concepts & concepts
    fp_concepts = concepts - gold_concepts
    print(f"TP: {tp_concepts};\n FP: {fp_concepts};\n FN: {fn_concepts}")
    tp_c, fp_c, fn_c = eval_deck(concepts, gold_concepts)
    tp += tp_c
    fp += fp_c
    fn += fn_c
  p = tp / (tp + fp) if tp + fp > 0 else 0
  r = tp / (tp + fn) if tp + fn > 0 else 0
  f1 = 2 * (p * r) / (p + r) if p + r > 0 else 0
  print(f"precision: {p}, recall: {r}, F1: {f1}")
  return {"concept_precision": p, "concept_recall": r, "concept_f1": f1}
