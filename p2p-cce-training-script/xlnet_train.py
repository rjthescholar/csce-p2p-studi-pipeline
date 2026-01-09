import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from eval import classification_report
import sklearn.metrics as sklmet
from sklearn.dummy import DummyClassifier
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import XLNetTokenizer, XLNetConfig, XLNetForTokenClassification
from transformers import BertTokenizer, BertConfig, BertForTokenClassification
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

from helpers import *

# Defining the training function on the 80% of the dataset for tuning the xlnet model
def train(model, optimizer, loader, epoch, gold=True):
    tr_loss, tr_accuracy = 0, 0
    nb_tr_examples, nb_tr_steps = 0, 0
    tr_preds, tr_labels = [], []
    field = 'targets' if gold else 'predictions'
    # put model in training mode
    model.train()

    for idx, batch in enumerate(loader):

        ids = batch['ids'].to(device, dtype = torch.long)
        mask = batch['mask'].to(device, dtype = torch.long)
        targets = batch[field].to(device, dtype = torch.long)

        outputs = model(input_ids=ids, attention_mask=mask, labels=targets)
        loss, tr_logits = outputs.loss, outputs.logits
        tr_loss += loss.item()

        nb_tr_steps += 1
        nb_tr_examples += targets.size(0)

        if idx % 100==0:
            loss_step = tr_loss/nb_tr_steps
            print(f"Training loss per 100 training steps: {loss_step}")

        # compute training accuracy
        flattened_targets = targets.view(-1) # shape (batch_size * seq_len,)
        active_logits = tr_logits.view(-1, model.num_labels) # shape (batch_size * seq_len, num_labels)
        flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size * seq_len,)
        # now, use mask to determine where we should compare predictions with targets (includes [CLS] and [SEP] token predictions)
        active_accuracy = mask.view(-1) == 1 # active accuracy is also of shape (batch_size * seq_len,)
        targets = torch.masked_select(flattened_targets, active_accuracy)
        predictions = torch.masked_select(flattened_predictions, active_accuracy)

        tr_preds.extend(predictions)
        tr_labels.extend(targets)

        tmp_tr_accuracy = accuracy_score(targets.cpu().numpy(), predictions.cpu().numpy())
        tr_accuracy += tmp_tr_accuracy

        # gradient clipping
        torch.nn.utils.clip_grad_norm_(
            parameters=model.parameters(), max_norm=MAX_GRAD_NORM
        )

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    epoch_loss = tr_loss / nb_tr_steps
    tr_accuracy = tr_accuracy / nb_tr_steps
    print(f"Training loss epoch: {epoch_loss}")
    print(f"Training accuracy epoch: {tr_accuracy}")

from time import sleep
def valid(model, loader):
    # put model in evaluation mode
    model.eval()

    eval_loss, eval_accuracy = 0, 0
    nb_eval_examples, nb_eval_steps = 0, 0
    eval_preds, eval_labels = [], []

    with torch.no_grad():
        for idx, batch in enumerate(loader):

            ids = batch['ids'].to(device, dtype = torch.long)
            mask = batch['mask'].to(device, dtype = torch.long)
            targets = batch['targets'].to(device, dtype = torch.long)

            outputs = model(input_ids=ids, attention_mask=mask, labels=targets)
            loss, eval_logits = outputs.loss, outputs.logits

            eval_loss += loss.item()

            nb_eval_steps += 1
            nb_eval_examples += targets.size(0)

            if idx % 100==0:
                loss_step = eval_loss/nb_eval_steps
                print(f"Validation loss per 100 evaluation steps: {loss_step}")

            # compute evaluation accuracy
            flattened_targets = targets.view(-1) # shape (batch_size * seq_len,)
            active_logits = eval_logits.view(-1, model.num_labels) # shape (batch_size * seq_len, num_labels)
            flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size * seq_len,)
            # now, use mask to determine where we should compare predictions with targets (includes [CLS] and [SEP] token predictions)
            active_accuracy = mask.view(-1) == 1 # active accuracy is also of shape (batch_size * seq_len,)
            targets = torch.masked_select(flattened_targets, active_accuracy)
            predictions = torch.masked_select(flattened_predictions, active_accuracy)

            eval_labels.extend(targets)
            eval_preds.extend(predictions)

            tmp_eval_accuracy = accuracy_score(targets.cpu().numpy(), predictions.cpu().numpy())
            eval_accuracy += tmp_eval_accuracy

    #print(eval_labels)
    #print(eval_preds)

    labels = [id2label[id.item()] for id in eval_labels]
    predictions = [id2label[id.item()] for id in eval_preds]

    #print(labels)
    #print(predictions)

    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_steps

    label_chunks, pred_chunks = set(get_chunks(labels)), set(get_chunks(predictions))
    correct_preds = len(label_chunks & pred_chunks)
    total_preds = len(pred_chunks)
    total_correct = len(label_chunks)

    p   = correct_preds / total_preds if correct_preds > 0 else 0
    r   = correct_preds / total_correct if correct_preds > 0 else 0
    new_F  = 2 * p * r / (p + r) if correct_preds > 0 else 0

    chunk_eval_stats = {"precision": p, "recall": r, "F1_score": new_F}
    print(f"Validation Loss: {eval_loss}")
    print(f"Validation Accuracy: {eval_accuracy}")
    print(f"Chunk Precision: {p}")
    print(f"Chunk Recall: {r}")
    print(f"Chunk F1: {new_F}")

    return eval_loss, labels, predictions, chunk_eval_stats

def train_epochs(model, optimizer, train_loader, dev_loader, testing_loader, num_epochs):
  early_stopper = EarlyStopper(patience=2, min_delta=0.01)
  for epoch in range(num_epochs):
    print(f"Training epoch: {epoch + 1}")
    train(model, optimizer, train_loader, epoch)
    loss, labels, predictions, _ = valid(model, dev_loader)
    print(sklmet.classification_report(labels, predictions, target_names=["B", "I", "O"]))
    print(classification_report([labels], [predictions]))
    if early_stopper.early_stop(loss):
        break
    loss, labels, predictions, _ = valid(model, testing_loader)
    print(sklmet.classification_report(labels, predictions, target_names=["B", "I", "O"]))
    print(classification_report([labels], [predictions]))
