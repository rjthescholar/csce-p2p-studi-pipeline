import random
import numpy as np
import torch
from torch import cuda
from sklearn.metrics import ConfusionMatrixDisplay
from transformers import XLNetTokenizer

import copy

device = 'cuda' if cuda.is_available() else 'cpu'

tokenizer = XLNetTokenizer.from_pretrained('xlnet-large-cased')

MAX_LEN = 512
TRAIN_BATCH_SIZE = 4
VALID_BATCH_SIZE = 2
EPOCHS = 50
LEARNING_RATE = 1e-05
MAX_GRAD_NORM = 10

best_model_init = False
ROUNDS = 4
SELF_TRAIN_EPOCH = 5

label2id = {
	'B': 0,
	'I': 1,
	'O': 2,
    'X': 3
 }
id2label = {
	0: 'B',
	1: 'I',
	2: 'O',
    3: 'X'
}

train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

valid_params = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

test_params = {'batch_size': 1,
                'shuffle': False,
                'num_workers': 0
                }


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# gets the locations of the chunks (i.e concept boundaries)
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
