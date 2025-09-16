from torch.utils.data import Dataset
import math

from helpers import *

CLS_TOKEN = "<cls>" #if model_type == 'XLNet' else "[CLS]"
SEP_TOKEN = "<sep>" #if model_type == 'XLNet' else "[SEP]"
PAD_TOKEN = "<pad>" #if model_type == 'XLNet' else "[PAD]"

class dataset(Dataset):
	def __init__(self, dataframe, tokenizer, max_len, model=None, model_name=""):
		self.len = len(dataframe)
		self.data = dataframe
		self.tokenizer = tokenizer
		self.max_len = max_len
		self.model = model
		self.model_name = model_name

	def set_model(self, model, model_name=""):
		del self.model
		torch.cuda.empty_cache()
		self.model = copy.deepcopy(model)
		self.model_name = model_name

	def __getitem__(self, index):
		# step 1: tokenize (and adapt corresponding labels)
		sentence = self.data.sentence[index]
		word_labels = self.data.word_labels[index]
		tokenized_sentence, labels = tokenize_and_preserve_labels(sentence, word_labels, self.tokenizer)

		# step 2: add special tokens (and corresponding labels)
		tokenized_sentence = tokenized_sentence + [SEP_TOKEN] + [CLS_TOKEN]# if model_type =='XLNet' else [CLS_TOKEN] + tokenized_sentence + [SEP_TOKEN] # add special tokens
		#if model_type == "XLNet":
		labels.append("O") # add outside label for [SEP] token
		labels.append("O") # add outside label for [CLS] token
		#else:
		#  labels.insert(0, "O") # add outside label for [CLS] token
		#  labels.append("O") # add outside label for [SEP] token

		# step 3: truncating/padding
		maxlen = self.max_len

		if (len(tokenized_sentence) > maxlen):
		  	# truncate
			tokenized_sentence = tokenized_sentence[:maxlen]
			labels = labels[:maxlen]
		else:
			# pad
			tokenized_sentence = tokenized_sentence + [PAD_TOKEN for _ in range(maxlen - len(tokenized_sentence))]
			labels = labels + ["O" for _ in range(maxlen - len(labels))]

		# step 4: obtain the attention mask
		attn_mask = [1 if tok != PAD_TOKEN else 0 for tok in tokenized_sentence]

		# step 5: convert tokens to input ids
		ids = self.tokenizer.convert_tokens_to_ids(tokenized_sentence)

		label_ids = [label2id[label] for label in labels]

		if self.model is not None:
			self.model.eval()
			ids_t = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
			mask_t = torch.tensor(attn_mask, dtype=torch.long).unsqueeze(0).to(device)
			label_ids_t =  torch.tensor(label_ids, dtype=torch.long).unsqueeze(0).to(device)
			outputs = self.model(input_ids=ids_t, attention_mask=mask_t, labels=label_ids_t)
			loss, eval_logits = outputs.loss, outputs.logits

			active_logits = eval_logits.view(-1, self.model.num_labels) # shape (batch_size * seq_len, num_labels)
			flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size * seq_len,)
		  # now, use mask to determine where we should compare predictions with targets (includes [CLS] and [SEP] token predictions)
		  # active_accuracy = mask_t.view(-1) == 1 # active accuracy is also of shape (batch_size * seq_len,)
		  # prediction_ids = torch.masked_select(flattened_predictions, active_accuracy)
			prediction_ids = flattened_predictions
		else:
			prediction_ids = [label2id['O'] for label in labels]
		# the following line is deprecated
		#label_ids = [label if label != 0 else -100 for label in label_ids]

		return {
			'ids': torch.tensor(ids, dtype=torch.long),
			'mask': torch.tensor(attn_mask, dtype=torch.long),
			#'token_type_ids': torch.tensor(token_ids, dtype=torch.long),
			'targets': torch.tensor(label_ids, dtype=torch.long),
			'predictions': prediction_ids
		}

	def __len__(self):
		return self.len

	def __iter__(self):
		worker_info = torch.utils.data.get_worker_info()
		if worker_info is None:
			return map(self.__getitem__, range(self.__len__()))

		per_worker = int(math.ceil((self.__len__()) / float(worker_info.num_workers)))
		worker_id = worker_info.id
		iter_start = worker_id * per_worker
		iter_end = min(iter_start + per_worker, self.__len__())
		return map(self.__getitem__, range(iter_start, iter_end))
