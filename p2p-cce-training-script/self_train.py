from torch.utils.data import DataLoader

from helpers import *
from xlnet_train import train, valid
from eval import get_predictions, classification_report
import sklearn.metrics as sklmet

def p2p_self_train(model_1, model_2, model_best, optimizer_1, optimizer_2, self_training_loader_1, self_training_loader_2, validation_loader, rounds, self_epochs, best_model_init=False):
	predicted_training_loader_1 = DataLoader(get_predictions(model_1, self_training_loader_1, model_name="model_1"), **train_params)
	predicted_training_loader_2 = DataLoader(get_predictions(model_2, self_training_loader_2, model_name="model_2"), **train_params)
	early_stopper = EarlyStopper(patience=3, min_delta=0.01)
	for round in range(rounds):
		print(f"Training round: {round + 1}")
		for epoch in range(self_epochs):
			print(f"Training epoch: {epoch + 1}")
			print(f"...Training Model 1 on {predicted_training_loader_2.dataset.model_name}'s predictions")
			train(model_1, optimizer_1, predicted_training_loader_2, epoch, gold=False)
			print(f"...Training Model 2 on {predicted_training_loader_1.dataset.model_name}'s predictions")
			train(model_2, optimizer_2, predicted_training_loader_1, epoch, gold=False)
		
			print("MODEL 1 VALIDATION")
			loss_1, labels, predictions, _ = valid(model_1, validation_loader)
			print(sklmet.classification_report(labels, predictions, target_names=["B", "I", "O"]))
			print(classification_report([labels], [predictions]))
			#eval_concepts(model_1, test_block_loader)
			print("MODEL 2 VALIDATION")
			loss_2, labels, predictions, _ = valid(model_2, validation_loader)
			print(sklmet.classification_report(labels, predictions, target_names=["B", "I", "O"]))
			print(classification_report([labels], [predictions]))
			#eval_concepts(model_2, test_block_loader)

		self_training_loader_1, self_training_loader_2 = self_training_loader_2, self_training_loader_1
		print("MODEL 1 VALIDATION")
		loss_1, labels, predictions, _ = valid(model_1, validation_loader)
		print(sklmet.classification_report(labels, predictions, target_names=["B", "I", "O"]))
		print(classification_report([labels], [predictions]))
		#eval_concepts(model_1, test_block_loader)
		print("MODEL 2 VALIDATION")
		loss_2, labels, predictions, _ = valid(model_2, validation_loader)
		print(sklmet.classification_report(labels, predictions, target_names=["B", "I", "O"]))
		print(classification_report([labels], [predictions]))
		#eval_concepts(model_2, test_block_loader)
		model_best = copy.deepcopy(model_1 if loss_1 < loss_2 else model_2)
		model_best.to(device)
		#if early_stopper.early_stop(min(loss_1, loss_2)):
		#	break
		#eval_concepts(model_best, test_block_loader)
		if best_model_init and (round % 2 == 1 or round == ROUNDS - 1):
			model_1 = copy.deepcopy(model_best)
			model_2 = copy.deepcopy(model_best)
			model_1.to(device)
			model_2.to(device)
			optimizer_1 = torch.optim.Adam(params=model_1.parameters(), lr=LEARNING_RATE)
			optimizer_2 = torch.optim.Adam(params=model_2.parameters(), lr=LEARNING_RATE)

	predicted_training_loader_1 = DataLoader(get_predictions(model_1, self_training_loader_1, model_name="model_1"), **train_params)
	predicted_training_loader_2 = DataLoader(get_predictions(model_2, self_training_loader_2, model_name="model_2"), **train_params)
	return model_best

def ts_self_train(teacher_model, student_model, teacher_optimizer, student_optimizer, self_training_loader, validation_loader, rounds, self_epochs):
	predicted_training_loader = DataLoader(get_predictions(teacher_model, self_training_loader, model_name="teacher_model"), **train_params)
	early_stopper_round = EarlyStopper(patience=3, min_delta=0.01)
	for round in range(rounds):
		print(f"Training round: {round + 1}")
		early_stopper = EarlyStopper(patience=3, min_delta=0.01)
		for epoch in range(self_epochs):
			print(f"Training epoch: {epoch + 1}")
			print(f"...Training student_model on {predicted_training_loader.dataset.model_name}'s predictions")
			train(student_model, student_optimizer, predicted_training_loader, epoch, gold=False)
			print("STUDENT VALIDATION")
			loss, labels, predictions, _ = valid(student_model, validation_loader)
			print(sklmet.classification_report(labels, predictions, target_names=["B", "I", "O"]))
			print(classification_report([labels], [predictions]))
			#eval_concepts(student_model, test_block_loader)
			#if early_stopper.early_stop(loss):
			#	break
		print(f"ROUND {round} COMPLETE!")
		teacher_model = copy.deepcopy(student_model)
		print("STUDENT VALIDATION")
		loss, labels, predictions, _ = valid(student_model, validation_loader)
		print(sklmet.classification_report(labels, predictions, target_names=["B", "I", "O"]))
		print(classification_report([labels], [predictions]))
		#eval_concepts(student_model, test_block_loader)
		if early_stopper_round.early_stop(loss):
			break
		predicted_training_loader = DataLoader(get_predictions(teacher_model, self_training_loader, model_name="teacher_model"), **train_params)
	return student_model
