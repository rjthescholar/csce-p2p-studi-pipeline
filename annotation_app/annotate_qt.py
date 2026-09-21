#!/usr/bin/env python

from functools import partial
import re
import ast
import os
import signal
from PyQt6.QtWidgets import (
	QApplication,
	QMainWindow,
	QSizePolicy,
	QWidget,
	QVBoxLayout,
	QInputDialog,
	QFileDialog,
	QMessageBox,
	QHBoxLayout,
	QScrollArea,
	QPushButton,
	QCheckBox,
	QFrame
)
from PyQt6.QtCore import *
from PyQt6.QtGui import *
import sys

NEWLINE_TOKEN = "__NEWLINE__"
FORMFEED_TOKEN = "__FORMFEED__"


def tokenize(text):
	tokens = []

	for chunk in re.split(r'(\n|\f)', text):

		if chunk == "":
			continue

		if chunk == "\n":
			tokens.append(NEWLINE_TOKEN)

		elif chunk == "\f":
			tokens.append(FORMFEED_TOKEN)

		else:
			tokens.extend(
				re.findall(
					r"\w+|[^\w\s]",
					chunk,
					re.UNICODE
				)
			)

	return tokens

class TokenButton(QPushButton):
	leftClicked = pyqtSignal(int)
	rightClicked = pyqtSignal(int)

	def __init__(self, text, index):
		super().__init__(text)
		self.index = index

	def mousePressEvent(self, event):
		if event.button() == Qt.MouseButton.LeftButton:
			self.leftClicked.emit(self.index)
		elif event.button() == Qt.MouseButton.RightButton:
			self.rightClicked.emit(self.index)

		super().mousePressEvent(event)


class ConceptAnnotator(QMainWindow):

	def __init__(self, tokens=None):
		super().__init__()

		self.setWindowTitle("BIO Concept Annotator")

		self.tokens = tokens or []
		self.labels = ["O"] * len(self.tokens)
		self.selectable_tokens = []

		self.concepts = set()
		self.last_edited_concept = (-1, -1)

		self.buttons = []
		self.token_indices = []

		self.central = QWidget()
		self.setCentralWidget(self.central)

		self.main_layout = QVBoxLayout(self.central)
		self.token_font = QFont("Lucida Sans Unicode", pointSize=10)

		self.metadata = {
			"segments": None,
			"course": None,
			"lecture": None
		}

		self.create_ui()

		if self.tokens:
			self.load_tokens(self.tokens)

	def enter_meta(self):
		segs, ok = QInputDialog.getText(self, "Segments", "Input the segments you want this file to be a part of, separated by commas:", text="labeled")
		self.metadata['segments'] = segs.split(',') if ok else ['labeled']
		self.metadata['course'], ok = QInputDialog.getText(self, "Course", "Enter the course number (e. g. cs0441)")
		self.metadata['lecture'], ok = QInputDialog.getText(self, "Lecture", "Enter the lecture number (e. g. lec01)")

	def display_meta(self):
		QMessageBox.information(self,"Metadata", f"Segments: {self.metadata['segments']}\nCourse: {self.metadata['course']}\nLecture: {self.metadata['lecture']}")

	"""def get_concepts(self):
		concepts = []

		i = 0
		while i < len(self.labels):

			if self.labels[i] == "B-CONCEPT":

				phrase = [self.tokens[i]]

				j = i + 1
				while (
					j < len(self.labels)
					and self.labels[j] == "I-CONCEPT"
				):
					phrase.append(self.tokens[j])
					j += 1

				concepts.append(tuple(phrase))
				i = j

			else:
				i += 1

		return concepts

	def clear_span(self, idx):

		if self.labels[idx] == "O":
			return

		start = idx

		while (
			start > 0
			and self.labels[start] == "I-CONCEPT"
		):
			start -= 1

		end = start + 1

		while (
			end < len(self.labels)
			and self.labels[end] == "I-CONCEPT"
		):
			end += 1

		for i in range(start, end):
			self.labels[i] = "O"

	def get_span(self, idx):

		if self.labels[idx] == "O":
			return None

		start = idx

		while (
			start > 0 and
			self.labels[start] == "I-CONCEPT"
		):
			start -= 1

		end = start + 1

		while (
			end < len(self.labels) and
			self.labels[end] == "I-CONCEPT"
		):
			end += 1

		return start, end

	def propagate_concept(self, concept_tokens, source_start):
		\"\"\"
		Propagate one newly-created concept to other occurrences.

		concept_tokens: tuple/list of tokens in concept
		source_start: start index of manually annotated concept
		\"\"\"

		n = len(concept_tokens)

		concept_norm = tuple(
			token.casefold()
			for token in concept_tokens
		)

		for start in range(len(self.tokens) - n + 1):

			# skip original occurrence
			if start == source_start:
				continue

			candidate = self.tokens[start:start + n]

			candidate_norm = tuple(
				token.casefold()
				for token in candidate
			)

			if candidate_norm != concept_norm:
				continue

			# only annotate completely unlabeled spans
			if any(
				self.labels[start + k] != "O"
				for k in range(n)
			):
				continue

			self.labels[start] = "B-CONCEPT"

			for k in range(1, n):
				self.labels[start + k] = "I-CONCEPT"
	"""

	def get_span(self, idx):
	
			if self.labels[self.token_indices[idx]] == "O":
				return None
	
			start = idx
	
			while (
				start > 0 and
				self.labels[self.token_indices[start]] == "I-Concept"
			):
				start -= 1
	
			end = start + 1
	
			while (
				end < len(self.token_indices) and
				self.labels[self.token_indices[end]] == "I-Concept"
			):
				end += 1
	
			return start, end

	def find_occurrences(self, concept, search_start):

		n = len(concept)

		for start in range(search_start, len(self.selectable_tokens) - n - 1):

			candidate = tuple(
				w.casefold()
				for w in self.selectable_tokens[start:start+n]
			)

			if candidate == concept:
				yield start

	def remove_concept_occurrences(self, concept):

		n = len(concept)

		if concept in self.concepts:
			self.concepts.remove(concept)

		for start in self.find_occurrences(concept, 0):
			if (start+n+1 < len(self.token_indices) and self.labels[self.token_indices[start]] == "B-Concept" and self.labels[self.token_indices[start+n+1]] != "I-Concept"):
				self.labels[self.token_indices[start]] = "O"

				for i in range(1, n):
					self.labels[self.token_indices[start+i]] = "O"

	def apply_concept_occurrences(self, concept, app_start):
		print(concept)
		n = len(concept)

		for start in self.find_occurrences(concept, app_start):

			tok_start = self.token_indices[start]

			if self.labels[tok_start] == "I-Concept":
				continue

			self.labels[tok_start] = "B-Concept"

			for i in range(1, n):
				self.labels[self.token_indices[start+i]] = "I-Concept"

	def create_concept(self, start, end):

		concept = tuple(
			w.casefold()
			for w in self.selectable_tokens[start:end]
		)

		if concept in self.concepts:
			return

		self.concepts.add(concept)

		self.apply_concept_occurrences(concept, start)

	def split_concept(self, idx):

		start, end = self.get_span(idx)

		concept = tuple(
			w.casefold()
			for w in self.selectable_tokens[start:end]
		)

		if concept not in self.concepts:
			return

		offset = idx - start

		left = concept[:offset]
		right = concept[offset:]

		self.concepts.remove(concept)

		self.remove_concept_occurrences(concept)

		if left:
			self.concepts.add(left)
			self.apply_concept_occurrences(left, start)

		if right:
			self.concepts.add(right)
			self.apply_concept_occurrences(right, offset)

	def merge_concept(self, idx):

		current_start, current_end = self.get_span(idx)

		prev_end = idx
		prev_start = idx - 1

		while (
			prev_start > 0 and
			self.labels[self.token_indices[prev_start]] == "I-CONCEPT"
		):
			prev_start -= 1

		left = tuple(
			w.casefold()
			for w in self.selectable_tokens[prev_start:prev_end]
		)

		right = tuple(
			w.casefold()
			for w in self.selectable_tokens[current_start:current_end]
		)

		if left not in self.concepts:
			return

		if right not in self.concepts:
			return

		merged = left + right

		self.concepts.remove(left)
		self.concepts.remove(right)

		self.remove_concept_occurrences(left)
		self.remove_concept_occurrences(right)

		self.concepts.add(merged)

		self.apply_concept_occurrences(merged, prev_start)

	def replace_concept(
		self,
		old_concept,
		new_concept
	):

		self.concepts.discard(old_concept)

		if old_concept:
			self.remove_concept_occurrences(
				old_concept
			)

		self.concepts.add(new_concept)

		self.apply_concept_occurrences(
			new_concept, 0
		)

		self.last_edited_concept = new_concept

	def create_ui(self):

		# ===== Scrollable annotation area =====

		

		central = QWidget()
		self.setCentralWidget(central)

		main_layout = QVBoxLayout(central)

		self.scroll_area = QScrollArea()
		self.scroll_area.setWidgetResizable(True)

		self.token_widget = QWidget()
		self.token_layout = QVBoxLayout(self.token_widget)

		self.scroll_area.setWidget(self.token_widget)

		main_layout.addWidget(self.scroll_area)

		controls = QHBoxLayout()

		btn = QPushButton("Load Text")
		btn.clicked.connect(self.load_text_file)
		controls.addWidget(btn)

		btn = QPushButton("Load BIO")
		btn.clicked.connect(self.load_bio)
		controls.addWidget(btn)

		btn = QPushButton("Save BIO")
		btn.clicked.connect(self.save_bio)
		controls.addWidget(btn)

		btn = QPushButton("Clear All")
		btn.clicked.connect(self.clear_all)
		controls.addWidget(btn)

		btn = QPushButton("Enter Metadata")
		btn.clicked.connect(self.enter_meta)
		controls.addWidget(btn)

		btn = QPushButton("Display Metadata")
		btn.clicked.connect(self.display_meta)
		controls.addWidget(btn)

		btn = QPushButton("Done Annotating")
		btn.clicked.connect(self.finished)
		controls.addWidget(btn)

		self.auto_propagate = QCheckBox(
			"Auto Annotate Matching Concepts"
		)
		self.auto_propagate.setChecked(True)

		controls.addWidget(self.auto_propagate)

		main_layout.addLayout(controls)

	def get_previous_span(self, idx):

		pos = idx - 1

		while pos >= 0:

			if self.labels[pos] == "B-CONCEPT":
				return self.get_span(pos)

			pos -= 1

		return None

	def commit_concept(self, start, end):

		concept = tuple(
			token.casefold()
			for token in self.selectable_tokens[start:end]
		)

		if concept in self.concepts:
			return

		self.concepts.add(concept)

		self.last_edited_concept = (-1, -1)

		self.apply_concept_occurrences(concept, start)

	def finished(self):
		self.commit_concept(self.last_edited_concept[0], self.last_edited_concept[1])

	def left_click(self, idx):

		# Remove label
		if self.labels[self.token_indices[idx]] != "O":
			start, end = self.get_span(idx)
			self.labels[self.token_indices[idx]] = "O"
			if self.auto_propagate.isChecked():
				concept = tuple(
					token.casefold()
					for token in self.selectable_tokens[start:end]
				)

				if concept in self.concepts or (start < idx and end > idx):
					self.remove_concept_occurrences(concept)
					if start < idx:
						self.create_concept(start, idx)
					if end > idx + 1:
						print(f"({start}, {idx}, {end})")
						self.create_concept(idx + 1, end)


			if self.auto_propagate.isChecked() and self.last_edited_concept[0] >= 0 and start != self.last_edited_concept[0]:
				self.commit_concept(self.last_edited_concept[0], self.last_edited_concept[1])
			self.last_edited_concept = (-1, -1)
			print(self.last_edited_concept)
			
			self.repair_bio()
			self.refresh()
			return

		# Add to concept
		if idx > 0 and self.labels[self.token_indices[idx - 1]] in (
			"B-Concept",
			"I-Concept"
		):
			self.labels[self.token_indices[idx]] = "I-Concept"

		elif idx < len(self.buttons) - 1 and \
			self.labels[self.token_indices[idx + 1]] in (
				"B-Concept",
				"I-Concept"
			):
			self.labels[self.token_indices[idx]] = "B-Concept"

		else:
			self.labels[self.token_indices[idx]] = "B-Concept"
			if self.auto_propagate.isChecked() and self.last_edited_concept[0] >= 0:
				self.commit_concept(self.last_edited_concept[0], self.last_edited_concept[1])

		self.last_edited_concept = self.get_span(idx)
		print(self.last_edited_concept)

		self.repair_bio()

		self.refresh()



	def right_click(self, idx):
		if self.labels[self.token_indices[idx]] == "B-Concept" and self.labels[self.token_indices[idx - 1]] != 'O':
			if self.auto_propagate.isChecked():
				start, end = self.get_span(idx)
				start_p, end_p = self.get_span(idx-1)
				if self.auto_propagate.isChecked() and self.last_edited_concept[0] >= 0 and \
					  self.last_edited_concept[0] != start and self.last_edited_concept[0] != start_p:
					self.commit_concept(self.last_edited_concept[0], self.last_edited_concept[1])
				self.last_edited_concept = (start_p, end)
			self.labels[self.token_indices[idx]] = "I-Concept"
		elif self.labels[self.token_indices[idx]] == "I-Concept":
			if self.auto_propagate.isChecked():
				start, end = self.get_span(idx)
				self.labels[self.token_indices[idx]] = "B-Concept"

				concept_w = tuple(
					token.casefold()
					for token in self.selectable_tokens[start:end]
				)

				self.remove_concept_occurrences(concept_w)

				self.create_concept(start, idx)
				self.create_concept(idx, end)

		else:
			if self.auto_propagate.isChecked() and self.last_edited_concept[0] >= 0:
				self.commit_concept(self.last_edited_concept[0], self.last_edited_concept[1])
			self.labels[self.token_indices[idx]] = "B-Concept"
			self.last_edited_concept = (idx, idx + 1)
		self.repair_bio()
		self.refresh()

	def clear_all(self):
		self.labels = ['O'] * len(self.tokens)
		self.concepts = set()
		self.last_edited_concept = (-1, -1)
		self.refresh()

	def refresh(self):

		for i, btn in enumerate(self.buttons):

			label = self.labels[
				self.token_indices[i]
			]

			if label == "O":

				btn.setStyleSheet("""
					QPushButton {
						background-color: lightgray;
						color: black;
					}
				""")

			elif label.startswith("B-"):

				btn.setStyleSheet("""
					QPushButton {
						background-color: #2ecc71;
						color: black						;
					}
				""")

			elif label.startswith("I-"):

				btn.setStyleSheet("""
					QPushButton {
						background-color: #3498db;
						color: black;
					}
				""")

	def repair_bio(self):

		i = 0

		while i < len(self.buttons):

			if self.labels[self.token_indices[i]] == "O":
				i += 1
				continue

			self.labels[self.token_indices[i]] = "B-Concept"

			j = i + 1

			while j < len(self.token_indices) and \
				self.labels[self.token_indices[j]] != "O":
				j += 1

			i = j

	def save_bio(self):
		if self.metadata['segments'] is None:
			self.enter_meta()
		filename, _ = QFileDialog.getSaveFileName(
			self,
			"Save BIO",
			"",
			"CoNLL Files (*.conll)",
		)
		if not filename:
			return
		with open(filename, "w", encoding="utf-8") as f:
			f.write(f"{self.metadata['segments']}|{self.metadata['course']}|{self.metadata['lecture']}\n")
			f.write("-DOCSTART- -X- -X- O\n\n")
			for token, label in zip(self.tokens, self.labels):
				if token == NEWLINE_TOKEN:
					f.write("\n")
					continue
				
				if token == FORMFEED_TOKEN:
					f.write("\f\n")
					continue
				f.write(f"{token} _ _ {label}\n")

	def load_bio(self):

		filename, _ = QFileDialog.getOpenFileName(
			self,
			"Open BIO File",
			"",
			"BIO Files (*.bio *.conll)"
		)

		if not filename:
			return

		words = []
		labels = []
		first_new = True
		first = True

		self.concepts = set()
		self.last_edited_concept = (-1, -1)

		extracted_concept = None

		try:

			with open(filename, "r", encoding="utf-8") as f:

				for line in f:

					stripped = line.rstrip("\n")

					if first:
						segs, self.metadata['course'], self.metadata['lecture'] = stripped.rsplit("|", maxsplit=2)
						self.metadata['segments'] = ast.literal_eval(segs)
						first = False
						continue
					
					# blank line -> newline marker
					if stripped == "":
						if not first_new:
							words.append(NEWLINE_TOKEN)
							labels.append("O")
						first_new = False
						continue

					# form feed marker
					if stripped == "\f":
						words.append(FORMFEED_TOKEN)
						labels.append("O")
						continue

					try:
						token, _, _, label = stripped.rsplit(maxsplit=3)
					except ValueError:
						raise ValueError(
							f"Invalid BIO line:\n{stripped}"
						)
					if token == "-DOCSTART-":
						continue
					if label not in {
						"O",
						"B-Concept",
						"I-Concept"
					}:
						raise ValueError(
							f"Invalid BIO label: {label}"
						)

					if label == 'B-Concept':
						if extracted_concept is not None:
							self.concepts.add(tuple(extracted_concept))
						extracted_concept = [token]
					elif label == 'O':
						if extracted_concept is not None:
							self.concepts.add(tuple(extracted_concept))
						extracted_concept = None
					else:
						extracted_concept.append(token)
				
					words.append(token)
					labels.append(label)

			self.load_tokens(words)

			self.labels = labels

			self.refresh()

		except Exception as e:

			QMessageBox.critical(
				self,
				"Load Error",
				str(e)
			)

	def load_tokens(self, tokens):

		self.tokens = tokens
		self.labels = ["O"] * len(tokens)

		while self.token_layout.count():
			item = self.token_layout.takeAt(0)

			if item.widget():
				item.widget().deleteLater()

		self.buttons = []
		self.token_indices = []

		row_widget = QWidget()
		row_layout = QHBoxLayout(row_widget)
		row_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
		row_layout.setSpacing(1)
		row_layout.setContentsMargins(2, 2, 2, 2)

		self.token_layout.addWidget(row_widget)

		for idx, token in enumerate(tokens):

			if token == NEWLINE_TOKEN:

				row_widget = QWidget()
				row_layout = QHBoxLayout(row_widget)
				row_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
				row_layout.setSpacing(1)
				row_layout.setContentsMargins(2, 2, 2, 2)

				self.token_layout.addWidget(row_widget)

				continue

			if token == FORMFEED_TOKEN:

				sep = QFrame()
				sep.setFrameShape(QFrame.Shape.HLine)

				self.token_layout.addWidget(sep)

				row_widget = QWidget()
				row_layout = QHBoxLayout(row_widget)
				row_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
				row_layout.setSpacing(1)
				row_layout.setContentsMargins(2, 2, 2, 2)

				self.token_layout.addWidget(row_widget)

				continue

			button = TokenButton(
				token,
				len(self.buttons)
			)

			button.setFont(self.token_font)

			button.setSizePolicy(
				QSizePolicy.Policy.Fixed,
				QSizePolicy.Policy.Fixed
			)

			button.setStyleSheet("""
			QPushButton {
				padding: 1px 3px;
			}
			""")

			button.leftClicked.connect(
				self.left_click
			)

			button.rightClicked.connect(
				self.right_click
			)

			row_layout.addWidget(button)

			self.buttons.append(button)
			self.token_indices.append(idx)

		self.selectable_tokens = [
			tokens[i]
			for i in self.token_indices
		]

		self.refresh()
	def load_text_file(self):

		filename, _ = QFileDialog.getOpenFileName(
			self,
			"Select text file",
			"",
		)

		if not filename:
			return

		with open(filename, "r", encoding="utf-8") as f:
			text = f.read()

		tokens = tokenize(text)

		self.metadata = {
			"segments": None,
			"course": None,
			"lecture": None
		}

		self.concepts = set()
		self.last_edited_concept = (-1, -1)

		self.load_tokens(tokens)

if __name__ == "__main__":

	app = QApplication(sys.argv)

	window = ConceptAnnotator([])
	window.resize(1400, 900)
	window.show()

	sys.exit(app.exec())
