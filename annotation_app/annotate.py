#!/usr/bin/env python

import tkinter as tk
from tkinter import filedialog, messagebox, ttk, font, simpledialog
from functools import partial
import re
import ast

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

class ConceptAnnotator:
	def __init__(self, root, tokens):
		self.root = root
		self.root.title("BIO Concept Annotator")

		self.tokens = tokens or []
		self.selectable_tokens = []
		# concept id per token (None = O)
		self.labels = ['O'] * len(tokens)

		self.concepts = set()
		self.last_edited_concept = (-1, -1)

		self.buttons = []
		self.token_indices = []
		self.token_font = font.Font(
			family="Lucida Sans Unicode",
			size=10
		)

		self.create_ui()
		self.metadata = {
			"segments": None,
			"course": None,
			"lecture": None
		}
		if self.tokens:
			self.load_tokens(self.tokens)

	def enter_meta(self):
		segs = simpledialog.askstring("Segments", "Input the segments you want this file to be a part of, separated by commas:", initialvalue="labeled")
		self.metadata['segments'] = segs.split(',') if segs else ['labeled']
		self.metadata['course'] = simpledialog.askstring("Course", "Enter the course number (e. g. cs0441)")
		self.metadata['lecture'] = simpledialog.askstring("Lecture", "Enter the lecture number (e. g. lec01)")

	def display_meta(self):
		messagebox.showinfo("Metadata", f"Segments: {self.metadata['segments']}\nCourse: {self.metadata['course']}\nLecture: {self.metadata['lecture']}")

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

		for start in range(search_start, len(self.selectable_tokens) - n + 1):

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
			if (start+n+1 < len(self.token_indices) and self.labels[self.token_indices[start+n+1]] != "I-Concept"):
				self.labels[self.token_indices[start]] = "O"

				for i in range(1, n):
					self.labels[self.token_indices[start+i]] = "O"

	def apply_concept_occurrences(self, concept, app_start):
		print(concept)
		n = len(concept)

		for start in self.find_occurrences(concept, app_start):

			tok_start = self.token_indices[start]

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

		canvas_frame = tk.Frame(self.root)
		canvas_frame.pack(fill="both", expand=True)

		v_scroll = tk.Scrollbar(canvas_frame, orient="vertical")
		v_scroll.pack(side="right", fill="y")

		h_scroll = tk.Scrollbar(canvas_frame, orient="horizontal")
		h_scroll.pack(side="bottom", fill="x")

		self.canvas = tk.Canvas(
			canvas_frame,
			yscrollcommand=v_scroll.set,
			xscrollcommand=h_scroll.set
		)

		self.canvas.pack(side="left", fill="both", expand=True)

		v_scroll.config(command=self.canvas.yview)
		h_scroll.config(command=self.canvas.xview)

		# Frame inside canvas
		self.token_frame = tk.Frame(self.canvas)

		self.canvas_window = self.canvas.create_window(
			(0, 0),
			window=self.token_frame,
			anchor="nw"
		)

		def configure_scroll_region(event):
			self.canvas.configure(
				scrollregion=self.canvas.bbox("all")
			)

		self.token_frame.bind(
			"<Configure>",
			configure_scroll_region
		)

		# ===== token buttons =====

		row = 0
		col = 0

		for idx, token in enumerate(self.tokens):

			btn = tk.Button(
				self.token_frame,
				text=token,
				width=12,
				bg="lightgray"
			)

			btn.bind("<Button-1>", partial(self.left_click, idx))
			btn.bind("<Button-3>", partial(self.right_click, idx))

			btn.grid(
				row=row,
				column=col,
				padx=2,
				pady=2,
				sticky="nsew"
			)

			self.buttons.append(btn)

			col += 1
			if col >= 10:
				row += 1
				col = 0

		# ===== Mouse wheel scrolling =====

		self.canvas.bind_all(
			"<MouseWheel>",
			lambda e: self.canvas.yview_scroll(
				int(-e.delta / 120),
				"units"
			)
		)

		# ===== Controls =====

		control_frame = tk.Frame(self.root)
		control_frame.pack(fill="x", pady=5)

		tk.Button(
			control_frame,
			text="Clear All",
			command=self.clear_all
		).pack(side="left", padx=5)

		tk.Button(
			control_frame,
			text="Load Text",
			command=self.load_text_file
		).pack(side="left", padx=5)

		tk.Button(
			control_frame,
			text="Load BIO",
			command=self.load_bio
		).pack(side="left", padx=5)

		tk.Button(
			control_frame,
			text="Save BIO",
			command=self.save_bio
		).pack(side="left", padx=5)

		tk.Button(
			control_frame,
			text="Enter Metadata",
			command=self.enter_meta
		).pack(side="left", padx=5)

		tk.Button(
			control_frame,
			text="Display Metadata",
			command=self.display_meta
		).pack(side="left", padx=5)

		tk.Button(
			control_frame,
			text="Done Annotating",
			command=self.finished
		).pack(side="left", padx=5)

		self.auto_propagate = tk.BooleanVar(
			value=True
		)
		
		tk.Checkbutton(
			control_frame,
			text="Auto Annotate Matching Concepts",
			variable=self.auto_propagate
		).pack(side="left")

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

	def left_click(self, idx, event):

		# Remove label
		if self.labels[self.token_indices[idx]] != "O":
			start, end = self.get_span(idx)
			self.labels[self.token_indices[idx]] = "O"
			if self.auto_propagate.get():
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
			if self.auto_propagate.get() and self.last_edited_concept[0] >= 0:
				self.commit_concept(self.last_edited_concept[0], self.last_edited_concept[1])

		self.last_edited_concept = self.get_span(idx)
		print(self.last_edited_concept)

		self.repair_bio()

		self.refresh()



	def right_click(self, idx, event):
		if self.labels[self.token_indices[idx]] == "B-Concept" and self.labels[self.token_indices[idx - 1]] != 'O':
			if self.auto_propagate.get():
				start, end = self.get_span(idx)
				start_p, end_p = self.get_span(idx-1)
				if self.auto_propagate.get() and self.last_edited_concept[0] >= 0 and \
					  self.last_edited_concept[0] != start and self.last_edited_concept[0] != start_p:
					self.commit_concept(self.last_edited_concept[0], self.last_edited_concept[1])
				self.last_edited_concept = (start_p, end)
			self.labels[self.token_indices[idx]] = "I-Concept"
		elif self.labels[self.token_indices[idx]] == "I-Concept":
			if self.auto_propagate.get():
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
			if self.auto_propagate.get() and self.last_edited_concept[0] >= 0:
				self.commit_concept(self.last_edited_concept[0], self.last_edited_concept[1])
			self.labels[self.token_indices[idx]] = "B-Concept"
			self.last_edited_concept = (idx, idx)
		self.repair_bio()
		self.refresh()

	def clear_all(self):
		self.labels = ['O'] * len(self.tokens)
		self.concepts = set()
		self.last_edited_concept = (-1, -1)
		self.refresh()

	def refresh(self):
		for i, btn in enumerate(self.buttons):

			label = self.labels[self.token_indices[i]]

			if label == "O":
				btn.configure(bg="lightgray", fg="black")

			elif label.startswith("B-"):
				btn.configure(bg="#2ecc71", fg="white")

			elif label.startswith("I-"):
				btn.configure(bg="#3498db", fg="white")

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
		filename = filedialog.asksaveasfilename(
		defaultextension=".txt"
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

		filename = filedialog.askopenfilename(
			title="Open BIO File",
			filetypes=[
				("BIO files", ["*.bio", "*.conll"]),
				("Text files", "*.txt"),
				("All files", "*.*")
			]
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

			messagebox.showerror(
				"Load Error",
				str(e)
			)


	def load_tokens(self, tokens):

		self.tokens = tokens
		self.labels = ["O"] * len(tokens)

		for widget in self.token_frame.winfo_children():
			widget.destroy()

		self.buttons = []
		self.token_indices = []

		max_tokens_per_row = 20
		row_frame = tk.Frame(self.token_frame)
		row_frame.pack(anchor="w")

		for idx, token in enumerate(self.tokens):

			if token == NEWLINE_TOKEN:
				row_frame = tk.Frame(self.token_frame)
				row_frame.pack(anchor="w")
				continue

			if token == FORMFEED_TOKEN:
				
				ttk.Separator(
					self.token_frame,
					orient="horizontal"
				).pack(fill="x", pady=10)

				row_frame = tk.Frame(self.token_frame)
				row_frame.pack(anchor="w")
				
				continue

			btn = tk.Button(
				row_frame,
				text=f"{token}",
				font=self.token_font,
				bg="lightgray",
				padx=4,
				pady=2
			)

			self.token_indices.append(idx)

			btn.bind("<Button-1>", partial(self.left_click, len(self.buttons)))
			btn.bind("<Button-3>", partial(self.right_click, len(self.buttons)))

			btn.pack(side="left", padx=2, pady=1)

			self.buttons.append(btn)

		self.selectable_tokens = [self.tokens[i] for i in self.token_indices]
		self.refresh()

	def load_text_file(self):

		filename = filedialog.askopenfilename(
			title="Select text file",
			filetypes=[("Text files", "*.txt")]
		)

		if not filename:
			return

		with open(filename, "r", encoding="utf-8") as f:
			text = f.read()

		tokens = tokenize(text)

		self.concepts = set()
		self.last_edited_concept = (-1, -1)

		self.load_tokens(tokens)

if __name__ == "__main__":

    root = tk.Tk()

    app = ConceptAnnotator(root, [])

    root.mainloop()