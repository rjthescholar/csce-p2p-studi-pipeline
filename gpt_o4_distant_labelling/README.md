# Concept-GPT-Annotation Pipeline
A small benchmark and automation framework for extracting key concepts
from university slide decks with OpenAI models.

---

## 1 · Lecture-Level Runs

| ID | Model & Prompting Setup | Notes |
|----|-------------------------|-------|
| 1  | **GPT-3.5** + minimal prompt | Baseline |
| 2  | **GPT-3.5** + prompt + annotation-guide | Adds rubric |
| 3  | **o 3** + minimal prompt | Stronger reasoning model |
| 4  | **o 3** + prompt + annotation-guide | Rubric + o-series |

---

## 2 · Course-Level Runs

| ID | Strategy | Model & Prompts | Prior-Knowledge Window |
|----|----------|-----------------|------------------------|
| 5  | **Sequential** – annotate one lecture at a time | o 3 + prompt + *prior-knowledge filter* | Skip the most recent **4** lectures when building prior terms |
| 6  | **All-at-once** – concatenate the entire course and annotate in a single request | o 3 + prompt + annotation-guide | *Not required* (all lectures in one context) |

---

## 3 · Evaluation

* Computed **precision, recall, and F1** against human-curated concept lists.  
* Exact string match, case-normalised, to keep scoring transparent.  

---

## 4 · Outcome

* **Best overall F1:** **o 3 + prompt + prior-knowledge filter** (sequential run).  
* Adopted this configuration as the default for annotating additional courses.

