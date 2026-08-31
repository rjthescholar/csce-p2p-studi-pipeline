# Simple Annotation App for Course Concepts

## Requirements

The only requirement to run this annotation tool is Python 3.8 or greater. To run the application, either invoke it in the command line with `./annotate.py` in Linux or `.\annotate.py` in Windows, or run it with the python interpreter in the usual manner.

## Usage

To begin annotating, either press the "Load Text" button to select a textfile (converted from a lecture PDF) or press the "Load BIO" button, if you already have annotated text.

Concepts are displayed with the first word in the concept being colored in green and the subsequent words colored in blue. Words that are not concepts are rendered in grey. To mark a word as a concept, simply left-click the button corresponding to the word. To mark a word as a new concept, right-click that word's button. If you wish to revert the word to being part of the previous concept, simply right-click the word's button again. To clear all of the concepts, simply hit "Clear All".

WARNING: There is no autosave, so save regularly or risk losing data.

To enter in the Metadata for the lecture, simply press the "Enter Metadata" button. This includes the segments of the dataset the lecture belongs to (if unknown, the default is 'labeled'), the course number, and the lecture. To display this metadata, simply press the "Display Metadata" button.

To save your annotations, simply hit the "Save BIO" button. If you have not entered the metadata, it will prompt you to do so before saving the file.

### Auto-annotation

To enable or disable auto-annotation, use the check box in the bottom right. Your annotations will propagate when you begin marking a new concept. If you decide a concept is actually multiple, smaller concepts, and split it, that will also propagate. Hit done annotating if you wish to manually propogate the last edited concept.
