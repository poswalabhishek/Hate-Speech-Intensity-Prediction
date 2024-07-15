import pandas as pd
import spacy 
from better_profanity import profanity

# SVO labeling, adding 3 columns with binary arrays in them indicating the presence of SVO
# For ex: "Cows eat grass"
# Subject:[1, 0, 0]
# Verb:   [0, 1, 0]
# Object: [0, 0, 1]
# The sentence has a subject, verb and object

# create class svo in python that returns the SVO tags for a given sentence
class SVO:
    # Using Roberta Base Model Tokenizer https://spacy.io/models/en#en_core_web_trf
    nlp = spacy.load("en_core_web_trf")
    
    def __init__(self, sentence):
        self.sentence = sentence
        self.doc = self.nlp(sentence)
        self.subject = [1 if tok.dep_ == "nsubj" else 0 for tok in self.doc]
        self.verb = [1 if tok.pos_ == "VERB" else 0 for tok in self.doc]
        self.object = [1 if tok.dep_ == "dobj" else 0 for tok in self.doc]

    def get_subject(self):
        return self.subject

    def get_verb(self):
        return self.verb

    def get_object(self):
        return self.object