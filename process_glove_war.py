
import tensorflow as tf
import tensorflow.keras as keras

import pandas as pd
import numpy as np
import pathlib
import re
from nltk.tokenize import sent_tokenize

# Loading data
filepath = './war_peace.txt'
with open(filepath, encoding='UTF-8') as f:
    war = f.read()


def get_sentences(text):
    
    # Remove blank lines or lines with CHAPTER
    sentences = text.split("\n")
    sentences = [sentence for sentence in sentences if sentence != '']
    sentences = [sentence for sentence in sentences if sentence.find('CHAPTER')!=0]
    # Rebuilds the text as it is (easier for sent_tokenize)
    text_clean = '\n'.join(sentences)
    # nltk function
    sentences = sent_tokenize(text_clean)
    # remove \n and quotes 
    sentences = [sentence.replace('\n', '').lower() for sentence in sentences]
    sentences = [re.sub('”|“','', sentence) for sentence in sentences]
    
    return sentences

sentences = get_sentences(war)

# Vocabulary
tokenizer = keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
total_words = len(word_index) + 1 # To account for padding

# Parsing GloVe file
glove_path = pathlib.Path(r'C:/Users/USUARIO/glove')

embedding_index = {}
with open(glove_path / 'glove.6B.100d.txt', encoding='UTF-8') as f:
	for line in f:
		values = line.split()
		word = values[0]
		coefs = np.asarray(values[1:], dtype='float32')
		embedding_index[word] = coefs

# Building embedding matrix for our vocabulary
embedding_dim = 100


embedding_matrix = np.zeros((total_words, embedding_dim))
for word, i in word_index.items():
	if i < total_words:
		embedding_vector = embedding_index.get(word)
		if embedding_vector is not None:
			embedding_matrix[i] = embedding_vector


# Saving embedding matrix
np.save('./embedding_matrix_war', embedding_matrix)