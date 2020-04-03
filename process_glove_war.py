import pickle
import pathlib
import numpy as np

# Parsing GloVe file
embedding_index = {}

# Local directory
# glove_path = pathlib.Path(r'C:/Users/USUARIO/glove')
# with open(glove_path / 'glove.6B.100d.txt', encoding='UTF-8') as f:
# GCP
with open('./glove/glove.6B.100d.txt', encoding='UTF-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = coefs
        
# Building embedding matrix for our vocabulary
with open("idx2word", "rb") as f:
    idx2word = pickle.load(f)
    
EMBEDDING_DIM = 100

n_matched = 0
embedding_matrix = np.zeros((len(idx2word) + 1, EMBEDDING_DIM))
for idx, word in idx2word.items():
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[idx] = embedding_vector
        n_matched += 1
print(n_matched)

# Saving embedding matrix
np.save('./embedding_matrix_war', embedding_matrix)