#! /usr/bin/python3

import sys
from os import system

from tensorflow.keras.models import load_model

from dataset import *
from codemaps import *

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score, precision_score, recall_score
import os
from tensorflow import keras
from keras.utils.vis_utils import plot_model
from keras.initializers import Constant
from codemaps import *
from dataset import *
from tensorflow.keras.layers import Embedding, Dense, Dropout, Conv1D, MaxPool1D, Reshape, Concatenate, Flatten, Bidirectional, LSTM
from tensorflow.keras.models import Model
from keras.models import Sequential
from tensorflow.keras import regularizers, Input
from keras.preprocessing.text import Tokenizer
from contextlib import redirect_stdout
import random
import sys
import tensorflow as tf
from matplotlib import pyplot as plt
from numpy.random import seed
from keras.wrappers.scikit_learn import KerasClassifier
from keras.constraints import maxnorm
seed(240993)
# from tensorflow import set_random_seed
tf.compat.v1.set_random_seed(60299)


# --------- MAIN PROGRAM -----------
# --
# -- Usage:  baseline-NER.py target-dir
# --
# -- Extracts Drug NE from all XML files in target-dir
# --
# directory with files to process
trainfile = sys.argv[1]
validationfile = sys.argv[2]
modelname = sys.argv[3]
GloVe = int(sys.argv[4])
optimization = int(sys.argv[5])

print('GloVe', GloVe)

# load train and validation data
traindata = Dataset(trainfile)
valdata = Dataset(validationfile)

config_maxlen = 100
config_batch_size = 32
config_filters = 32
config_kernel_size = 2
config_epochs = 5
n_words = 100
config_vocab_size = n_words

config_hidden_dims = 250

# The more numbers too big overfitting to small under fitting miss info
config_embeddings_dims = 300

config_activation = 'relu'
config_padding = 'same'
config_kernel_regularizer_l2 = 0.001

glove_dir = '/mnt/c/Users/DanielAR.SKYLINE2/Documents/GitHub/MDS/2022/MUD/lab3/06-DDI-nn/DDI/data/glove'

# create indexes from training data
max_len = config_maxlen
suf_len = 5
codes = Codemaps(traindata, max_len)

REGULARIZATION_PATH = "/mnt/c/Users/DanielAR.SKYLINE2/Documents/GitHub/MDS/2022/MUD/lab3/06-DDI-nn/regularization/"
MODELS_PATH = REGULARIZATION_PATH + "models/"
INDEX_PATH = REGULARIZATION_PATH + "Index/"
PARAMS_PATH = REGULARIZATION_PATH + "results/best/"

with open(PARAMS_PATH + modelname + '.txt', 'w') as f:
   f.write("Best paramas: ")
   
def load_glove_embedding(EMBEDDING_DIM, word2index: dict):
    glove_dir_path = f'{glove_dir}/glove.6B.{EMBEDDING_DIM}d.txt'

    n_words = len(word2index)
    embedding_matrix = np.zeros((n_words, EMBEDDING_DIM))

    with open(glove_dir_path, "r") as f:
        for _line in f:
            line = _line.split()
            word = line[0]
            if word in word2index:
                idx = word2index[word]
                embedding_vector = np.array(line[1:], dtype=np.float32)
                embedding_matrix[idx] = embedding_vector

    return Embedding(n_words, EMBEDDING_DIM, weights=[embedding_matrix], trainable=False)


def output_BP(outfile, best_score, best_params):

   outf = open(outfile, 'w')
   print("Best: %f using %s" %
         (grid_result.best_score_, grid_result.best_params_), file=outf)

# Function to create model, required for KerasClassifier
def create_model_epoch_batch():
    # sizes
    n_words = codes.get_n_words()
    max_len = codes.maxlen
    n_labels = codes.get_n_labels()

    # create model
    model = Sequential()
    model.add(Input(shape=(max_len,)))
    model.add(Embedding(input_dim=n_words, output_dim=100,
            input_length=max_len, mask_zero=False))
    model.add(Conv1D(config_filters, config_kernel_size,
            padding='valid', activation='relu',
            kernel_regularizer=regularizers.l2(config_kernel_regularizer_l2)))
    model.add(MaxPool1D())
    model.add(Conv1D(config_filters, config_kernel_size,
            padding='valid', activation='relu',
            kernel_regularizer=regularizers.l2(config_kernel_regularizer_l2)))
    model.add(MaxPool1D())
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    return model

# Function to create model, required for KerasClassifier


def create_model_activation(activation='relu'):
   # create model
 	# create model
   model = Sequential()
   model.add(Dense(5, input_dim=n_words, kernel_initializer='uniform', activation=activation))
   model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
   # Compile model
   model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
   return model

def create_model_drop(dropout_rate=0.0, weight_constraint=0):
	# create model
	model = Sequential()
	model.add(Dense(5, input_dim=n_words, kernel_initializer='uniform', activation='linear', kernel_constraint=maxnorm(weight_constraint)))
	model.add(Dropout(dropout_rate))
	model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

Xt = codes.encode_words(traindata)
Yt = codes.encode_labels(traindata)

param_grid = {}

# define the grid search parameters
if optimization == 1:
    batch_size = [8, 16, 32]
    epochs = [5, 10, 15]
   #  epochs = [5]
    param_grid = dict(batch_size=batch_size, epochs=epochs)
    if GloVe == 1:
        print('glove_embeddings_index')
        glove_embeddings_index = load_glove_embedding(
            config_embeddings_dims, codes.lc_word_index)
        print('finished glove_embeddings_index')
        # model = build_network(codes, glove_embeddings_index)
        model = KerasClassifier(build_fn=create_model_epoch_batch,
                                epochs=5, batch_size=10, verbose=0)
    else:
        model = KerasClassifier(build_fn=create_model_epoch_batch,
                                epochs=5, batch_size=10, verbose=0)
# ------------------------------------------------------------------
if optimization == 2:
    activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'elu']
    param_grid = dict(activation=activation)
    if GloVe == 1:
        print('glove_embeddings_index')
        glove_embeddings_index = load_glove_embedding(
            config_embeddings_dims, codes.lc_word_index)
        print('finished glove_embeddings_index')
        # model = build_network(codes, glove_embeddings_index)
        model = KerasClassifier(build_fn=create_model_activation,
                                epochs=5, batch_size=10, verbose=0)
    else:
        model = KerasClassifier(build_fn=create_model_activation,
                                epochs=5, batch_size=10, verbose=0)
# ------------------------------------------------------------------
if optimization == 3:
    weight_constraint = [1, 2, 3, 4, 5]
    dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    param_grid = dict(dropout_rate=dropout_rate,
                      weight_constraint=weight_constraint)
    if GloVe == 1:
        print('glove_embeddings_index')
        glove_embeddings_index = load_glove_embedding(
            config_embeddings_dims, codes.lc_word_index)
        print('finished glove_embeddings_index')
        # model = build_network(codes, glove_embeddings_index)
        model = KerasClassifier(build_fn=create_model_drop,
                                epochs=5, batch_size=10, verbose=0)
    else:
        model = KerasClassifier(build_fn=create_model_drop,
                                epochs=5, batch_size=10, verbose=0)
# ------------------------------------------------------------------

    # model = build_network(codes, '')

# with redirect_stdout(sys.stderr):
#     model.summary()

grid = GridSearchCV(estimator=model, param_grid=param_grid,
                    scoring='accuracy', refit='accuracy', n_jobs=-1, cv=3)

print("Xt.shape: ", len(Xt))
print("Yt.shape: ", len(Yt))
print("Yt type: ", type(Yt))
print("Yt values: ", Yt[:5])
Yt = Yt.sum(axis=1)
Yt = np.where(Yt > 3, 0, Yt)
# Yt = Yt.flatten()
print("Yt values: ", Yt[:5])

grid_result = grid.fit(Xt, Yt)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
output_BP(PARAMS_PATH + modelname + '.txt',
          grid_result.best_score_, grid_result.best_params_)
# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#     print("%f (%f) with: %r" % (mean, stdev, param))
