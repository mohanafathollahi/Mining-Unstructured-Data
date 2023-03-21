#! /usr/bin/python3
import os
from tensorflow import keras
from keras.utils.vis_utils import plot_model
from keras.initializers import Constant
from codemaps import *
from dataset import *
from tensorflow.keras.layers import Embedding, Dense, Dropout, Conv1D, MaxPool1D, Reshape, Concatenate, Flatten, Bidirectional, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers, Input
from keras.preprocessing.text import Tokenizer
from contextlib import redirect_stdout
import random
import sys
import tensorflow as tf
from matplotlib import pyplot as plt
from numpy.random import seed
seed(240993)
# from tensorflow import set_random_seed
tf.compat.v1.set_random_seed(60299)


config_maxlen = 100
config_batch_size = 32
config_filters = 16
config_kernel_size = 2
config_epochs = 10
n_words = 1000
config_vocab_size = n_words

config_hidden_dims = 250
config_embeddings_dims = 300

config_activation = 'relu'
config_padding = 'same'
config_kernel_regularizer_l2 = 0.001

glove_dir = '/mnt/c/Users/DanielAR.SKYLINE2/Documents/GitHub/MDS/2022/MUD/lab3/06-DDI-nn/DDI/data/glove'

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
    
def build_network(idx, glove_embeddings_index):

    # sizes
    n_words = codes.get_n_words()
    max_len = codes.maxlen
    n_labels = codes.get_n_labels()

    # -----------------------------------------------------------------------------------------------------------------------------------
    inptW = Input(shape=(max_len,))
    # x = Embedding(input_dim=n_words, output_dim=config_embeddings_dims,
    #               input_length=max_len, mask_zero=False)(inptW)

    x = glove_embeddings_index(inptW)
    # -----------------------------------------------------------------------------------------------------------------------------------

    inptL = Input(shape=(max_len,))
    y = Embedding(input_dim=n_words, output_dim=config_embeddings_dims,
                     input_length=max_len, mask_zero=False)(inptL)

    inptP = Input(shape=(max_len,))
    z = Embedding(input_dim=n_words, output_dim=config_embeddings_dims,
                     input_length=max_len, mask_zero=False)(inptP)

    # -----------------------------------------------------------------------------------------------------------------------------------
    x = Conv1D(filters=config_filters, kernel_size=config_kernel_size,
               strides=1, activation='relu', padding='same', name="Conv1D_x")(x)
    maxpool_0 = MaxPool1D(pool_size=(max_len), strides=1, padding='valid')(x)

    # -----------------------------------------------------------------------------------------------------------------------------------
    y = Conv1D(filters=config_filters, kernel_size=config_kernel_size,
               strides=1, activation='relu', padding='same', name="Conv1D_y")(y)
    maxpool_1 = MaxPool1D(pool_size=(max_len), strides=1, padding='valid')(y)
    # -----------------------------------------------------------------------------------------------------------------------------------
    z = Conv1D(filters=config_filters, kernel_size=config_kernel_size,
               strides=1, activation='relu', padding='same', name="Conv1D_z")(y)
    maxpool_2 = MaxPool1D(pool_size=(max_len), strides=1, padding='valid')(z)
    # -----------------------------------------------------------------------------------------------------------------------------------

    final = Concatenate(axis=1, name='Concatenate_MaxPool')([maxpool_0,maxpool_1,maxpool_2])

    # -----------------------------------------------------------------------------------------------------------------------------------

    final = Flatten(name='Flatten')(final)

    final = Dropout(0.1)(final)
    final = Dense(n_labels, activation='softmax', name="Final_Dense")(final)
    
    # -----------------------------------------------------------------------------------------------------------------------------------
    
    # final = Model(inputs=[inptW], outputs=final)
    final = Model(inputs=[inptW, inptL, inptP], outputs=final)
    # -----------------------------------------------------------------------------------------------------------------------------------

    final.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    return final

# --------- MAIN PROGRAM -----------
# --
# -- Usage:  train.py ../data/Train ../data/Devel  modelname
# --

# --------- MAIN PROGRAM -----------
# --
# -- Usage:  train.py ../data/Train ../data/Devel  modelname
# --

# --------------------------------------------------------------------------------


# --------------------------------------------------------------------------------

# directory with files to process
trainfile = sys.argv[1]
validationfile = sys.argv[2]
modelname = sys.argv[3]

# load train and validation data
traindata = Dataset(trainfile)
valdata = Dataset(validationfile)

# create indexes from training data
max_len = config_maxlen
suf_len = 5
codes = Codemaps(traindata, max_len)

# build network
glove_embeddings_index = load_glove_embedding(config_embeddings_dims, codes.lc_word_index)

model = build_network(codes, glove_embeddings_index)
with redirect_stdout(sys.stderr):
    model.summary()

# encode datasets
Xt = codes.encode_words(traindata)
Yt = codes.encode_labels(traindata)
Xv = codes.encode_words(valdata)
Yv = codes.encode_labels(valdata)

# train model
with redirect_stdout(sys.stderr):
    history = model.fit(Xt, Yt, batch_size=config_batch_size,
                        epochs=config_epochs, validation_data=(Xv, Yv), verbose=1)

# save model and indexs

REGULARIZATION_PATH = "/mnt/c/Users/DanielAR.SKYLINE2/Documents/GitHub/MDS/2022/MUD/lab3/06-DDI-nn/regularization/"
MODELS_PATH = REGULARIZATION_PATH + "models/"
INDEX_PATH = REGULARIZATION_PATH + "Index/"

model.save(MODELS_PATH + modelname)
codes.save(MODELS_PATH + modelname)

# path
model_plots_path = REGULARIZATION_PATH + 'plots/' + modelname

# Create the directory
# 'GeeksForGeeks' in
# '/home / User / Documents'
try:
    os.mkdir(model_plots_path)
except OSError as error:
    print(error)


plot_model(model, model_plots_path + '/' +
           modelname + '.png', show_shapes=True)

plt.clf()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig(model_plots_path + '/accuracy' + "_" + modelname + '.png')

plt.clf()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig(model_plots_path + '/loss' + "_" + modelname + '.png')
