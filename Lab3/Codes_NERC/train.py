#! /usr/bin/python3

import sys, os
from contextlib import redirect_stdout
import pickle
from tensorflow.keras import Input, utils
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, concatenate, Reshape,Conv1D, Conv2D, MaxPool2D, Concatenate, Flatten, MaxPooling1D, GRU
from keras.callbacks import EarlyStopping
from dataset import *
from codemaps import *
seed_value= 1
utils.set_random_seed(seed_value)

## --------- MAIN PROGRAM ----------- 
## --
## -- Usage:  train.py ../data/Train ../data/Devel  modelname
## --

# directory with files to process
traindir = sys.argv[1]
validationdir = sys.argv[2]
modelname = sys.argv[3]

emb_dim = int(sys.argv[4])
lstm_units = int(sys.argv[5])
max_len = int(sys.argv[6])
fc_dim = int(sys.argv[7])
use_prefix = sys.argv[8].lower() == 'true'
use_pos = sys.argv[9].lower() == 'true'
suf_len = int(sys.argv[10])

use_casing = sys.argv[11].lower() == 'true'

pref_len = 3
num_epoch = 5
batch_size = 64
emb_dim_suffix = emb_dim//2
drop = 0.3
LSTM_DROPOUT = 0.1

glove_dir = os.path.join("../glove_allDims")
def load_glove_embedding(glove_dir_path, EMBEDDING_DIM):
    if EMBEDDING_DIM == 100:
        path_to_glove_file = os.path.join(glove_dir, "glove.6B.100d.txt")
    elif EMBEDDING_DIM == 200:
        path_to_glove_file = os.path.join(glove_dir, "glove.6B.200d.txt")
    elif EMBEDDING_DIM == 300:
        path_to_glove_file = os.path.join(glove_dir, "glove.6B.300d.txt")
    elif EMBEDDING_DIM == 50:
        path_to_glove_file = os.path.join(glove_dir, "glove.6B.50d.txt")
    else:
        assert False, print(f'{EMBEDDING_DIM} is not supported in Glove')

    pickle_path = f'{glove_dir}/glove_{EMBEDDING_DIM}.pkl'
    print(pickle_path)
    if os.path.exists(pickle_path):
        with open(pickle_path, 'rb') as handle:
            glove_embeddings_index = pickle.load(handle)
            print(f'glove embedding loaded from {pickle_path}')
            return glove_embeddings_index
        
    glove_embeddings_index = {}
    with open(path_to_glove_file) as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, "f", sep=" ")
            glove_embeddings_index[word] = coefs

    with open(pickle_path, 'wb') as handle:
        pickle.dump(glove_embeddings_index, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'glove embedding is saved at {pickle_path}')
        
    print("Found %s word vectors in glove." % len(glove_embeddings_index))
    return glove_embeddings_index

def get_preTrained_embedding(glove_embeddings_index, embedding_dim, feature_index):
    words_not_found = []
    vocab = len(feature_index)
    embedding_matrix = np.random.uniform(-0.25, 0.25, size=(vocab, embedding_dim))
    for word, i in feature_index.items():
        if i >= vocab:
            continue
        embedding_vector = glove_embeddings_index.get(word)
        if (embedding_vector is not None) and len(embedding_vector) > 0:
            embedding_matrix[i] = embedding_vector
        else:
            words_not_found.append(word)
        # print('Number of null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))
    print("\tShape of embedding matrix: %s" % str(embedding_matrix.shape))
    print("\tNo. of words not found in GloVe: ", len(words_not_found))
    return embedding_matrix

def build_network(codes, glove_embeddings_index):
    n_lc_words = codes.get_n_lc_words()

    n_sufs = codes.get_n_sufs()
    n_labels = codes.get_n_labels()   
    max_len = codes.maxlen

    n_pos = codes.get_n_pos()
    n_casing = len(codes.case2Idx)
    caseEmbeddings = np.identity(n_casing, dtype='float32')

    
    #n_words = codes.get_n_words()
    # glove_word_emb = get_preTrained_embedding(glove_embeddings_index, emb_dim, codes.word_index)
    #inptW = Input(shape=(max_len,)) # word input layer & embeddings
    #embW = Embedding(input_dim=n_words, output_dim=emb_dim,
    #                  input_length=max_len, mask_zero=True) (inptW)#weights=[glove_word_emb])(inptW)
    
    glove_Lword_emb = get_preTrained_embedding(glove_embeddings_index, emb_dim, codes.lc_word_index)
    inptLW = Input(shape=(max_len,))  # word input layer & embeddings
    embLW = Embedding(input_dim=n_lc_words, output_dim=emb_dim,
                      mask_zero=True,input_length=max_len, weights=[glove_Lword_emb])(inptLW)

    inptS = Input(shape=(max_len,))  # suf input layer & embeddings
    embS = Embedding(input_dim=n_sufs, output_dim=emb_dim_suffix,
                     mask_zero=True,input_length=max_len)(inptS)
        
    dropLW = Dropout(drop)(embLW)
    dropS = Dropout(drop)(embS)
    drops = [dropLW, dropS]

    if use_casing:
        inptc = Input(shape=(max_len,))
        embc = Embedding(input_dim=n_casing, output_dim=n_casing,weights=[caseEmbeddings],
                         input_length=max_len, trainable=False)(inptc)
        drops.append(embc)

    if use_prefix:
        n_prefs = codes.get_n_prefs()
        inptpr = Input(shape=(max_len,))
        embpr = Embedding(input_dim=n_prefs, output_dim=emb_dim_suffix,
                          input_length=max_len)(inptpr)
        dropPr = Dropout(drop)(embpr)
        drops.append(dropPr)

    if use_pos:
        inptp = Input(shape=(max_len,))
        embp = Embedding(input_dim=n_pos, output_dim=emb_dim_suffix,
                         input_length=max_len, mask_zero=True)(inptp)
        dropP = Dropout(drop)(embp)
        drops.append(dropP)

    drops = concatenate(drops)
    bilstm_1 = Bidirectional(LSTM(units=lstm_units, recurrent_dropout=LSTM_DROPOUT, return_sequences=True))(drops)                                  
    bilstm_2 = Bidirectional(LSTM(units=lstm_units, recurrent_dropout=LSTM_DROPOUT, return_sequences=True,))(bilstm_1)
    #timeDistributed = TimeDistributed(Dense(fc_dim, activation="relu"))(bilstm_2)
    out = TimeDistributed(Dense(n_labels, activation="softmax"))(bilstm_2)#(timeDistributed)
      

    input_features = [inptLW, inptS]
    if use_casing:
        input_features.append(inptc)
    if use_prefix:
        input_features.append(inptpr)
    if use_pos:
        input_features.append(inptp)
        
    model = Model(inputs=input_features, outputs=out)
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
   
    return model
   



# load train and validation data
traindata = Dataset(traindir)
valdata = Dataset(validationdir)

# create indexes from training data
codes  = Codemaps(traindata, max_len, suf_len, pref_len)
# encode datasets
Xt = codes.encode_words(traindata)
Yt = codes.encode_labels(traindata)

Xv = codes.encode_words(valdata)
Yv = codes.encode_labels(valdata)

# build network
glove_embeddings_index = load_glove_embedding(glove_dir, emb_dim)
model = build_network(codes, glove_embeddings_index)
with redirect_stdout(sys.stderr) :
   model.summary()




Xt_list = [Xt['lc_word'], Xt['suffix']]
Xv_list = [Xv['lc_word'], Xv['suffix']]
if use_prefix:
    Xt_list.append(Xt['prefix'])
    Xv_list.append(Xv['prefix'])
if use_pos:
    Xt_list.append(Xt['pos'])
    Xv_list.append(Xv['pos'])
if use_casing:
    Xt_list.append(Xt['casing'])
    Xv_list.append(Xv['casing'])
    
# train model
with redirect_stdout(sys.stderr) :
    model.fit(Xt_list, Yt, batch_size=batch_size, epochs=num_epoch, validation_data=(Xv_list,Yv), verbose=1)


model.save(modelname)
codes.save(modelname)

