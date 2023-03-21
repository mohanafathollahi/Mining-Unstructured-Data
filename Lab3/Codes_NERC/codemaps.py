
import string
import re

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

from dataset import *

extra_feature=True
class Codemaps :
    # --- constructor, create mapper either from training data, or
    # --- loading codemaps from given file
    def __init__(self, data, maxlen=None, suflen=None, preflen=None) :
        self.case2Idx = {'numeric': 0, 'allLower':1, 'allUpper':2, 'initialUpper':3, 'other':4,'mainly_numeric':5, 'contains_digit': 6, 'PADDING_TOKEN':7, 'contains_dash':8}
        if isinstance(data,Dataset) and maxlen is not None and suflen is not None:
            self.__create_indexs(data, maxlen, suflen,preflen)

        elif type(data) == str and maxlen is None and suflen is None:
            self.__load(data)

        else:
            print('codemaps: Invalid or missing parameters in constructor')
            exit()

    def getCasing(self, word, caseLookup):
        casing = 'other'
        numDigits = 0
        for char in word:
            if char.isdigit():
                numDigits += 1

        digitFraction = numDigits / float(len(word))

        if word.isdigit():  # Is a digit
            casing = 'numeric'
        elif digitFraction > 0.5:
            casing = 'mainly_numeric'
        elif '-' in word:
            casing = 'contains_dash'
        elif word.islower():  # All lower case
            casing = 'allLower'
        elif word.isupper():  # All upper case
            casing = 'allUpper'
        elif word[0].isupper():  # is a title, initial char upper, then all lower
            casing = 'initialUpper'
        elif numDigits > 0:
            casing = 'contains_digit'

        return caseLookup[casing]

    # --------- Create indexs from training data
    # Extract all words and labels in given sentences and 
    # create indexes to encode them as numbers when needed
    def __create_indexs(self, data, maxlen, suflen,preflen) :

        self.maxlen = maxlen
        self.suflen = suflen
        self.preflen = preflen
        lc_words = set([])
        sufs = set([])
        labels = set([])
        if extra_feature:
            prefs = set([])
            pos = set([])
            lemma = set([])
            words = set([])

        for s in data.sentences():
            for t in s :
                lc_words.add(t['lc_form'])#.replace('-', ''))
                suffix = t['lc_form'][-self.suflen:]
                sufs.add(suffix)
                labels.add(t['tag'])
                if extra_feature:
                    prefix = t['lc_form'][:self.preflen]
                    pos.add(t['Pos'])
                    #lemma.add(t['lemma'])
                    #prefs.add(prefix)
                    words.add(t['form'])
                    
        self.label_index = {t: i+1 for i,t in enumerate(list(labels))}
        self.label_index['PAD'] = 0 # Padding

        self.lc_word_index = {w: i + 2 for i, w in enumerate(list(lc_words))}
        self.lc_word_index['PAD'] = 0  # Padding
        self.lc_word_index['UNK'] = 1  # Unknown words

        self.suf_index = {s: i+2 for i,s in enumerate(list(sufs))}
        self.suf_index['PAD'] = 0  # Padding
        self.suf_index['UNK'] = 1  # Unknown suffixes
        
        if extra_feature:
            self.pref_index = {s: i + 2 for i, s in enumerate(list(prefs))}
            self.pref_index['PAD'] = 0  # Padding
            self.pref_index['UNK'] = 1  # Unknown suffixes

            self.pos_index = {s: i + 2 for i, s in enumerate(list(pos))}
            self.pos_index['PAD'] = 0  # Padding
            self.pos_index['UNK'] = 1  # Unknown suffixes
            
            self.lemma_index = {s: i + 2 for i, s in enumerate(list(lemma))}  # length:9180
            self.lemma_index['PAD'] = 0  # Padding
            self.lemma_index['UNK'] = 1  # Unknown suffixes

            self.word_index = {w: i+2 for i,w in enumerate(list(words))}
            self.word_index['PAD'] = 0 # Padding
            self.word_index['UNK'] = 1 # Unknown words

            
        
    ## --------- load indexs ----------- 
    def __load(self, name) : 
        self.maxlen = 0
        self.suflen = 0
        self.preflen = 0

        self.word_index = {}
        self.suf_index = {}
        self.label_index = {}
        self.pos_index = {}
        self.lemma_index = {}
        self.lc_word_index = {}
        self.pref_index = {}


        with open(name+".idx") as f :
            for line in f.readlines():
                (t,k,i) = line.split()
                if t == 'MAXLEN' : self.maxlen = int(k)
                elif t == 'SUFLEN' : self.suflen = int(k)                
                elif t == 'SUF': self.suf_index[k] = int(i)
                elif t == 'LABEL': self.label_index[k] = int(i)
                elif t == 'lc_word': self.lc_word_index[k] = int(i)
                if extra_feature:
                    if t == 'POS': self.pos_index[k] = int(i)
                    #if t == 'CASING': self.case2Idx[K]= int(i)
                    #elif t == 'WORD': self.word_index[k] = int(i)
                    #elif t == 'LEMMA': self.lemma_index[k] = int(i)
                    #elif t == 'PREF': self.pref_index[k] = int(i)
                            
    
    ## ---------- Save model and indexs ---------------
    def save(self, name) :
        # save indexes
        with open(name+".idx","w") as f :
            print('MAXLEN', self.maxlen, "-", file=f)
            print('SUFLEN', self.suflen, "-", file=f)
            for key in self.label_index: print('LABEL', key, self.label_index[key], file=f)
            for key in self.suf_index : print('SUF', key, self.suf_index[key], file=f)
            for key in self.lc_word_index: print('lc_word', key, self.lc_word_index[key], file=f)
            if extra_feature:
                #for key in self.word_index: print('WORD', key, self.word_index[key], file=f)
                #for key in self.pref_index : print('PREF', key, self.pref_index[key], file=f)
                #for key in self.lemma_index:  print('LEMMA', key, self.lemma_index[key], file=f)
                #for key in self.case2Idx:  print('CASING', key, self.case2Idx[key], file=f)
                for key in self.pos_index:  print('POS', key, self.pos_index[key], file=f)



    ## --------- encode X from given data ----------- 
    def encode_words(self, data) :        
        # encode and pad sentence words
        #Xw = [[self.word_index[w['form']] if w['form'] in self.word_index else self.word_index['UNK'] for w in s] for s in data.sentences()]
        #Xw = pad_sequences(maxlen=self.maxlen, sequences=Xw, padding="post", value=self.word_index['PAD'])

        Xcasing = [[self.getCasing(w['form'], self.case2Idx) for w in s] for s in data.sentences()]
        Xcasing = pad_sequences(maxlen=self.maxlen, sequences=Xcasing, padding="post", value=self.case2Idx['PADDING_TOKEN'])
        
        Xs = [[self.suf_index[w['lc_form'][-self.suflen:]] if w['lc_form'][-self.suflen:] in self.suf_index
               else self.suf_index['UNK'] for w in s] for s in data.sentences()]
        Xs = pad_sequences(maxlen=self.maxlen, sequences=Xs, padding="post", value=self.suf_index['PAD'])

        # encode and pad lc_word
        Xlc = [[self.lc_word_index[w['lc_form']] if w['lc_form'] in self.lc_word_index else
               self.lc_word_index['UNK'] for w in s] for s in data.sentences()]
        Xlc = pad_sequences(maxlen=self.maxlen, sequences=Xlc, padding="post", value=self.lc_word_index['PAD'])

        out = dict()
        out['lc_word']=Xlc
        out['suffix']= Xs
        if extra_feature:
            Xp = [[self.pos_index[w['Pos']] if w['Pos'] in self.pos_index else self.pos_index['UNK']
                   for w in s] for s in data.sentences()]
            Xp = pad_sequences(maxlen=self.maxlen, sequences=Xp, padding="post", value=self.pos_index['PAD'])
            #Xpr = [[self.pref_index[w['lc_form'][:self.preflen]] if w['lc_form'][:self.preflen] in self.pref_index
            #        else self.pref_index['UNK'] for w in s] for s in data.sentences()]
            #Xpr = pad_sequences(maxlen=self.maxlen, sequences=Xpr, padding="post", value=self.pref_index['PAD'])
            #Xl = [[self.lemma_index[w['lemma']] if w['lemma'] in self.lemma_index else self.lemma_index['UNK']
                    #for w in s] for s in data.sentences()]
            #Xl = pad_sequences(maxlen=self.maxlen, sequences=Xl, padding="post", value=self.lemma_index['PAD'])
            #out['prefix'] = Xpr
            #out['lemma'] = Xl
            out['pos'] = Xp
            out['casing'] = Xcasing
        return out
        

    ## --------- encode Y from given data -----------
    def encode_labels(self, data) :
        # encode and pad sentence labels 
        Y = [[self.label_index[w['tag']] for w in s] for s in data.sentences()]
        Y = pad_sequences(maxlen=self.maxlen, sequences=Y, padding="post", value=self.label_index["PAD"])
        return np.array(Y)

    ## -------- get word index size ---------
    def get_n_words(self) :
        return len(self.word_index)

    def get_n_lc_words(self):
        return len(self.lc_word_index)

    ## -------- get suf index size ---------
    def get_n_sufs(self) :
        return len(self.suf_index)

    ## -------- get pref index size ---------
    def get_n_prefs(self):
        return len(self.pref_index)

    ## -------- get label index size ---------
    def get_n_labels(self) :
        return len(self.label_index)

    ## -------- get pos index size ---------
    def get_n_pos(self):
        return len(self.pos_index)

    ## -------- get lemma index size ---------
    def get_n_lemma(self):
        return len(self.lemma_index)

    ## -------- get index for given word ---------
    def word2idx(self, w) :
        return self.word_index[w]
    ## -------- get index for given suffix --------
    def suff2idx(self, s) :
        return self.suff_index[s]
    ## -------- get index for given label --------
    def label2idx(self, l) :
        return self.label_index[l]
    ## -------- get label name for given index --------
    def idx2label(self, i) :
        for l in self.label_index :
            if self.label_index[l] == i:
                return l
        raise KeyError

