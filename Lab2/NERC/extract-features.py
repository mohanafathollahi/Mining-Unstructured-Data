#! /usr/bin/python3

import sys
import re
from os import listdir

from xml.dom.minidom import parse
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
most_frequent_n_grams = ["in","ne","ro","id","ol","et","am","ri","en","ine","ide","ami","ate","min"]
import pandas as pd

import string


#nltk.download('averaged_perceptron_tagger')

path = "Resources"
dn = "Drug.csv"
db = "Brand.csv"
dg = "Group.csv"
dh = "HSDB.csv"

df_n = pd.read_csv(path+'/'+dn, header=None, sep='|')
df_b = pd.read_csv(path+'/'+db, header=None, sep='|')
df_g = pd.read_csv(path+'/'+dg, header=None, sep='|')
df_h = pd.read_csv(path+'/'+dh, header=None, sep='|')

df_n.iloc[:, 0] = df_n.iloc[:, 0].apply(lambda x : x.lower())
df_b.iloc[:, 0] = df_b.iloc[:, 0].apply(lambda x : x.lower())
df_g.iloc[:, 0] = df_g.iloc[:, 0].apply(lambda x : x.lower())
df_h.iloc[:, 0] = df_h.iloc[:, 0].apply(lambda x : x.lower())

dg_n_list = df_n.iloc[:, 0].values.tolist()
dg_b_list = df_b.iloc[:, 0].values.tolist()
dg_g_list = df_g.iloc[:, 0].values.tolist()
dg_h_list = df_h.iloc[:, 0].values.tolist()

def is_in_external_source_drug(string):
   return string.lower() in dg_h_list or string.lower() in dg_n_list

def is_in_external_source_brand(string):
   return string.lower() in dg_b_list

def is_in_external_source_group(string):
   return string.lower() in dg_g_list


def hasNumbers(inputString):
   return str(any(char.isdigit() for char in inputString))

def chunkstring(string, length):
   return list(set(string[0 + i:length + i] for i in range(0, len(string), length)))

def count_n_gram(inputString):
   two_gram = chunkstring(inputString, 2)
   three_gram = chunkstring(inputString, 3)
   n_gram = two_gram + three_gram
   sum_gram = sum(el in most_frequent_n_grams for el in n_gram)
   if sum_gram > 5:
      return True
   else:
      return False



## --------- tokenize sentence -----------
## -- Tokenize sentence, returning tokens and span offsets


def tokenize(txt):
    offset = 0
    tks = []
    pos =[]
    pos.append(nltk.pos_tag(word_tokenize(txt)))
    for t in word_tokenize(txt):
      offset = txt.find(t, offset)
      tks.append((t,offset, offset+len(t)-1))
      offset += len(t)

    ## tks is a list of triples (word,start,end)
    return tks,pos



## --------- get tag ----------- 
##  Find out whether given token is marked as part of an entity in the XML

def get_tag(token, spans) :
   (form,start,end) = token

   for (spanS,spanE,spanT) in spans :
      if start==spanS and end<=spanE :
         return "B-"+spanT
      elif start>=spanS and end<=spanE :
         return "I-"+spanT

   return "O"


def is_roman_number(num):
   roman_pattern = re.compile(r"""^M{0,3}(CM|CD|D?C{0,3})?(XC|XL|L?X{0,3})?(IX|IV|V?I{0,3})?$""", re.VERBOSE)
   return re.match(roman_pattern, num)
## --------- Feature extractor ----------- 
## -- Extract features for each token in given sentence

def extract_features(tokens_1) :
   lemmatizer = WordNetLemmatizer()


   # for each token, generate list of features and add it to the result
   result = []
   for k in range(0,len(tokens_1[0])):
      tokenFeatures = []
      t = tokens_1[0][k]
      t_word = t[0]
      t_pos = t[1]
      tokenFeatures.append("form="+t_word.lower())
      tokenFeatures.append("suf3="+t_word[-3:])
      tokenFeatures.append("suf2="+t_word[-2:])
      tokenFeatures.append("form.pos="+t_pos)
      tokenFeatures.append("containshyphen=" + str("-" in t_word))
      tokenFeatures.append("isNN=" + str(t_pos == 'NN' or t_pos == 'NNS'))
      tokenFeatures.append("isNNP=" + str(t_pos == 'NNP' or t_pos == 'NNPS'))
      tokenFeatures.append("stemm=" + SnowballStemmer('english').stem(t_word))
      tokenFeatures.append("count_n_gram=" + str(count_n_gram(t_word)))
      tokenFeatures.append("inExtSource_Drug=" + str(is_in_external_source_drug(t_word)))
      tokenFeatures.append("inExtSource_Brand=" + str(is_in_external_source_brand(t_word)))
      tokenFeatures.append("inExtSource_Group=" + str(is_in_external_source_group(t_word)))

      if k>0 : #k = 1,2,...
         tPrev = tokens_1[0][k-1]
         tPrev_word = tPrev[0]
         tPrev_pos = tPrev[1]
         tokenFeatures.append("formPrev="+tPrev_word.lower())
         tokenFeatures.append("formPrev.pos=" + tPrev_pos)
         tokenFeatures.append("lemmaPrev=" + lemmatizer.lemmatize(tPrev_word))
      else :   #k =0
         tokenFeatures.append("BoS")
      if k<len(tokens_1[0])-1 :   #k= 0,1,...n-1
         tNext = tokens_1[0][k+1] #next word
         tNext_word = tNext[0]
         tNext_pos = tNext[1]
         tokenFeatures.append("formNext="+tNext_word.lower())
         tokenFeatures.append("formNext.pos=" + tNext_pos)
         tokenFeatures.append("lemmaNext=" + lemmatizer.lemmatize(tNext_word))


      else:
         tokenFeatures.append("EoS")
    
      result.append(tokenFeatures)
    
   return result


## --------- MAIN PROGRAM ----------- 
## --
## -- Usage:  baseline-NER.py target-dir
## --
## -- Extracts Drug NE from all XML files in target-dir, and writes
## -- them in the output format requested by the evalution programs.
## --


# directory with files to process
datadir = sys.argv[1]
# process each file in directory
for f in listdir(datadir):
   
   # parse XML file, obtaining a DOM tree
   tree = parse(datadir+"/"+f)
#   'Activation of an effector immediate-early gene arc by methamphetamine.\r\n
   # process each sentence in the file
   sentences = tree.getElementsByTagName("sentence")
   for s in sentences :
      sid = s.attributes["id"].value   # get sentence id
      spans = []
      stext = s.attributes["text"].value   # get sentence text
      entities = s.getElementsByTagName("entity")
      for e in entities :
         # for discontinuous entities, we only get the first span
         # (will not work, but there are few of them)
         (start,end) = e.attributes["charOffset"].value.split(";")[0].split("-")
         typ =  e.attributes["type"].value
         spans.append((int(start),int(end),typ))

      # convert the sentence to a list of tokens
      tokens_1 = tokenize(stext)[1]
      tokens_0 = tokenize(stext)[0]

      # extract sentence features
      features = extract_features(tokens_1)

      # print features in format expected by crfsuite trainer
      for i in range (0,len(tokens_0)) :
         # see if the token is part of an entity
         tag = get_tag(tokens_0[i], spans)
         print (sid, tokens_0[i][0], tokens_0[i][1], tokens_0[i][2], tag, "\t".join(features[i]), sep='\t')

      # blank line to separate sentences
      print()
