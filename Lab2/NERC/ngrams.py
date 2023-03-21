from os import listdir
from xml.dom.minidom import parse

import pandas as pd
from pyparsing import col

datadir = "MUD//Lab2//data//"

devel = "devel"
test = "test"
train = "train"

drug_names = set()

two_gram = set()
three_gram = set()

two_occurrences = {}
three_occurrences = {}

def chunkstring(string, length):
    return set(string[0+i:length+i].lower() for i in range(0, len(string), length))

def add_n_grams(datadir, drug_names, two_gram, three_gram):
    
    for f in listdir(datadir) :
       tree = parse(datadir+"/"+f)
   
       sentences = tree.getElementsByTagName("sentence")
       for s in sentences :
          entities = s.getElementsByTagName("entity")
          for e in entities :
             typ =  e.attributes["type"].value
             if typ in ["drug", "drug_n"]:
                drug = e.attributes["text"].value
                drug_names.add(drug)
            
                two_gram.update(chunkstring(drug, 2))
                three_gram.update(chunkstring(drug, 3))

add_n_grams(f'{datadir}{devel}' , drug_names, two_gram, three_gram)
add_n_grams(f'{datadir}{test}' , drug_names, two_gram, three_gram)
add_n_grams(f'{datadir}{train}' , drug_names, two_gram, three_gram)
            
print(len(two_gram))
print(len(three_gram))

def n_gram_occurrenses(drug_names, two_gram, two_occurrences):
    for drug in drug_names:
        for n_gram in two_gram:
            n = drug.count(n_gram)
            if  n_gram in two_occurrences.keys():
                two_occurrences[n_gram] += n
            else:
                two_occurrences[n_gram] = n

n_gram_occurrenses(drug_names, two_gram, two_occurrences)
n_gram_occurrenses(drug_names, three_gram, three_occurrences)

df_2 = pd.DataFrame.from_dict(two_occurrences, orient='index')
df_3 = pd.DataFrame.from_dict(three_occurrences, orient='index')

df_2.to_csv(f"{datadir}two_occurrences.csv")
df_3.to_csv(f"{datadir}three_occurrences.csv")

# df_2 = pd.DataFrame.from_dict(two_occurrences, orient='index', columns=["n_gram", "occurrence"])
# df_3 = pd.DataFrame.from_dict(three_occurrences, orient='index', columns=["n_gram", "occurrence"])

# print(df_2.columns)
# print(df_3.columns)

# cols = ["occurrence"]

# df_2.columns = cols
# df_3.columns = cols

# print(df_2.columns)
# print(df_3.columns)

# print(df_2.describe())
# print(df_3.describe())

# print(df_2.sort_values(by='occurrence', ascending=True).head(10))
# print(df_3.sort_values(by='occurrence', ascending=True).head(10))
# print()
