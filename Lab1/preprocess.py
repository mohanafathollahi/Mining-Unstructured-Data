from enum import Enum
from xml.dom.pulldom import CHARACTERS
from nltk.corpus import stopwords
import nltk
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer
from utils import Method

# Tokenizer function. You can add here different preprocesses.

def preprocess(rows, labels, method):
    '''
    Task: Given a sentences apply all the required preprocessing steps
    to compute train our classifier, such as sentences splitting, 
    tokenization or sentences splitting.

    Input: Sentences in string format
    Output: Preprocessed sentences either as a list or a string
    '''
    # Place your code here
    # Keep in mind that sentences splitting affectes the number of sentencess
    # and therefore, you should replicate labels to match.

    # tokenization
    nltk.data.load

    nltk.download('stopwords')

    preprocessSentences = {}
    return_labels = {}
    stop_words_by_lang = {}
    supported_langs = stopwords.fileids()

    if method == Method.STEAMING._value_:
        supported_stemm_langs = SnowballStemmer.languages

    if method == Method.LEMATIZATION._value_:
        nltk.download('omw-1.4')
        nltk.download('wordnet')
        lemmatizer = WordNetLemmatizer()

    for lang in set(map(lambda lang: str.lower(lang), labels)):
        if lang in supported_langs:
            stop_words_by_lang[lang] = stopwords.words(lang)

    stop_words_keys = stop_words_by_lang.keys()

    indexes = set(map(int, rows.keys()))

    max_i = max(indexes) + 1

    # rows is a series -> {index, value}
    for index, row in rows.items():

        row = GeneralPreprocess(row)
        label = labels[index].lower()
        return_labels[index] = label

        if method == Method.LEMATIZATION._value_:
            Lematization(lemmatizer, row, preprocessSentences, index)
        if method == Method.STEAMING._value_:
            Stemming(supported_stemm_langs, row,
                     label, preprocessSentences, index)
        if method == Method.SENTENCE._value_:
            max_i = SentenceTokenization(
                return_labels, preprocessSentences, max_i, row, label, index)
        if method == Method.TOKENIZATION._value_:
            RemoveStopWords(stop_words_by_lang, row, label,
                            stop_words_keys, preprocessSentences, index)        
        if method == Method.NOTHING._value_:
            preprocessSentences[index] = row

    # return series of sentences {index, value}, series of labels {index, value}
    return pd.Series(preprocessSentences), pd.Series(return_labels)


def SentenceTokenization(return_labels, preprocessSentences, max_i, row, label,index):
    # Sentence Tokenize
    sents_words = nltk.sent_tokenize(row)
    preprocessSentences[index] = row
    if len(sents_words) > 1:
        for sentence in sents_words:
            preprocessSentences[max_i] = sentence
            return_labels[max_i] = label
            max_i += 1
    return max_i

def RemoveStopWords(stop_words_by_lang, row, label, stop_words_keys, preprocessSentences, index):
    filtered = []

    if label in stop_words_keys:
        words = nltk.word_tokenize(row)
        for w in words:
            if not w in stop_words_by_lang[label]:
                filtered.append(w)
    if filtered:
        preprocessSentences[index] = ' '.join(
            map(str, nltk.word_tokenize(row))).strip()
    else:
        preprocessSentences[index] = row


def GeneralPreprocess(row):
    # To lower, remove special characters, numbers, words with - and '
    return row.lower().replace(r'\d+', '').replace(r'[^A-Za-z0-9]+', '').replace(r"(?:[a-z][a-z'\-_]+[a-z])",  '')

def Lematization(lemmatizer, row, preprocessSentences, index):
    words = nltk.word_tokenize(row)
    lemmatize_words = [lemmatizer.lemmatize(word) for word in words]
    preprocessSentences[index] = ' '.join(map(str, lemmatize_words)).strip()
    

def Stemming(supported_stemm_langs, row, label, preprocessSentences, index):
    if label in supported_stemm_langs:
        stemmed = [SnowballStemmer(label).stem(word)
                   for word in nltk.word_tokenize(row)]
        preprocessSentences[index] = ' '.join(map(str, stemmed)).strip()
    preprocessSentences[index] = row


def error():
    print('error')
