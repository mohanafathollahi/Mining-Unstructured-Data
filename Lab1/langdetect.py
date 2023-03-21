from enum import Enum
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
import random
from utils import *
from classifiers import *
from preprocess import preprocess
from time import process_time

seed = 42
random.seed(seed)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input",
                        help="Input data in csv format", type=str)
    parser.add_argument("-v", "--voc_size",
                        help="Vocabulary size", type=int)
    parser.add_argument("-m", "--method",
                        help="preprocess method", type=int)
    parser.add_argument("-c", "--classifier",
                        help="classifier method", type=int)
    parser.add_argument("-a", "--analyzer",
                        help="Tokenization level: {word, char}",
                        type=str, choices=['word', 'char'])
    return parser


def WriteCoverage(args, method, classifier, coverage):
    with open("LangDetect/data/Coverage.csv", "a") as file_object:
        file_object.write(
            f"char_{Method(method).name}, {Classifier(classifier).name}, {coverage}, {args.voc_size}")
        file_object.write('\n')

if __name__ == "__main__":
    t1_start = process_time()
    parser = get_parser()
    args = parser.parse_args()
    raw = pd.read_csv(args.input)
    method = args.method
    classifier = args.classifier

    # Languages
    languages = [x.lower() for x in set(raw['language'])]
    print('========')
    print('Languages', languages)
    print('========')

    # Split Train and Test sets
    X = raw['Text']
    y = raw['language']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed)

    print('========')
    print('Split sizes:')
    print('Train:', len(X_train))
    print('Test:', len(X_test))
    print('========')

    # Preprocess text (Word granularity only)
    if args.analyzer == 'word':
        X_train, y_train = preprocess(X_train, y_train, method)
        X_test, y_test = preprocess(X_test, y_test, method)

    # Compute text features
    features, X_train_raw, X_test_raw = compute_features(X_train,
                                                         X_test,
                                                         analyzer=args.analyzer,
                                                         max_features=args.voc_size)

    print('========')
    print('Number of tokens in the vocabulary:', len(features))
    coverage = compute_coverage(
        features, X_test.values, analyzer=args.analyzer)
    print('Coverage: ', coverage)
    print('========')

    WriteCoverage(args, method, classifier, coverage)

    # Apply Classifier
    X_train, X_test = normalizeData(X_train_raw, X_test_raw)
    y_predict = applyClassifier(X_train, y_train, X_test, classifier)

    print('========')
    print('Prediction Results:')
    plot_F_Scores(y_test, y_predict, method, args.voc_size, classifier)
    print('========')

    plot_Confusion_Matrix(y_test, y_predict, method, args.voc_size, classifier, "Greens")

    # Plot PCA
    print('========')
    print('PCA and Explained Variance:')
    plotPCA(X_train, X_test, y_test, languages, method, args.voc_size, classifier)
    print('========')
    t1_stop = process_time()

    print("Elapsed time:", t1_stop, t1_start)

    print("Elapsed time during the whole program in seconds:", t1_stop-t1_start)
