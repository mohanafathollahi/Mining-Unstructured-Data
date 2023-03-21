from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, kneighbors_graph
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from utils import toNumpyArray, Classifier

# You may add more classifier methods replicating this function
def applyClassifier(X_train, y_train, X_test, classifier):
    '''
    Task: Given some features train a Naive Bayes classifier
          and return its predictions over a test set
    Input; X_train -> Train features
           y_train -> Train_labels
           X_test -> Test features 
    Output: y_predict -> Predictions over the test set
    '''
    trainArray = toNumpyArray(X_train)
    testArray = toNumpyArray(X_test)
    
    if  classifier == Classifier.NB._value_:
      clf = MultinomialNB()
      clf.fit(trainArray, y_train)
      return clf.predict(testArray)
      
    if  classifier == Classifier.LR._value_:
      clf = LogisticRegression(class_weight='balanced', multi_class='multinomial', solver='saga')
      clf.fit(trainArray, y_train)
      return clf.predict(testArray)

    if  classifier == Classifier.NEIGHBORS._value_:
      clf = KNeighborsClassifier(metric="manhattan")
      clf.fit(trainArray, y_train)
      return clf.predict(testArray)

    if  classifier == Classifier.RANDOMFOREST._value_:
      clf = RandomForestClassifier()
      clf.fit(trainArray, y_train)
      return clf.predict(testArray)

    if  classifier == Classifier.SVM._value_:
      clf = SVC()
      clf.fit(trainArray, y_train)
      return clf.predict(testArray)

#     clf.fit(trainArray, y_train)
    
#     return clf.predict(testArray)