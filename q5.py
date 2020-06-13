import numpy as np
import pandas as pd
from numpy.random import RandomState
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import svm
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
import re
from nltk.corpus import stopwords

class AuthorClassifier:
    def __init__(self):
        self.text_clf = Pipeline([('vect', TfidfVectorizer()),
                     ('clf', SVC(C=1000)),
        ])
      
    def accuracy(self,y_real, y_pred):
        accuracy = np.sum(y_real == y_pred) / len(y_real)
        return accuracy
    
    def train(self,path):
        df = pd.read_csv(str(path))
        df = df.iloc[:,1:]
        df.loc[:,"text"] = df.text.apply(lambda x : str.lower(x))
        df.loc[:,"text"] = df.text.apply(lambda x : " ".join(re.findall('[\w]+',x)))
        df = df.replace(to_replace =["and","And","this","This","or","So","in","In","to","To"],value ="") 
        train = df
        X_train,Y_train_temp = train.iloc[:,:], train.iloc[:,:]
        Y_train = []
        for i in np.array(Y_train_temp):
            Y_train.append(i[0])
        
        self.text_clf.fit(X_train.text, X_train.author)
        
    def predict(self, path):
        df = pd.read_csv(str(path))
        df.loc[:,"text"] = df.text.apply(lambda x : str.lower(x))
        df.loc[:,"text"] = df.text.apply(lambda x : " ".join(re.findall('[\w]+',x)))
        df = df.replace(to_replace =["and","And","this","This","or","So","in","In","to","To"],value ="")
        X_validation = df
        predicted = self.text_clf.predict(X_validation.text)
        return predicted