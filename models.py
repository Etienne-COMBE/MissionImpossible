import os
import scipy
import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import GridSearchCV ,train_test_split
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.linear_model import LogisticRegression # Regression logistique
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import SGDClassifier
# EX: On va travaille sur l'arbre de Decision
from sklearn import tree
# On va charge la classe "GridSearchCV"
from sklearn.model_selection import GridSearchCV
# Remise à meme echele
from sklearn.preprocessing import StandardScaler
# Dans l'evaluation on trouve plusieurs metriques : pour verifier
from sklearn.metrics import confusion_matrix
# Charge la matrice de confusion + un rapport de classification
from sklearn.metrics import classification_report
# 'mean', 'median', 'most_frequent', 'constant'
from sklearn.impute import SimpleImputer
# remplace des valeurs manquantes: Trouve le nombre de voisin optimals
from sklearn.impute import KNNImputer
# Indiquer ou est ce qu'il manque les données dans la dataset
from sklearn.impute import MissingIndicator
# Regroupe les deux
from sklearn.pipeline import make_union

def arbreDecision(X_train, y_train,X_test, y_test):
    model = DecisionTreeClassifier(criterion='entropy')
    X_test.fillna(X_train.mean(), inplace=True)
    X_train.fillna(X_train.mean(), inplace=True)
    model.fit(X_train, y_train)
    model.predict(X_test)
    model.score(X_test,y_test)
    return model.score(X_test,y_test)


def logisticRegression(X_train, y_train,X_test, y_test):
    model_linear = LogisticRegression()
    model_linear.fit(X_train, y_train)
    model_linear.predict(X_test)
    print("score test : ",model_linear.score(X_test, y_test))
    print("score test : ",model_linear.score(X_train, y_train))
    return model_linear

def foretAleatoire(X_train, y_train,X_test, y_test):
    model_foret = RandomForestClassifier(criterion='gini',n_estimators=100, n_jobs=2, random_state=2, max_depth=6, verbose=2)
    X_test.fillna(X_train.mean(), inplace=True)
    X_train.fillna(X_train.mean(), inplace=True)
    model_foret.fit(X_train, y_train)
    y_pred = model_foret.predict(X_test)
    print("score test : ",model_foret.score(X_test, y_test))
    print("score test : ",model_foret.score(X_train, y_train))
    return model_foret

def knnVoisinage(X_train, y_train,X_test, y_test):
    model_knn = KNeighborsClassifier(n_neighbors= 3)
    model_knn.fit(X_train, y_train)
    model_knn.predict(X_test)
    print("score test : ",model_knn.score(X_test, y_test))
    print("score test : ",model_knn.score(X_train, y_train))
    return model_knn

def entropy(target: pd.Series):
    pdata = target.value_counts()
    # On lui dit utilise moi le logarithme en base 2
    e = scipy.stats.entropy(pdata, base=2)
    return e

def survie(model, pclass=3, sex=0, age=28):
    x = np.array([pclass, sex, age]).reshape(1, 3) 
    print(model.predict(x))
    print(model.predict_proba(x))