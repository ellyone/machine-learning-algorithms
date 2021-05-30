from matplotlib import pyplot as plt
import numpy as np
import pandas as pnd
from pandas import DataFrame
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from sklearn.decomposition import PCA
import warnings

def hog(data: DataFrame):
    result = np.ndarray()

train_set = pnd.read_csv("fashion-mnist_train.csv", sep=',', encoding="cp1252")
#np.array(train_set)
x_train = train_set.iloc[:, 1:]
y_train = train_set.iloc[:, 0]

test_set = pnd.read_csv("fashion-mnist_test.csv", sep=',', encoding="cp1252")
#np.array(test_set)
x_test = test_set.iloc[:, 1:]
y_test = test_set.iloc[:, 0]

print("Data prepared")

def prepare_data(data: DataFrame):
    data.sort_values(by='class', inplace=True)
    data.reset_index(drop=True, inplace=True)
    y_true = data['class'].to_numpy()
    data.drop('class', inplace=True, axis='columns')
    X = hog(data)
    return X, y_true

bayes = SVC(C=10000.0, degree=2,gamma=0.00005,kernel='poly')
bayes.fit(x_train, y_train)
print("Model was fited\n")
#print(bayes.best_params_)
#model = bayes.best_estimator_
y_fit=bayes.predict(x_test)
res = accuracy_score(y_test, y_fit)
print("Точность = ", round(np.mean(100 * res), 2), "%", sep="")