import numpy as np
import pandas as pnd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

set = pnd.read_csv("fashion-mnist.csv", sep=',', encoding="cp1252")

y = set['class']
set.drop('class', inplace=True, axis='columns')
X = set
PCA(0.87).fit_transform(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1429, random_state=69)
print("Data prepared")

forest = RandomForestClassifier(n_estimators=250,bootstrap=False,n_jobs=-1,max_features="auto")
pipe = Pipeline(steps=[
    ('poly', PolynomialFeatures(3)),
     ('clf', forest)
])
forest.fit(X_train, y_train)
print("Model was fited")
y_predict = forest.predict(X_test)
print("Точность", round(np.mean(100 * accuracy_score(y_test, y_predict))))
