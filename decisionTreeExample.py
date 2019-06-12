import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import tree, svm
from sklearn.ensemble import RandomForestClassifier


fileName = './datasets/weatherPerth.csv'
df = pd.read_csv(fileName)

#data sanisation
df = df.drop(['RISK_MM'], axis=1)
# print(df.mean())

df = df.fillna(df.mean())

determine = 'RainTomorrow'
# split the data int x(training data) and y (results)
y = df[determine]
x = df.drop([determine], axis=1)
x = pd.get_dummies(x)
y = pd.get_dummies(y)

#Split test and training
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2)

#Decision Tree created, fitted and tested
decTree = tree.DecisionTreeClassifier(max_leaf_nodes=3,max_depth=3, random_state=0)
decTree.fit(X_train, y_train)
print(decTree.score(X_test, y_test))
