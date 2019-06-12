import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm


df = pd.read_csv('train_clean.csv')
determine = 'Survived'

# split the data int x(training data) and y (results)
y = df[determine]
x = df.drop([determine], axis=1)
x = pd.get_dummies(x)
#y = pd.get_dummies(y)

#Split test and training
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)

estimator = svm.LinearSVC(random_state=0, tol=1e-5)
estimator.fit(X_train, y_train)
print(estimator.score(X_test, y_test))