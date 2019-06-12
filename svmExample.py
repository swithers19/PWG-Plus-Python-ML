import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm

#Training on titatnc
fileName = './datasets/titanic_train.csv'
df = pd.read_csv(fileName)

determine = 'Survived'

# split the data int x(training data) and y (results)
y = df[determine]
x = df.drop([determine], axis=1)
x = pd.get_dummies(x)
#y = pd.get_dummies(y)

#Split test and training
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)

svmModel = svm.LinearSVC(random_state=0, tol=1e-5)
svmModel.fit(X_train, y_train)
print(svmModel.score(X_test, y_test))