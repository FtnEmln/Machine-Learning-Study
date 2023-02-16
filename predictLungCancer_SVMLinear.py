
# classify the whether have lung cancer or not using SVM

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

cancerdata = pd.read_csv ('survey lung cancer.csv')
X = cancerdata.drop('LUNG_CANCER', axis=1)
y = cancerdata['LUNG_CANCER']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.20)

#Training SVM kenel 'linear','poly', 'rbf'
svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)

#Make prediction
y_pred = svclassifier.predict(X_test)

#Evaluate SVM accuracy performance
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))