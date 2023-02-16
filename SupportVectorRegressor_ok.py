
#https://stackabuse.com/implementing-svm-and-kernel-svm-with-pythons-scikit-learn/
#predict dengue case in 2021- continuous data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import seaborn as sns

denguedata = pd.read_csv("useCovid19RelatedData.csv")
X = denguedata.drop('new_deaths', axis=1)
y = denguedata['new_deaths']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)

#Training SVM kenel 'linear','poly', 'rbf', 'sigmoid'
svclassifier = SVR(kernel='poly')
svclassifier.fit(X_train, y_train)

#Make prediction
y_pred = svclassifier.predict(X_test)

#Evaluate SVM accuracy performance

#RMSE is square root of average squared difference between actual value and predicted value
mse = mean_squared_error(y_test, y_pred)
rmse = (np.sqrt(mse))

#MAE is mean (actual value - predicted value)
mae=mean_absolute_error(y_test, y_pred)
print ("RMSE : ", rmse)
print ("MAE : ", mae)


