import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error,median_absolute_error, r2_score, explained_variance_score, max_error,mean_absolute_percentage_error

def forecast_accuracy(forecast, actual):
    print("R2 Score: %.3f" % r2_score(actual, forecast))
    print("Mean Squared Error (MSE): %.3f" % mean_squared_error(actual, forecast))
    print("Root Mean Squared Error (RMSE): %.3f" % mean_squared_error(actual, forecast, squared=False))
    print("Mean Absolute Error (MAE): %.3f" % mean_absolute_error(actual, forecast))
    print("Mean Absolute Percentage Error (MAPE): %.3f" % mean_absolute_percentage_error(actual, forecast))
    print("Median Absolute Error (MAE): %.3f" % median_absolute_error(actual, forecast))
    print("Explained Variance Score: %.3f" % explained_variance_score(actual, forecast))
    print("Max Error: %.3f" % max_error(actual, forecast))

dataset = pd.read_csv('useCovid19RelatedData.csv')

X = dataset.drop(columns=['new_deaths'])
y = dataset['new_deaths']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nRANDOM FOREST REGRESSOR\n")
forecast_accuracy(y_pred, y_test)
