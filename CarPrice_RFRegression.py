import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error,median_absolute_error, r2_score, explained_variance_score, max_error,mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor
from pprint import pprint

def forecast_accuracy(forecast, actual):
    print("R2 Score: %.3f" % r2_score(actual, forecast))
    print("Mean Squared Error (MSE): %.3f" % mean_squared_error(actual, forecast))
    print("Root Mean Squared Error (RMSE): %.3f" % mean_squared_error(actual, forecast, squared=False))
    print("Mean Absolute Error (MAE): %.3f" % mean_absolute_error(actual, forecast))
    print("Mean Absolute Percentage Error (MAPE): %.3f" % mean_absolute_percentage_error(actual, forecast))
    print("Median Absolute Error (MAE): %.3f" % median_absolute_error(actual, forecast))
    print("Explained Variance Score: %.3f" % explained_variance_score(actual, forecast))
    print("Max Error: %.3f" % max_error(actual, forecast))


#Attributes:'CarName','fueltype','aspiration','doornumber',
#'carbody','drivewheel','enginelocation','wheelbase','carlength',
#'carwidth','carheight','curbweight','enginetype','cylindernumber',
#'enginesize','fuelsystem','boreratio','stroke','compressionratio',
#'horsepower','peakrpm','citympg','highwaympg','price'


dataset = pd.read_csv('CarPrice_Assignment.csv')
X= dataset.drop(columns=['price'])

#Experiment 1:Feature selection
#Experiment A (SelectKBest)
#X = dataset.drop(columns=['CarName','fueltype','aspiration','doornumber','carbody','carheight','stroke','compressionratio','peakrpm','price'])
#Experiment B (Random Forest Regressor)
#X = dataset.drop(columns=['CarName','enginelocation','carwidth','cylindernumber','compressionratio','price'])
#Experiment C  (Linear Regression)
#X = dataset.drop(columns=['CarName','doornumber','carlength','carwidth','cylindernumber','enginesize','compressionratio','citympg','price'])

y = dataset['price']

#Test size : 0.2,0.3,0.4
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#change hyperparameter
#max_depth
#min_samples_split
#max_leaf_nodes
#min_samples_leaf
#n_estimators
#model = RandomForestRegressor(max_depth,min_samples_split,max_leaf_nodes,
                        #min_samples_leaf,n_estimators, max_sample,max_features)
params = {'max_depth':10 ,
          'min_samples_split':2,
	   'max_leaf_nodes':125,
	  'n_estimators':125
	}
model = RandomForestRegressor(**params)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print('Parameters currently in use:\n')
pprint(model.get_params())

print("\nRANDOM FOREST REGRESSOR\n")
forecast_accuracy(y_pred, y_test)

