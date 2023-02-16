 #random forest for feature importance on a regression problem
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from matplotlib import pyplot
# Feature Selection with Univariate Statistical Tests
from pandas import read_csv
from numpy import set_printoptions


# load data
filename = 'CarPrice_Assignment.csv'
dataframe = read_csv(filename)
array = dataframe.values

X = array[:,0:22]
y = array[:,23]
# define dataset
X, y = make_regression(n_samples=1000, n_features=23, n_informative=15, random_state=42)
# define the model
model = RandomForestRegressor()
# fit the model
model.fit(X, y)
# get importance
importance = model.feature_importances_
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.title('Random Forest Regressor Feature Importance')
pyplot.xlabel("feature index")
pyplot.ylabel("Mean decrease in impurity")
pyplot.xticks()
pyplot.show()
