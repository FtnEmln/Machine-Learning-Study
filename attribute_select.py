# Feature Selection with Univariate Statistical Tests
from pandas import read_csv
from numpy import set_printoptions
# linear regression feature importance
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot

# load data
filename = 'CarPrice_Assignment.csv'
dataframe = read_csv(filename)
array = dataframe.values

X = array[:,0:22] # all feature except Price
Y = array[:,23] #rarget variable: Price

# define dataset
X, Y = make_regression(n_samples=1000, n_features=23, n_informative=15, random_state=42)
# define the model
model = LinearRegression()
# fit the model
model.fit(X, Y)
# get importance
importance = model.coef_
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.title('Linear Regression Feature Importance')
pyplot.xlabel("feature index")
pyplot.ylabel("Coefficient Value")
pyplot.xticks()
pyplot.show()


