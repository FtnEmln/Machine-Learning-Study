# Feature Selection with Univariate Statistical Tests
from pandas import read_csv
from numpy import set_printoptions
# linear regression feature importance
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_regression
from sklearn.model_selection import train_test_split
from matplotlib import pyplot

# load data
filename = 'CarPrice_Assignment.csv'
dataframe = read_csv(filename)
array = dataframe.values
head=dataframe.columns
print(head)

X = array[:,0:23]
y = array[:,23]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
# feature selection
f_selector = SelectKBest(score_func=f_regression, k='all')
# learn relationship from training data
f_selector.fit(X_train, y_train)
# transform train input data
X_train_fs = f_selector.transform(X_train)
# transform test input data
X_test_fs = f_selector.transform(X_test)

# get importance
importance = f_selector.scores_
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# Plot the scores for the features
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.title('SelectKBest Feature Importance')
pyplot.xlabel("feature index")
pyplot.ylabel("F-value (transformed from the correlation values)")
pyplot.xticks()
pyplot.show()
