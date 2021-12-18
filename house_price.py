import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import rc
import unittest
from openpyxl.workbook import Workbook

sns.set(style='whitegrid', palette='muted', font_scale=1.5)

rcParams['figure.figsize'] = 14, 8
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

df_train = pd.read_csv('train.csv')
print(df_train['SalePrice'].describe())
sns.distplot(df_train['SalePrice'])
plt.show()
var = 'GrLivArea'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
g1 = data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000), s=32)
var = 'OverallQual'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(14, 8))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000)
corrmat = df_train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True)
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars']
sns.pairplot(df_train[cols], size=4)
plt.show()

total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/df_train.isnull().count()
           ).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
print(missing_data.head(20))
x = df_train[['OverallQual', 'GrLivArea',
              'GarageCars', 'LotArea', 'OverallCond', 'BsmtUnfSF', 'GarageArea']]
y = df_train['SalePrice']
df_test = pd.read_csv('test.csv')
x_test = df_test[['OverallQual', 'GrLivArea',
                  'GarageCars', 'LotArea', 'OverallCond', 'BsmtUnfSF', 'GarageArea']]

x = (x - x.mean()) / x.std()
x = np.c_[np.ones(x.shape[0]), x]

x_test = (x_test - x_test.mean()) / x_test.std()
x_test = np.c_[np.ones(x_test.shape[0]), x_test]


def loss(h, y):
    sq_error = (h - y)**2
    n = len(y)
    return 1.0 / (2*n) * sq_error.sum()


class LinearRegression:

    def predict(self, X):
        return np.dot(X, self._W)

    def _gradient_descent_step(self, X, targets, lr):

        predictions = self.predict(X)

        error = predictions - targets
        gradient = np.dot(X.T,  error) / len(X)

        self._W -= lr * gradient

    def fit(self, X, y, n_iter=100000, lr=0.01):

        self._W = np.zeros(X.shape[1])

        self._cost_history = []
        self._w_history = [self._W]
        for i in range(n_iter):

            prediction = self.predict(X)
            cost = loss(prediction, y)

            self._cost_history.append(cost)

            self._gradient_descent_step(x, y, lr)

            self._w_history.append(self._W.copy())
        return self


clf = LinearRegression()
clf.fit(x, y, n_iter=2000, lr=0.01)
print(clf._W)
plt.title('Cost Function J')
plt.xlabel('No. of iterations')
plt.ylabel('Cost')
plt.plot(clf._cost_history)
plt.show()
print(clf._cost_history[-1])

col1 = [i for i in range(1461, 2920)]

pred = clf.predict(x_test)
col2 = list(pred)

df = pd.DataFrame.from_dict({'Id': col1, 'SalePrice': col2})
df.to_excel('answer.xlsx', header=True, index=False)
