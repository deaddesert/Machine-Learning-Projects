# Linear Regression using Gradient Descent

import numpy as np
import pandas as pd

class GradientDescent:

	def __init__(self, learning_rate, iterations):
		self.learning_rate = learning_rate
		self.iterations = iterations

	def fit(self, X, y):
		# Scaling Features
		min_of_feature = X.min(axis=0)
		max_of_feature = X.max(axis=0)
		for j in range(X.shape[1]):
			X[:, j] = (np.abs(X[:, j] - min_of_feature[j]))/(max_of_feature[j] - min_of_feature[j])

		one = np.zeros((X.shape[0], 1))
		X = np.concatenate([one, X], axis=1)
		y = y.reshape(y.shape[0], 1)

		self.W = np.zeros((X.shape[1], 1))
		self.dW = np.zeros((X.shape[1], 1))

		for it in range(self.iterations):
			for j in range(X.shape[1]):
				temp = np.dot(X, self.W) - y
				self.dW[j] = (np.dot(temp.T , X[:,j]))/(X.shape[0])

			self.W = self.W - self.learning_rate*self.dW

	def predict(self, X):
		# Scaling Features
		min_of_feature = X.min(axis=0)
		max_of_feature = X.max(axis=0)
		for j in range(X.shape[1]):
			X[:, j] = (np.abs(X[:, j] - min_of_feature[j]))/(max_of_feature[j] - min_of_feature[j])

		one = np.zeros((X.shape[0], 1))
		X = np.concatenate([one, X], axis=1)

		return np.dot(X, self.W)


df = pd.read_csv("house_price.txt")
print(df)

dataset = np.reshape(df.values, (len(df.index), len(df.columns))).astype(dtype=float)
np.random.shuffle(dataset)

train = dataset[:10]
np.random.shuffle(train)
train_data = train[:, :-1]
train_label = train[:, -1]

test = dataset[10:]
np.random.shuffle(test)
test_data = test[:, :-1]
test_label = test[:, -1]

model = GradientDescent(0.01, 1000)

# Training Data
model.fit(train_data, train_label)

# Testing Data
prediction = model.predict(test_data)
print("Predicted labels: ",prediction[:3].T)
print("Actual labels: ",test_label[:3].T)

prediction_error = np.abs(test_label.reshape(10,1) - prediction)
print("Prediction Error: ",prediction_error[:3].T)









