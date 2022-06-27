import numpy as np
import pandas as pd


class LinearRegression:

	def training(self, X, y):

		# Add x0 = [1 1 1 ... 1]
		one = np.ones((X.shape[0], 1))
		X = np.concatenate([one, X], axis=1)

		A = np.dot(X.T, X)
		B = np.dot(X.T, y)

		self.w = np.dot(np.linalg.pinv(A), B) # np.linalg.pinv(A) <-> A^(-1)

	def testing(self, X):

		# Add x0 = [1 1 1 ... 1]
		one = np.ones((X.shape[0], 1))
		X = np.concatenate([one, X], axis=1)

		return np.dot(X, self.w)


	def model_accuracy(self, y_pred, y_test):

		return np.abs(np.subtract(y_test, y_pred))



df = pd.read_csv("slump.data.txt")
df.replace('?', -99999, inplace=True)
df.drop(['No'], axis=1, inplace=True)

dataset = np.reshape(df.values, (103, 10)).astype(dtype=float)
np.random.shuffle(dataset)

# Scaling Features
data = dataset[:, :7]
min_of_feature = data.min(axis=0)
max_of_feature = data.max(axis=0)
for j in range(data.shape[1]):
	data[:, j] = (np.abs(data[:, j] - min_of_feature[j]))/(max_of_feature[j] - min_of_feature[j])

dataset[:, :7] = data

# Splitting dataset into training set and testing set
train = dataset[:80]
np.random.shuffle(train)
train_data = train[:, :7]
train_label = train[:, 7:]

test = dataset[80:]
np.random.shuffle(test)
test_data = test[:,:7]
test_label = test[:, 7:]

model = LinearRegression()

# Training model
model.training(train_data, train_label)

# Testing model
pred_label = model.testing(test_data)

print("Predicted labels: ",list(pred_label)[:2])
print("Actual labels: ",list(test_label)[:2])

print("Prediction Error: ",list(model.model_accuracy(pred_label, test_label))[:2])
