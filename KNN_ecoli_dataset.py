import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

class KNearestNeighbors:

	def fit(self, X, y):
		self.train = dict()
		for c in np.unique(y):
			self.train[c] = X[y == c]


	def predict(self, X, k):

		predictions = list()

		for i in X:
			distances = list()
			votes = list()
			for group in self.train.keys():
				for data in self.train[group]:
					euclidean_distance = np.linalg.norm(np.array(data) - np.array(i))
					# np.linalg.norm(np.array(data) - np.array(i)) = np.sqrt(np.sum((np.array(data) - np.array(i))**2))
					distances.append([euclidean_distance, group])
			for j in sorted(distances)[:k]:
				votes.append(j[1]) # add group of K Nereast to 'votes'

			nested_dict = dict()
			for vote in votes:
				if vote not in nested_dict:
					nested_dict[vote] = 1
				else:
					nested_dict[vote] += 1

			predictions.append(pd.Series(nested_dict).idxmax())

		return predictions


	def accuracy_score(self, y_pred, y_test, classes):

		confusion_matrix = np.zeros((len(classes), len(classes)))
		correct = 0

		for i in range(len(y_pred)):
			if y_pred[i] == y_test[i]:
				correct += 1
				confusion_matrix[classes.index(y_pred[i])][classes.index(y_pred[i])] = confusion_matrix[classes.index(y_pred[i])][classes.index(y_pred[i])] + 1 
			else:
				confusion_matrix[classes.index(y_test[i])][classes.index(y_pred[i])] = confusion_matrix[classes.index(y_test[i])][classes.index(y_pred[i])] + 1
	
		plt.figure(figsize=(18,10))
		uniform_data = np.random.rand(10, 12)
		ax = sns.heatmap(confusion_matrix, annot=True, cmap="YlGnBu")
		bottom,top=ax.get_ylim()
		ax.set_ylim(bottom+0.5,top-0.5)
		plt.xlabel('Predicted class value')
		plt.ylabel('Actual class value')
		plt.show()


		return [correct, round(correct/len(y_test), 4)]



df = pd.read_csv('ecoli.data.txt')
df.replace('?', -99999, inplace=True)
df.drop(['sequence_name'], 1, inplace=True)
dataset = np.reshape(df.values, (336, 8))
n_class = list(np.unique(dataset[:, -1]))
np.random.shuffle(dataset)

train = dataset[:300]
np.random.shuffle(train)
train_data = train[:, :-1]
train_label = train[:, -1]

test = dataset[300:]
np.random.shuffle(test)
test_data = test[:, :-1]
test_label = test[:, -1]

model = KNearestNeighbors()

model.fit(train_data, train_label)

pred_label = model.predict(test_data, 11)

print("Predicted labels: ",pred_label[:10])
print("Actual labels:    ",list(test_label[:10]))

score = model.accuracy_score(pred_label, test_label, n_class)

print("Number of correct predictions: ",score[0])

print("Accuracy score: ",score[1])
















