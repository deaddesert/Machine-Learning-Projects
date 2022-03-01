'''Recognize/Classify image of handwritten digit using Naive Bayes Classifier
and MNIST dataset'''

# Loading MNIST dataset throughout Keras lib
from keras.datasets import mnist

import numpy as np
# Note: MNIST's data is numpy arrays

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

# Load training and testing data
(train_image, train_label), (test_image, test_label) = mnist.load_data()



class NaiveBayesClassifier:

	def fit(self, X, Y, smoothing=1e+2):
		self.GaussianNB = dict() # P(image|digit)
		self.prior = dict() # P(digit)

		for c in np.unique(Y):
			images_of_c = X[Y == c]
			# Caculate mean and variance for image of each class
			self.GaussianNB[c] = {
				'mean': images_of_c.mean(axis=0), # calculate 'mean' of each class along column
				'variance': images_of_c.var(axis=0) + smoothing # calculate 'variance' of each class along column
			}
			self.prior[c] = float(len(Y[Y == c])/len(Y))

	def predict(self, X, Y):

		images_and_preds = []
		for i in range(X.shape[0]):
			list_of_probs = []
			for c, g in self.GaussianNB.items():
				likelihood = (np.exp(-((X[i]- g['mean'])**2)/(2*(g['variance']))))/(np.sqrt(2*np.pi*g['variance']))
				prob_of_class = np.sum(np.log(likelihood) + np.log(self.prior[c]))

				list_of_probs.append(prob_of_class)

			pred = np.argmax(list_of_probs) # return the index of max prob

			images_and_preds.append([X[i], pred])

		return images_and_preds

	def accuracy(self, y_pred, y_test, digits):

		confusion_matrix = np.zeros((len(digits), len(digits)))

		for i in range(len(y_pred)):
			confusion_matrix[digits.index(y_test[i])][digits.index(y_pred[i])] = confusion_matrix[digits.index(y_test[i])][digits.index(y_pred[i])] + 1

		return confusion_matrix



# Shuffle data before training and testing model
train_image = train_image.reshape(60000, 784)
test_image = test_image.reshape(10000, 784)

total_image = np.zeros([70000, 784])
total_image[:60000, :] = train_image
total_image[60000:, :] = test_image

total_label = np.zeros(70000)
total_label[:60000] = train_label
total_label[60000:] = test_label

total_data = np.zeros([70000, 785])
total_data[:, :-1] = total_image
total_data[:, -1] = total_label
np.random.shuffle(total_data) # Multi-dimesional arrays are only shuffled along the first axis (x-axis)

train_data = total_data[:60000, :]
np.random.shuffle(train_data)

test_data = total_data[60000:, :]
np.random.shuffle(test_data)

classes = list(np.unique(total_label))

# Train NBC Model
model = NaiveBayesClassifier()
model.fit(train_data[:, :-1], train_data[:, -1])

# Test NBC Model
prediction = model.predict(test_data[:, :-1], test_data[:, -1])
pred_image = []
pred_label = []

for i in range(len(prediction)):
	pred_image.append(prediction[i][0])
	pred_label.append(prediction[i][1])


scores_matrix = model.accuracy(pred_label, test_data[:, -1], classes)

accuracy_score = 0
list_of_accuracy = []
for i in range(10):
	accuracy_score += scores_matrix[i][i]
	img_of_digit = 0
	for j in range(10):
		img_of_digit += scores_matrix[i][j]
	list_of_accuracy.append(round(100*(scores_matrix[i][i])/img_of_digit))

# Accuracy Score for each Digit
NBC_scores = pd.DataFrame(np.array([classes, list_of_accuracy]).T,
	 columns = ['Digit', 'Accuracy Score (%)'])
print("\nAccuracy Score of each digit: \n",NBC_scores)

# Overal Score
print('\n Overall Accuracy Score of NBC (%): ', round(100*(accuracy_score)/np.sum(scores_matrix), 4))

# Plot the confusion matrix
plt.figure(figsize=(50,50))
ax = sns.heatmap(scores_matrix, annot=True, cmap="YlGnBu")
bottom,top=ax.get_ylim()
ax.set_ylim(bottom+0.5,top-0.5)
plt.xlabel('Predicted class value')
plt.ylabel('Actual class value')

# Represent the Classification
plt.figure(figsize=(50, 20))
for i in range(20):
	j = np.random.randint(len(test_data[:, -1]))
	plt.subplot(2, 10, i+1)
	img = pred_image[j].reshape(28, 28)
	plt.imshow(img, cmap='Greys')
	plt.xlabel('Image of digit '+str(pred_label[j]), fontsize=9)

plt.show()



















