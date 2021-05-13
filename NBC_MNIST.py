'''Recognize/Classify image of handwritten digit using Naive Bayes Classifier
and MNIST dataset'''

# Loading MNIST dataset throughout Keras lib
from keras.datasets import mnist

import numpy as np
# Note: MNIST's data is numpy arrays

import pandas as pd

import matplotlib.pyplot as plt

# Load training and testing data
(train_image, train_label), (test_image, test_label) = mnist.load_data()

digits = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


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
		self.digit_accuracy = dict()
		self.true_predictions = 0
		images_and_preds = []
		for i in range(X.shape[0]):
			list_of_probs = []
			for c, g in self.GaussianNB.items():
				mean, variance = g['mean'], g['variance']
				likelihood = (np.exp(-((X[i]- g['mean'])**2)/(2*(g['variance']))))/(np.sqrt(2*np.pi*g['variance']))
				prob_of_class = np.sum(np.log(likelihood) + np.log(self.prior[c]))
				list_of_probs.append(prob_of_class)

			pred = np.argmax(list_of_probs) # return the index of max prob

			if pred == Y[i]:
				self.true_predictions += 1
				if Y[i] not in self.digit_accuracy:
					self.digit_accuracy[Y[i]] = 1
				else:
					self.digit_accuracy[Y[i]] += 1

			images_and_preds.append([X[i], pred])

		return images_and_preds

	def overall_score(self, X):
		return round(100*(self.true_predictions)/(X.shape[0]),4)

	def digit_accuracy_score(self, X, Y):
		list_of_accuracy = []
		for i in self.digit_accuracy.keys():
			list_of_accuracy.append(round(100*(self.digit_accuracy[i]/len(X[Y == i])), 4))
		return list_of_accuracy


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

# Train NBC Model
model = NaiveBayesClassifier()
model.fit(train_data[:, :-1], train_data[:, -1])

# Test NBC Model
image_prediction = model.predict(test_data[:, :-1], test_data[:, -1])

# Accuracy Score for each Digit
NBC_scores = pd.DataFrame(np.array([digits, model.digit_accuracy_score(test_data[:, :-1], test_data[:, -1])]).T, 
	 columns = ['Digit', 'Accuracy Score (%)'])
print("\nAccuracy Score of each digit: \n",NBC_scores)

# Overall Accuracy Score for NBC
print("\n Overall Accuracy Score of NBC (%): ",model.overall_score(test_data[:, :-1]))

# Represent the Classification
plt.figure(figsize=(50, 20))
for i in range(20):
	j = np.random.randint(len(test_data[:, -1]))
	plt.subplot(2, 10, i+1)
	img = image_prediction[j][0].reshape(28, 28)
	plt.imshow(img, cmap='Greys')
	plt.xlabel('Image of digit '+str(image_prediction[j][1]), fontsize=9)

plt.show()


















