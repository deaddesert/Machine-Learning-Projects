import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class KMeansClustering:

	def __init__(self, k, tol, max_iter):
		self.k = k 
		self.tol = tol
		self.max_iter = max_iter


	def fit(self, X):
		self.centroids = dict()
		self.clusters = dict()
		cost_function_list = list()

		for i in range(self.max_iter):

			self.centroids[i] = []
			self.clusters[i] = []
			cost_function = 0
			optimization = 0

			for j in range(self.k):
				self.centroids[i].append(X[j])

			while (optimization == 0):
				K_clusters = dict()
				for j in range(self.k):
					K_clusters[j] = []

				for featureset in X:
					distances = [np.linalg.norm(featureset - centroid) for centroid in self.centroids[i]]
					cluster = np.argmin(distances)
					K_clusters[cluster].append(featureset)

				prev_centroids = self.centroids[i]

				for cluster in K_clusters:
					self.centroids[i][cluster] = np.mean(K_clusters[cluster], axis=0)


				for j in range(len(prev_centroids)):
					if np.sum((self.centroids[i][j] - prev_centroids[j])/prev_centroids[j]*100.0) <= self.tol:
						optimization = 1

			for cluster in K_clusters:
				self.clusters[i].append(K_clusters[cluster])

			for j in range(len(self.clusters[i])):
				for x in self.clusters[i][j]:
					cost_function += ((1/len(X))*((x - self.centroids[i][j])**2))

			cost_function_list.append(cost_function)

		self.min = np.argmin(cost_function_list)

		return [self.centroids[self.min], self.clusters[self.min]]


	def predict(self, X):
		K_clusters = dict()
		for j in range(self.k):
			K_clusters[j] = []

		for featureset in X:
			distances = [np.linalg.norm(featureset - centroid) for centroid in self.centroids[self.min]]
			cluster = np.argmin(distances)
			K_clusters[cluster].append(featureset)

		return K_clusters


# Data
df = pd.read_csv('Live_20210128.csv')
df.drop(['status_id', 'status_published'], axis=1, inplace=True)

text_digit_vals = dict()

def convert_to_digit(val):
	return text_digit_vals[val]

post_types = np.unique(df['status_type'].values)
digit = 0

for post_type in post_types:
	if post_type not in text_digit_vals:
		text_digit_vals[post_type] = digit 
		digit += 1
df['status_type'] = list(map(convert_to_digit, df['status_type']))

dataset = np.reshape(df.values, (len(df.index), len(df.columns)))
np.random.shuffle(dataset)

train_data = dataset[:6550]
np.random.shuffle(train_data)

test_data = dataset[6550:]

model = KMeansClustering(2, 0.0001, 100)

# Training model
training = model.fit(train_data)
centroids = training[0]
clusters = training[1]

for centroid in centroids:
	print([centroid[1], centroid[2]])

# We will represent clusters and centroids based on num_reactions and num_comments

plt.axis([0, 2500, 0, 1600])
for features in clusters[0][:200]:
	plt.plot(features[1], features[2], 'ro')

for features in clusters[1][:200]:
	plt.plot(features[1], features[2], 'bo')

for centroid in centroids:
	plt.scatter(centroid[1], centroid[2], marker="x",s=150, linewidths=5)

plt.xlabel("Number of Reactions")
plt.ylabel("Number of Comments")

plt.show()

# Testing model
prediction = model.predict(test_data)

plt.axis([0, 2500, 0, 1600])
for features in prediction[0]:
	plt.plot(features[1], features[2], 'ro')

for features in prediction[1]:
	plt.plot(features[1], features[2], 'bo')

for centroid in centroids:
	plt.scatter(centroid[1], centroid[2], marker="x",s=150, linewidths=5)

plt.xlabel("Number of Reactions")
plt.ylabel("Number of Comments")

plt.show()







			







