import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class KMeansClustering:

	def __init__(self, k, max_iter):
		self.k = k
		self.max_iter = max_iter


	def training(self, X):
		list_of_centroids = list()
		list_of_clusters = list()
		cost_function_values = list()

		for n in range(self.max_iter):

			M = np.zeros((self.k, X.shape[1])) # centroids
			cost_function = 0
			optimized = 0

			# Initializing centroids
			M = X[np.random.choice(X.shape[0], self.k, replace=False)]

			# Assigning Label to Data

			while (optimized == 0):
				K_clusters = dict()
				y = np.zeros((X.shape[0], self.k)) # Label vectors
				X_list = X.tolist()
				for j in range(self.k):
					K_clusters[j] = []

				for x in X_list:
					distances = [np.linalg.norm(x - m) for m in M]
					cluster = np.argmin(distances)
					index = X_list.index(x)
					K_clusters[cluster].append(x)
					y[index][cluster] = 1

				# Calculate new centroids
				prev_centroids = M

				for cluster in K_clusters:
					M[cluster] = np.mean(K_clusters[cluster], axis=0)

				# Test the convegence
				comparision = M == prev_centroids
				if comparision.all():
					optimized = 1

			list_of_clusters.append(K_clusters)
			list_of_centroids.append(M)

			for i in range(X.shape[0]):
				for j in range(self.k):
					cost_function += (1/len(X))*np.sum((y[i][j]*((X[i] - M[j])**2)))

			cost_function_values.append(cost_function)

		cost_function_min_index = np.argmin(cost_function_values)
		print(cost_function_min_index)

		# Final result of K-Mean Clustering
		self.centroids = list_of_centroids[cost_function_min_index]
		self.clusters = list_of_clusters[cost_function_min_index]

		return [self.centroids, self.clusters, y]


	def testing(self, X):
		K_clusters = dict()
		y = np.zeros((X.shape[0], self.k)) # Label vectors
		X_list = X.tolist()
		for j in range(self.k):
			K_clusters[j] = []

		for x in X_list:
			distances = [np.linalg.norm(x - m) for m in self.centroids]
			cluster = np.argmin(distances)
			index = X_list.index(x)
			K_clusters[cluster].append(x)
			y[index][cluster] = 1

		return [K_clusters, y]


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

model = KMeansClustering(2, 60)

# Training model
centroid_results, train_clusters, train_labels = model.training(train_data)

for centroid in centroid_results:
	print([centroid[1], centroid[2]])

# We will represent clusters and centroids based on num_reactions and num_comments

plt.axis([0, 2500, 0, 5000])
for features in train_clusters[0][:200]:
	plt.plot(features[1], features[2], 'ro')

for features in train_clusters[1][:200]:
	plt.plot(features[1], features[2], 'bo')

for centroid in centroid_results:
	plt.scatter(centroid[1], centroid[2], marker="x",s=150, linewidths=5)

plt.xlabel("Number of Reactions")
plt.ylabel("Number of Comments")

print(train_labels[:200])

plt.show()

print("\n")

# Testing model
predicted_clusters, test_labels = model.testing(test_data)

plt.axis([0, 2500, 0, 5000])
for features in predicted_clusters[0]:
	plt.plot(features[1], features[2], 'ro')

for features in predicted_clusters[1]:
	plt.plot(features[1], features[2], 'bo')

for centroid in centroid_results:
	plt.scatter(centroid[1], centroid[2], marker="x",s=150, linewidths=5)

plt.xlabel("Number of Reactions")
plt.ylabel("Number of Comments")

print(test_labels)

plt.show()
