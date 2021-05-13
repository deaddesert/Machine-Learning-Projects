# Using MultinomialNB module form sklearn
from sklearn.naive_bayes import MultinomialNB

import numpy as np 

# Traning data
document = {'d1': 'hanoi pho thudo bundau',
			'd2': 'hanoi bundau pho omai',
			'd3': 'saigon banhcanh cholon',
			'd4': 'saigon hutiu banhcanh',
			'd5': 'cholon banhcanh hutiu'}
print (document.values())

# Bag of Words
V = ['hanoi', 'pho', 'thudo', 'bundau', 'omai', 'banhcanh', 'saigon', 'hutiu', 'cholon']

train_data = []

for string in document.values():
	Words = {}
	for word in V:
		if word in string:
			for element in string.split(' '):
				if word == element:
					if word not in Words:
						Words[word] = 1
					else:
						Words[word] += 1
		else:
			Words[word] = 0

	train_data.append(list(Words.values()))

print(train_data)

label = np.array(['N', 'N', 'S', 'S', 'S'])

# call MultinomialNB
clf = MultinomialNB()
# Training
clf.fit(train_data, label)

# Test data
d6 = 'hanoi hanoi bundau hutiu'

test_word = {}
test_data = []

for word in V:
	if word in d6:
		for element in d6.split(' '):
			if word == element:
				if word not in test_word:
					test_word[word] = 1
				else:
					test_word[word] += 1
	else:
		test_word[word] = 0

test_data.append(list(test_word.values()))

# Predict class of data
print('Probability of d5 in each class: ', clf.predict_proba(test_data))
print('Predict class of d6: ', str(clf.predict(test_data)[0]))