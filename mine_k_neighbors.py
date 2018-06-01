import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from collections import Counter
from matplotlib import style
import random
style.use('fivethirtyeight')


# plt1 = [1, 3]
# plt2 = [2, 5]
# euclidean_distance = sqrt((plt1[0] - plt2[0])**2 + (plt1[1] - plt2[1])**2)
# print(euclidean_distance)

dataset = {'k': [[1, 2], [2, 3], [3, 1]], 'r': [[6, 5], [7, 7], [8, 6]]}
new_ft = [5, 7]




def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K must be greater or equal then data')
    distances = []
    for group in data:
        for features in data[group]:
            # euclidian_distance = np.sqrt(np.sum((np.array(features) - np.array(predict))**2))
            euclidian_distance = np.linalg.norm(np.array(features) - np.array(predict))
            distances.append([euclidian_distance, group])

    votes = [i[1] for i in sorted(distances)[:k]]
    # print(votes)
    # print(Counter(votes).most_common(1))
    vote_result = Counter(votes).most_common(1)[0][0]
    return vote_result


result = k_nearest_neighbors(dataset, new_ft, k=3)
# print(result)
#
# [[plt.scatter(ii[0], ii[1], s=100, color=i)for ii in dataset[i]]for i in dataset]
# plt.scatter(new_ft[0], new_ft[1], s=100, color=result)
# plt.show()

df = pd.read_csv('breast.csv')
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)
full_data = df.astype(float).values.tolist()
random.shuffle(full_data)
test_size = 0.2
train_set = {2: [], 4: []}
test_set = {2: [], 4: []}
train_data = full_data[:-int(test_size*len(full_data))]
test_data = full_data[-int(test_size*len(full_data)):]

for i in train_data:
    train_set[i[-1]].append(i[:-1])
for i in test_data:
    test_data[i[-1]].append(i[:-1])

correct = 0
total = 0
for group in test_set:
    for data in test_set[group]:
        vote = k_nearest_neighbors(train_set, data, k=5)
        if group == vote:
            correct += 1
        total +=1

print('acc:', correct / total)
