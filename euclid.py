import random

import numpy as np
#import matplotlib.pyplot as plt
import warnings
#from matplotlib import style
from collections import Counter
import pandas as pd
from math import sqrt
#style.use('fivethirtyeight')

#euclidian_distance = sqrt(((plot1[0]-plot1[0])**2) + ((plot1[1]-plot2[1])**2))
#euclidian_distance = np.sqrt(np.sum(np.array(feature)-np.array(predict))**2)

datasets = {'k': [[1, 2], [2, 3], [3,1]], 'r': [[6, 5], [7, 7], [8, 6]]}
new_features = [5, 7]


def k_nearest_neighbours (data, predict, k=3):
    if len(data) >= k:
        warnings.warn('k value yako imezidi bana!')
    distances=[]
    for group in data:
        for features in data[group]:
            euclidian_distance = np.linalg.norm(np.array(features)-np.array(predict))
            distances.append([euclidian_distance, group])

    votes = [i[1] for i in sorted(distances) [:k]]
    #print(Counter(votes).most_common(1))
    vote_result = Counter(votes).most_common(1) [0] [0]

    return vote_result


result = k_nearest_neighbours(datasets, new_features, k=3)
print(result)

#[[plt.scatter(ii[0], ii[1], s=100, color=i) for ii in datasets[i]] for i in datasets]
#plt.scatter(new_features[0], new_features[1], s=75, color=result)
#plt.show()

file_path = r"C:\Users\Admin\Desktop\necessities\breast-cancer-wisconsin.data"
df = pd.read_csv(file_path)
df.columns = ["id", "clump_thickness", "unif_cell_size", "unif_cell_shape", "marg_adhesion",
                "single_epith_cell_size", "bare_nuclei", "bland_chrom", "norm_nucleoli", "mitoses", "class"]
df.replace('?', -99999, inplace=True)
df.drop(['id'], axis=1, inplace=True)
#print(df.head())
full_data = df.astype(float).values.tolist()
#print(full_data[:5])
random.shuffle(full_data)
#print(20*'#')
#print(full_data[:5])

test_size = 0.2
train_set = {2:[], 4:[]}
test_set = {2:[], 4:[]}
train_data = full_data[:-int(test_size*len(full_data))]
test_data = full_data[-int(test_size*len(full_data)):]

for i in train_data:
    train_set[i[-1]].append(i[:-1])

for i in test_data:
    test_set[i[-1]].append(i[:-1])

correct = 0
total = 0

for group in test_set:
    for data in test_set[group]:
        vote = k_nearest_neighbours(train_set, data, k=5)
        if group == vote:
            correct += 1
        total += 1

print('Accuracy: ', correct/total)




