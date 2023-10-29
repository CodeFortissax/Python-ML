import numpy as np
from sklearn import preprocessing, model_selection, neighbors, svm
import pandas as pd
import matplotlib.pyplot as plt
import random as rd
from sklearn.decomposition import PCA

#genrate an array of 100 gene names
genes = ['gene' + str(i) for i in range(1, 101)]

#create an array of sample names
wt = ['wt' + str(i) for i in range(1, 6)]
ko = ['ko' + str(i) for i in range(1, 6)]

data = pd.DataFrame(columns=[*wt, *ko], index=genes)

#create random data
for gene in data.index:
    data.loc[gene, 'wt1':'wt5'] = np.random.poisson(lam=rd.randrange(10, 1000), size=5)
    data.loc[gene, 'ko1':'ko5'] = np.random.poisson(lam=rd.randrange(10, 1000), size=5)

#print(data.head())

#center and scale data
scaled_data = preprocessing.scale(data.T) #we are passing in the transpose of our data because scale function expects it to be in rows instead of columns.
# Alternatively we could use:
#StandardScaler().fit_transform(data.T)

pca = PCA()
pca.fit(scaled_data) #calculate loading scores and each pc accounts for
pca_data = pca.transform(scaled_data) #generate coordinates for a PCA graph based on the loading scores and scaled data

#calculate percentage of variation
per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)
labels = ['PC' + str(x) for x in range(1, len(per_var)+1)] # labels for SCREE plot

plt.bar(x=range(1, len(per_var)+1), height=per_var, tick_label=labels)
plt.ylabel('Percentage of Explained Variance')
plt.xlabel('Principal Componenet')
plt.title('Scree Plot')
#plt.show()

#draw a PCA plot
pca_df = pd.DataFrame(pca_data, index=[*wt, *ko], columns=labels)

plt.scatter(pca_df.PC1, pca_df.PC2)
plt.title('My PCA Graph')
plt.xlabel('PC1 - {0}%'.format(per_var[0]))
plt.xlabel('PC2 - {0}%'.format(per_var[1]))

for sample in pca_df.index:
    plt.annotate(sample, (pca_df.PC1.loc[sample], pca_df.PC2.loc[sample]))

plt.show()

#examine the loading scores
loading_scores = pd.Series(pca.components_[0], index=genes)
sorted_loading_scores = loading_scores.abs().sort_values(ascending=False)

top_10_genes = sorted_loading_scores[0:10].index.values

print(loading_scores[top_10_genes])









