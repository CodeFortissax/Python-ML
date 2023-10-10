import numpy as np
from sklearn import preprocessing, model_selection, neighbors, svm
import pandas as pd

file_path = r"C:\Users\Admin\Desktop\necessities\breast-cancer-wisconsin.data"
df = pd.read_csv(file_path)
df.columns = ["id", "clump_thickness", "unif_cell_size", "unif_cell_shape", "marg_adhesion",
                "single_epith_cell_size", "bare_nuclei", "bland_chrom", "norm_nucleoli", "mitoses", "class"]

df.replace('?', -99999, inplace=True)
df.drop(['id'], axis=1, inplace=True)

X = np.array(df.drop(['class'], axis=1))
y = np.array(df['class'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size=0.2)

clf = svm.SVC(gamma='auto')
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(accuracy)

example_measures = np.array([[4, 2, 1, 1, 1, 2, 3, 2, 1], [4, 2, 1, 2, 1, 1, 3, 2, 1]])
example_measures = example_measures.reshape( len(example_measures), -1)

prediction = clf.predict(example_measures)
print(prediction)