import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as colors
from sklearn.utils import resample #down sample the dataset
from sklearn.model_selection import train_test_split #split data into training and testing set
from sklearn.preprocessing import scale #scale and center data
from sklearn.svm import SVC #this will make a support vector machine for classification
from sklearn.model_selection import GridSearchCV #this will do cross validation
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA #to perform PCA to plot the data #PCA stands for Principle Component Analysis

#C:\Users\Admin\Downloads\default+of+credit+card+clients.zip
#"C:\Users\Admin\Desktop\necessities\Copy of default of credit card clients.xls"
#file_path = r"C:\Users\Admin\Downloads\default+of+credit+card+clients.zip"
file_path1 = r"C:\Users\Admin\Desktop\necessities\Copy of default of credit card clients.xls"
df = pd.read_excel(file_path1, header=1) #sep='\t' #encoding='cp1252'

#print("Shape of DataFrame:", df.shape)
df.rename({'default payment next month' : 'DEFAULT'}, axis='columns', inplace=True)
df.drop('ID', axis='columns', inplace=True)
#print(df.dtypes)
#print(len(df.loc[(df['EDUCATION'] == 0) | (df['MARRIAGE'] == 0)]))
#print(df.head())
df_no_missing = df.loc[(df['EDUCATION'] != 0) & (df['MARRIAGE'] != 0)]
#print(df_no_missing)
df_no_default = df_no_missing[df_no_missing['DEFAULT'] == 0]
df_default = df_no_missing[df_no_missing['DEFAULT'] == 1]

df_no_default_downsampled = resample(df_no_default,
                                     replace=False,
                                     n_samples=1000,
                                     random_state=42)
#print(df_no_default_downsampled)
df_default_downsampled = resample(df_default,
                                     replace=False,
                                     n_samples=1000,
                                     random_state=42)
df_downsample = pd.concat([df_no_default_downsampled, df_default_downsampled])
#print(df_downsample)

#now for training and testing data
X = df_downsample.drop('DEFAULT', axis=1).copy() #training data
y = df_downsample['DEFAULT'].copy() #testing data

X_encoded = pd.get_dummies(X, columns=['SEX',
                                       'EDUCATION',
                                       'MARRIAGE',
                                       'PAY_0',
                                       'PAY_2',
                                       'PAY_3',
                                       'PAY_4',
                                       'PAY_5',
                                       'PAY_6'])
#print(X_encoded.head())

#Centering and scaling
#RBF assumes each column has a mean of 0 and a standard deviation of 1

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, random_state=42)
X_train_scale = scale(X_train)
X_test_scale = scale(X_test)

#Build a preliminary support vector machine

clf_svm = SVC(random_state=42)
clf_svm.fit(X_train_scale, y_train)

# Generate predictions
y_pred = clf_svm.predict(X_test_scale)

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred)
#Drawing confusion matrix from test dataset

# Plot confusion matrix
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()

classes = ['Did not default', 'Defaulted']
tick_marks = np.arange(len(classes))

plt.xticks(tick_marks, classes)
plt.yticks(tick_marks, classes)

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

#disp = plot_confusion_matrix(clf_svm,
                      #X_test_scale,
                      #y_test,
                      #values_format='d',
                      #display_labels=['Did not default', 'Defaulted'],
                       #      cmap=plt.cm.Blues)

# Set the title of the plot
#disp.ax_.set_title('Confusion Matrix')

# Display the plot
#plt.show()

#Optmize Parameters with Cross Validation and GridSearchCV()

param_grid = [{'C' : [0.5, 1, 10, 100],
               'gamma' : ['scale', 1, 0.1, 0.001], #0.0001],
               'kernel' : ['rbf']}]

optimal_params = GridSearchCV(SVC(),
                              param_grid,
                              cv=5,
                              scoring='accuracy',
                              verbose=0 )

optimal_params.fit(X_train_scale, y_train)
#print(optimal_params.best_params_)

#Building, Evaluating, Drawing and Interpreting the Final SVM

clf_svm = SVC(random_state=42, C=100, gamma=0.01)
clf_svm.fit(X_train_scale, y_train)

#Draw another confusion matrix to see if the optimized SVM does better

# Generate predictions
y_pred = clf_svm.predict(X_test_scale)

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred)
#Drawing confusion matrix from test dataset

# Plot confusion matrix
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()

classes = ['Did not default', 'Defaulted']
tick_marks = np.arange(len(classes))

plt.xticks(tick_marks, classes)
plt.yticks(tick_marks, classes)

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

#Draw a SVM Decision Boundary
len(df_downsample.columns)

#Use PCA to combine the 24 features into 2 orthogonal meta-features that we can use as axes for a graph
#plot a 'scree plot' to see if the PCA graph will be useful

pca = PCA()
X_train_pca = pca.fit_transform(X_train_scale)

per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)
labels = [str(x) for x in range(1, len(per_var)+1)]

plt.bar(x=range(1, len(per_var)+1), height=per_var)
plt.tick_params(axis='x',
                which='both',
                bottom=False,
                top=False,
                labelbottom=False)
plt.ylabel('Percentage of Explained Variance')
plt.xlabel('Principle Components')
plt.title('Scree Plot')
plt.show()

#Now draw the PCA graph

train_pc1_coords = X_train_pca[:, 0] #contains x axis coordinates
train_pc2_coords = X_train_pca[:, 1] #contains y axis coordinates

#Now center and scale PCs...
pca_train_scaled = scale(np.column_stack((train_pc2_coords, train_pc1_coords)))

#now we optimize SVM to fit x and y-axis coordinates

param_grid = [{'C' : [1, 10, 100],#, 1000],
               'gamma' : ['scale', 1, 0.1, 0.01],#, 0.001],
               'kernel' : ['rbf']},]

optimal_params = GridSearchCV(SVC(),
                              param_grid,
                              cv=5,
                              scoring='accuracy',
                              verbose=0)

optimal_params.fit(pca_train_scaled, y_train)
print(optimal_params.best_params_)

#Now that we have optimal values for C and gamma, draw the graph with the decision boundary

clf_svm = SVC(random_state=42, C=100, gamma=0.01)
clf_svm.fit(pca_train_scaled, y_train)

#Transform the datatset with PCA
X_test_pca = pca.transform(X_train_scale)

test_pc1_coords = X_test_pca[:, 0]
test_pc2_coords = X_test_pca[:, 1]

x_min = test_pc1_coords.min() - 1
x_max = test_pc1_coords.max() + 1

y_min = test_pc2_coords.max() - 1
y_max = test_pc2_coords.max() + 1

xx, yy = np.meshgrid(np.arange(start=x_min, stop=x_max, step=0.1),
                     np.arange(start=y_min, stop=y_max, step=0.1))

Z = clf_svm.predict(np.column_stack((xx.ravel(), yy.ravel())))

Z = Z.reshape(xx.shape)

fig, ax = plt.subplots(figsize=(10, 10))

ax.contour(xx, yy, Z, alpha=0.1)

cmap = colors.ListedColormap(['#e41a1c', '#4daf4a'])

scatter = ax.scatter(test_pc1_coords, test_pc2_coords, c=y_train,
                     cmap=cmap,
                     s=100,
                     edgecplors='k',
                     alpha=0.7)

legend = ax.legend(scatter.legend_elements()[0],
                   scatter.legend_elements()[1],
                   loc='upper right')

legend.get_texts()[0].set_text("No Default")
legend.get_texts()[1].set_text("Yes Default")

ax.set_ylabel('PC2')
ax.set_xlabel('PC1')
ax.set_title('Decision surface using the PCA transformed/projected features')

plt.show()















