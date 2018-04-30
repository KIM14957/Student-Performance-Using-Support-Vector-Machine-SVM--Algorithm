#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 00:53:34 2018

@author: clive
"""

# Support Vector Machine for Student Performance dataset on (SVM)

# Import libraries
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from matplotlib.colors import ListedColormap


# Import the dataset
dataset = pd.read_csv("/Users/clive/Desktop/student/student-mat.csv", sep=";")
X = dataset.iloc[:, [30, 31]].values
y = dataset.iloc[:, -1].values

#Analysis and Visualisation of dataset
print("Violin plot")
sns.violinplot(dataset['sex'], dataset['G3'])
sns.despine()
 
   
fig=plt.figure()
ax = fig.add_subplot(1,1,1)
ax.boxplot(dataset['G3'])
plt.show()

print(dataset.describe())


# This function converts GB and MP to 0&1
def convert(School):
    if(School=='GB'):
        return 1
    else :
        return 0

dataset['school'] = dataset['school'].apply(convert)

# This function converts all the pass and fail 0&1
def convert(g3):
    if(g3>=10):
        return 1
    else :
        return 0

dataset['G3'] = dataset['G3'].apply(convert)

# This function converts all the yes/no columns to 1/0
def yes_or_no(parameter):
    if parameter == 'yes' :
        return 1
    else :
        return 0

def yn(c) :
    dataset[c] = dataset[c].apply(yes_or_no)
    
Colum = ['schoolsup','famsup','paid','activities','nursery','higher','internet','romantic']


for c in Colum :
    yn(c)
    
        
# Splitting the dataset into the Training set and Test set
X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size = 0.25, random_state = 17)

# Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_validation = sc.transform(X_validation)

# Fitting Model to Train set
classifier = SVC(kernel = 'poly', degree=3, random_state = 0)
classifier.fit(X_train, y_train)

# Predict the Test set results
prediction = classifier.predict(X_validation)

# Making the Confusion Matrix
cmatrix = confusion_matrix(y_validation, prediction)
print("=============================")
print("Confusion matrix :")
print(cmatrix)
print("=============================")
print("Accuracy: ",100*accuracy_score(y_validation, prediction),"%")
print("=============================")
print(classification_report(y_validation, prediction))
print("=============================")


# Visualising the results
X_set, y_set = X_validation, y_validation
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, 
                               stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, 
                               stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('yellow', 'blue'))(i), label = j)
plt.title('SVM (Student Prediction)')
plt.xlabel('G1')
plt.ylabel('G2')
plt.legend()
plt.show()



