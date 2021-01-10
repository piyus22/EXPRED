#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 12:37:22 2020

@author: piyus
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle


cell_df=pd.read_csv('extractAAC_negative_1isto1_with_pos.csv')
#cell_df.head()
#cell_df.shape
#expansin_df=cell_df[cell_df['class']==0]
#non_expansin_df=cell_df[cell_df['class']==1[0:120]]
#expansin_df.plot(kind='scatter', x='A', y='R', color='red', label='Expansin', ax=axes)
#cell_df.dtypes
cell_df.columns
feature_df=cell_df[['A',	'R',	'N',	'D',	'C',	'E',	'Q',	'G',	'H',	'I',	'L',	'K',	'M',	'F',	'P',	'S',	'T',	'W',	'Y',	'V']]
x=np.asarray(feature_df)
y=np.asarray(cell_df['class'])
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y, test_size=0.2, random_state=0)
print(type(x_train))
#y_test.shape=51
#x_train.shape=(201,20)
#x_test.shape=51,20
from sklearn import svm
classifier=svm.SVC(C=10, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma=1, kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
#classifier=svm.SVC(kernel='linear',gamma='auto', C=2)
classifier.fit(x_train,y_train)
#print(classifier)
y_predict=classifier.predict(x_test)
#print(y_predict)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_predict))
from sklearn.metrics import confusion_matrix
#print("Confusion matrix: \n",confusion_matrix(y_test, y_predict))
from sklearn.metrics import accuracy_score
print("Accuracy:",accuracy_score(y_test, y_predict))
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
#mod_ROC=roc_auc_score(y_test,classifier.predict(x_test))
#fpr,tpr, _ =roc_curve(y_train, y_test)
#probs=classifier.predict_proba(x_test)
#print(probs)
#probs=probs[:,1]
#fpr, tpr, _ = roc_curve(y_test, probs)
#fpr, tpr, threshold = roc_curve(y_test, probs)
#plt.plot([0,1],[0,1], linestyle='--')
#plt.plot(fpr,tpr,marker='.')

#print(y_train)


#print(probs)
#roc_auc_score(y_test, probs)
pickle.dump(classifier, open('SVM_model.pkl','wb'))
model = pickle.load(open('SVM_model.pkl','rb'))
#x=model.predict([[0.1271186441,0.0169491525,0.0169491525,0.0254237288,0.0423728814,0.0254237288,0.0254237288,0.1101694915,0.0084745763,0.0508474576,0.0593220339,0.0677966102,0.0084745763,0.0593220339,0.0508474576,0.0677966102,0.1101694915,0.0084745763,0.0338983051,0.0847457627]])
x=model.predict([[0.0716845878,0.0286738351,0.0573476703,0.0501792115,0.0286738351,0.0322580645,0.0215053763,0.1111111111,0.0107526882,0.0609318996,0.0609318996,0.0681003584,0.0394265233,0.0501792115,0.0465949821,0.0322580645,0.082437276,0.0322580645,0.0501792115,0.064516129]])
print(x)
if x==0:
    print("Is an expansin sequence")
else:
    print("Is not an expansin sequence")
from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(classifier, x_test, y_test)
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg')
plt.savefig('Confusion_aa_1isto1.png')
    

