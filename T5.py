import librosa
import os
from sklearn import svm
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import fbeta_score
p1="C:\\Users\\BAHI\\Desktop\\t5\\data\\Train\\spk1"
d=os.listdir(p1)
x_t=[]
y_t=[]
s=0
for d1 in d:
    s+=1
    data_p,fs=librosa.load(p1+'\\'+d1)

    x=librosa.feature.mfcc(data_p, sr=fs)
    for i in range(len(x[0])):

        x1=x[:,i]
        j = len(x1)
        while j < 20:
            x1.append(0)
            j += 1
        if(len(x1)==20):
            x_t.append(x1)
            y_t.append(1)
print('here')
p2="C:\\Users\\BAHI\\Desktop\\t5\\data\\Train\\spk2"
d=os.listdir(p2)
for d1 in d:
    data_p,fs=librosa.load(p2+'\\'+d1)

    x=librosa.feature.mfcc(data_p, sr=fs)
    for i in range(len(x[0])):
        x1=x[:,i]
        j = len(x1)
        while j < 20:
            x1.append(0)
            j += 1
        if (len(x1) == 20):
            x_t.append(x1)
            y_t.append(0)

p1="C:\\Users\\BAHI\\Desktop\\t5\\data\\Test\\spk1"
d=os.listdir(p1)
x_s=[]
y_s=[]
for d1 in d:
    data_p,fs=librosa.load(p1+'\\'+d1)

    x=librosa.feature.mfcc(data_p, sr=fs)
    for i in range(len(x[0])):
        x1=x[:,i]
        j = len(x1)
        while j < 20:
            x1.append(0)
            j += 1
        if (len(x1) == 20):
            x_s.append(x1)
            y_s.append(1)

print('here')
p2="C:\\Users\\BAHI\\Desktop\\t5\\data\\Test\\spk2"
d=os.listdir(p2)
for d1 in d:
    data_p,fs=librosa.load(p2+'\\'+d1)

    x=librosa.feature.mfcc(data_p, sr=fs)
    for i in range(len(x[0])):
        x1=x[:,i]
        j=len(x1)
        while j < 20:
            x1.append(0)
            j+=1
        if (len(x1) == 20):
            x_s.append(x1)
            y_s.append(0)
print('errorrrrrrrrrrr')
print(x_t[0],y_t[0])
print(np.array(x_t).shape,np.array(y_t).shape)
svm_model= svm.SVC(kernel='linear')
params={'C':[1,2]}
cv_sets = ShuffleSplit(n_splits = 10, test_size = 0.20, random_state = 0)
scorer = make_scorer(fbeta_score, beta=1)
scoring_fnc = make_scorer(r2_score)
print('fitting')
grid_obj = GridSearchCV(svm_model, params,scoring=scorer, cv=cv_sets)
grid_fit = grid_obj.fit(x_t, y_t)
print('testing')
best_clf = grid_fit.best_estimator_
best_predictions = best_clf.predict(x_s)
acc=accuracy_score(y_s, best_predictions)
print('accuracy',acc)

data_p,fs=librosa.load("C:\\Users\\BAHI\\Desktop\\t5\\data\\Test\\spk1\\arctic_b0048.wav")
x=librosa.feature.mfcc(data_p, sr=fs)
for i in range(len(x[0])):
    x1=x[:,i]
    j=len(x1)
    while j < 20:
        x1.append(0)
        j+=1
    if (len(x1) == 20):
        x_s.append(x1)
        y_s.append(0)

my_prec = best_clf.predict(x_s)
o=0
z=0
for x in my_prec:
    if x==1:
        o+=1
    else :
        z+=1
if(o>=z):
    print('prediction is 1')
else:
    print('prediction is 0')

