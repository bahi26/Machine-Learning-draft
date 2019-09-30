from sklearn import svm
import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import fbeta_score
#read the data
data = pd.read_csv('housing.csv')
prices = data['MEDV']
features = data.drop('MEDV', axis = 1)
#split your data
X_train, X_test, y_train, y_test = train_test_split(features, prices, test_size=0.2, random_state=42)

#train the model
#kernel linear rbf polynomial
svm_model= svm.SVC(kernel='linear')
params={'C':[1,2]}
#svm_model.fit(X, y)
#scores = cross_val_score(svm_model, X_train, y_train, cv=cv_sets)
#prediction=svm_model.predict([[2., 2.]])
cv_sets = ShuffleSplit(n_splits = 10, test_size = 0.20, random_state = 0)
scorer = make_scorer(fbeta_score, beta=1)
scoring_fnc = make_scorer(r2_score)
grid_obj = GridSearchCV(svm_model, params,scoring=scoring_fnc, cv=cv_sets)
grid_fit = grid_obj.fit(X_train, y_train)
best_clf = grid_fit.best_estimator_
best_predictions = best_clf.predict(X_test)
print(best_predictions)