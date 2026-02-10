# %%
import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import load_breast_cancer

# %% Loading data:

# Directly
X,y = load_breast_cancer(return_X_y=True, as_frame=True)

# Other\
# breast_cancer = load_breast_cancer()
# X = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
# y = breast_cancer.target
# %% Input some Naan
np.random.seed(seed=42)
mask = np.random.rand(*X.shape) < 0.05
X = X.mask(mask)
X
# %%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
X_train

# %% There are various methods to fill in values depending on the nature of data:

# # Fill na
# X_train.fillna(0)

# With mean
training_means = X_train.mean()
X_train = X_train.fillna(training_means)
X_test = X_test.fillna(training_means)

# # If dealing with time series, can interpolate
# X_train = X_train.interpolate(method='linear')
# %%
# Setting up pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, fbeta_score

model = make_pipeline(
    StandardScaler(), 
    LogisticRegression(max_iter=500)
    )

param_grid = [{
    'logisticregression__C' : [10**i for i in range(-2, 2)],
    'logisticregression__penalty' : ['l1'],
    'logisticregression__solver' : ['saga', 'liblinear']
},
{
    'logisticregression__C' : [10**i for i in range(-2, 2)],
    'logisticregression__penalty' : ['l2'],
    'logisticregression__solver' : ['saga', 'liblinear', 'lbfgs']
},
{
    'logisticregression__C' : [10**i for i in range(-2, 2)],
    'logisticregression__penalty' : ['elasticnet'],
    'logisticregression__solver' : ['saga'],
    'logisticregression__l1_ratio' : [0.5]
}]

clf1 = GridSearchCV(model, param_grid=param_grid, scoring='accuracy')
clf1.fit(X_train, y_train)

f2_score = make_scorer(fbeta_score, beta=2)
clf2 = GridSearchCV(model, param_grid=param_grid, scoring=f2_score)
clf2.fit(X_train, y_train)

# %%
# CHecking the scores
print(clf1.best_params_, clf1.best_score_)

# CHecking the scores
print(clf2.best_params_, clf2.best_score_)
# %%
# Testing on training data

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

y_pred = clf1.predict(X_test)
y_probs = clf1.predict_proba(X_test)

# For medical applications, reduce the threshold for guessing cancer positive ---> Reduce the number of false negatices
y_pred_custom = (y_probs[:, 1] > 0.4).astype(int)

acc = accuracy_score(y_test, y_pred_custom)
f1 = f1_score(y_test, y_pred_custom)
f2 = fbeta_score(y_test, y_pred_custom, beta=2)
cm = confusion_matrix(y_test, y_pred_custom)

print('Accuracy score:' , acc)
print('F1 score:' , f1)
print('F2 score:' , f2)
print('Confusion matrix:' , cm)
print(classification_report(y_test, y_pred_custom))
