# %%
import numpy as np
import pandas as pd
# %%
from sklearn.datasets import load_breast_cancer

X,y = load_breast_cancer(return_X_y=True, as_frame=True)
# %%
# Naning the data:

np.random.seed(seed=4)
mask = np.random.rand(*X.shape) < 0.05
X = X.mask(mask, np.nan)
X

# %%
# Preprocess the data

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

training_means = X_train.mean()
X_train = X_train.fillna(training_means)
X_test = X_test.fillna(training_means)


# %%
# Pipelining
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import make_scorer, fbeta_score
from sklearn.model_selection import GridSearchCV

model = make_pipeline(RandomForestClassifier(random_state=4))

param_grid = {
    'randomforestclassifier__n_estimators' : [100,200],
    'randomforestclassifier__max_depth' : [10,None],
    'randomforestclassifier__min_samples_split' : [2,5],
    'randomforestclassifier__criterion' : ['gini', 'entropy'],
    'randomforestclassifier__max_features' : ['sqrt', 'log2']
}

f2_score = make_scorer(fbeta_score, beta=2)
scoring = {'accuracy' : 'accuracy', 'f2' : f2_score}

clf = GridSearchCV(model, param_grid=param_grid, scoring=scoring, refit='f2')
clf.fit(X_train, y_train)

# %%
print(clf.best_params_, clf.best_score_)
# %%
# Testin

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

y_pred = clf.predict(X_test)
y_probs = clf.predict_proba(X_test)
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
# %%
