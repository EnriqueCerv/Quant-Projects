# %%
import numpy as np
import pandas as pd
import sklearn
# %%
## Load data
from sklearn.datasets import load_digits

nums = load_digits()
nums_vis = pd.DataFrame(data=nums.data, columns=nums.feature_names)
nums_vis['Target'] = nums.target

X, y = load_digits(return_X_y=True)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2) # Stratify ensures class distribution is consistent accross split

## Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

model = make_pipeline(
    StandardScaler(),
    SVC()
)

param_grid = {
    'svc__C': [0.1,1,10],
    'svc__kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'svc__gamma': ['scale', 'auto'],
    'svc__degree': [2,3,4]
}

svm = GridSearchCV(model, param_grid, scoring='accuracy', cv=5)
svm.fit(X_train, y_train)
print("Best parameters:", svm.best_params_)
print("Best CV score:", svm.best_score_)

## Test data
from sklearn.metrics import classification_report, confusion_matrix
y_pred = svm.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))