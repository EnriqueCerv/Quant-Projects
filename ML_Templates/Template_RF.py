# %%
import numpy as np
import pandas as pd
import sklearn
# %%
### Classifier

## Load Data

from sklearn.datasets import load_wine, load_breast_cancer
wine = load_wine()
wine_df = pd.DataFrame(data = wine.data, columns = wine.feature_names)
wine_df['Target'] = wine.target_names[wine.target]

X,y = load_wine(return_X_y=True)
X,y = load_breast_cancer(return_X_y=True)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)


## Pipeline 1
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],
    'bootstrap': [True, False]
}
rf = GridSearchCV(model, param_grid=param_grid, cv=5, scoring='accuracy')
rf.fit(X_train, y_train)

print("Best parameters:", rf.best_params_)
print("Best CV score:", rf.best_score_)


# ## Pipeline 2
# from sklearn.model_selection import cross_validate
# from sklearn.ensemble import RandomForestClassifier

# model = RandomForestClassifier(n_estimators=100)
# cv_results = cross_validate(model, X_train, y_train, cv=5, return_estimator=True, scoring='accuracy')
# scores = cv_results['test_score'] # All scores
# models = cv_results['estimator'] # All models
# rf = models[np.argmax(scores)]

## Test 
from sklearn.metrics import classification_report, confusion_matrix
y_pred = rf.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
# %%

### Regressor

## Load Data

from sklearn.datasets import load_diabetes
X,y = load_diabetes(return_X_y=True)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

## Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor()
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],
    'bootstrap': [True, False]
}

rf = GridSearchCV(model, param_grid=param_grid, scoring='r2', cv=5)
rf.fit(X_train, y_train)

print('Best parameters:', rf.best_params_)
print('Best score:', rf.best_score_)

## Test data
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
y_pred = rf.predict(X_test)
print('MSE:', mean_squared_error(y_test, y_pred))
print('MAE:', mean_absolute_error(y_test, y_pred))
print('R2:', r2_score(y_test, y_pred))