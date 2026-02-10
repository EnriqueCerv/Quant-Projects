# %%
import numpy as np
import pandas as pd
import sklearn
# %%

### Normal regressor

## Load Data
from sklearn.datasets import load_diabetes

diab = load_diabetes()
diab_vis = pd.DataFrame(data=diab.data, columns=diab.feature_names)
diab_vis['Target'] = diab.target

X, y = load_diabetes(return_X_y=True)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

## Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LinearRegression

model = make_pipeline(StandardScaler(), LinearRegression())

cv_results = cross_validate(model, X_train, y_train, cv=5, return_estimator=True, scoring='r2')
scores = cv_results['test_score'] # All cv scores
models = cv_results['estimator'] # All cv models
grid = models[np.argmax(scores)] # Best model


## Test model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
y_pred = grid.predict(X_test)
print('MSE:', mean_squared_error(y_test, y_pred))
print('MAE:', mean_absolute_error(y_test, y_pred))
print('R2:', r2_score(y_test, y_pred))


# %%
### Ridge/Lasso

## Load Data

from sklearn.datasets import load_diabetes

diab = load_diabetes()
diab_vis = pd.DataFrame(data=diab.data, columns=diab.feature_names)
diab_vis['Target'] = diab.target

X, y = load_diabetes(return_X_y=True)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

## Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge, Lasso

model = make_pipeline(StandardScaler(), Lasso())

param_grid = {
    'lasso__alpha': [0.001, 0.1, 1, 10]
}

grid = GridSearchCV(model, param_grid=param_grid, scoring='r2', cv=5)
grid.fit(X_train, y_train)
print('Best parameters:', grid.best_params_)
print('Best CV score:', grid.best_score_)


## Test model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
y_pred = grid.predict(X_test)
print('MSE:', mean_squared_error(y_test, y_pred))
print('MAE:', mean_absolute_error(y_test, y_pred))
print('R2:', r2_score(y_test, y_pred))
# %%

### SGD regressor that does both l1 and l2

## Load Data

from sklearn.datasets import load_diabetes

diab = load_diabetes()
diab_vis = pd.DataFrame(data=diab.data, columns=diab.feature_names)
diab_vis['Target'] = diab.target

X, y = load_diabetes(return_X_y=True)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

## Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDRegressor

model = make_pipeline(StandardScaler(), SGDRegressor())

param_grid = {
    'sgdregressor__alpha': [0.001, 0.1, 1, 10],
    'sgdregressor__penalty': ['l1', 'l2', 'elasticnet']
}

grid = GridSearchCV(model, param_grid=param_grid, scoring='r2', cv=5)
grid.fit(X_train, y_train)
print('Best parameters:', grid.best_params_)
print('Best CV score:', grid.best_score_)


## Test model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
y_pred = grid.predict(X_test)
print('MSE:', mean_squared_error(y_test, y_pred))
print('MAE:', mean_absolute_error(y_test, y_pred))
print('R2:', r2_score(y_test, y_pred))
# %%
