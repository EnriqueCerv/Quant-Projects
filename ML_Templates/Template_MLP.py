# %%
import numpy as np
import pandas as pd
import sklearn
# %%
### Classifier

## Load data
from sklearn.datasets import load_digits
X, y = load_digits(return_X_y=True)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, stratify=y, test_size=0.2)

## Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier

model = make_pipeline(StandardScaler(), MLPClassifier(max_iter=400))

param_grid = {
    'mlpclassifier__hidden_layer_sizes': [(50,), (100,), (50,50), (33,33,34)],
    'mlpclassifier__activation': ['relu', 'tanh', 'logistic'],
    'mlpclassifier__solver': ['adam', 'sgd'],
    'mlpclassifier__alpha': [0.0001, 0.001, 0.01],
    'mlpclassifier__learning_rate': ['adaptive']
}

grid = GridSearchCV(model, param_grid=param_grid, scoring='accuracy', cv=5)
grid.fit(X_train, y_train)
print('Best parameters: ', grid.best_params_)
print('Best CV score: ', grid.best_score_)

## Test data
from sklearn.metrics import classification_report, confusion_matrix
y_pred = grid.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
# %%

### Regressor

## Load data
from sklearn.datasets import load_diabetes
X, y = load_diabetes(return_X_y=True)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, stratify=y, test_size=0.2)

## Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPRegressor

model = make_pipeline(StandardScaler(), MLPRegressor(max_iter=400))

param_grid = {
    'mlpregressor__hidden_layer_sizes': [(50,), (100,), (50,50), (33,33,34)],
    'mlpregressor__activation': ['relu', 'tanh', 'logistic'],
    'mlpregressor__solver': ['adam', 'sgd'],
    'mlpregressor__alpha': [0.0001, 0.001, 0.01],
    'mlpregressor__learning_rate': ['adaptive']
}

grid = GridSearchCV(model, param_grid=param_grid, scoring='accuracy', cv=5)
grid.fit(X_train, y_train)
print('Best parameters: ', grid.best_params_)
print('Best CV score: ', grid.best_score_)

## Test data
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
y_pred = grid.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.3f}, MAE: {mae:.3f}, R²: {r2:.3f}")