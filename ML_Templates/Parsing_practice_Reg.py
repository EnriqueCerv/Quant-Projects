# %%
import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes
# %%
X,y = load_diabetes(return_X_y=True, as_frame=True)
# %%
# Naning the data
np.random.seed(42)
mask = np.random.rand(*X.shape) < 0.05
X = X.mask(mask)
X
# %%
# Train test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

#%%
# Fill na
train_means = X_train.mean()
X_train = X_train.fillna(train_means)
X_test = X_test.fillna(train_means)
# %%
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
# %%
model_lr = make_pipeline(
    StandardScaler(),
    LinearRegression()
)
param_grid = {
    'linearregression__fit_intercept': [True, False]
}
scoring = {
    'r2':'r2',
    'neg_mean_squared_error':'neg_mean_squared_error',
    'neg_rms': 'neg_root_mean_squared_error'
}
reg_lr = GridSearchCV(model_lr, param_grid, scoring=scoring, cv=5, refit='r2')
reg_lr.fit(X_train, y_train)


y_pred = reg_lr.predict(X_test)
print('LinReg RMSE', root_mean_squared_error(y_test, y_pred))
print('LinReg MSE', mean_squared_error(y_test, y_pred))
print('LinReg MAE', mean_absolute_error(y_test, y_pred))
print('LinReg r2', r2_score(y_test, y_pred))
# %%

model_rf = make_pipeline(
    RandomForestRegressor()
)

param_grid = {
    'randomforestregressor__n_estimators' : [500],
    'randomforestregressor__max_depth' : [10,25],
    'randomforestregressor__min_samples_split' : [2,5],
    'randomforestregressor__criterion' : ['squared_error', 'absolute_error'],
    'randomforestregressor__min_samples_leaf': [2,5],
    'randomforestregressor__max_features' : ['sqrt', 'log2']
}

reg_rf = GridSearchCV(model_rf, param_grid=param_grid, cv=5, scoring=scoring, refit='r2')
reg_rf.fit(X_train, y_train)

y_pred = reg_rf.predict(X_test)
print('RFReg RMSE', root_mean_squared_error(y_test, y_pred))
print('RFReg MSE', mean_squared_error(y_test, y_pred))
print('RFReg MAE', mean_absolute_error(y_test, y_pred))
print('RFReg r2', r2_score(y_test, y_pred))

# %%
model_mlp = make_pipeline(
    StandardScaler(),
    MLPRegressor(max_iter=5000, early_stopping=True, random_state=42, learning_rate_init=0.0001)
)

# param_grid = {
#     'mlpregressor__solver': ['lbfgs', 'adam'],
#     'mlpregressor__alpha': [0.0001, 0.001],
#     'mlpregressor__activation': ['relu', 'tanh'],
#     'mlpregressor__hidden_layer_sizes': [(100,),(50,50)]
# }

param_grid = {
    'mlpregressor__solver': ['adam'],
    'mlpregressor__alpha': [0.0001, 0.1],
    'mlpregressor__activation': ['relu'],
    'mlpregressor__hidden_layer_sizes': [(100,),(50,50),(30,30,30)]
}

reg_mlp = GridSearchCV(model_mlp, param_grid=param_grid, scoring='r2', refit=True, cv=5)
reg_mlp.fit(X_train, y_train)


y_pred = reg_mlp.predict(X_test)
print('MLPReg RMSE', root_mean_squared_error(y_test, y_pred))
print('MLPReg MSE', mean_squared_error(y_test, y_pred))
print('MLPReg MAE', mean_absolute_error(y_test, y_pred))
print('MLPReg r2', r2_score(y_test, y_pred))

# %%
model_lsgd = make_pipeline(
    StandardScaler(),
    SGDRegressor(max_iter=2500)
)

param_grid = {
    'sgdregressor__alpha': [0.0001, 0.001, 0.01],
    'sgdregressor__penalty': [None, 'l1','l2','elasticnet'],
    'sgdregressor__l1_ratio': [0.15, 0.5]
}

reg_lsgd = GridSearchCV(model_lsgd, param_grid=param_grid, scoring=scoring, cv=5, refit='r2')
reg_lsgd.fit(X_train, y_train)

y_pred = reg_lsgd.predict(X_test)
print('LSGDReg RMSE', root_mean_squared_error(y_test, y_pred))
print('LSGDReg MSE', mean_squared_error(y_test, y_pred))
print('LSGDReg MAE', mean_absolute_error(y_test, y_pred))
print('LSGDReg r2', r2_score(y_test, y_pred))
