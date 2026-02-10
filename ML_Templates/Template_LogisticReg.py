# %%
import numpy as np
import pandas as pd
import sklearn

# %%
## Load data
from sklearn.datasets import load_iris
X, y = load_iris(return_X_y=True)

## Data preprocess

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


## Data pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

model = make_pipeline(
    StandardScaler(),
    LogisticRegression()
)

param_grid = {
    'logisticregression__C': [0.01, 0.1, 1, 10],
    'logisticregression__penalty': ['l1', 'l2'],
    'logisticregression__solver': ['liblinear']
}
grid = GridSearchCV(model, param_grid, scoring='accuracy', cv = 5)
grid.fit(X_train, y_train)

# ## Alternative to print cross_val scores
# from sklearn.model_selection import cross_val_score, cross_validate
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import make_pipeline
# from sklearn.linear_model import LogisticRegression

# model = make_pipeline(
#     StandardScaler(),
#     LogisticRegression()
# )

# cv_results = cross_validate(model, X_train, y_train, cv=5, return_estimator=True, scoring='accuracy')
# scores = cv_results['test_score'] # All cv scores
# models = cv_results['estimator'] # All cv models
# grid = models[np.argmax(scores)] # Best model


## Data test
from sklearn.metrics import accuracy_score, classification_report 
y_pred = grid.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))