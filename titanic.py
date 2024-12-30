# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 11:32:46 2024

@author: LENOVO
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier



#from pycaret.regression import *
from pycaret.classification import *

'''
#==========================================XGBoost Model==========================================================
#Cross Validation Score To Evaluate Model by its features best or not
#Lower Scores the Better
from xgboost import XGBRegressor
model1 = XGBRegressor(n_estimators=400,learning_rate=0.01)
scores = -1 * cross_val_score(model1, X, y,
                              cv=5,
                              scoring='neg_root_mean_squared_error')
scores.mean()

#==========================================LinearRegression Model==========================================================
from sklearn.linear_model import LinearRegression
model2 = LinearRegression()

scores2 = -1 * cross_val_score(model2, X, y,
                              cv=5,
                              scoring='neg_root_mean_squared_error')
scores2.mean()


#==========================================Ridge Regression Model==========================================================
from sklearn.linear_model import Ridge
model3 = Ridge(alpha=1.0)

scores3 = -1 * cross_val_score(model3, X, y,
                              cv=5,
                              scoring='neg_root_mean_squared_error')
scores3.mean()

#==========================================Lasso Regression Model==========================================================
from sklearn.linear_model import Lasso
model4 = Lasso(alpha=0.1)

scores4 = -1 * cross_val_score(model4, X, y,
                              cv=5,
                              scoring='neg_root_mean_squared_error')
scores4.mean()

#==========================================Support Vector Regression (SVR)===================================================
from sklearn.svm import SVR
model5 = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
scores5 = -1 * cross_val_score(model5, X, y.values.ravel(),
                              cv=5,
                              scoring='neg_root_mean_squared_error')
scores5.mean()

#==========================================Decision Tree Regressor===================================================
from sklearn.tree import DecisionTreeRegressor
model6 = DecisionTreeRegressor(max_depth=3)
scores6 = -1 * cross_val_score(model6, X, y,
                              cv=5,
                              scoring='neg_root_mean_squared_error')
scores6.mean()

#==========================================Random Forest===================================================
from sklearn.ensemble import RandomForestRegressor
model7 = RandomForestRegressor(n_estimators=100)
scores7 = -1 * cross_val_score(model7, X, y.values.ravel(),
                              cv=5,
                              scoring='neg_root_mean_squared_error')
scores7.mean()

#==========================================Gradient Boosting===================================================
from sklearn.ensemble import GradientBoostingRegressor
model8 = GradientBoostingRegressor(n_estimators=1000,learning_rate=0.01)
scores8 = -1 * cross_val_score(model8, X, y.values.ravel(),
                              cv=5,
                              scoring='neg_root_mean_squared_error')
scores8.mean()

#==========================================Neural Network===================================================
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

model9 = MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000,solver='lbfgs')

scores9 = -1 * cross_val_score(model9, X, y.values.ravel(),
                              cv=5,
                              scoring='neg_root_mean_squared_error')
scores9.mean()

#=============================================FINDING BEST MODEL WITH AUTO ML=================================================
from pycaret.regression import *

# Setup untuk regresi
reg_automl = setup(df, target='EV_HC',fold=5, fold_strategy='kfold')
best_model = compare_models()
tuned_model = tune_model(best_model)
scores10 = -1 * cross_val_score(tuned_model, X, y,
                              cv=5,
                              scoring='neg_root_mean_squared_error')
scores10.mean()
model10 = tuned_model
#===========================================HUBER REGRESSOR===============================================================
from sklearn.linear_model import HuberRegressor
model11 = HuberRegressor(epsilon=1.2, alpha=0.005, max_iter=200, fit_intercept=False)
scores11 = -1 * cross_val_score(model11, X, y.values.ravel(),
                              cv=5,
                              scoring='neg_root_mean_squared_error')
scores11.mean()

#==========================================Create Score Table===================================================

data_scores = {
    "model": ["XGboost", "Linear", "Ridge","Laso","SVR","Decision Tree",
              "Random Forest","Gradient Boosting","MultiLayerPerceptron","AutoML","HuberRegressor"],
    "score": [scores.mean(), scores2.mean(), scores3.mean(),scores4.mean(),scores5.mean(),scores6.mean(),
              scores7.mean(),scores8.mean(),scores9.mean(),0,scores11.mean()]}#scores10.mean()
score_table = pd.DataFrame(data_scores)

'''

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

#Check The Rate of Survive of Woman
women = train_data[(train_data['Sex']=='female')]['Survived']
#women.head()
ratewomen = sum(women)/len(women)

#Check The Rate of Survive Men
men =  train_data[(train_data['Sex']=='male')]['Survived']
ratemen = sum(men)/len(men)

features= train_data.drop(columns=['PassengerId','Name','Ticket','Cabin','Survived']).columns
target = ['Survived']

#datatypeColumns
train_data.dtypes

X = train_data[features]
y = train_data[target]

#auto ML
train_data_clean = train_data.drop(columns=['PassengerId','Name','Ticket','Cabin'])
reg_automl = setup(train_data_clean, target='Survived',fold=5, fold_strategy='kfold')
best_model = compare_models()
tuned_model = tune_model(best_model)
clf = RandomForestClassifier(random_state=42)

scores10 = cross_val_score(clf, X, y,
                              cv=5,
                              scoring='accuracy')
scores10.mean()
model = tuned_model


