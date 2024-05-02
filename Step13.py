# STEP 13
# Investigating multiple Regression algorithms

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder

file_path = 'C:\\Users\\kevin\\Downloads\\Laptop_price.csv'
df = pd.read_csv(file_path)

# Setting 'Price' as the target variable
X = df.drop('Price', axis=1)
y = df['Price']

# One-hot encode the categorical variables
X = pd.get_dummies(X)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Up until now is the same as step 12, prepareing the data for model training and evaluation.
#__________________________________________________________________________________________________________________#

# MSE = Mean Squared Error, measures the average squared difference between the actual and predicted values. The Lower the better. 
# R^2 = R squared, measures how well the model explains the variability of the target variable, the higher the better, max is 1. 
# Running one test for all, interpretations are based on the ONE test. (Every time you run the code, the output is different, because its running individual tests)

# Linear Regression
linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)
linear_reg_pred = linear_reg.predict(X_test)
linear_reg_mse = mean_squared_error(y_test, linear_reg_pred)
linear_reg_r2 = r2_score(y_test, linear_reg_pred)

print("Linear Regression:")
print(f"MSE: {linear_reg_mse}")
print(f"R^2: {linear_reg_r2}\n")

print("\n"+"======================================"+"\n")

# MSE: 32304.737335988277
# R^2: 0.9996442074396727

# Lowest MSE (refer to step 12) of all tested regressions, highest R^2 of all tested regressions. Means model fits well. 

#__________________________________________________________________________________________________________________#

# Random Forest Regression
random_forest_reg = RandomForestRegressor()
random_forest_reg.fit(X_train, y_train)
random_forest_reg_pred = random_forest_reg.predict(X_test)
random_forest_reg_mse = mean_squared_error(y_test, random_forest_reg_pred)
random_forest_reg_r2 = r2_score(y_test, random_forest_reg_pred)

print("Random Forest Regression:")
print(f"MSE: {random_forest_reg_mse}")
print(f"R^2: {random_forest_reg_r2}\n")

print("\n"+"======================================"+"\n")

# MSE: 39448.15518205505
# R^2: 0.999565532448494

# Low MSE, high R^2, not better than linear but still very good. 

#__________________________________________________________________________________________________________________#

# Decision Tree Regression
decision_tree_reg = DecisionTreeRegressor()
decision_tree_reg.fit(X_train, y_train)
decision_tree_reg_pred = decision_tree_reg.predict(X_test)
decision_tree_reg_mse = mean_squared_error(y_test, decision_tree_reg_pred)
decision_tree_reg_r2 = r2_score(y_test, decision_tree_reg_pred)

print("Decision Tree Regression:")
print(f"MSE: {decision_tree_reg_mse}")
print(f"R^2: {decision_tree_reg_r2}\n")

print("\n"+"======================================"+"\n")

# MSE: 76331.11571385809
# R^2: 0.9991593170125479

# HIGHEST MSE, meaning more average errors. And, lowest R^2.    

#__________________________________________________________________________________________________________________#

# K-Neighbours Regression
knn_reg = KNeighborsRegressor()
knn_reg.fit(X_train, y_train)
knn_reg_pred = knn_reg.predict(X_test)
knn_reg_mse = mean_squared_error(y_test, knn_reg_pred)
knn_reg_r2 = r2_score(y_test, knn_reg_pred)

print("K-Neighbours Regression:")
print(f"MSE: {knn_reg_mse}")
print(f"R^2: {knn_reg_r2}\n")

print("\n"+"======================================"+"\n")

# MSE: 43009.520859210876
# R-squared: 0.9995263088696312

# Average. 

#__________________________________________________________________________________________________________________#

# XGBoost Regression
xgb_reg = XGBRegressor()
xgb_reg.fit(X_train, y_train)
xgb_reg_pred = xgb_reg.predict(X_test)
xgb_reg_mse = mean_squared_error(y_test, xgb_reg_pred)
xgb_reg_r2 = r2_score(y_test, xgb_reg_pred)

print("XGBoost Regression:")
print(f"MSE: {xgb_reg_mse}")
print(f"R^2s: {xgb_reg_r2}")

# MSE: 43257.62310296692
# R-squared: 0.9995235763622714

# Average. 


