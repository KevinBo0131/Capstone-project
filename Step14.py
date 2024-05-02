# STEP 14
# Selection of the best Model

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

file_path = 'C:\\Users\\kevin\\Downloads\\Laptop_price.csv'
df = pd.read_csv(file_path)

# Setting 'Price' as the target variable
X = df.drop('Price', axis=1)
y = df['Price']

# One-hot encode the categorical variables
X = pd.get_dummies(X)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Different types of testing regression models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest Regression": RandomForestRegressor(),
    "Decision Tree Regression": DecisionTreeRegressor(),
    "K-Neighbours Regression": KNeighborsRegressor(),
    "XGBoost Regression": XGBRegressor()
}

results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results.append({"Model": name, "MSE": mse, "R-squared": r2})

results_df = pd.DataFrame(results)

# Find the best model based on MSE
best_model = results_df.loc[results_df["MSE"].idxmin()]

print("Best Model based on MSE:")
print(best_model)

