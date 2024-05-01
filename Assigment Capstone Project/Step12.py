# STEP 12
# Training/Testing Sampling and K-fold cross validation

import LibrariesPackages as lp
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

file_path = 'C:\\Users\\kevin\\Downloads\\Laptop_price.csv'
df = lp.pd.read_csv(file_path)

# Split the data into features (X) and target variable (y)
X = df.drop(["Price"], axis=1) 
y = df["Price"].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Creates two sets of data: one for training the model and one for testing the model.
# Allows us to evaluate the model's performance on unseen data.
# Helps prevent overfitting by evaluating the model on data it hasn't seen during training.

#__________________________________________________________________________________________________________________#

# Converts categorical variables into a numerical format suitable for model training.
# Ensures all features are represented in a consistent numerical format

# One-hot encode the categorical features in the training and testing sets
X_train_encoded = lp.pd.get_dummies(X_train)
X_test_encoded = lp.pd.get_dummies(X_test)

# Function to calculate the performance of KNN for different k values
def knn_performance(X_train, y_train, X_test):
    k_values = range(1, 21)  # Test k values from 1 to 20
    mse_values = []

    for k in k_values:
        # Initialize and fit the KNN regressor
        knn = KNeighborsRegressor(n_neighbors=k)
        knn.fit(X_train, y_train)
        
        # Make predictions on the testing set
        y_pred = knn.predict(X_test)
        
        # Calculate Mean Squared Error (MSE)
        mse = mean_squared_error(y_test, y_pred)
        mse_values.append(mse)

    return k_values, mse_values

#__________________________________________________________________________________________________________________#

# Call the knn_performance function
k_values, mse_values = knn_performance(X_train_encoded, y_train, X_test_encoded)

# Plot MSE values for different k values
plt.plot(k_values, mse_values, marker='o')
plt.title('KNN Performance')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Mean Squared Error (MSE)')
plt.xticks(np.arange(1, 21, 1))
plt.grid(True)
plt.show()

# Perform K-fold cross-validation (e.g., with 5 folds)
cv_scores = cross_val_score(KNeighborsRegressor(), X_train_encoded, y_train, cv=5, scoring='neg_mean_squared_error')

# Convert negative MSE scores to positive
cv_scores = -cv_scores

print("Cross-Validation Scores:")
print(cv_scores)
print(f"Mean CV Score: {cv_scores.mean()}")

# The graph shows the average error rate at each 'k'. The lower the error rate the better.
# The output scores tell you how well the model is performing across different folds of the data.
# The lower the MSE, the better the model's performance
