# STEP 9
# Removal of outliers and missing values

# Outliers are removed based on many reasons.
# They can significantly affect the mean and standard deviation, leading to biased statistical analysis.
# Models may try to fit the outliers, leading to overfitting and poor generalization to new data.
# Outliers can sometimes represent errors in the data or rare events that are not representative of the population.

# The following code will identify outliers in each attribute. Using Z-score.
import pandas as pd

file_path = 'C:\\Users\\kevin\\Downloads\\Laptop_price.csv'
df = pd.read_csv(file_path)

# Function to detect outliers using z-score
def detect_outliers_zscore(data, threshold=2):
    from scipy.stats import zscore
    z_scores = zscore(data)
    return data.index[(z_scores > threshold) | (z_scores < -threshold)]

# Select numerical columns
numerical_columns = df.select_dtypes(include='number')

# Detect outliers
outliers_index = {}
for col in numerical_columns.columns:
    outliers_index[col] = detect_outliers_zscore(df[col])

# Print outliers
print("Outliers detected:")
for col, index in outliers_index.items():
    if len(index) > 0:
        print(f"Column: {col}")
        print(df.iloc[index])
        print()
    else:
        print(f"No outliers found in column: {col}")

# Z-score is used to find outliers.
# Z-score measure the standard divieation to a data.
# This code goes over the data and checks if that value is too far from the mean. 
# If the z-score is greater than a certain threshold, it indicates that the data point is significantly different from the rest of the dataset and is considered an outlier.
# As the code says, there exists no outliers, therfor nothing to remove.

print("\n"+"======================================"+"\n")

# Identification and removal of missing values. 
# The folloing code will identify any missing values. 
missing_values = df.isnull().sum()
print("Missing Values:")
print(missing_values)

# Similar to outliers, missing values are also reconmanded to be removed to prevent bias analysis. 
# Removing also increase accuracy as there are more complete and accurate data to base prdictions on.
# No missing values in this data, so no removal of any data samples(rows) is needed.


