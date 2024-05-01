# STEP 11
# Data Conversion to numeric values for machine learning/predictive analysis

# The understanding of this question.
# The goals is to turn catagorical vlaues (non-numerical) to numbered values.
# In this case, turning the attribute 'Brand' to numerical values.
# Meaning, instead of the laptop falling under the catagory Asus, it'll go under a numeric catagory for that brand.

# The following code turns non-numerical value 'Brand' to numbered value. 
import LibrariesPackages as lp
from sklearn.preprocessing import LabelEncoder

file_path = 'C:\\Users\\kevin\\Downloads\\Laptop_price.csv'
df = lp.pd.read_csv(file_path)

# Identify categorical variables
categorical_cols = df.select_dtypes(include=['object']).columns
print("Categorical Columns:")
print(categorical_cols)

# Handle missing values in categorical columns
df[categorical_cols] = df[categorical_cols].fillna('Unknown')

# Apply Label Encoding
label_encoder = LabelEncoder()
for col in categorical_cols:
    df[col] = label_encoder.fit_transform(df[col])

# Display the first 25 rows of the encoded DataFrame
print("Encoded DataFrame:")
print(df.head(25))

# As you could see in the termical output. All data is now numerical.
# Laptop brands have the following brand_codes.

# brand_code - Brand
# 0 - Acer
# 1 - Asus
# 2 - Dell
# 3 - HP
# 4 - Lenovo

