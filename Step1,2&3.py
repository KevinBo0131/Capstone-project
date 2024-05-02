# ST1/ST1G Assignment 9 (Capstone Programming Project)
# Laptop_price.csv StudentID:u3261037
# Available at https://www.kaggle.com/datasets/mrsimple07/laptoppriceprediction 
# It contains data of 999 laptop specs and other details.
# The goal is to make a calculator that can predict laptop prices based on the data, it does that by compareing specs to price.

# Step 1: Reading the data Reading the data with python

# Specify file path
file_path = 'C:\\Users\\kevin\\Downloads\\Laptop_price.csv'

import pandas as pd

# Read the CSV file into a DataFrame
file_path = 'C:\\Users\\kevin\\Downloads\\Laptop_price.csv'
df = pd.read_csv(file_path)

# Number of unique attributes
num_unique_attributes = df.nunique()

# Check for duplicates
duplicates = df.duplicated().any()

print("Number of Unique Attributes:")
print(num_unique_attributes)
print("\nDuplicates:")
print(duplicates)

# The DataFram(df) counts 1001 lines, one of them being the attributes. Therefore 1000 input datas. Input starts at line 0. 
# These attributes being, Brand, Processor_Speed, RAM_Size, Storage_Capacity, Screen_Size, Weight, Price
# There are no duplications of data. 

#__________________________________________________________________________________________________________________#

# STEP 2
# Problem statement definition

# Creating  a prediction model to predict the price of laptops
# Target Variable: Price Predictors/Features: Brand, Processor_Speed, RAM_Size, Storage_Capacity, Screen_Size, Weight

#__________________________________________________________________________________________________________________#

# STEP 3
# Target variable identification

# Selecting the target variable, the attribute which the calculator targets to solve.
# In this case it would be the *PRICE* of the laptop


