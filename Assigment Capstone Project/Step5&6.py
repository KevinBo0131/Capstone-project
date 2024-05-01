# STEP 5
# Data exploration at basic level

# There are four commands which are used for Basic data exploratory Analysis in Python
# head() : This helps to see a few sample rows of the data
# info() : This provides the summarized information of the data
# describe() : This provides the descriptive statistical details of the data
# nunique(): This helps us to identify if a column is categorical or continuous

import LibrariesPackages as lp
file_path = 'C:\\Users\\kevin\\Downloads\\Laptop_price.csv'
df = lp.pd.read_csv(file_path)

# See first and last 3 lines of the database.
lp.display(lp.pd.concat([df.head(3), df.tail(3)]))

print("\n"+"======================================"+"\n")

# Shows summary of information. 
# Shows if theres any missing values. 
# Remove Qualitative variables which cannot be used in Machine Learning.
lp.display(df.info())

print("\n"+"======================================"+"\n")

# See the data base.
# shows more
def calculate_statistics(df):
    stats_df = df.describe()
    return stats_df

stats_df = calculate_statistics(df)
lp.display(stats_df)

import LibrariesPackages as lp

print("\n"+"======================================"+"\n")

# Based on the summarized data, you can now create reports of the data base.
# with reports and supporting graphs, it'll allow users to pridict future costs. 

#__________________________________________________________________________________________________________________#

# STEP 6
# Identifying and rejecting useless columns

# With this we can confirm the required columns that are required to continue this report. 
# All columns are significant and contributes to the price difference of a laptop, therefore no rejection of any data. 
