# STEP 10
# Visual and Statistic Correlation analysis for selection of best features.

# First step is to creat a dot plot graph for each attribute. 
# The goal is to choose the best columns(Features) which are correlated to the Target variable (Price).

import LibrariesPackages as lp
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway

file_path = 'C:\\Users\\kevin\\Downloads\\Laptop_price.csv'
df = lp.pd.read_csv(file_path)

# Define numerical attributes
numerical_attributes = ['Processor_Speed', 'RAM_Size', 'Storage_Capacity', 'Screen_Size', 'Weight']

# Create dot plots for each attribute against the price
plt.figure(figsize=(18, 10))

for i, attribute in enumerate(numerical_attributes, 1):
    plt.subplot(2, 3, i)
    sns.scatterplot(data=df, x=attribute, y='Price', color='blue')
    plt.title(f'{attribute} vs Price')
    plt.xlabel(attribute)
    plt.ylabel('Price')

plt.tight_layout()
plt.show()

# The reason to creat dot plot is to seek for any trends. 
# As a result, there is no *clear* trend as they all seem to be at any price with any specs. 
# Except, Storage_Capacity. From the plot we can clearly see as the storage capacity increase the price value also increases substantially and more visable. 
# Every other attributes seem to be the opposite, even if processor speed is on the higher end, between 3.5 - 4.0, the price of the laptop can still be on the cheaper end. 
# That also applies to Ram_Size, Screen_size and weight. 
# Storage capacity has the highest correlation.

#__________________________________________________________________________________________________________________#

# Statistical Feature Selection
# The following code creates a graph of a correlation matrix.

# Column order
df = df[['Price', 'Brand', 'Processor_Speed', 'RAM_Size', 'Storage_Capacity', 'Screen_Size', 'Weight']]

# Remove non-numeric columns
df_numeric = df.drop(columns=['Brand'])

# Calculate correlation matrix
correlation_matrix = df_numeric.corr()

# Plot correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

# Due to the low correlation numbers, any correlation greater or equals to ±0.05 will be considered as significant
# These are the results...
# Price: 1 (Very Significant)
# Processor_speed: -0.05 (Significant)
# RAM_Size: 0.06 (Significant)
# Storage_Capacity: 1 (Very Significant)
# Screen_Size: -0.03 (NOT Significant)
# Weight: 0.04 (NOT Significant)

# So the final selected features are Storage_Capacity, RAM_Size, Processor_speed.

#__________________________________________________________________________________________________________________#

# ANOVA testing

def anova_comparison(df, columns, target_variable):
    results = {}
    for column in columns:
        groups = df.groupby(target_variable)[column].apply(list)
        anova_result = f_oneway(*groups)
        results[column] = anova_result
    return results

# Define columns and target variable
columns = ['Processor_Speed', 'RAM_Size', 'Storage_Capacity', 'Screen_Size', 'Weight']
target_variable = 'Brand'

# Perform ANOVA test
anova_results = anova_comparison(df, columns, target_variable)

# Display ANOVA results
for column, result in anova_results.items():
    print(f"ANOVA result for {column}: {result}")

# The result of the ANOVA test suggests that there are significant differences in RAM size between the different brands.

