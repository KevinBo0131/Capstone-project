# STEP 4
# Visualising the distribution of Target variable

# Understand the distribution of the target variable ('Price' in this case).
# Identify the central tendency and spread of the target variable.
# Use a histogram to visualize the distribution of the target variable.

import LibrariesPackages as lp

# Read the CSV file into a DataFrame
file_path = 'C:\\Users\\kevin\\Downloads\\Laptop_price.csv'
df = lp.pd.read_csv(file_path)

# Visualizing the distribution of the target variable
lp.plt.figure(figsize=(8, 6))
lp.sns.histplot(df['Price'], bins=30, kde=True, color='blue')
lp.plt.title('Distribution of Laptop Prices')
lp.plt.xlabel('Price')
lp.plt.ylabel('Frequency')
lp.plt.show()


# The distribution is right-skewed, indicating that there are fewer laptops at higher price points.
# There are a few laptops priced significantly higher (outliers) beyond the main distribution. These could be premium or specialized models/higher specs.
# Smaller peaks around $500-$600 and $1200-$1300. These clusters represent popular price points.