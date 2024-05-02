# STEP 4
# Visualising the distribution of Target variable

# Understand the distribution of the target variable ('Price' in this case).
# Identify the central tendency and spread of the target variable.
# Use a histogram to visualize the distribution of the target variable.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_path = 'C:\\Users\\kevin\\Downloads\\Laptop_price.csv'
df = pd.read_csv(file_path)

# Visualizing the distribution of the target variable
plt.figure(figsize=(8, 6))
sns.histplot(df['Price'], bins=30, kde=True, color='blue')
plt.title('Distribution of Laptop Prices')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()


# The distribution is right-skewed, indicating that there are fewer laptops at higher price points.
# There are a few laptops priced significantly higher (outliers) beyond the main distribution. These could be premium or specialized models/higher specs.
# Smaller peaks around $500-$600 and $1200-$1300. These clusters represent popular price points.


