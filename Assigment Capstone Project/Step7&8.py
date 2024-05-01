# STEP 7
# Visual Exploratory Data Analysis of data (with Histogram and Barcharts)

# Visualising the data with histogram and bar charts

import LibrariesPackages as lp
import matplotlib.pyplot as plt
file_path = 'C:\\Users\\kevin\\Downloads\\Laptop_price.csv'
df = lp.pd.read_csv(file_path)

# Histogram
# Plotting histograms of multiple columns together
df[['Processor_Speed', 'RAM_Size', 'Storage_Capacity', 'Screen_Size', 'Weight', 'Price']].hist(figsize=(18, 10))
plt.tight_layout()
plt.show()

#Barchart of RAM_Size
# Visualizing the count of a categorical variable (RAM size)
plt.figure(figsize=(8, 6))
df['RAM_Size'].value_counts().plot(kind='bar', color='blue', alpha=0.7)
plt.title('Count of RAM_Size')
plt.xlabel('RAM_Size')
plt.ylabel('Count')
plt.grid(axis='y')
plt.show()


# Pie chart of Brand
# Visualizing the count of a categorical variable (Brand)
plt.figure(figsize=(8, 6))
df['Brand'].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=140)
plt.title('Distribution of Brands')
plt.ylabel('')
plt.show()

# This gives a more visual summary of the database in the 3 stated attributes. 

# Histogram Interpretation
# Each histograms shows us the data distribution for a single continuous variable.
# The X-axis shows the range of values and Y-axis represent the number of values in that range.

#__________________________________________________________________________________________________________________#

# STEP 8
# Feature Selection based on data distribution

# This step asks to identify and reject useless columns from your dataset.
# These *useless* columns are those that do not provide any meaningful information for the analysis
# These usless colums could have a high rate of missing values, being constent values, or irrelevant and doesn;t impact the price of the laptop.

# Similar to question 6, we can see that there is no missing value or duplcations of data. 
# And, each attribute has dirrect impact to the price of the laptop, so there will be no removals.

# Brand, brand value will directly impact the price of the item. 
# Processor_Speed, faster processor speeds generally indicate better performance, therfore influencing price.
# RAM_Size, larger RAM sizes allow laptops to handle more tasks simultaneously and run more demanding applications smoothly. IMpacts price.
# Storage_Capacity, laptops with larger storage capacities can store more data, higher storage capacity often cost more to manufactuee, impacting the price.
# Screen_Size, laptops with larger screens tend to be more expensive due to the higher cost of manufacturing larger displays. Impacting price.
# Weight, lightweight laptops can be more appealing to users who travel frequently. To acchive lighter weight, advanced materials and engineering are required to reduce weight while maintaining durability and performance.
# Price, Having the 'Price' column allows us to train machine learning models to learn the relationship between features and the price of a laptop.

# With these reasons, all attributes are to remain and continue to serve a purpose.

