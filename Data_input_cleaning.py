# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your data
# Replace 'your_data.csv' with the path to your actual data file
df = pd.read_csv('your_data.csv')

# Display the first few rows of the dataframe
print(df.head())

# Display summary statistics
print(df.describe())

# Count the number of purchases made by each user
user_purchase_counts = df['user_id'].value_counts()
print(user_purchase_counts)

# Plot a histogram of the purchase amounts
plt.hist(df['purchase_amount'], bins=50)
plt.xlabel('Purchase Amount')
plt.ylabel('Frequency')
plt.title('Distribution of Purchase Amounts')
plt.show()

# Count the number of purchases for each product
product_purchase_counts = df['product_id'].value_counts()

# Plot the top 10 most purchased products
product_purchase_counts[:10].plot(kind='bar')
plt.xlabel('Product ID')
plt.ylabel('Number of Purchases')
plt.title('Top 10 Most Purchased Products')
plt.show()

# More complex visualizations might involve looking at the relationship between different variables
# For example, you might want to see if users who buy product A are also likely to buy product B
# This would require more complex data manipulation and is beyond the scope of this basic example