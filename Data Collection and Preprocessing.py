import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Assume you have a CSV file "data.csv"
df = pd.read_csv('data.csv')

# Fill missing values
df = df.fillna(method='fill')

# Standardize features by removing the mean and scaling to unit variance
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# Convert to DataFrame
df_scaled = pd.DataFrame(df_scaled, columns=df.columns)

# Data Visualization
df.hist(bins=50, figsize=(20, 15))
plt.show()

# Correlation matrix
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True)
plt.show()
