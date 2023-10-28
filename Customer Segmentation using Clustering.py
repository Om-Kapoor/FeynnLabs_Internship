from sklearn.cluster import KMeans

# Assume we choose 5 clusters
kmeans = KMeans(n_clusters=5, random_state=0).fit(df_scaled)

# Get cluster labels
labels = kmeans.labels_

# Add labels to dataframe
df_scaled['Cluster'] = labels

# Visualizing Clusters
sns.pairplot(df_scaled, hue='Cluster')
