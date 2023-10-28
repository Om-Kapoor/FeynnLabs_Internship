from sklearn.metrics.pairwise import cosine_similarity

# Calculate the cosine similarity
cosine_sim = cosine_similarity(df_scaled, df_scaled)

# Get the pairwsie similarity scores of all items with item of interest
sim_scores = list(enumerate(cosine_sim[0]))

# Sort the items based on the similarity scores
sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
