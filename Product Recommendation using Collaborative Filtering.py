from surprise import KNNWithMeans
from surprise import Dataset
from surprise import accuracy
from surprise.model_selection import train_test_split

# Load the movielens-100k dataset (just for example)
data = Dataset.load_builtin('ml-100k')

# sample random trainset and testset, test set is made of 25% of the ratings.
trainset, testset = train_test_split(data, test_size=.25)

# Use user_based true/false to switch between user-based or item-based collaborative filtering
algo = KNNWithMeans(k=50, sim_options={'name': 'pearson_baseline', 'user_based': True})
algo.fit(trainset)

# Test the trained model
predictions = algo.test(testset)

# Compute RMSE
accuracy.rmse(predictions)
