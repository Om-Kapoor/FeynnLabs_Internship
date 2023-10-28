# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Assuming df is your DataFrame and it has columns 'feature1', 'feature2'... as predictors and 'rating' as the target variable
X = df[['feature1', 'feature2']]  # replace these with your actual feature columns
y = df['rating']

# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Predict product ratings for the test set
y_pred = model.predict(X_test)

#Classification

from sklearn.linear_model import LogisticRegression

# Assuming df has 'feature1', 'feature2'... as predictors and 'segment' as the target variable
X = df[['feature1', 'feature2']]  # replace these with your actual feature columns
y = df['segment']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create a logistic regression model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Predict customer segments for the test set
y_pred = model.predict(X_test)
