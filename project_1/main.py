# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the Iris dataset from sklearn
iris = load_iris()

# Create a pandas DataFrame with the Iris data
data = pd.DataFrame(iris.data, columns=iris.feature_names)

# Add the 'species' column to the DataFrame
data['species'] = iris.target

# Display the first few rows of the dataset to understand its structure
print("First few rows of the dataset:")
print(data.head())

# Create a pairplot to visualize the relationships between the features
sns.pairplot(data, hue='species')
plt.show()

# Split the features (X) and the target variable (y)
X = data.drop('species', axis=1)  # Features (sepal length, sepal width, petal length, petal width)
y = data['species']  # Target variable (species)

# Split the dataset into training and test sets (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the Random Forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions with the test data
y_pred = model.predict(X_test)

# Calculate the model's accuracy with the test data
accuracy = accuracy_score(y_test, y_pred)
print(f'Model accuracy: {accuracy:.2f}')

# Optional: Display the feature importance to understand what influences the model's decisions
feature_importance = model.feature_importances_
print("Feature importance:")
for feature, importance in zip(iris.feature_names, feature_importance):
    print(f"{feature}: {importance:.4f}")
