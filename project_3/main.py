import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy.stats import randint

# Load the housing dataset from OpenML
housing = fetch_openml(name="house_prices", as_frame=True)

# Input features (X) and target (y)
X = housing.data
y = housing.target

# Filter out outliers in the target variable (house prices)
y = pd.to_numeric(y, errors='coerce')  # Ensure y is numeric
threshold = y.quantile(0.99)  # Limit to the 99th percentile
X = X[y <= threshold]
y = y[y <= threshold]

# 2.1 **Visualizing the features**
# Print descriptive statistics of the features
print(X.describe())

# Histogram of house prices to visualize the distribution
plt.figure(figsize=(10, 6))
sns.histplot(y, kde=True, color='blue')
plt.title('Distribution of House Prices')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()

# 2.2 **Correlation analysis**
# Define relevant columns for correlation analysis
relevant_columns = ['LotArea', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotRmsAbvGrd', 'YearBuilt']

# Check if all relevant columns exist in the dataset
print("Available columns in the dataset:", X.columns)
print("Relevant columns selected:", relevant_columns)

# If any column is missing from the dataset, print an error message
missing_columns = [col for col in relevant_columns if col not in X.columns]
if missing_columns:
    print(f"The following columns are not present in the dataset: {missing_columns}")
else:
    # Select relevant columns and add the target variable (price)
    X_relevant = X[relevant_columns].copy()
    X_relevant['Price'] = y

    # Check for missing values in the relevant columns
    print("Missing values in the relevant columns:", X_relevant.isnull().sum())

    # Ensure that the data is numeric for correlation calculation
    X_relevant = X_relevant.apply(pd.to_numeric, errors='coerce')

    # Calculate the correlation matrix
    correlation_matrix = X_relevant.corr()

    # Plot the heatmap of the correlation matrix
    plt.figure(figsize=(10, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix Between Selected Features and Price')
    plt.show()

# 2.3 **Checking for missing values**
print("Missing values before imputation:")
print(X.isnull().sum())
print("Missing values in target (y):", y.isnull().sum())

# 2.4 **Identifying categorical columns**
categorical_columns = X.select_dtypes(include=['object']).columns
print("Categorical columns:", categorical_columns)

# 2.5 **Imputation of missing values**
# Impute numerical features with the median
imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(imputer.fit_transform(X.select_dtypes(exclude=['object'])), columns=X.select_dtypes(exclude=['object']).columns)

# Impute categorical features with the most frequent value
imputer_cat = SimpleImputer(strategy='most_frequent')
X_imputed[categorical_columns] = pd.DataFrame(imputer_cat.fit_transform(X[categorical_columns]), columns=categorical_columns)

print("Missing values after imputation:")
print(X_imputed.isnull().sum())

# Normalize numerical features
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed.select_dtypes(exclude=['object'])), columns=X_imputed.select_dtypes(exclude=['object']).columns)

# One-Hot Encoding for categorical variables
X_encoded = pd.get_dummies(X_imputed, drop_first=True)

# 4. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# 5. Hyperparameter tuning using RandomizedSearchCV
# Set up a random forest model and hyperparameter grid for tuning
rf = RandomForestRegressor(random_state=42)
param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],  # Control maximum depth of trees
    'min_samples_split': [5, 10],  # Larger minimum number of samples to split a node
    'min_samples_leaf': [2, 4],  # Larger minimum number of samples in a leaf node
    'max_features': ['sqrt', 'log2'],  # Try a smaller subset of features
    'bootstrap': [True, False]  # Whether to use bootstrap sampling
}

# Perform hyperparameter search with RandomizedSearchCV and 10-fold cross-validation
random_search = RandomizedSearchCV(
    rf, param_distributions=param_dist, n_iter=20, cv=10, verbose=2, n_jobs=-1, scoring='neg_mean_squared_error', random_state=42
)
random_search.fit(X_train, y_train)

# Print the best hyperparameters
print(f"Best hyperparameters: {random_search.best_params_}")

# Train and predict using the best model
best_rf_model = random_search.best_estimator_
y_pred = best_rf_model.predict(X_test)

# Model evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"R²: {r2}")
