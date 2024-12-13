# Project 3 - Housing Price Prediction with Random Forest

## Description
This project uses a **Random Forest** model to predict housing prices based on various features, using the **House Prices** dataset available on **OpenML**.

The goal is to create a prediction model that estimates a house's price based on features such as lot area, number of rooms, material quality, and others.

## Results Obtained
After tuning the model's hyperparameters, the best results obtained are as follows:

- **MAE (Mean Absolute Error):** 14,005.95
- **MSE (Mean Squared Error):** 423,934,876.17
- **RMSE (Root Mean Squared Error):** 20,589.68
- **R²:** 0.896 (the model explains approximately 89.6% of the variability in housing prices).

While the results are quite good, there is room for improvement. An **R²** of 0.896 suggests that the model is performing well in prediction, but **MAE** and **RMSE** might seem relatively high depending on the range of house prices.

## Possible Improvements
### 1. Try Other Machine Learning Algorithms
- **Gradient Boosting** (e.g., **XGBoost** or **LightGBM**) are known for superior performance in regression tasks and could improve the results of this model.
- **Deep Neural Networks** or **Support Vector Machines (SVM)** could also be explored as alternatives.

### 2. Further Hyperparameter Tuning
While parameters like `max_depth`, `min_samples_split`, and `min_samples_leaf` have been adjusted, it may be helpful to test a greater number of combinations or perform a **GridSearchCV** instead of random search to explore more possible values.

Increasing the number of estimators (`n_estimators`) might also improve accuracy, although this could increase computation time.

### 3. Improve Feature Engineering
- Exploring new feature transformations could help improve the model. Some ideas include:
    - Creating new features from existing ones, such as interactions between features.
    - Transforming features like house size or year of construction using logarithms or different scales.
- Perform **feature importance analysis** to ensure all relevant variables are being used.

### 4. More Data
Training the model with more data or a different dataset that is more representative of a wider variety of homes could help improve the model's generalization and reduce errors.

### 5. More Exhaustive Cross-Validation
Using more exhaustive cross-validation (e.g., 10-fold instead of 5-fold) could help provide a more robust evaluation and avoid overfitting.

## How to Improve the Results on Your Machine
If you want to improve the results, you can try the following recommendations in your environment:

- **Try XGBoost:**
  Install **XGBoost** and train a similar model to Random Forest but using XGBoost-specific parameters.

- **Perform more detailed hyperparameter tuning:**
  Use **GridSearchCV** for a more exhaustive search for the best hyperparameters.

- **Explore regularization techniques and feature tuning:**
  Try **feature normalization** or **feature selection** using techniques such as **PCA** (Principal Component Analysis) or **Recursive Feature Elimination (RFE)**.

## Data and Preprocessing
This model uses the **House Prices** dataset from OpenML, which contains various housing features. The following steps were carried out during preprocessing:

1. **Outlier removal** for housing prices.
2. **Handling missing values**:
   - Numerical features were imputed with the **median**.
   - Categorical features were imputed with the **most frequent value**.
3. **Feature scaling** using `StandardScaler` for numerical variables.
4. **One-Hot Encoding** for categorical variables.

## Model Used
The model used for predictions is a **Random Forest Regressor**, tuned with **RandomizedSearchCV** to optimize hyperparameters. The key hyperparameters adjusted include:
- `n_estimators`: Number of trees in the forest.
- `max_depth`: Maximum depth of trees.
- `min_samples_split`: Minimum samples required to split a node.
- `min_samples_leaf`: Minimum samples required to be a leaf.
- `max_features`: Maximum number of features to consider for splitting a node.

## Evaluation Metrics
The model's performance was evaluated using the following metrics:

- **MAE (Mean Absolute Error):** Average of the absolute differences between the predictions and actual values.
- **MSE (Mean Squared Error):** Average of the squared differences between the predictions and actual values.
- **RMSE (Root Mean Squared Error):** Square root of MSE, gives a measure of error magnitude in the same units as the target variable.
- **R² (Coefficient of Determination):** The proportion of the variance in the target variable that is explained by the model. An R² of 0.896 means the model explains approximately 89.6% of the variance in house prices.

## Instructions to Run the Project
1. Clone this repository to your local machine:
```bash
git clone https://github.com/PepeLavadoRalo/projects.git
```
2. Navigate to the project directory:
```bash
cd projects/project_3
```
3. Install the necessary dependencies:
```bash
pip install -r requirements.txt
```
4.Run the main file to predict housing prices:
```bash
python main.py
```
