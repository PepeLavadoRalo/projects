# Project 1: Iris Classification with Random Forest

## Description
This project is an example of machine learning using the famous Iris dataset. The Iris dataset contains measurements of the sepal and petal lengths and widths for three different species of Iris flowers. The goal of this project is to predict the species of a flower based on these measurements using a Random Forest classifier.

### Key Steps:
1. **Data Loading**: The Iris dataset is loaded from `sklearn.datasets`.
2. **Data Exploration**: A pairplot is created to visualize the relationships between the features (sepal length, sepal width, petal length, petal width).
3. **Model Training**: A Random Forest classifier is trained to predict the flower species based on the input features.
4. **Model Evaluation**: The model's accuracy is calculated, and feature importance is displayed to understand which features influence the predictions the most.

## Requirements
To install the necessary packages for this project, use the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## Project Structure

```bash
project_1/
│
├── main.py          # Main Python script that runs the machine learning model
├── requirements.txt # List of required Python packages
└── .gitignore       # Git ignore file to exclude unwanted files from version control
```

## How to Run
1. Clone the repository:
```bash
git clone https://github.com/PepeLavadoRalo/projects.git
cd projects/project_1
```

2. Install the dependencies:
```bash
pip install -r requirements.txt
```

3.Run the project
```bash
python main.py
```

## Output
- A pairplot visualizing the Iris dataset.
- The accuracy of the Random Forest model on the test data.
- Feature importance showing how much each feature (sepal length, sepal width, etc.) contributes to the model's predictions.
