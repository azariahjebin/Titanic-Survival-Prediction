# Titanic Survival Prediction - Machine Learning Project

This repository contains a machine learning project focused on predicting the survival of passengers on the Titanic based on the famous Titanic dataset. This project uses several machine learning models to classify passengers into survivors and non-survivors, and evaluates their performance to identify the best model. The implementation includes Logistic Regression, Random Forest, Gradient Boosting, and other classifiers. We also perform feature engineering, data preprocessing, and hyperparameter tuning to enhance model accuracy.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation and Setup](#installation-and-setup)
- [Dependencies](#dependencies)
- [Features and Models](#features-and-models)
- [Model Performance](#model-performance)
- [Acknowledgements](#acknowledgements)

---

## Overview

The sinking of the Titanic is one of the most infamous disasters in history. In this project, we aim to predict whether a passenger would survive or not based on features like gender, age, class, and more. We experiment with various machine learning algorithms to create a model that delivers the best possible predictions, leveraging both classical and ensemble learning techniques.

### Objective

The primary goal of this project is to predict passenger survival by building, training, and evaluating different machine learning models on the Titanic dataset. The models are evaluated based on accuracy, precision, recall, and F1-score to ensure optimal performance.

---

## Dataset

The dataset used in this project is the [Titanic Dataset](https://www.kaggle.com/c/titanic/data) provided by Kaggle. It contains various passenger details such as name, age, gender, socio-economic class, and more.

### Features

The key features used in this project include:

- `PassengerId`: A unique identifier for each passenger
- `Pclass`: Ticket class (1st, 2nd, or 3rd)
- `Name`: Passenger's name
- `Sex`: Passenger's gender
- `Age`: Passenger's age
- `SibSp`: Number of siblings or spouses aboard
- `Parch`: Number of parents or children aboard
- `Ticket`: Ticket number
- `Fare`: Amount of money paid for the ticket
- `Cabin`: Cabin number
- `Embarked`: Port of embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)
- `Survived`: The target variable, indicating whether the passenger survived (1) or not (0).

---

## Project Structure

```
├── data
│   └── titanic.csv              # Titanic dataset
├── notebooks
│   └── Titanic.ipynb            # Jupyter notebook containing the full workflow
├── models
│   └── Titanic Survival Prediction.pkl                # Saved trained model
├── requirements.txt             # Required dependencies
├── README.md                    # Project documentation
```

- **data/**: Contains the dataset used in the project.
- **notebooks/**: Contains the notebook with the entire analysis and model training process.
- **models/**: Contains saved model files, such as trained models in `.pkl` format.
- **requirements.txt**: File specifying the Python libraries required for the project.

---

## Installation and Setup

### Prerequisites

Before you begin, ensure you have the following installed:

- Python 3.x
- Jupyter Notebook or Jupyter Lab

### Installation Steps

1. **Clone the repository**:

   ```bash
   git clone https://github.com/your-username/titanic-survival-prediction.git
   cd titanic-survival-prediction
   ```

2. **Create and activate a virtual environment**:

   ```bash
   python -m venv venv
   source venv/bin/activate   # For Linux and macOS
   venv\Scripts\activate      # For Windows
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the notebook**:

   Start Jupyter Notebook or Jupyter Lab and open the `Titanic.ipynb` file:

   ```bash
   jupyter notebook notebooks/Titanic.ipynb
   ```

---

## Dependencies

This project uses the following Python libraries:

- `numpy`: For numerical computation
- `pandas`: For data manipulation and analysis
- `matplotlib` and `seaborn`: For data visualization
- `scikit-learn`: For machine learning models
- `xgboost`: For gradient boosting
- `joblib`: For model saving and loading

You can find all dependencies in the `requirements.txt` file.

---

## Features and Models

### Feature Engineering

- **Handling Missing Values**: We handle missing values for features like `Age` and `Cabin`.
- **Encoding Categorical Variables**: Categorical features such as `Sex` and `Embarked` are label encoded.
- **Feature Scaling**: Numerical features are scaled to ensure consistent model performance.

### Models Used

The following machine learning models were implemented and evaluated:

- **Logistic Regression**: A baseline classifier for binary classification.
- **Decision Tree Classifier**: A tree-based model for classification tasks.
- **Random Forest Classifier**: An ensemble of decision trees to improve performance.
- **Gradient Boosting Classifier**: Boosting method for improving accuracy by reducing bias.
- **AdaBoost Classifier**: Adaptive boosting algorithm for enhancing model accuracy.
- **SVC**: Support Vector Classifier with a linear or radial kernel.
- **KNeighborsClassifier**: k-NN classifier for simple proximity-based classification.
- **GaussianNB**: Naive Bayes classifier for probabilistic predictions.
- **XGBoost**: A powerful gradient boosting algorithm optimized for performance.

---

## Model Performance

After training and testing the models, we compared their performance using accuracy, precision, recall, and F1 score. The **Random Forest Classifier** was the best-performing model with an accuracy of **82.68%**.

### Model Evaluation Metrics

- **Accuracy**: Proportion of correctly classified instances.
- **Precision**: True positives divided by the sum of true and false positives.
- **Recall**: True positives divided by the sum of true positives and false negatives.
- **F1-Score**: Harmonic mean of precision and recall, balancing false positives and false negatives.

The confusion matrix and ROC curves are also used to assess model performance.


---

## Acknowledgements

- [Kaggle](https://www.kaggle.com/) for providing the Titanic dataset.
- [scikit-learn](https://scikit-learn.org/) for machine learning utilities.
- The [Python](https://www.python.org/) and [Jupyter](https://jupyter.org/) communities for their continued support in data science.

---
