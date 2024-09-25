# Customer Churn Prediction using Random Forest Classifier

This project aims to predict customer churn for a subscription-based service using a Random Forest Classifier. By analyzing historical customer data, including features like customer demographics and usage behavior, the model identifies customers who are likely to discontinue the service.

## Table of Contents
- [Introduction](#introduction)
- [Project Overview](#project-overview)
- [Data](#data)
- [Model](#model)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Customer churn is a major concern for subscription-based businesses. The ability to predict churn allows businesses to take proactive measures to retain customers, enhance customer satisfaction, and improve revenue.

In this project, we use machine learning to predict whether a customer will churn based on historical data such as usage patterns and demographic information. The project implements a Random Forest Classifier, which is an ensemble learning method that combines multiple decision trees to improve prediction accuracy.

## Project Overview

The objective of this project is to build a predictive model to determine which customers are likely to churn. A Random Forest Classifier was chosen due to its robustness in handling complex datasets with multiple features and its ability to model non-linear relationships.

### Key Steps:
1. **Data Preprocessing**: Clean and prepare the dataset for training.
2. **Feature Engineering**: Extract relevant features from customer usage data and demographic information.
3. **Model Training**: Train the Random Forest Classifier on the preprocessed dataset.
4. **Evaluation**: Evaluate the model's performance using appropriate metrics like accuracy, precision, recall, and F1-score.
5. **Interpretation**: Analyze feature importance to understand which factors contribute most to customer churn.

## Data

The dataset used for this project includes historical customer data with the following features:
- **Demographics**: Age, gender, income, location, etc.
- **Subscription Info**: Subscription type, duration, renewal history, etc.
- **Usage Behavior**: Monthly usage statistics, feature usage, activity frequency, etc.
- **Churn Label**: Binary indicator where `1` represents a customer that has churned, and `0` indicates retention.

## Model

### Random Forest Classifier

The Random Forest algorithm works by building multiple decision trees during training and outputting the mode of the classes (classification) as the prediction. It's particularly effective for churn prediction because it:
- Handles large datasets with high dimensionality.
- Provides insights into feature importance.
- Is robust to overfitting, especially with an appropriate number of trees.

The model was tuned using hyperparameter optimization to ensure the best possible performance. Key hyperparameters like the number of estimators, maximum depth, and minimum samples split were fine-tuned.

### Key Libraries Used:
- **Scikit-learn**: For machine learning algorithms and evaluation.
- **Pandas**: For data manipulation.
- **NumPy**: For numerical operations.
- **Matplotlib & Seaborn**: For data visualization.

## Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/yourusername/customer-churn-prediction.git
    cd customer-churn-prediction
    ```

2. Install the required Python libraries:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the model training script:
    ```bash
    python train_model.py
    ```

## Usage

- **Train the Model**: Run `train_model.py` to train the Random Forest Classifier on the historical customer dataset.
- **Evaluate the Model**: After training, the model can be evaluated using the test dataset. The results will include accuracy, precision, recall, and F1-score metrics.
- **Feature Importance**: The model outputs feature importance to help identify key factors contributing to customer churn.

## Results

The trained Random Forest model achieved the following metrics on the test set:
- **Accuracy**: 86%

## Contributing

Contributions are welcome! If you'd like to contribute to this project, please fork the repository and create a pull request with your proposed changes.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more information.

