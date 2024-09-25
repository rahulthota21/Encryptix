# SMS Spam Classification using Naive Bayes

This project focuses on building an AI model that classifies SMS messages as either spam or legitimate using a Naive Bayes classifier. By processing the text messages and applying techniques like TF-IDF, the model aims to accurately predict whether a given message is spam. The model achieves an accuracy score of **98%**.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Project Overview](#project-overview)
- [Model](#model)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Spam messages are a common nuisance, and building a system that can automatically classify messages as spam or legitimate is critical for improving user experience. In this project, we leverage natural language processing (NLP) and machine learning techniques to develop a robust spam classification model.

We utilize the **Naive Bayes** algorithm, which is particularly well-suited for text classification tasks like this due to its simplicity and efficiency in handling large-scale data.

## Dataset

The dataset used in this project is the **[SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)** from Kaggle. It contains a collection of SMS messages, with each message labeled as either **spam** or **ham** (legitimate).

### Dataset Features:
- **Label**: Indicates whether the message is spam or legitimate.
- **Message**: The actual content of the SMS message.

The dataset includes 5,572 messages, with approximately 13.4% spam and 86.6% legitimate messages.

## Project Overview

The goal of this project is to classify SMS messages as spam or legitimate using a **Naive Bayes classifier**. We preprocess the text data and apply **TF-IDF (Term Frequency-Inverse Document Frequency)** to convert the text into a form that can be fed into the classifier. After training the model, we evaluate its performance using accuracy and other relevant metrics.

### Key Steps:
1. **Data Preprocessing**: Cleaning the text data, including removing punctuation, converting text to lowercase, and tokenization.
2. **Feature Extraction**: Converting text into numerical features using **TF-IDF** vectorization.
3. **Model Selection**: Training a Naive Bayes classifier on the transformed data.
4. **Model Evaluation**: Measuring the model's accuracy, precision, recall, and F1-score on the test dataset.
5. **Final Model**: Achieved 98% accuracy on the test dataset.

## Model

### Naive Bayes Classifier

Naive Bayes is a probabilistic classifier based on applying Bayes' Theorem with the assumption of independence between the features (words). It is particularly effective for text classification tasks because:
- It handles high-dimensional data efficiently.
- It works well with sparse datasets like the one produced by TF-IDF vectorization.
- It is computationally inexpensive, making it suitable for large-scale datasets.

We used **Multinomial Naive Bayes** for this task, which is well-suited for document classification where the data is represented as word frequency counts or TF-IDF values.

### Key Techniques:
- **TF-IDF Vectorization**: Converts text into numerical features based on the frequency of words in the document relative to their frequency across all documents. This helps to highlight important words while reducing the importance of common words.
  
### Key Libraries:
- **Scikit-learn**: For TF-IDF vectorization and machine learning model implementation.
- **Pandas**: For data manipulation.
- **NumPy**: For numerical operations.
- **Matplotlib & Seaborn**: For data visualization and performance analysis.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/sms-spam-classifier.git
    cd sms-spam-classifier
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

- **Train the Model**: Run `train_model.py` to train the Naive Bayes classifier on the SMS spam dataset.
- **Evaluate the Model**: The trained model will automatically evaluate performance on the test set and display accuracy, precision, recall, and F1-score metrics.
- **Classify New Messages**: Use the model to classify new SMS messages as spam or legitimate by running `predict.py` with a custom message.

## Results

The Naive Bayes model achieved the following performance on the test dataset:
- **Accuracy**: 98%


The high accuracy of the model demonstrates that Naive Bayes is an effective approach for spam classification, particularly when combined with TF-IDF feature extraction.

## Contributing

Contributions are welcome! If you'd like to contribute to this project, please fork the repository and create a pull request with your proposed changes.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more information.

