# Movie Genre Classification

## Overview

This project builds a machine learning model to classify movies into genres based on plot summaries.

## Objectives

- Predict movie genres using various machine learning techniques.
- Experiment with TF-IDF, word embeddings, Naive Bayes, Logistic Regression, and SVM.

## Dataset

- **Source:** [Genre Classification Dataset on Kaggle](https://www.kaggle.com/datasets/hijest/genre-classification-dataset-imdb)
- **Description:** Movie plot summaries with corresponding genres.

## Setup Instructions

### 1. Clone the repository:
   ```bash
   git clone https://github.com/rahulthota21/Encryptix.git
```
### 2.Activate your environment:
   ```bash
   Encryptix_Task1\Scripts\activate
```
### 3.Install required libraries:
```bash
pip install -r requirements.txt
```
### 4.Verify installations
```bash
pip list
```

## Data Exploration and Preparation
```bash
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
```

## Modeling
- TF-IDF Vectorization: Convert text data into numerical features.
- SVM Training: Train an naive_bayes model on the processed data.

## Evaluation
- Accuracy Score: Evaluate the model's performance.
- Classification Report: Review detailed metrics.

## Results
The model's accuracy and other performance metrics are discussed in the notebook.

## Contributing
Fork and create a pull request for improvements.

## License
This project is licensed under the MIT License.

## Acknowledgements
- Dataset: [Kaggle](https://www.kaggle.com/)

**For detailed explanations and code, refer to the Movie Classification.ipynb notebook.**








