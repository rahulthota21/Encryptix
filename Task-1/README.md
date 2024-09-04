# Task 1: Movie Genre Classification

## Part 1: Overview

This task involves building a machine learning model to classify movies into different genres based on their plot summaries or other textual data.

## Objectives

- **Create a Machine Learning model** that predicts movie genres.
- **Experiment with different techniques** such as TF-IDF, word embeddings, Naive Bayes, Logistic Regression, and Support Vector Machines (SVM).

## Dataset

- **Source:** [Genre Classification Dataset on Kaggle](https://www.kaggle.com/datasets/hijest/genre-classification-dataset-imdb)
- **Description:** The dataset contains movie plot summaries and their corresponding genres.

## Setup Instructions

1. **Clone the Repository:** If not already done, clone the main repository:
   ```bash
   git clone https://github.com/rahulthota21/Encryptix.git


### Part 2: Set Up Your Environment

**1. Activate Your Environment:**
   - Open Command Prompt or Terminal.
   - Navigate to your project directory.
   - Activate the environment with:
     ```bash
     Encryptix_Task1\Scripts\activate
     ```

**2. Install Required Libraries:**
   - Ensure the environment is activated.
   - Create a `requirements.txt` file with the following content:
     ```
     numpy
     pandas
     seaborn
     scikit-learn
     matplotlib
     ```
   - Install the libraries using:
     ```bash
     pip install -r requirements.txt
     ```

**3. Verify Installations:**
   - Check the installed libraries with:
     ```bash
     pip list
     ```
### Part 3: Loading Libraries and Dataset

**1. Load Required Libraries:**
   - Open your Jupyter Notebook.
   - Import the libraries you need for the project, such as NumPy, Pandas, Seaborn, Scikit-learn, and Matplotlib.

**2. Load the Dataset:**
   - Read the dataset files (`train_data.txt`, `test_data.txt`, etc.) into your Jupyter Notebook using Pandas.
   - Ensure you use the correct delimiters to properly parse the text files.

### Part 4: Data Exploration

**1. Check for Null Values:**
   - Inspect the dataset to identify if there are any missing values in the columns.
   - If any null values are found, decide on how to handle them (e.g., filling with default values or removing them).

**2. Check for Duplicates:**
   - Check if there are any duplicate entries in the dataset.
   - If duplicates are present, decide on how to handle them (e.g., removing duplicates).



   
