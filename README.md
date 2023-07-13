# Spam Classification using Multinomial Naive Bayes

This repository contains code for performing spam classification using Multinomial Naive Bayes. The code is implemented in Python using the scikit-learn library and demonstrates the process of training a model, evaluating its performance, and making predictions on new messages.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Code Explanation](#code-explanation)
- [Results](#results)
- [Prediction](#prediction)
- [Conclusion](#conclusion)

## Introduction
Spam classification is the task of automatically identifying whether a given message is spam (unsolicited bulk email) or ham (non-spam). This repository provides a Python implementation of spam classification using the Multinomial Naive Bayes algorithm. The code demonstrates how to load a dataset, preprocess the data, train the model, evaluate its performance, and make predictions on new messages.

## Installation
To run the code in this repository, you need to have Python installed along with the following libraries:
- pandas
- numpy
- scikit-learn

You can install these libraries using pip:

```shell
pip install pandas numpy scikit-learn
```

## Usage
1. Clone the repository:

```shell
git clone https://github.com/your-username/spam-classification.git
```

2. Navigate to the repository directory:

```shell
cd spam-classification
```

3. Run the Python script:

```shell
python spam_classification.py
```

## Dataset
The code uses a spam dataset stored in a CSV file. The dataset contains two columns: "Category" (ham or spam) and "Message" (text of the message). The script loads the dataset using the pandas library and performs data analysis and description.

## Code Explanation
The code is organized into several sections, each serving a specific purpose. Here's a brief explanation of each section:

1. **Loading the Dataset**: The script loads the spam dataset from a CSV file using the pandas library.

2. **Data Analysis and Description**: This section performs basic analysis and description of the loaded dataset, such as displaying the first few rows, computing descriptive statistics, and checking for null values.

3. **Preparing the Data for Training and Testing**: The code creates a target column indicating whether each message is spam or not. It then splits the dataset into training and testing sets using the `train_test_split` function from scikit-learn.

4. **Feature Extraction: CountVectorizer**: To convert text data into numerical features, the code uses the `CountVectorizer` class from scikit-learn. It creates an instance of `CountVectorizer` with lowercase conversion and English stop words removal.

5. **Model Training: Multinomial Naive Bayes**: The code initializes an instance of the `MultinomialNB` class, representing the Multinomial Naive Bayes model. It trains the model using the training set's transformed features and corresponding labels.

6. **Model Evaluation**: After training the model, the code makes predictions on the testing set and calculates the accuracy score, generates a confusion matrix, and a classification report using the scikit-learn metrics functions.

7. **Prediction on New Messages**: The code includes an example of predicting whether a new message is spam or ham. It takes a new message as input, transforms it using the trained `CountVectorizer` instance, and predicts its label using the trained model.

## Results
The results section provides the accuracy score after testing the model, a confusion matrix representing true positives (TP), false positives (FP), false negatives (FN), and true negatives (TN), and a classification report showing precision, recall, and F1-score for both classes.

## Prediction
The code includes an example of predicting whether a new message is spam or ham. It demonstrates how to transform a new message using the trained `CountVectorizer` instance and predict its label using the trained model.

## Conclusion
This repository provides a practical implementation of spam classification using Multinomial Naive Bayes. By following the code and the accompanying documentation, you can learn how to load a dataset, preprocess the data, train a model, evaluate its performance, and make predictions on new messages. You can also apply similar techniques to classify spam messages in your own datasets.
