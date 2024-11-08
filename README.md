# Online-Fraud-Detection

# Online Fraud Detection Model Comparison

This repository contains a comparative analysis of various machine learning models for detecting online transaction fraud. The primary goal of this project is to identify the most effective algorithm for fraud detection based on accuracy, precision, and F1-score.

## Table of Contents
- [Project Overview](#project-overview) ✨
- [Features](#features) 📦
- [Tech Stack](#tech-stack) 💻
- [Dataset](#dataset) 🔍
- [Usage](#usage) 📝
- [Model Comparison](#model-comparison) 📊
- [Results](#results) 📈
- [Findings](#findings) ✅

## Project Overview ✨
This project evaluates and compares the performance of multiple machine learning models on a dataset of online transactions to determine which algorithm is best suited for fraud detection. By analyzing each model's ability to correctly identify fraudulent transactions, we aim to provide insights on the strengths and weaknesses of each approach.

## Features 📦
- Preprocessing and feature engineering to optimize model input.
- Comparison of several machine learning algorithms, including:
  - Naive Bayes
  - Decision Tree
  - Logistic Regression
  - Random Forest
- Detailed evaluation of each model based on accuracy, precision, and F1-score.

## Tech Stack 💻
- **Programming Language**: Python
- **Machine Learning Libraries**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Notebook Environment**: Jupyter Notebook

## Dataset 🔍
This project uses a dataset containing transaction records, including features like transaction amount, timestamp, and other potentially relevant information for fraud detection. 

[Online Fraud Dataset](https://www.kaggle.com/datasets/rupakroy/online-payments-fraud-detection-dataset/data)

## Usage 📝
Open the Jupyter Notebook provided and follow the steps to load the dataset, preprocess the data, and evaluate each model in the code. Each model is trained and evaluated individually, with a summary comparison at the end to identify the best-performing algorithm.

## Model Comparison 📊
The project compares the following machine learning models:

- **Naive Bayes**: Achieved high accuracy, though it has limitations in capturing complex dependencies between features.
- **Decision Tree**: Known for interpretability and performed well in initial testing.
- **Logistic Regression**: A simple baseline model for binary classification.
- **Random Forest**: An ensemble model with strong accuracy and precision.
### Note: K-Nearest Neighbors (KNN) was excluded due to computational inefficiency with large datasets.

## Results 📈

| Model               | Accuracy | Precision | F1-Score |
|:--------------------|:--------:|:---------:|:--------:|
| Naive Bayes         | 99.87%   | 100.0%    | 0.096    |
| Decision Tree       | 99.97%   | 89.31%    | 88.55    |
| Logistic Regression | 99.77%   | 34.18%    | 47.92    |
| Random Forest       | 99.97%   | 97.01%    | 85.90    |

## Findings ✅
- Decision Tree achieved the highest accuracy at 99.97%, with a strong F1-score, making it the best-performing model in this comparison.
- Naive Bayes also achieved high accuracy and perfect precision but had a low F1-score, indicating challenges in handling imbalanced data.
- Random Forest showed high precision and a good F1-score, making it another strong contender for fraud detection.
- Logistic Regression served as a baseline model but struggled with low precision and F1-score compared to other models.
