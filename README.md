# Data Science Project Portfolio

This repository contains three independent data science / machine learning projects completed as part of academic coursework: **Retail Sales EDA**, **Sentiment Analysis**, and **Credit Card Fraud Detection**. Each project is self-contained with its own dataset, notebook, and methodology, summarized below.

## Table of Contents
1. [Retail Sales Exploratory Data Analysis](#1-retail-sales-exploratory-data-analysis)
2. [Sentiment Analysis on Twitter and Reddit Data](#2-sentiment-analysis-on-twitter-and-reddit-data)
3. [Credit Card Fraud Detection](#3-credit-card-fraud-detection)
4. [Repository Structure](#repository-structure)
5. [Setup & Installation](#setup--installation)
6. [Author](#ghulam-mustafa)

---

## 1. Retail Sales Exploratory Data Analysis

### Overview
Performs exploratory data analysis (EDA) on a retail sales dataset to uncover patterns in customer purchasing behavior, product performance, and sales trends, deriving actionable business insights through data cleaning, visualization, and statistical summarization.

### Objectives
- Understand the structure and quality of the retail sales dataset
- Identify sales trends over time (daily, monthly, seasonal patterns)
- Analyze product/category-wise performance
- Explore customer purchasing behavior and segmentation
- Detect anomalies, outliers, and missing data patterns

### Dataset
- **Source**: Kaggle (retail sales dataset)
- **Format**: CSV — transaction-level records including order date, product/category, quantity, price, and customer information

> Note: Add the exact Kaggle dataset name/link before final submission.

### Tools & Technologies
Python · Pandas · Matplotlib · Seaborn · NumPy · Jupyter Notebook

### Methodology
1. Data loading & inspection
2. Data cleaning (missing values, duplicates, formatting)
3. Univariate analysis (sales, price, quantity distributions)
4. Bivariate/multivariate analysis (category, time, customer segments)
5. Time-series trend visualization
6. Insight generation

### Key Findings
> Summarize top 3–5 insights here (e.g. peak sales months, best-performing categories, price–quantity correlation).

---

## 2. Sentiment Analysis on Twitter and Reddit Data

### Overview
Builds a text classification pipeline to determine sentiment (positive, negative, or neutral) of social media posts from Twitter and Reddit, combining TF-IDF feature extraction with classical machine learning classifiers.

### Objectives
- Preprocess and clean raw social media text
- Convert text into numerical features using TF-IDF vectorization
- Train and evaluate ML models for sentiment classification
- Compare model performance to identify the best approach

### Dataset
- **Source**: Kaggle — [Twitter and Reddit Sentimental Analysis Dataset](https://www.kaggle.com/datasets/cosmos98/twitter-and-reddit-sentimental-analysis-dataset)
- **Files**: `Twitter_Data.csv`, `Reddit_Data.csv`
- **Labels**: Positive, Negative, Neutral

### Tools & Technologies
Python · Scikit-learn (TF-IDF, Naive Bayes, Logistic Regression) · Pandas/NumPy · Matplotlib/Seaborn · NLTK/re · Jupyter Notebook

### Methodology
1. Data loading (merge Twitter + Reddit datasets)
2. Text preprocessing (lowercasing, stopword/punctuation removal, tokenization)
3. TF-IDF feature extraction
4. Model training — Naive Bayes and Logistic Regression
5. Evaluation — accuracy, precision, recall, F1-score, confusion matrix
6. Model comparison

### Results
> Summarize final accuracy/F1-scores for Naive Bayes vs. Logistic Regression, and note the best-performing model.

---

## 3. Credit Card Fraud Detection

### Overview
Builds a machine learning system to detect fraudulent credit card transactions, focusing on robust preprocessing, class-imbalance handling, and comparison of multiple classical ML models.

### Objectives
- Explore the characteristics of the credit card transaction dataset
- Handle severe class imbalance between fraud and legitimate transactions
- Train and compare multiple classification models
- Evaluate using metrics suited for imbalanced classification
- Identify the most effective model for fraud detection

### Dataset
- **Source**: Kaggle — Credit Card Fraud Detection dataset
- **Contents**: Anonymized PCA-transformed features (V1–V28), `Time`, `Amount`, and binary `Class` label (0 = legitimate, 1 = fraud)
- **Characteristic**: Highly imbalanced — fraud is a very small fraction of transactions

> Note: Confirm the exact Kaggle dataset link/version before final submission.

### Tools & Technologies
Python · Scikit-learn (Logistic Regression, Random Forest) · XGBoost · Pandas/NumPy · Matplotlib/Seaborn · imbalanced-learn (SMOTE, if used) · Jupyter Notebook

### Methodology
1. Data loading & exploration (class distribution, feature stats)
2. Preprocessing — scale `Amount`/`Time`, handle class imbalance
3. Model training — Logistic Regression, Random Forest, XGBoost
4. Evaluation — precision, recall, F1-score, ROC-AUC, confusion matrix
5. Model comparison

### Results
> Summarize precision/recall/F1/ROC-AUC across models and note the best performer.

---

## Repository Structure
```
project-portfolio/
├── retail-sales-eda/
│   ├── data/
│   ├── notebooks/
│   │   └── retail_sales_eda.ipynb
│   └── images/
├── sentiment-analysis/
│   ├── data/
│   │   ├── Twitter_Data.csv
│   │   └── Reddit_Data.csv
│   └── notebooks/
│       └── sentiment_analysis.ipynb
├── credit-card-fraud/
│   ├── data/
│   │   └── creditcard.csv
│   └── notebooks/
│       └── credit_card_fraud_detection.ipynb
├── README.md
└── requirements.txt
```

## Setup & Installation
```bash
# Clone the repository
git clone <repository-url>
cd project-portfolio

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter and open any project notebook
jupyter notebook
```

## Future Work
- **Retail Sales EDA**: Customer segmentation (RFM analysis), sales forecasting, interactive dashboard
- **Sentiment Analysis**: Deep learning approaches (LSTM, BERT), word embeddings, deployment as an API
- **Credit Card Fraud**: Anomaly detection (Isolation Forest, Autoencoders), cost-sensitive learning, real-time deployment

## Author
Submitted as part of academic coursework — Information Technology, CBIT (Osmania University)
