# Breast Cancer Diagnosis using Machine Learning

This project involves analyzing breast cancer data to classify tumors as benign or malignant using various machine learning algorithms including Naive Bayes, Logistic Regression, and Decision Tree classifiers. The dataset is preprocessed, scaled, and split into training and testing sets to evaluate model performance.

## Dataset

The dataset used is the Breast Cancer Wisconsin dataset, consisting of 569 samples with 30 numerical features describing various characteristics of breast tumors. The target variable is `diagnosis`, which indicates whether the tumor is malignant (`M`) or benign (`B`).

Preprocessing steps performed include:  
- Dropping the column `Unnamed: 32` which contained only null values.  
- Encoding the `diagnosis` column to numerical values: malignant as 1 and benign as 0.  
- Applying Min-Max scaling to all feature columns to normalize their ranges.  
- Splitting the dataset into training and testing subsets with an 80-20 ratio, ensuring stratification by the diagnosis label.
