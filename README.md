# SVM-DECISION-TREE

# Bank Customer Churn Prediction
Predicting whether a customer will leave a bank using machine learning models (Decision Tree, Logistic Regression, and SVM) with comprehensive Exploratory Data Analysis (EDA), feature engineering, and visualization.

## Table of Contents

Project Overview

Dataset

EDA & Preprocessing

Feature Engineering

Models

Evaluation Metrics

Visualizations

Insights

Future Scope

How to Run

## Project Overview

This project aims to predict customer churn in a bank using historical customer data. Churn prediction is critical for banks to retain valuable customers by proactively identifying potential leavers.

## We implement:

Exploratory Data Analysis (EDA)

Feature preprocessing and normalization

Outlier reduction using IQR method

Machine Learning Models:

Logistic Regression

Decision Tree Classifier (with hyperparameter tuning)

Support Vector Machine (SVM)

We also perform model evaluation, visualization of distributions, decision boundaries, and tree structure interpretation.

## Dataset

Source: Kaggle – Bank Customer Churn Prediction

## Columns Overview:

Column	Description
RowNumber	Index of the row
CustomerId	Unique customer ID
Surname	Customer surname
CreditScore	Customer credit score
Geography	Customer country (France, Germany, Spain)
Gender	Customer gender (Male/Female)
Age	Customer age
Tenure	Number of years with the bank
Balance	Account balance
NumOfProducts	Number of bank products held
HasCrCard	Credit card possession (1 = Yes, 0 = No)
IsActiveMember	Active member flag (1 = Yes, 0 = No)
EstimatedSalary	Customer estimated salary
Exited	Target variable: churn (1 = Yes, 0 = No)
EDA & Preprocessing

## Categorical Mapping:

Gender: Male = 0, Female = 1

Geography: France = 1, Germany = 2, Spain = 3

## Distribution Analysis:

### 📊 Correlation Heatmap Visualization

<img width="1008" height="782" alt="image" src="https://github.com/user-attachments/assets/98ae0b5e-e77e-4d45-9203-95b8d585ac0d" />

## Histograms and KDE plots for all numerical features
## Skewness calculated before and after Yeo-Johnson normalization

<img width="1189" height="490" alt="image" src="https://github.com/user-attachments/assets/70f3756f-12b6-4126-a558-0770d6012961" />

<img width="1189" height="490" alt="image" src="https://github.com/user-attachments/assets/374cea6d-d9ed-4a27-a3d6-de6e16c75a89" />


 <img width="1189" height="490" alt="image" src="https://github.com/user-attachments/assets/c59443c1-5712-4ba6-bfa0-c68cec6b5f98" />

## Outlier Handling:

Outliers detected and capped using IQR method

Side-by-side boxplots before and after outlier reduction

<img width="985" height="390" alt="image" src="https://github.com/user-attachments/assets/2d272f86-1608-4194-a8bd-0f0d50094b6d" />


<img width="989" height="390" alt="image" src="https://github.com/user-attachments/assets/d084cf3f-ef9c-45bc-a8f9-f2d104c8ae07" />

Counts of outliers before and after correction

Feature Engineering

Drop irrelevant columns (RowNumber, CustomerId, Surname)

Scale numerical features using StandardScaler

Prepare train-test split (80%-20%)

Features ready for ML models

## Models
### 1. Logistic Regression

Predicts probability of churn using linear decision boundary

Evaluated with accuracy, precision, recall, F1, ROC-AUC

### 2. Decision Tree Classifier

Hyperparameter tuning for max_depth and min_samples_split

Evaluated with accuracy, precision, recall, F1, ROC-AUC, RMSE

Visualized tree structure and actual vs predicted

2D decision boundary visualization using PCA

### 3. SVM (Optional)

Can be added for comparison with non-linear decision boundaries

Evaluation Metrics

Accuracy: Correct predictions over total predictions

Precision: Correct positive predictions over predicted positives

Recall: Correct positive predictions over actual positives

F1 Score: Harmonic mean of precision and recall

ROC-AUC: Area under the ROC curve

RMSE: Root Mean Squared Error for model performance

### Visualizations

## Actual vs Predicted scatter plots

<img width="691" height="547" alt="image" src="https://github.com/user-attachments/assets/14cf3a47-8625-4442-953a-fd5279627cf3" />

## Decision Tree structure visualization

<img width="1698" height="816" alt="image" src="https://github.com/user-attachments/assets/a19bf0b2-3806-41d0-8da2-5caef91c4b05" />

## Decision boundary in 2D PCA projection

<img width="689" height="547" alt="image" src="https://github.com/user-attachments/assets/bea296e5-b194-4cfc-81ea-a0c952e578a8" />

Confusion matrices

## Insights

Certain features like CreditScore, Age, Balance, and Tenure have skewed distributions that require normalization.

Outlier reduction reduces extreme impacts on models.

Decision Tree identifies key features contributing to churn.

Churn patterns vary by geography and account balance.

## Future Scope

Implement Random Forest or XGBoost for better accuracy

Use SMOTE or class balancing if target is imbalanced

Deploy as a web app dashboard for bank analysts

Include SVM kernel tuning for complex non-linear patterns

Perform feature importance analysis

## How to Run

### Clone the repository:

git clone <your-repo-url>
cd bank-customer-churn-prediction


### Install required packages:

pip install -r requirements.txt


### Run the main notebook/script:

jupyter notebook Bank_Customer_Churn.ipynb


All plots, metrics, and model outputs will be generated automatically.

### References

Kaggle Dataset: Bank Customer Churn Prediction

Sklearn Documentation: https://scikit-learn.org

Visualization with Seaborn and Matplotlib
