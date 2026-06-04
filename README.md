# ml-classification-project
# 🔍 ML Classification Project - Customer Churn Prediction

> Comparing Logistic Regression vs Random Forest for customer churn prediction

## 📊 Project Overview

This project builds and evaluates supervised classification models to predict customer churn for a telecommunications company. Two algorithms are compared using multiple performance metrics.

**Best Model:** Logistic Regression (81.03% Accuracy)

## 🎯 Key Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **Logistic Regression** | **81.03%** | **65.17%** | **55.35%** | **59.84%** | **84.64%** |
| Random Forest | 79.67% | 63.30% | 49.06% | 55.30% | 83.20% |

## 📈 Dataset

- **Source:** IBM Telco Customer Churn Dataset
- **Samples:** 7,043 customers
- **Features:** 20 (tenure, monthly charges, contract type, etc.)
- **Target:** Churn (Yes/No) - 26.6% churn rate

## 🛠️ Technologies Used

- Python 3.8+
- Pandas, NumPy (Data processing)
- Scikit-learn (ML models)
- Matplotlib, Seaborn (Visualizations)

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/sonal-yadav02/ml-classification-project.git
cd ml-classification-project

Key Findings
Tenure is the strongest predictor of churn

Month-to-month contracts have 3x higher churn risk

Higher monthly charges increase churn probability

New customers (<12 months) need retention focus

Visualizations Included
ROC Curves Comparison

Confusion Matrices

Feature Importance Plot

Churn Distribution

Correlation Heatmap
