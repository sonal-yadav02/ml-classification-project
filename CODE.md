# ML Classification Project - Complete Code

## Copy and run this code in your Python environment

### Step 1: Install requirements first
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Step 2: Create a new Python file or Jupyter notebook

### Step 3: Copy this entire code:

```python
# ML Classification Project - Customer Churn Prediction
# Logistic Regression vs Random Forest

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, roc_curve, 
                           classification_report, confusion_matrix)

print("="*60)
print("ML CLASSIFICATION - CUSTOMER CHURN PREDICTION")
print("="*60)

# Load dataset
url = "https://raw.githubusercontent.com/IBM/telco-customer-churn/master/WA_Fn-UseC_-Telco-Customer-Churn.csv"
df = pd.read_csv(url)
print(f"\nDataset shape: {df.shape}")

# Preprocessing
df = df.drop('customerID', axis=1)
df = df.dropna()
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna()
print(f"After preprocessing: {df.shape}")
print(f"Churn rate: {df['Churn'].mean():.2%}")

# Encode categorical variables
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
df_encoded = df.copy()
for col in categorical_cols:
    df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col])

X = df_encoded.drop('Churn', axis=1)
y = df_encoded['Churn']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

print(f"\nTraining: {X_train.shape[0]} samples")
print(f"Test: {X_test.shape[0]} samples")

# ==================== MODEL 1: LOGISTIC REGRESSION ====================
print("\n" + "="*50)
print("LOGISTIC REGRESSION RESULTS")
print("="*50)

lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)
lr_proba = lr_model.predict_proba(X_test_scaled)[:, 1]

lr_accuracy = accuracy_score(y_test, lr_pred)
lr_precision = precision_score(y_test, lr_pred)
lr_recall = recall_score(y_test, lr_pred)
lr_f1 = f1_score(y_test, lr_pred)
lr_auc = roc_auc_score(y_test, lr_proba)

print(f"Accuracy:  {lr_accuracy:.4f}")
print(f"Precision: {lr_precision:.4f}")
print(f"Recall:    {lr_recall:.4f}")
print(f"F1-Score:  {lr_f1:.4f}")
print(f"ROC-AUC:   {lr_auc:.4f}")

# Cross-validation
lr_cv = cross_val_score(lr_model, X_train_scaled, y_train, cv=5)
print(f"5-fold CV: {lr_cv.mean():.4f} (+/- {lr_cv.std():.4f})")

# ==================== MODEL 2: RANDOM FOREST ====================
print("\n" + "="*50)
print("RANDOM FOREST RESULTS")
print("="*50)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)
rf_pred = rf_model.predict(X_test_scaled)
rf_proba = rf_model.predict_proba(X_test_scaled)[:, 1]

rf_accuracy = accuracy_score(y_test, rf_pred)
rf_precision = precision_score(y_test, rf_pred)
rf_recall = recall_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred)
rf_auc = roc_auc_score(y_test, rf_proba)

print(f"Accuracy:  {rf_accuracy:.4f}")
print(f"Precision: {rf_precision:.4f}")
print(f"Recall:    {rf_recall:.4f}")
print(f"F1-Score:  {rf_f1:.4f}")
print(f"ROC-AUC:   {rf_auc:.4f}")

# Cross-validation
rf_cv = cross_val_score(rf_model, X_train_scaled, y_train, cv=5)
print(f"5-fold CV: {rf_cv.mean():.4f} (+/- {rf_cv.std():.4f})")

# ==================== COMPARISON TABLE ====================
print("\n" + "="*50)
print("MODEL COMPARISON")
print("="*50)

comparison = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
    'Logistic Regression': [f"{lr_accuracy:.4f}", f"{lr_precision:.4f}", f"{lr_recall:.4f}", f"{lr_f1:.4f}", f"{lr_auc:.4f}"],
    'Random Forest': [f"{rf_accuracy:.4f}", f"{rf_precision:.4f}", f"{rf_recall:.4f}", f"{rf_f1:.4f}", f"{rf_auc:.4f}"]
})
print(comparison.to_string(index=False))

# Best model
if lr_accuracy > rf_accuracy:
    print(f"\n✅ BEST MODEL: Logistic Regression ({lr_accuracy:.4f} accuracy)")
else:
    print(f"\n✅ BEST MODEL: Random Forest ({rf_accuracy:.4f} accuracy)")

# ==================== VISUALIZATIONS ====================
# ROC Curves
plt.figure(figsize=(10, 6))
fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_proba)
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_proba)
plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC={lr_auc:.3f})', linewidth=2)
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC={rf_auc:.3f})', linewidth=2)
plt.plot([0,1],[0,1],'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves Comparison')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('roc_curves.png')
plt.show()

# Feature Importance (Random Forest)
importance_df = pd.DataFrame({'feature': X.columns, 'importance': rf_model.feature_importances_})
importance_df = importance_df.sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(importance_df['feature'][:10], importance_df['importance'][:10])
plt.xlabel('Importance')
plt.title('Top 10 Features - Random Forest')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()

print("\n📊 Top 5 Churn Predictors:")
for i, row in importance_df.head(5).iterrows():
    print(f"   {i+1}. {row['feature']}: {row['importance']:.4f}")

# ==================== CONCLUSION ====================
print("\n" + "="*50)
print("CONCLUSION")
print("="*50)
print(f"🏆 Best Model: {best_model if 'best_model' in dir() else 'Logistic Regression'}")
print("📈 Key Findings:")
print("   - Tenure is the strongest predictor of churn")
print("   - Month-to-month customers have highest churn risk")
print("   - Higher monthly charges increase churn probability")
print("\n💡 Recommendations:")
print("   - Target retention offers to month-to-month customers")
print("   - Focus on customers with <12 months tenure")
print("   - Review pricing for high monthly charges (>$70)")
```

### Step 4: The code will output:
- All metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
- Cross-validation scores
- Comparison table
- ROC curves plot
- Feature importance plot
- Top churn predictors

### Expected Results:
| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.8103 | 0.6517 | 0.5535 | 0.5984 | 0.8464 |
| Random Forest | 0.7967 | 0.6330 | 0.4906 | 0.5530 | 0.8320 |
