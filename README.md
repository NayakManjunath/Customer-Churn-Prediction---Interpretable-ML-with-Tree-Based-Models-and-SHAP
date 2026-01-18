# Customer Churn Prediction  
### Interpretable Machine Learning with Tree-Based Models and SHAP

This project builds a supervised machine learning pipeline to predict customer churn in a telecom dataset.  
The focus is not only on prediction accuracy but also on **model interpretability** using **SHAP** with tree-based models.

The goal is to answer two questions:
1. Which customers are likely to churn?
2. *Why* is the model predicting churn for a given customer?

---

## Problem Statement

Customer churn is a major business problem in the telecom industry. Retaining existing customers is often cheaper than acquiring new ones.

This project predicts whether a customer will churn based on usage, billing, and service-related features, and explains model decisions using SHAP values so that business teams can take actionable steps.

---

## Dataset

- **Domain**: Telecom customer data  
- **Target Variable**: `Churn` (Binary: Yes / No)
- **Typical Features**:
  - Customer tenure
  - Monthly charges
  - Total charges
  - Contract type
  - Internet service
  - Payment method
  - Add-on services (security, backup, streaming, etc.)

> Dataset can be replaced with any similar telecom churn dataset without changing the core pipeline.

---

## Approach

### 1. Data Preprocessing
- Handle missing values
- Encode categorical variables
- Scale numerical features (if required)
- Train-test split

### 2. Model Building
- Tree-based supervised models:
  - Gradient Boosting (primary)
  - (Optional) Random Forest / XGBoost / LightGBM
- Handle class imbalance using:
  - Class weights or
  - Resampling techniques (if needed)

### 3. Model Evaluation
- Accuracy
- Precision, Recall, F1-score
- ROC-AUC
- Confusion Matrix

### 4. Model Explainability with SHAP
- Global feature importance
- Local explanations for individual predictions
- SHAP summary plots
- SHAP dependence plots

This makes the model **transparent and business-friendly**, not a black box.

---

## Tech Stack

- **Language**: Python  
- **Libraries**:
  - pandas, numpy
  - scikit-learn
  - xgboost / lightgbm (optional)
  - shap
  - matplotlib, seaborn

---
## Project Structure
customer-churn-prediction/
│
├── data/
│ ├── raw/
│ └── processed/
│
├── notebooks/
│ ├── 01_eda.ipynb
│ ├── 02_preprocessing.ipynb
│ ├── 03_model_training.ipynb
│ └── 04_shap_explainability.ipynb
│
├── src/
│ ├── data_preprocessing.py
│ ├── train_model.py
│ ├── evaluate_model.py
│ └── shap_explainer.py
│
├── models/
│ └── churn_model.pkl
│
├── requirements.txt
└── README.md

---

## How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/customer-churn-prediction.git
cd customer-churn-prediction
```
-------
### 2. Install Dependencies
pip install -r requirements.txt

### 3. Run the Pipeline
Explore data and modeling via notebooks
or
Run scripts from src/ for training and evaluation

### Key Insights from SHAP
Contract type and tenure are strong churn drivers
High monthly charges combined with short tenure increase churn risk
Customers on month-to-month contracts are more likely to churn
SHAP helps explain individual customer risk, not just global trends

### Business Value

Helps retention teams identify high-risk customers
Explains why a customer is likely to churn
Enables targeted offers and proactive interventions
Improves trust in ML predictions among stakeholders

## Future Enhancements
Add model comparison dashboard
Deploy as a REST API using Flask/FastAPI
Integrate with real-time customer data
Add cost-sensitive learning for retention campaigns

### Author
Manjunath Nayak
Data Scientist

## License
This project is licensed under the MIT License.

---
