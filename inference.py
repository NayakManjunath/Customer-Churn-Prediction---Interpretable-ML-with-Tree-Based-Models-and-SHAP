import joblib
import pandas as pd

# Load artifacts
model = joblib.load("artifacts/model.pkl")
preprocess = joblib.load("artifacts/preprocess.pkl")

def predict_churn(df_new):
    X_new = preprocess.transform(df_new)
    probs = model.predict_proba(X_new)[:, 1]
    return probs

# Example usage
if __name__ == "__main__":
    sample = pd.read_csv("data/sample_customers.csv")
    preds = predict_churn(sample)
    print(preds[:5])
