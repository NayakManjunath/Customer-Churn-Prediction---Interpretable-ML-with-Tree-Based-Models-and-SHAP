import streamlit as st
import joblib
import pandas as pd

# Load trained pipeline
model = joblib.load("artifacts/model.pkl")

st.title("Customer Churn Scoring")

uploaded_file = st.file_uploader(
    "Upload customer data (CSV)", type="csv"
)

df = None  # <-- SAFETY INITIALIZATION

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data Preview")
    st.dataframe(df.head())

    st.markdown("### Run Model")

    if st.button("Run Churn Scoring"):
        if df is None:
            st.error("No data loaded")
        else:
            probs = model.predict_proba(df)[:, 1]
            df["churn_probability"] = probs

            cols = ["churn_probability"] + [c for c in df.columns if c != "churn_probability"]
            df = df[cols]

            st.subheader("Scored Customers")
            st.dataframe(df.head())

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Scored File",
                csv,
                "churn_scored_customers.csv",
                "text/csv"
            )


# import streamlit as st
# import joblib
# import pandas as pd

# # Load trained pipeline (includes preprocessing)
# model = joblib.load("artifacts/model.pkl")

# st.title("Customer Churn Scoring")

# uploaded_file = st.file_uploader(
#     "Upload customer data (CSV)", type="csv"
# )
# if st.button("Run Churn Scoring"):
#     probs = model.predict_proba(df)[:, 1]
#     df["churn_probability"] = probs

#     # Move churn_probability to front
#     cols = ["churn_probability"] + [c for c in df.columns if c != "churn_probability"]
#     df = df[cols]

#     st.subheader("Scored Customers")
#     st.dataframe(df.head())

#     csv = df.to_csv(index=False).encode("utf-8")
#     st.download_button(
#         "Download Scored File",
#         csv,
#         "churn_scored_customers.csv",
#         "text/csv"
#     )


# if uploaded_file is not None:
#     df = pd.read_csv(uploaded_file)

#     st.subheader("Uploaded Data Preview")
#     st.dataframe(df.head())

#     st.markdown("### Run Model")

#     if st.button("Run Churn Scoring"):
#         probs = model.predict_proba(df)[:, 1]
#         df["churn_probability"] = probs

#         st.subheader("Scored Customers")
#         st.dataframe(df.head())

#         csv = df.to_csv(index=False).encode("utf-8")
#         st.download_button(
#             "Download Scored File",
#             csv,
#             "churn_scored_customers.csv",
#             "text/csv"
#         )


# import streamlit as st
# import joblib
# import pandas as pd

# # Load trained PIPELINE (includes preprocessing)
# model = joblib.load("artifacts/model.pkl")

# # Extract expected input columns from pipeline
# preprocess = model.named_steps["prep"]
# CATEGORICAL_COLS = preprocess.transformers_[0][2]
# NUMERIC_COLS = preprocess.transformers_[1][2]

# st.title("Customer Churn Predictor")
# st.subheader("Customer Details")

# input_data = {}

# # Categorical inputs
# for col in CATEGORICAL_COLS:
#     input_data[col] = st.text_input(col)

# # Numeric inputs
# for col in NUMERIC_COLS:
#     input_data[col] = st.number_input(col, value=0.0)

# if st.button("Predict"):
#     df = pd.DataFrame([input_data])

#     # ⚠️ NO manual preprocessing here
#     prob = model.predict_proba(df)[0, 1]

#     st.success(f"Churn Probability: {prob:.2f}")

