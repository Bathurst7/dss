import streamlit as st
import pandas as pd
import joblib

# Load model & scaler paths
model_path = '../model_building/model_folder/gbr.joblib'
scaler_path = '../model_building/scaler_folder/minmax_scaler.joblib'
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# order of columns when fit into model
expected_order = [
    "Company name",
    "Time",
    "Engineered_PRASM",
    "Engineered_RASM",
    "Engineered_CASM",
    "Engineered_Load_factor",
    "Engineered_Gross_profit_margin",
    "Engineered_Quick_ratio",
    "Engineered_D/E",
    "Engineered_ROA",
    "Engineered_EPS",
    "TSR_lag1",
    "TSR_lag2",
    "Engineered_PRASM_lag1",
    "Engineered_PRASM_lag2",
    "Engineered_RASM_lag1",
    "Engineered_RASM_lag2",
    "Engineered_CASM_lag1",
    "Engineered_CASM_lag2",
    "Engineered_Load_factor_lag1",
    "Engineered_Load_factor_lag2",
    "Engineered_Gross_profit_margin_lag1",
    "Engineered_Gross_profit_margin_lag2",
    "Engineered_Quick_ratio_lag1",
    "Engineered_Quick_ratio_lag2",
    "Engineered_D/E_lag1",
    "Engineered_D/E_lag2",
    "Engineered_ROA_lag1",
    "Engineered_ROA_lag2",
    "Engineered_EPS_lag1",
    "Engineered_EPS_lag2",
    "TSR_rolling_mean"
]
def scale_features(df, scaler, exclude_columns=None):
    """
    Scale numerical features with MinMaxScaler
    """
    if exclude_columns is None:
        exclude_columns = ["Company name", "Time"]

    scaled_features = [col for col in df.columns if col not in exclude_columns]
    df_scaled = scaler.transform(df[scaled_features])
    scaled_df = pd.DataFrame(df_scaled, columns=scaled_features, index=df.index)
    final_df = df[exclude_columns].join(scaled_df)
    return final_df


def main():
    st.title('Company TSR Prediction')
    st.markdown("""
    ### Provide the following details to predict TSR:
    """)

    # Mapping company name with corresponding label encoded
    mapping = {
        'US_Alaska': 0,
        'US_Allegiant': 1,
        'US_American Airlines': 2,
        'US_Delta Airlines': 3,
        'US_Hawaiian': 4,
        'US_JetBlue Airways': 5,
        'US_Southwest Airlines': 6,
        'US_Spirit Airlines': 7,
        'US_United Airlines': 8
    }

    # Map company with encoded value
    company_name = st.selectbox("Select Company Name:", list(mapping.keys()))
    encoded_company_name = mapping[company_name]

    # Input: Year and Quarter
    year = st.number_input("Select Year:", value=2023)
    quarter = st.selectbox("Select Quarter:", [1, 2, 3, 4])

    # Convert year & quarter to time
    time = year + (quarter / 4)

    # Input: Current Quarter Data
    st.subheader("Enter Current Quarter Data:")
    current_data = {
        "Engineered_PRASM": st.number_input("PRASM (Current):", min_value=0.0, max_value=100.0, value=0.0),
        "Engineered_RASM": st.number_input("RASM (Current):", min_value=0.0, max_value=100.0, value=0.0),
        "Engineered_CASM": st.number_input("CASM (Current):", min_value=0.0, max_value=100.0, value=0.0),
        "Engineered_Load_factor": st.number_input("Load Factor (Current):", min_value=0.0, max_value=100.0, value=0.0),
        "Engineered_Gross_profit_margin": st.number_input("Gross Profit Margin (Current):", min_value=0.0, max_value=100.0, value=0.0),
        "Engineered_Quick_ratio": st.number_input("Quick Ratio (Current):", min_value=0.0, max_value=10.0, value=0.0),
        "Engineered_D/E": st.number_input("D/E (Current):", min_value=0.0, max_value=10.0, value=0.0),
        "Engineered_ROA": st.number_input("ROA (Current):", min_value=0.0, max_value=10.0, value=0.0),
        "Engineered_EPS": st.number_input("EPS (Current):", min_value=0.0, max_value=10.0, value=0.0),
    }

    # Input: TSR Lagged Features
    st.subheader("Enter TSR Lagged Data:")
    tsr_lag1 = st.number_input("TSR Lag 1 (Last Quarter):", min_value=0.0, max_value=1.0, value=0.0)
    tsr_lag2 = st.number_input("TSR Lag 2 (Two Quarters Ago):", min_value=0.0, max_value=1.0, value=0.0)

    # Input: Last Quarter Data
    st.subheader("Enter Last Quarter Data:")
    lag1_data = {
        f"{col}_lag1": st.number_input(f"{col} Lag 1:", min_value=0.0, max_value=100.0, value=0.0)
        for col in current_data.keys()
    }

    # Input: Two Quarters Ago Data
    st.subheader("Enter Data for Two Quarters Ago:")
    lag2_data = {
        f"{col}_lag2": st.number_input(f"{col} Lag 2:", min_value=0.0, max_value=100.0, value=0.0)
        for col in current_data.keys()
    }

    # Input: TSR Rolling Mean
    tsr_rolling_mean = st.number_input("TSR Rolling Mean (Past 4 Quarters):", min_value=0.0, max_value=1.0, value=0.0)

    # Prepare Data for Prediction
    if st.button("Predict TSR"):
        input_data = {
            "Company name": encoded_company_name,
            "Time": time,
            **current_data,
            "TSR_lag1": tsr_lag1,
            "TSR_lag2": tsr_lag2,
            **lag1_data,
            **lag2_data,
            "TSR_rolling_mean": tsr_rolling_mean,
        }
        # Reorder data based on expected key order
        ordered_data = {key:input_data[key] for key in expected_order}
        input_df = pd.DataFrame([ordered_data])

        # Scale data
        scaled_df = scale_features(input_df, scaler)

        # Make prediction

        prediction = model.predict(scaled_df)
        st.success(f"The predicted TSR for {company_name} in Q{quarter}, {year} is: {prediction[0]:.4f}")


if __name__ == "__main__":
    main()
