# === AI-Driven Predictive Analytics Platform (Streamlit Web App) ===
# === AI-Driven Predictive Analytics Platform with Admin/User Login ===

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import streamlit as st
import io
import os
from datetime import datetime

# 1. File Upload Interface
st.title("📊 AI-Driven Predictive Analytics for Small Businesses")
st.write("Upload your historical monthly business data (CSV format)")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

# 2. Load Data

=======
# Dummy credentials
USER_CREDENTIALS = {"user": "user123", "admin": "admin123"}

# Session state to persist login
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.role = None
>>>>>>> Stashed changes

# Login function
def login():
    st.title("🔐 Login to Predictive Analytics Platform")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
            st.session_state.logged_in = True
            st.session_state.role = "admin" if username == "admin" else "user"
            st.success(f"Welcome, {st.session_state.role.title()}!")
            st.rerun()  # <- Updated from st.experimental_rerun()
        else:
            st.error("Invalid username or password.")

if not st.session_state.logged_in:
    login()
    st.stop()

# Common preprocessing function
def preprocess_data(df):
    required_columns = ['month', 'ads_spent', 'customer_visits', 'sales']
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"The following required columns are missing: {missing_columns}")

    df['month'] = pd.to_datetime(df['month'], errors='coerce')
    duplicates_removed = df.duplicated(subset=['month']).sum()
    df = df.drop_duplicates(subset=['month'], keep='last')
    df = df.sort_values('month')
    df['month_number'] = range(1, len(df) + 1)
    return df, duplicates_removed

# 3. Train or Load Model

# Model training
def train_model(data):
    X = data[['month_number', 'ads_spent', 'customer_visits']]
    y = data['sales']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = GradientBoostingRegressor()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    joblib.dump(model, 'monthly_sales_forecast_model.pkl')
    return model, mse

# 4. Forecasting Function (Enhanced)

# Forecast function
def forecast(model, last_month_num, last_month_date, forecast_months, ads_spent, customer_visits):
    next_months = pd.DataFrame({
        'month_number': range(last_month_num + 1, last_month_num + forecast_months + 1),
        'ads_spent': [ads_spent] * forecast_months,
        'customer_visits': [customer_visits] * forecast_months
    })
    predictions = model.predict(next_months)
    next_months['forecasted_sales'] = predictions
    next_months['month'] = pd.date_range(
        start=last_month_date + pd.DateOffset(months=1),
        periods=forecast_months,
        freq='MS'
    )
    return next_months

# 5. Visualization

# Visualization
def plot_forecast(original, forecast):
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(original['month'], original['sales'], label='Historical Sales')
    ax.plot(forecast['month'], forecast['forecasted_sales'], label='Forecasted Sales', linestyle='--')
    ax.set_xlabel('Month')
    ax.set_ylabel('Sales')
    ax.set_title('Sales Forecast (Monthly)')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

# 6. Main Execution

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df, duplicates_removed = preprocess_data(df)

    st.subheader("🧹 Cleaned & Processed Data Preview")
    st.write(df.head())

    if duplicates_removed > 0:
        st.info(f"ℹ️ {duplicates_removed} duplicate entries (based on month) were removed from your uploaded data.")

# Save logs (for admin view)
def save_to_log(dataframe):
    os.makedirs("logs", exist_ok=True)

    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"logs/forecast_log_{now}.csv"
    dataframe.to_csv(filename, index=False)
# ===================== USER VIEW =====================
if st.session_state.role == "user":
    st.title("📊 AI-Driven Predictive Analytics for Small Businesses")
    uploaded_file = st.file_uploader("Upload your historical monthly business data (CSV format)", type="csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        try:
            df, duplicates_removed = preprocess_data(df)
        except Exception as e:
            st.error(str(e))
            st.stop()

        st.subheader("🧹 Cleaned & Processed Data Preview")
        st.write(df.head())
        if duplicates_removed > 0:
            st.info(f"ℹ️ {duplicates_removed} duplicate entries (based on month) were removed.")

<<<<<<< Updated upstream
    # 📅 Show All Monthly Data from CSV
    st.subheader("📅 Historical Monthly Data Overview")
    st.dataframe(df[['month', 'ads_spent', 'customer_visits', 'sales']])

    # Optional line chart of historical sales
    st.line_chart(df.set_index('month')['sales'])

    # User Inputs for Forecasting
    st.subheader("🛠 Forecast Settings")
    forecast_months = st.slider("How many months do you want to forecast?", min_value=1, max_value=24, value=6)
    future_ads = st.number_input("Estimated Ads Spent for Future Months", min_value=0, value=4000)
    future_visits = st.number_input("Estimated Customer Visits for Future Months", min_value=0, value=600)

    model, mse = train_model(df)
    st.success(f"✅ Model trained successfully! Mean Squared Error: {mse:.2f}")

    forecast_df = forecast(
        model,
        df['month_number'].iloc[-1],
        df['month'].iloc[-1],
        forecast_months,
        future_ads,
        future_visits
    )

    st.subheader(f"📈 Forecasted Sales for Next {forecast_months} Month(s)")
    st.write(forecast_df[['month', 'forecasted_sales']])

    st.download_button(
        label="📥 Download Forecast Data as CSV",
        data=forecast_df.to_csv(index=False).encode('utf-8'),
        file_name='forecasted_sales.csv',
        mime='text/csv'
    )

    st.subheader("📊 Visualization")
    plot_forecast(df, forecast_df)
else:
    st.warning("Please upload a CSV file with 'month', 'sales', 'ads_spent', and 'customer_visits' columns.")
=======
        st.subheader("📅 Historical Monthly Data Overview")
        st.dataframe(df[['month', 'ads_spent', 'customer_visits', 'sales']])
        st.line_chart(df.set_index('month')['sales'])
>>>>>>> Stashed changes

        st.subheader("🛠 Forecast Settings")
        forecast_months = st.slider("How many months to forecast?", 1, 24, 6)
        future_ads = st.number_input("Estimated Ads Spent", min_value=0, value=4000)
        future_visits = st.number_input("Estimated Customer Visits", min_value=0, value=600)

        model, mse = train_model(df)
        st.success(f"✅ Model trained. MSE: {mse:.2f}")

        forecast_df = forecast(model, df['month_number'].iloc[-1], df['month'].iloc[-1], forecast_months, future_ads, future_visits)

        st.subheader(f"📈 Forecasted Sales for Next {forecast_months} Month(s)")
        st.write(forecast_df[['month', 'forecasted_sales']])

        st.download_button("📥 Download Forecast Data (CSV)", forecast_df.to_csv(index=False).encode('utf-8'), file_name='forecasted_sales.csv', mime='text/csv')

        st.subheader("📊 Visualization")
        plot_forecast(df, forecast_df)

        save_to_log(forecast_df)

# ===================== ADMIN VIEW =====================
elif st.session_state.role == "admin":
    st.title("🛠 Admin Dashboard")
    st.markdown("Manage system activity and logs.")

    import os
    import glob

    log_files = glob.glob("logs/forecast_log_*.csv")
    if not log_files:
        st.warning("No forecast logs found.")
    else:
        selected_log = st.selectbox("Select a forecast log file to view:", log_files)
        if selected_log:
            df_log = pd.read_csv(selected_log)
            st.subheader("📄 Forecast Log Preview")
            st.write(df_log.head())
            st.download_button("📥 Download this Log", df_log.to_csv(index=False).encode('utf-8'), file_name=selected_log.split("/")[-1])

# Logout button
if st.button("🔒 Logout"):
    st.session_state.logged_in = False
    st.session_state.role = None
    st.rerun()  # <- Updated from st.experimental_rerun()
