import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Stock Volatility ML Analyzer", layout="wide")

st.title("📊 Stock Market Volatility Analysis & Prediction")

uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if uploaded_file is not None:
    
    df = pd.read_csv(uploaded_file, encoding='latin1')


    st.subheader("Raw Dataset")
    st.write(df.head())

    # Preprocessing 
    
    df.drop_duplicates(inplace=True)
    
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.sort_values('Date')
    
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    
    # Feature Engineering 
    
    if 'Close' in df.columns:
        
        df['Return'] = df['Close'].pct_change()
        df['Volatility'] = df['Return'].rolling(20).std()
        df['MA20'] = df['Close'].rolling(20).mean()
        df['Lag1'] = df['Return'].shift(1)
        df['Lag2'] = df['Return'].shift(2)
        
        df.dropna(inplace=True)

        st.subheader("Processed Dataset")
        st.write(df.head())

        #  Descriptive Stats 
        
        st.subheader("📊 Descriptive Statistics")
        st.write(df['Return'].describe())
        st.write("Skewness:", df['Return'].skew())
        st.write("Kurtosis:", df['Return'].kurtosis())

        # Inferential Stats 
        
        st.subheader("📉 Inferential Statistics")
        t_stat, p_value = stats.ttest_1samp(df['Return'], 0)
        st.write("T-Test p-value:", p_value)

        # ML Section 
        
        st.subheader("🤖 Machine Learning Models")

        features = ['Lag1', 'Lag2', 'MA20']
        X = df[features]
        y = df['Volatility']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

        # Linear Regression
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        lr_pred = lr.predict(X_test)

        # Random Forest
        rf = RandomForestRegressor()
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)

        # Evaluation
        
        lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
        rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))

        lr_r2 = r2_score(y_test, lr_pred)
        rf_r2 = r2_score(y_test, rf_pred)

        st.write("Linear Regression RMSE:", lr_rmse)
        st.write("Linear Regression R2:", lr_r2)

        st.write("Random Forest RMSE:", rf_rmse)
        st.write("Random Forest R2:", rf_r2)

        # Plot Predictions 
        
        st.subheader("📈 Model Predictions vs Actual Volatility")
        
        fig, ax = plt.subplots()
        ax.plot(y_test.values, label="Actual")
        ax.plot(lr_pred, label="Linear Regression")
        ax.plot(rf_pred, label="Random Forest")
        ax.legend()
        st.pyplot(fig)
        
        st.subheader("📈 Closing Price Over Time")

        fig1, ax1 = plt.subplots()
        ax1.plot(df['Date'], df['Close'])
        ax1.set_title("Closing Price")
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Price")
        st.pyplot(fig1)

        st.subheader("📊 Daily Returns")

        fig2, ax2 = plt.subplots()
        ax2.plot(df['Date'], df['Return'])
        ax2.set_title("Daily Returns")
        st.pyplot(fig2)
        
        st.subheader("📉 Rolling Volatility (20-Day)")

        fig3, ax3 = plt.subplots()
        ax3.plot(df['Date'], df['Volatility'])
        ax3.set_title("20-Day Rolling Volatility")
        st.pyplot(fig3)
        
        st.subheader("📊 Return Distribution")

        fig4, ax4 = plt.subplots()
        sns.histplot(df['Return'], kde=True, ax=ax4)
        st.pyplot(fig4)
        
        st.subheader("📦 Boxplot of Returns")

        fig5, ax5 = plt.subplots()
        sns.boxplot(y=df['Return'], ax=ax5)
        st.pyplot(fig5)





    else:
        st.warning("Dataset must contain 'Close' column.")
