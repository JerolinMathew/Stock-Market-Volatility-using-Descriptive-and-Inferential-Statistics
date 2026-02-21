📊 Stock Market Volatility Analysis & Prediction
📌 Project Overview

This project focuses on analyzing and predicting stock market volatility using historical stock price data. The dataset includes daily trading information such as Open, High, Low, Close prices, and Volume.

The project performs:

Exploratory Data Analysis (EDA)

Correlation Analysis

Time Series Visualization

Statistical Testing

Machine Learning-based Volatility Prediction

The application is built using Streamlit for interactive visualization and model evaluation.

Objectives

Analyze relationships between stock price variables.

Identify trends, seasonal patterns, and irregular fluctuations.

Compute rolling volatility using statistical techniques.

Build and compare machine learning models for volatility prediction.

Visualize insights using interactive charts.

Dataset Description

The dataset contains historical stock data with the following columns:

Date – Trading date

Open – Opening price

High – Highest price of the day

Low – Lowest price of the day

Close – Closing price

Volume – Number of shares traded

⚙️ Features Implemented
🔹 Data Preprocessing

Duplicate removal

Missing value handling

Date conversion and sorting

Forward and backward filling

🔹 Feature Engineering

Daily Returns

20-Day Rolling Volatility

20-Day Moving Average (MA20)

Lag Variables (Lag1, Lag2)

🔹 Statistical Analysis

Descriptive statistics (mean, standard deviation, skewness, kurtosis)

One-sample t-test for returns

🔹 Machine Learning Models

Linear Regression

Random Forest Regressor

🔹 Evaluation Metrics

RMSE (Root Mean Squared Error)

R² Score

🔹 Visualizations

Closing Price Over Time

Daily Returns Plot

Rolling Volatility Plot

Return Distribution Histogram

Boxplot of Returns

Model Predictions vs Actual Volatility

Technologies Used

Python

Streamlit

Pandas

NumPy

Matplotlib

Seaborn

SciPy

Scikit-learn

How to Run the Project
1️⃣ Clone the Repository
git clone https://github.com/your-username/repository-name.git
cd repository-name
2️⃣ Create Virtual Environment (Recommended)
python -m venv venv
venv\Scripts\activate   # Windows
3️⃣ Install Dependencies
pip install -r requirements.txt

If requirements.txt is not available:

pip install streamlit pandas numpy matplotlib seaborn scipy scikit-learn
4️⃣ Run the Application
streamlit run app.py
Key Insights

Strong positive correlation among Open, High, Low, and Close prices.

Moderate negative correlation between Volume and price variables.

Clear long-term upward trend in stock prices.

No strong visible seasonal pattern.

Presence of short-term irregular fluctuations due to market volatility.

Random Forest model performs better than Linear Regression in volatility prediction.

Future Improvements

Implement ARIMA or LSTM for advanced time-series forecasting.

Add real-time stock data integration.

Deploy application on cloud (Streamlit Cloud / Heroku).

Include hyperparameter tuning for model optimization.

👨‍💻 Author

Your Name
B.Tech – Computer Science (Data Science)
Christ University
