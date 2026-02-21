# 📊 Stock Market Volatility Analysis & Prediction

## 📌 Project Overview

This project focuses on analyzing and predicting stock market volatility using historical stock price data. The dataset consists of daily trading information including Open, High, Low, Close prices, and Volume.

The application performs statistical analysis, correlation studies, time series visualization, and machine learning-based volatility prediction using Streamlit.

---

## 🎯 Objectives

- Analyze relationships among stock price variables.
- Generate correlation heatmaps for numerical variables.
- Create time series plots to identify trends and fluctuations.
- Compute rolling volatility and moving averages.
- Build and compare machine learning models for volatility prediction.

---

## 🗂 Dataset Description

The dataset contains the following variables:

- **Date** – Trading date  
- **Open** – Opening price  
- **High** – Highest price of the day  
- **Low** – Lowest price of the day  
- **Close** – Closing price  
- **Volume** – Number of shares traded  

---

## ⚙️ Methodology

### 1️⃣ Data Preprocessing
- Removal of duplicate values  
- Handling missing values  
- Conversion of Date column to datetime format  
- Sorting dataset chronologically  

### 2️⃣ Feature Engineering
- Daily Returns calculation  
- 20-Day Rolling Volatility  
- 20-Day Moving Average (MA20)  
- Lag variables (Lag1, Lag2)  

### 3️⃣ Statistical Analysis
- Descriptive statistics  
- Skewness and kurtosis  
- One-sample t-test on returns  

### 4️⃣ Machine Learning Models
- Linear Regression  
- Random Forest Regressor  

### 5️⃣ Evaluation Metrics
- Root Mean Squared Error (RMSE)  
- R² Score  

---

## 📈 Visualizations

- Closing Price over Time  
- Daily Returns Plot  
- Rolling Volatility Plot  
- Correlation Heatmap  
- Return Distribution Histogram  
- Boxplot of Returns  
- Model Predictions vs Actual Volatility  

---

## 🛠 Technologies Used

- Python  
- Streamlit  
- Pandas  
- NumPy  
- Matplotlib  
- Seaborn  
- SciPy  
- Scikit-learn  

---

## 🚀 How to Run the Project

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/repository-name.git
cd repository-name
```

### Step 2: Create Virtual Environment (Recommended)

```bash
python -m venv venv
venv\Scripts\activate   # For Windows
```

### Step 3: Install Required Libraries

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not available:

```bash
pip install streamlit pandas numpy matplotlib seaborn scipy scikit-learn
```

### Step 4: Run the Application

```bash
streamlit run app.py
```

---

## 🔍 Key Findings

- Strong positive correlation among Open, High, Low, and Close prices.
- Moderate negative correlation between Volume and price variables.
- Clear long-term upward trend in stock prices.
- No strong seasonal pattern observed in daily data.
- Presence of short-term irregular fluctuations.
- Random Forest performs better than Linear Regression in predicting volatility.

---

## 🔮 Future Enhancements

- Implementation of ARIMA or LSTM models.
- Real-time stock data integration.
- Deployment using Streamlit Cloud.
- Hyperparameter tuning for improved prediction accuracy.

---

## 👨‍💻 Author

Jerolin Mathew
B.Tech – Computer Science (Data Science)  
Christ University
