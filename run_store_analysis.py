"""
Superstore Sales Analysis and Profit Prediction
Standalone execution script adapted from 'Store analysis and profit prediction.ipynb'
Includes both:
1. The original notebook's model pipeline (for reproducibility).
2. The corrected Data Analyst best practice (time-series monthly sales forecasting).
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
import sklearn.metrics as metrics

# Set styling
sns.set_theme(style='whitegrid')

def main():
    data_file = 'Superstore.csv'
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"File '{data_file}' not found in current directory.")

    print(f"Loading dataset from {data_file}...")
    data = pd.read_csv(data_file, encoding='latin-1')
    print(f"Dataset loaded: {data.shape[0]} rows, {data.shape[1]} columns")

    # Date Engineering
    data["Order Date"] = pd.to_datetime(data["Order Date"])
    data["Ship Date"] = pd.to_datetime(data["Ship Date"])
    data["order_month"] = data["Order Date"].dt.month_name()
    data["order_year"] = data["Order Date"].dt.year
    data["Ship_month"] = data["Ship Date"].dt.month_name()
    data["Ship_year"] = data["Ship Date"].dt.year

    # 1. Monthly Sales Analysis & Visualization
    month_order = ["January", "February", "March", "April", "May", "June", 
                   "July", "August", "September", "October", "November", "December"]
    monthly_sales = data.groupby("order_month")["Sales"].sum().reindex(month_order).reset_index()

    plt.figure(figsize=(10, 5))
    sns.barplot(data=monthly_sales, x="order_month", y="Sales", color='#3498db')
    plt.title("Total Superstore Sales by Month")
    plt.xlabel("Month")
    plt.ylabel("Total Sales ($)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("monthly_sales_trend.png", dpi=300)
    plt.close()
    print("Saved 'monthly_sales_trend.png'")

    # 2. Category Sales & Profit Analysis
    cat_summary = data.groupby("Category")[["Sales", "Profit"]].sum().reset_index()
    print("\n--- Sales & Profit by Category ---")
    print(cat_summary)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    x = np.arange(len(cat_summary['Category']))
    width = 0.35
    ax1.bar(x - width/2, cat_summary['Sales'], width, label='Sales', color='#2980b9')
    ax1.bar(x + width/2, cat_summary['Profit'], width, label='Profit', color='#27ae60')
    ax1.set_xlabel('Category')
    ax1.set_ylabel('Amount ($)')
    ax1.set_title('Superstore Sales vs Profit by Product Category')
    ax1.set_xticks(x)
    ax1.set_xticklabels(cat_summary['Category'])
    ax1.legend()
    plt.tight_layout()
    plt.savefig("category_sales_profit.png", dpi=300)
    plt.close()
    print("Saved 'category_sales_profit.png'")

    # 3. Monthly Sales Forecasting using Linear Regression (Best Practice for Analysts)
    print("\n--- Model 1: Monthly Sales Trend Forecasting (Linear Regression) ---")
    data['YearMonth'] = data['Order Date'].dt.to_period('M')
    ts_monthly = data.groupby('YearMonth')['Sales'].sum().reset_index()
    ts_monthly['MonthIndex'] = np.arange(len(ts_monthly)) + 1

    X_ts = ts_monthly[['MonthIndex']]
    y_ts = ts_monthly['Sales']

    ts_model = LinearRegression()
    ts_model.fit(X_ts, y_ts)
    y_ts_pred = ts_model.predict(X_ts)

    # Forecast next 6 months
    future_indices = np.arange(len(ts_monthly) + 1, len(ts_monthly) + 7).reshape(-1, 1)
    future_sales = ts_model.predict(future_indices)

    print("Historical Monthly Sales Count:", len(ts_monthly))
    print("Upcoming 6-Month Sales Forecast:")
    last_period = ts_monthly['YearMonth'].iloc[-1]
    for i, pred in enumerate(future_sales, 1):
        future_period = last_period + i
        print(f"  {future_period}: ${pred:,.2f}")

    # Plot Forecast
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(ts_monthly) + 1), y_ts, marker='o', label='Actual Monthly Sales', color='#2c3e50')
    plt.plot(range(1, len(ts_monthly) + 1), y_ts_pred, label='Fitted Trend Line', color='#e74c3c', linestyle='--')
    plt.plot(range(len(ts_monthly) + 1, len(ts_monthly) + 7), future_sales, marker='s', label='6-Month Forecast', color='#27ae60', linestyle=':')
    plt.title("Superstore Monthly Sales Trend & Linear Regression Forecast")
    plt.xlabel("Sequential Month Number")
    plt.ylabel("Sales ($)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("sales_forecast_trend.png", dpi=300)
    plt.close()
    print("Saved 'sales_forecast_trend.png'")

    # 4. Author's Original Notebook Pipeline (Multi-feature Regression with Label Encoding)
    print("\n--- Model 2: Original Notebook Multi-Model Comparison (Replication) ---")
    Num1 = data[['Row ID', 'Postal Code', 'Sales', 'Quantity', 'Discount', 'Profit', 'Ship_year']].copy()
    Cat1 = data[['Ship Mode', 'Segment', 'Country', 'City', 'State', 'Region', 'Category', 'Sub-Category']].copy()
    
    Mod1 = Num1.copy()
    for col in Cat1.columns:
        Mod1[col] = Cat1[col]

    # Replicate author's encoding
    encoder = LabelEncoder()
    for col in Mod1.columns:
        Mod1[col] = encoder.fit_transform(Mod1[col])

    y_orig = Mod1['Profit']
    X_orig = Mod1.drop(columns=['Profit', 'Country'])

    X_train, X_test, y_train, y_test = train_test_split(X_orig, y_orig, test_size=0.3, random_state=2)

    lr_orig = LinearRegression().fit(X_train, y_train)
    r2_lr = metrics.r2_score(y_test, lr_orig.predict(X_test))

    lasso_orig = Lasso().fit(X_train, y_train)
    r2_lasso = lasso_orig.score(X_test, y_test)

    ridge_orig = Ridge().fit(X_train, y_train)
    r2_ridge = ridge_orig.score(X_test, y_test)

    rf_orig = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
    r2_rf = rf_orig.score(X_test, y_test)

    knn_orig = KNeighborsRegressor().fit(X_train, y_train)
    r2_knn = knn_orig.score(X_test, y_test)

    print(f"Replicated Notebook Scores on Test Split (30%):")
    print(f"  Linear Regression R² : {r2_lr:.4f} (55.96%)")
    print(f"  Lasso Regression Score: {r2_lasso:.4f} (55.96%)")
    print(f"  Ridge Regression Score: {r2_ridge:.4f} (55.96%)")
    print(f"  Random Forest Score   : {r2_rf:.4f} (81.89%)")
    print(f"  KNN Regressor Score   : {r2_knn:.4f} (16.86%)")

    print("\nProject 1 execution finished successfully!")

if __name__ == '__main__':
    main()
