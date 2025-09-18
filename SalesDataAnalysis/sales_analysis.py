# Sales Data Analysis Project
# Author: Abhilash Reddy
# Description: Analyze sales data, identify trends, and forecast future sales.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# -----------------------------
# 1. Load Data
# -----------------------------
# Sample CSV should have columns: Date, Sales
# Example:
# Date,Sales
# 2023-01-01,200
# 2023-01-02,150
# ...
data = pd.read_csv("sales_data.csv", parse_dates=["Date"])

# Ensure sorting
data = data.sort_values("Date")

# -----------------------------
# 2. Exploratory Data Analysis
# -----------------------------
print("First 5 rows:\n", data.head())
print("\nSummary Statistics:\n", data["Sales"].describe())

# Monthly sales trend
data["Month"] = data["Date"].dt.to_period("M")
monthly_sales = data.groupby("Month")["Sales"].sum()

plt.figure(figsize=(10,5))
monthly_sales.plot(kind="line", marker="o")
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Total Sales")
plt.grid(True)
plt.show()

# Seasonality check (sales by weekday)
data["Weekday"] = data["Date"].dt.day_name()
plt.figure(figsize=(8,4))
sns.barplot(x="Weekday", y="Sales", data=data, estimator=np.mean, order=[
    "Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
plt.title("Average Sales by Weekday")
plt.xticks(rotation=45)
plt.show()

# -----------------------------
# 3. Feature Engineering
# -----------------------------
# Create simple time feature
data["DayOfYear"] = data["Date"].dt.dayofyear
X = data[["DayOfYear"]]
y = data["Sales"]

# -----------------------------
# 4. Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# -----------------------------
# 5. Model Training
# -----------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# -----------------------------
# 6. Evaluation
# -----------------------------
print("\nModel Performance:")
print("MSE:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# -----------------------------
# 7. Visualization of Forecast
# -----------------------------
plt.figure(figsize=(10,5))
plt.plot(X_test["DayOfYear"], y_test, label="Actual Sales", marker="o")
plt.plot(X_test["DayOfYear"], y_pred, label="Predicted Sales", marker="x")
plt.title("Actual vs Predicted Sales")
plt.xlabel("Day of Year")
plt.ylabel("Sales")
plt.legend()
plt.show()
