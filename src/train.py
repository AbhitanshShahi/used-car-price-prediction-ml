import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt


df = pd.read_csv('data/used_car_dataset.csv')

# Clean numerical strings to proper datatypes
df['kmDriven'] = df['kmDriven'].astype(str).str.replace('km', '', case=False).str.replace(',', '').str.strip()
df['kmDriven'] = pd.to_numeric(df['kmDriven'], errors='coerce')

df['AskPrice'] = df['AskPrice'].astype(str).str.replace('₹', '', regex=False).str.replace(',', '').str.strip()
df['AskPrice'] = pd.to_numeric(df['AskPrice'], errors='coerce')

# handle missing values
df['kmDriven'] = df['kmDriven'].fillna(df['kmDriven'].median())

# drop unnecessary columns
df = df.drop(columns=['PostedDate', 'Year', 'AdditionInfo'])

X = df.drop(columns=['AskPrice'])
y = df['AskPrice']

X = pd.get_dummies(
    X,
    columns=['Brand', 'model', 'Transmission', 'Owner', 'FuelType'],
    drop_first=True
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = X_train.copy()
X_test = X_test.copy()

numerical_cols = ['Age', 'kmDriven']
scaler = StandardScaler()

X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])


linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)
linear_mae = mean_absolute_error(y_test, y_pred_linear)
linear_mse = mean_squared_error(y_test, y_pred_linear)
linear_rmse = root_mean_squared_error(y_test, y_pred_linear)
linear_r2 = r2_score(y_test, y_pred_linear)


ridge_model = Ridge()
ridge_model.fit(X_train, y_train)
y_pred_ridge = ridge_model.predict(X_test)
ridge_mae = mean_absolute_error(y_test, y_pred_ridge)
ridge_mse = mean_squared_error(y_test, y_pred_ridge)
ridge_rmse = root_mean_squared_error(y_test, y_pred_ridge)
ridge_r2 = r2_score(y_test, y_pred_ridge)

lasso_model = Lasso(max_iter=10000, random_state=42)
lasso_model.fit(X_train, y_train)
y_pred_lasso = lasso_model.predict(X_test)
lasso_mae = mean_absolute_error(y_test, y_pred_lasso)
lasso_mse = mean_squared_error(y_test, y_pred_lasso)
lasso_rmse = root_mean_squared_error(y_test, y_pred_lasso)
lasso_r2 = r2_score(y_test, y_pred_lasso)

results = pd.DataFrame({
    'Model': ['Linear Regression', 'Ridge Regression', 'Lasso Regression'],
    'MAE': [linear_mae, ridge_mae, lasso_mae],
    'MSE': [linear_mse, ridge_mse, lasso_mse],
    'RMSE': [linear_rmse, ridge_rmse, lasso_rmse],
    'R2 Score': [linear_r2, ridge_r2, lasso_r2]
})

plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_pred_linear)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r--'
)

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Price")

plt.savefig("plot/actual_vs_predicted.png")
plt.show()

models = [
    "Linear Regression",
    "Ridge Regression",
    "Lasso Regression"
]

r2_scores = [
    linear_r2,
    ridge_r2,
    lasso_r2
]

plt.figure(figsize=(8, 6))

plt.bar(
    models,
    r2_scores
)

plt.ylabel("R² Score")
plt.title("Model Comparison")

plt.savefig("plot/model_comparison.png")
plt.show()

joblib.dump(lasso_model, 'models/lasso_model.joblib')
joblib.dump(X.columns.tolist(), 'models/columns.joblib')
joblib.dump(ridge_model, 'models/ridge_model.joblib')
joblib.dump(linear_model, 'models/linear_model.joblib')
joblib.dump(scaler, 'models/scaler.joblib')


print(sorted(df["Brand"].unique()))
print(sorted(df["FuelType"].unique()))
print(sorted(df["Transmission"].unique()))
print(sorted(df["Owner"].unique()))
print(sorted(df["model"].unique()))