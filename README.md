# 🚗 Used Car Price Prediction using Machine Learning

Predict the market value of used cars using Machine Learning and an interactive Streamlit web application.

## 🌐 Live Demo

**Deployed Application**

https://used-car-price-prediction-ml-v1.streamlit.app/

---

## Overview

This project predicts used car prices based on:

* Brand
* Model
* Age
* Kilometers Driven
* Fuel Type
* Transmission
* Ownership History

The project includes:

* Exploratory Data Analysis (EDA)
* Data Cleaning and Preprocessing
* Feature Engineering
* Linear Regression
* Ridge Regression
* Lasso Regression
* Model Comparison
* Streamlit Deployment

---

## Dataset

The dataset contains **9,582 car listings** with 11 columns.

Features:

* Brand
* model
* Year
* Age
* kmDriven
* Transmission
* Owner
* FuelType
* PostedDate
* AdditionInfo

Target:

* AskPrice

---

## Machine Learning Workflow

```text
Load Dataset
↓
Exploratory Data Analysis
↓
Data Cleaning
↓
Feature Engineering
↓
Encoding
↓
Train-Test Split
↓
Linear Regression
↓
Ridge Regression
↓
Lasso Regression
↓
Model Evaluation
↓
Model Comparison
↓
Save Model
↓
Build Streamlit App
↓
Deploy
```

---

## Data Preprocessing

### Missing Values

Missing values in `kmDriven` were filled using the median.

### Removed Columns

* PostedDate
* AdditionInfo
* Year

### Numerical Cleaning

Examples:

```python
98,000 km → 98000

₹ 5,25,000 → 525000
```

### One-Hot Encoding

Applied on:

* Brand
* model
* FuelType
* Transmission
* Owner

### Feature Scaling

StandardScaler was used on:

* Age
* kmDriven

---

## Models Trained

### Linear Regression

Baseline model.

### Ridge Regression

Linear Regression with L2 Regularization.

### Lasso Regression

Linear Regression with L1 Regularization.

---

## Performance

| Model             | R² Score |
| ----------------- | -------: |
| Linear Regression |    0.792 |
| Lasso Regression  |    0.780 |
| Ridge Regression  |    0.755 |

---

## Visualizations

The project generates:

* Histograms Distribution
* Boxplots for Outlier Detection
* Correlation Heatmap
* Actual vs Predicted Plot
* Model Comparison Plot

---

## Tech Stack

### Python

* Pandas
* NumPy
* Scikit-Learn
* Joblib
* Streamlit
* Altair
* Matplotlib

---

## Project Structure

```text
used-car-price-prediction-ml
│
├── data
│   └── used_car_dataset.csv
│
├── models
│   ├── columns.joblib
│   ├── linear_model.joblib
│   ├── ridge_model.joblib
│   ├── lasso_model.joblib
│   └── scaler.joblib
│
├── plot
│   ├── actual_vs_predicted.png
│   ├── boxplots_outliers.png
│   ├── correlation_heatmap.png
│   ├── histograms_distribution.png
│   └── model_comparison.png
│
├── src
│   ├── app.py
│   ├── eda.py
│   └── train.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/AbhitanshShahi/used-car-price-prediction-ml.git
```

Move into the project:

```bash
cd used-car-price-prediction-ml
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run src/app.py
```

---

## Future Improvements

* Hyperparameter Tuning
* XGBoost Regressor
* Random Forest Regressor
* CatBoost
* Interactive Plotly Visualizations
* Improved UI/UX
* Better Feature Engineering

---

## Author

### Abhitansh Shahi

GitHub:

https://github.com/AbhitanshShahi

---

## Live Demo

https://used-car-price-prediction-ml-v1.streamlit.app/
