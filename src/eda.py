import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('data/used_car_dataset.csv')

sns.set_theme(style="whitegrid")

# Create Histograms for distributions
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.histplot(df['Age'], bins=20, ax=axes[0], kde=True, color='skyblue')
axes[0].set_title('Distribution of Age')
axes[0].set_xlabel('Age (Years)')

# Temporarily clean values just for plotting distribution
temp_km = pd.to_numeric(df['kmDriven'].astype(str).str.replace('km', '', case=False).str.replace(',', '').str.strip(), errors='coerce')
temp_price = pd.to_numeric(df['AskPrice'].astype(str).str.replace('₹', '', regex=False).str.replace(',', '').str.strip(), errors='coerce')

sns.histplot(temp_km.dropna(), bins=30, ax=axes[1], kde=True, color='salmon')
axes[1].set_title('Distribution of kmDriven')
axes[1].set_xlabel('km Driven')

sns.histplot(temp_price, bins=30, ax=axes[2], kde=True, color='green')
axes[2].set_title('Distribution of AskPrice (Log Scale)')
axes[2].set_xlabel('Ask Price (₹)')
axes[2].set_xscale('log')
plt.tight_layout()
plt.savefig('plot/histograms_distribution.png')
plt.close()

# Create Boxplots for outlier detection
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.boxplot(y=df['Age'], ax=axes[0], color='skyblue')
axes[0].set_title('Boxplot of Age')

sns.boxplot(y=temp_km, ax=axes[1], color='salmon')
axes[1].set_title('Boxplot of kmDriven')

sns.boxplot(y=temp_price, ax=axes[2], color='green')
axes[2].set_title('Boxplot of AskPrice')
plt.tight_layout()
plt.savefig('plot/boxplots_outliers.png')
plt.close()

# Correlation Heatmap
numeric_df = pd.DataFrame({
    'Age': df['Age'],
    'Year': df['Year'],
    'kmDriven': temp_km,
    'AskPrice': temp_price
})
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
ax.set_title('Correlation Heatmap')
plt.tight_layout()
plt.savefig('plot/correlation_heatmap.png')
plt.close()