# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import pickle
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
plt.style.use('fivethirtyeight')

# ==================== READ THE DATASET ====================
# Read the dataset from CSV file
df = pd.read_csv('dataset.csv')

print("Dataset loaded successfully!")
print(f"Dataset shape: {df.shape}\n")

# ==================== EXPLORE INPUT FEATURES ====================
# Display all columns in the dataset
print("Input Features (Columns):")
print(df.columns)
print("\n")

# ==================== REMOVE UNNECESSARY COLUMNS ====================
# Drop superfluous columns (modify as per your dataset)
# Example: df = df.drop(['unnecessary_column1', 'unnecessary_column2'], axis=1)
# df = df.drop(['id', 'timestamp'], axis=1)

# ==================== DISPLAY FIRST FIVE ROWS ====================
# Display first 5 rows of the dataset
print("First 5 rows of the dataset:")
print(df.head())
print("\n")

# ==================== DISPLAY LAST FIVE ROWS ====================
# Display last 5 rows of the dataset
print("Last 5 rows of the dataset:")
print(df.tail())
print("\n")

# ==================== SET STYLE TO GGPLOT ====================
# Change visualization style to ggplot
plt.style.use('ggplot')

# ==================== CORRELATION ANALYSIS ====================
# Calculate correlation between all features
correlation = df.corr()
print("Correlation Matrix:")
print(correlation)
print("\n")

# ==================== HEATMAP VISUALIZATION ====================
# Create a heatmap to visualize correlations
plt.figure(figsize=(12, 10))
sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0, fmt='.2f', cbar_kws={'label': 'Correlation'})
plt.title('Correlation Heatmap - Feature Relationships', fontsize=16, fontweight='bold')
plt.xlabel('Features', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.tight_layout()
plt.show()

print("All libraries imported successfully!")
