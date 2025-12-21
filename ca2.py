# ROAD ACCIDENT SEVERITY PREDICTION 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

sns.set_style("whitegrid")

df = pd.read_csv("C:/Users/abhilash/Downloads/Road.csv")
df.columns = [c.strip() for c in df.columns]

TARGET = "Accident_severity"

print("Dataset Shape:", df.shape)
print(df[TARGET].value_counts())

# EDA — Simple Distribution + Boxplots

numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

time_cols = [c for c in df.columns if "time" in c.lower()]

if time_cols:
    tcol = time_cols[0]
    print(f"Extracting Accident_Hour from column: {tcol}")

    df[tcol] = pd.to_datetime(df[tcol], format="%H:%M:%S", errors="coerce")
    df["Accident_Hour"] = df[tcol].dt.hour

    print(df["Accident_Hour"].head())
else:
    print("No time column found. Accident_Hour not created.")


if "Accident_Hour" in df.columns:

    plt.figure(figsize=(12, 4))
    sns.histplot(df["Accident_Hour"].dropna(), bins=24, kde=True, color='salmon', edgecolor=None)
    plt.title("Distribution of Accident_Hour", fontsize=14)
    plt.xlabel("Accident_Hour")
    plt.ylabel("Count")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 2.8))
    sns.boxplot(x=df["Accident_Hour"].dropna(), color='salmon')
    plt.title("Boxplot of Accident_Hour", fontsize=14)
    plt.xlabel("Accident_Hour")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
else:
    print("Accident_Hour not found in dataset.")

numeric_cols = [c for c in numeric_cols if c != TARGET]
plot_cols = numeric_cols[:6]

plot_cols = [c for c in plot_cols if c != "Number_of_vehicles_involved"]

for col in plot_cols:
    plt.figure(figsize=(10, 4))
    sns.histplot(df[col].dropna(), kde=True, color='salmon')
    plt.title(f"Distribution of {col}")
    plt.show()

    plt.figure(figsize=(10, 3))
    sns.boxplot(x=df[col].dropna(), color='skyblue')
    plt.title(f"Boxplot of {col}")
    plt.show()

# ENCODING

cat_cols = df.select_dtypes(include=['object']).columns.tolist()
cat_cols = [c for c in cat_cols if c != TARGET]

small_cat = [c for c in cat_cols if df[c].nunique() <= 12]
df = pd.get_dummies(df, columns=small_cat, drop_first=True)

for c in cat_cols:
    if c not in small_cat:
        df[c] = LabelEncoder().fit_transform(df[c].astype(str))

severity_mapping = {
    "Slight Injury": 0,
    "Serious Injury": 1,
    "Fatal injury": 2
}
df[TARGET] = df[TARGET].map(severity_mapping)


# Heatmap

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if TARGET in numeric_cols:
    numeric_cols.remove(TARGET)

if numeric_cols:
    corr = df[numeric_cols].corr().abs()  

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr,
                annot=True,
                fmt=".2f",
                cmap='YlOrRd', 
                linewidths=0.5,
                square=True,
                cbar_kws={"shrink": .8})
    plt.title("Interactive Heatmap (Absolute Correlations Only)", fontsize=14)
    plt.tight_layout()
    plt.show()

if TARGET in df.columns and np.issubdtype(df[TARGET].dtype, np.number):
    corr_with_target = df[numeric_cols + [TARGET]].corr()[TARGET].abs().sort_values(ascending=False)
    print("\nTop correlations with target (abs):")
    print(corr_with_target.head(6))

# TRAIN–TEST SPLIT

if "Time" in df.columns:
    df = df.drop(columns=["Time"]) 

X = df.drop(columns=[TARGET])
y = df[TARGET]

X = X.fillna(X.median())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# SCALING

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

