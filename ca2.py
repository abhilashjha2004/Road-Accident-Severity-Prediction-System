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


# TRAIN MODELS

lr = LogisticRegression(max_iter=2000)
lr.fit(X_train_s, y_train)
y_pred_lr = lr.predict(X_test_s)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_s, y_train)
y_pred_knn = knn.predict(X_test_s)

dt = DecisionTreeClassifier(max_depth=8, random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

# Performance 

def evaluate_model(name, y_true, y_pred):
    print(f"\n===== {name} =====")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred, average='weighted', zero_division=0))
    print("Recall:", recall_score(y_true, y_pred, average='weighted', zero_division=0))
    print("F1 Score:", f1_score(y_true, y_pred, average='weighted', zero_division=0))
    print("confusion matrix: ",confusion_matrix(y_test,y_pred))
    print("\n", classification_report(y_true, y_pred, zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix - {name}")
    plt.show()

evaluate_model("Logistic Regression", y_test, y_pred_lr)
evaluate_model("KNN", y_test, y_pred_knn)
evaluate_model("Decision Tree", y_test, y_pred_dt)

#  ACTUAL vs PREDICTED GRAPH 

def plot_actual_vs_pred_scatter(y_true, y_pred, model_name):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    x = np.arange(len(y_true))

    plt.figure(figsize=(12,4))
    plt.scatter(x, y_true + np.random.uniform(-0.05, 0.05, len(x)),
                label="Actual", alpha=0.5, s=10)
    plt.scatter(x, y_pred + np.random.uniform(-0.05, 0.05, len(x)),
                label="Predicted", alpha=0.5, s=10)

    plt.title(f"Actual vs Predicted (Scatter) - {model_name}")
    plt.xlabel("Test Sample Index")
    plt.ylabel("Severity Class (0=Slight, 1=Serious, 2=Fatal)")
    plt.yticks([0,1,2], ["Slight", "Serious", "Fatal"])
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

plot_actual_vs_pred_scatter(y_test, y_pred_lr, "Logistic Regression")
plot_actual_vs_pred_scatter(y_test, y_pred_knn, "KNN")
plot_actual_vs_pred_scatter(y_test, y_pred_dt, "Decision Tree")

