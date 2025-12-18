# ROAD ACCIDENT SEVERITY PREDICTION – CLEAN VERSION (UPDATED)
# Added: Correlation Heatmap + Actual vs Predicted Graphs

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

# -----------------------------------------------------
# 1. LOAD DATA
# -----------------------------------------------------
df = pd.read_csv("C:/Users/abhilash/Downloads/Road.csv")
df.columns = [c.strip() for c in df.columns]

TARGET = "Accident_severity"

print("Dataset Shape:", df.shape)
print(df[TARGET].value_counts())

# Select meaningful numeric cols (excluding target)
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

# -----------------------------------------------
# Extract Accident_Hour from Time column
# -----------------------------------------------
time_cols = [c for c in df.columns if "time" in c.lower()]

if time_cols:
    tcol = time_cols[0]
    print(f"Extracting Accident_Hour from column: {tcol}")

    # UNIVERSAL TIME PARSER (fixes empty plots)
    df[tcol] = pd.to_datetime(df[tcol], format="%H:%M:%S", errors="coerce")
    df["Accident_Hour"] = df[tcol].dt.hour

    print(df["Accident_Hour"].head())
else:
    print("⚠ No time column found. Accident_Hour not created.")

# -----------------------------------------------------
# 3. EDA — Simple Distribution + Boxplots
# -----------------------------------------------------
# -----------------------------------------------------
# EXTRA EDA — Accident Time (Accident_Hour) if exists
# -----------------------------------------------------

if "Accident_Hour" in df.columns:

    # Distribution of Accident Hour
    plt.figure(figsize=(12, 4))
    sns.histplot(df["Accident_Hour"].dropna(), bins=24, kde=True, color='salmon', edgecolor=None)
    plt.title("Distribution of Accident_Hour", fontsize=14)
    plt.xlabel("Accident_Hour")
    plt.ylabel("Count")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Boxplot of Accident Hour
    plt.figure(figsize=(12, 2.8))
    sns.boxplot(x=df["Accident_Hour"].dropna(), color='salmon')
    plt.title("Boxplot of Accident_Hour", fontsize=14)
    plt.xlabel("Accident_Hour")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
else:
    print("⚠ Accident_Hour not found in dataset.")

# -----------------------------------------------------
# 3. EDA — Simple Distribution + Boxplots
# -----------------------------------------------------
numeric_cols = [c for c in numeric_cols if c != TARGET]
plot_cols = numeric_cols[:6]

# REMOVE Number_of_vehicles_involved from plots
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


# -----------------------------------------------------
# 4. ENCODING
# -----------------------------------------------------
cat_cols = df.select_dtypes(include=['object']).columns.tolist()
cat_cols = [c for c in cat_cols if c != TARGET]

# One-hot for small categories
small_cat = [c for c in cat_cols if df[c].nunique() <= 12]
df = pd.get_dummies(df, columns=small_cat, drop_first=True)

# Label encode rest
for c in cat_cols:
    if c not in small_cat:
        df[c] = LabelEncoder().fit_transform(df[c].astype(str))

# Encode target
severity_mapping = {
    "Slight Injury": 0,
    "Serious Injury": 1,
    "Fatal injury": 2
}
df[TARGET] = df[TARGET].map(severity_mapping)


# Heatmap
# -----------------------------------------------------
# CLEAN & INTERACTIVE HEATMAP (ABS CORRELATION)
# -----------------------------------------------------
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if TARGET in numeric_cols:
    numeric_cols.remove(TARGET)

if numeric_cols:
    corr = df[numeric_cols].corr().abs()   # <--- ONLY POSITIVE CORRELATIONS

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr,
                annot=True,
                fmt=".2f",
                cmap='YlOrRd',    # more attractive color palette
                linewidths=0.5,
                square=True,
                cbar_kws={"shrink": .8})
    plt.title("Interactive Heatmap (Absolute Correlations Only)", fontsize=14)
    plt.tight_layout()
    plt.show()


# Show top predictors correlated with target if possible
if TARGET in df.columns and np.issubdtype(df[TARGET].dtype, np.number):
    # target numeric, compute correlations
    corr_with_target = df[numeric_cols + [TARGET]].corr()[TARGET].abs().sort_values(ascending=False)
    print("\nTop correlations with target (abs):")
    print(corr_with_target.head(6))


# -----------------------------------------------------
# 5. TRAIN–TEST SPLIT
# -----------------------------------------------------
# -----------------------------------------------------
# 5. REMOVE DATETIME COLUMN BEFORE SPLITTING
# -----------------------------------------------------
if "Time" in df.columns:
    df = df.drop(columns=["Time"])   # <--- IMPORTANT FIX

# TRAIN–TEST SPLIT
X = df.drop(columns=[TARGET])
y = df[TARGET]

X = X.fillna(X.median())


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------------------------------
# 6. SCALING
# -----------------------------------------------------
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# -----------------------------------------------------
# 7. TRAIN MODELS
# -----------------------------------------------------
lr = LogisticRegression(max_iter=2000)
lr.fit(X_train_s, y_train)
y_pred_lr = lr.predict(X_test_s)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_s, y_train)
y_pred_knn = knn.predict(X_test_s)

dt = DecisionTreeClassifier(max_depth=8, random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

# -----------------------------------------------------
# 8. EVALUATION FUNCTION
# -----------------------------------------------------
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

# Evaluate all
evaluate_model("Logistic Regression", y_test, y_pred_lr)
evaluate_model("KNN", y_test, y_pred_knn)
evaluate_model("Decision Tree", y_test, y_pred_dt)

# -----------------------------------------------------
# 9. ACTUAL vs PREDICTED GRAPH (Simple Line Plot)
# -----------------------------------------------------
# ---------- 1) Cleaner Scatter Plot with Jitter ----------
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


# ---------- 2) Actual vs Predicted Counts (Bar Chart) ----------
def plot_counts(y_true, y_pred, model_name):
    actual = pd.Series(y_true).value_counts().sort_index()
    pred = pd.Series(y_pred).value_counts().sort_index()

    df_plot = pd.DataFrame({
        "Actual": actual,
        "Predicted": pred
    })

    df_plot.plot(kind="bar", figsize=(6,4))
    plt.title(f"Actual vs Predicted Count Comparison - {model_name}")
    plt.xlabel("Severity Class")
    plt.ylabel("Count")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()


# ==============================================================
#  Call plots for each model
# ==============================================================

plot_actual_vs_pred_scatter(y_test, y_pred_lr, "Logistic Regression")
plot_counts(y_test, y_pred_lr, "Logistic Regression")

plot_actual_vs_pred_scatter(y_test, y_pred_knn, "KNN")
plot_counts(y_test, y_pred_knn, "KNN")

plot_actual_vs_pred_scatter(y_test, y_pred_dt, "Decision Tree")
plot_counts(y_test, y_pred_dt, "Decision Tree")

# -----------------------------------------------------
# 10. DECISION TREE FEATURE IMPORTANCE
# -----------------------------------------------------
dt = DecisionTreeClassifier(random_state=42)
best_dt = dt.fit(X_train, y_train)
try:
    if isinstance(X_train, pd.DataFrame):
        feat_names = X_train.columns
    else:
        feat_names = list(range(X_train.shape[1]))
    fi = pd.Series(best_dt.feature_importances_, index=feat_names).sort_values(ascending=False)
    print(fi.head(20))
    plt.figure(figsize=(10,6))
    sns.barplot(x=fi.values[:20], y=fi.index[:20])
    plt.title("Decision Tree - Top 20 Features")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()
except Exception as e:
    print("Could not compute feature importances:", e)

# -----------------------------------------------------
# PERFORMANCE COMPARISON (TEXT OUTPUT)
# -----------------------------------------------------
performance_df = pd.DataFrame({
    "Model": ["Logistic Regression", "KNN", "Decision Tree"],
    "Accuracy": [
        accuracy_score(y_test, y_pred_lr),
        accuracy_score(y_test, y_pred_knn),
        accuracy_score(y_test, y_pred_dt)
    ],
    "Precision": [
        precision_score(y_test, y_pred_lr, average='weighted', zero_division=0),
        precision_score(y_test, y_pred_knn, average='weighted', zero_division=0),
        precision_score(y_test, y_pred_dt, average='weighted', zero_division=0)
    ],
    "Recall": [
        recall_score(y_test, y_pred_lr, average='weighted', zero_division=0),
        recall_score(y_test, y_pred_knn, average='weighted', zero_division=0),
        recall_score(y_test, y_pred_dt, average='weighted', zero_division=0)
    ],
    "F1 Score": [
        f1_score(y_test, y_pred_lr, average='weighted', zero_division=0),
        f1_score(y_test, y_pred_knn, average='weighted', zero_division=0),
        f1_score(y_test, y_pred_dt, average='weighted', zero_division=0)
    ]
})

print("\n========== MODEL PERFORMANCE COMPARISON ==========")
print(performance_df.round(4))

best_model = performance_df.loc[performance_df["F1 Score"].idxmax()]

print("\nBEST PERFORMING MODEL:")
print(f"Model Name : {best_model['Model']}")
print(f"Accuracy   : {best_model['Accuracy']:.4f}")
print(f"F1 Score   : {best_model['F1 Score']:.4f}")


# -----------------------------------------------------
# 11. PERFORMANCE COMPARISON GRAPH
# -----------------------------------------------------
results = {
    "Model": ["Logistic Regression", "KNN", "Decision Tree"],
    "Accuracy": [
        accuracy_score(y_test, y_pred_lr),
        accuracy_score(y_test, y_pred_knn),
        accuracy_score(y_test, y_pred_dt)
    ],
    "F1 Score": [
        f1_score(y_test, y_pred_lr, average='weighted'),
        f1_score(y_test, y_pred_knn, average='weighted'),
        f1_score(y_test, y_pred_dt, average='weighted')
    ]
}

perf_df = pd.DataFrame(results)

plt.figure(figsize=(8, 5))
sns.barplot(data=perf_df, x="Model", y="Accuracy", hue="Model", dodge=False)
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.show()

plt.figure(figsize=(8, 5))
sns.barplot(data=perf_df, x="Model", y="F1 Score", hue="Model", dodge=False)
plt.title("Model F1 Score Comparison")
plt.ylabel("F1 Score")
plt.ylim(0, 1)
plt.show()

print("\nScript Completed Successfully!")



