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

