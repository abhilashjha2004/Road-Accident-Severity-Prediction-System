"""
Flask Backend for Road Accident Severity Prediction
Updated for your folder structure
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import json
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# Global variables to store models and data
models = {}
scaler = None
feature_columns = []
label_encoder = None
X_test = None
y_test = None
processed_data = None

# UPDATE THIS PATH TO YOUR DATASET LOCATION
DATA_PATH = "Road.csv"
TARGET_COL = "Accident_severity"

def train_models():
    """Train all models and store them"""
    global models, scaler, feature_columns, label_encoder, X_test, y_test, processed_data

    print("=" * 70)
    print("Loading and preprocessing data...")
    print("=" * 70)

    # Load data
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip() for c in df.columns]

    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    # Store original data stats
    processed_data = {
        'total_records': int(df.shape[0]),
        'total_features': int(df.shape[1]),
        'target_distribution': {str(k): int(v) for k, v in df[TARGET_COL].value_counts().to_dict().items()}
    }

    print(f"Target column distribution:\n{df[TARGET_COL].value_counts()}")

    # Feature engineering - Extract time features
    print("\n" + "=" * 70)
    print("Feature Engineering...")
    print("=" * 70)

    time_cols = [c for c in df.columns if 'time' in c.lower() or 'hour' in c.lower()]
    if time_cols:
        tcol = time_cols[0]
        print(f"Found time column: {tcol}")
        df[tcol] = pd.to_datetime(df[tcol], errors='coerce')
        df['Accident_Hour'] = df[tcol].dt.hour

    date_cols = [c for c in df.columns if 'date' in c.lower()]
    if date_cols:
        dcol = date_cols[0]
        print(f"Found date column: {dcol}")
        df[dcol] = pd.to_datetime(df[dcol], errors='coerce')
        df['Accident_DayOfWeek'] = df[dcol].dt.dayofweek

    # Encoding categorical variables
    print("\n" + "=" * 70)
    print("Encoding Categorical Variables...")
    print("=" * 70)

    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    cat_cols = [c for c in cat_cols if c != TARGET_COL]
    print(f"Found {len(cat_cols)} categorical columns")

    # One-hot encoding for low cardinality
    onehot_cols = [c for c in cat_cols if df[c].nunique() <= 12]
    label_cols = [c for c in cat_cols if c not in onehot_cols]

    if onehot_cols:
        print(f"One-hot encoding: {len(onehot_cols)} columns")
        df = pd.get_dummies(df, columns=onehot_cols, drop_first=True, dtype=int)

    # Label encoding for high cardinality
    if label_cols:
        print(f"Label encoding: {len(label_cols)} columns")
        for c in label_cols:
            try:
                le = LabelEncoder()
                df[c] = le.fit_transform(df[c].astype(str))
            except Exception as e:
                print(f"Warning: Could not encode {c}: {e}")
                df.drop(columns=[c], inplace=True)

    # Encode target variable
    print("\n" + "=" * 70)
    print("Encoding Target Variable...")
    print("=" * 70)

    if df[TARGET_COL].dtype == 'object' or str(df[TARGET_COL].dtype) == 'category':
        label_encoder = LabelEncoder()
        df[TARGET_COL] = label_encoder.fit_transform(df[TARGET_COL].astype(str))
        print(f"Target classes: {list(label_encoder.classes_)}")

    # Prepare X and y
    print("\n" + "=" * 70)
    print("Preparing Features and Target...")
    print("=" * 70)

    X = df.drop(columns=[TARGET_COL]).copy()
    y = df[TARGET_COL].copy()

    # Drop identifier and datetime columns
    drop_cols = []
    for col in X.columns:
        if any(keyword in col.lower() for keyword in ['id', 'index', 'sno', 'date', 'time']):
            if not any(feat in col.lower() for feat in ['hour', 'day', 'week', 'month']):
                drop_cols.append(col)

    if drop_cols:
        print(f"Dropping identifier/datetime columns: {drop_cols}")
        X = X.drop(columns=drop_cols)

    # Convert all remaining object columns to numeric
    print("\nConverting remaining columns to numeric...")
    for col in X.columns:
        if X[col].dtype == 'object':
            try:
                X[col] = pd.to_numeric(X[col], errors='coerce')
            except:
                try:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
                except Exception as e:
                    print(f"Warning: Dropping column {col} due to encoding error")
                    X = X.drop(columns=[col])

    # Handle missing values
    print("\nHandling missing values...")
    X = X.apply(pd.to_numeric, errors='coerce')  # Force all to numeric
    X = X.fillna(X.median())  # Fill with median
    y = y.fillna(y.mode().iloc[0] if len(y.mode()) > 0 else 0)

    # Ensure all data is proper numeric type
    X = X.astype(np.float64)
    y = y.astype(np.int32)

    print(f"\nFinal feature shape: {X.shape}")
    print(f"Feature data types: {X.dtypes.value_counts().to_dict()}")

    feature_columns = X.columns.tolist()
    print(f"Total features: {len(feature_columns)}")

    # Train-test split
    print("\n" + "=" * 70)
    print("Splitting Data (80% Train, 20% Test)...")
    print("=" * 70)

    X_train, X_test_global, y_train, y_test_global = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_test = X_test_global
    y_test = y_test_global

    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples: {X_test.shape[0]}")

    # Scaling
    print("\n" + "=" * 70)
    print("Scaling Features...")
    print("=" * 70)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test_global)

    print("Scaling completed using StandardScaler")

    # Train models
    print("\n" + "=" * 70)
    print("Training Machine Learning Models...")
    print("=" * 70)

    # 1. Logistic Regression
    print("\n1. Training Logistic Regression...")
    lr = LogisticRegression(max_iter=3000, random_state=42, C=1, class_weight='balanced', solver='lbfgs')
    lr.fit(X_train_scaled, y_train)
    models['logistic'] = lr
    y_pred_lr = lr.predict(X_test_scaled)
    acc_lr = accuracy_score(y_test_global, y_pred_lr)
    print(f"   ✓ Logistic Regression trained - Accuracy: {acc_lr:.4f}")

    # 2. KNN
    print("\n2. Training K-Nearest Neighbors...")
    knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
    knn.fit(X_train_scaled, y_train)
    models['knn'] = knn
    y_pred_knn = knn.predict(X_test_scaled)
    acc_knn = accuracy_score(y_test_global, y_pred_knn)
    print(f"   ✓ KNN trained - Accuracy: {acc_knn:.4f}")

    # 3. Decision Tree
    print("\n3. Training Decision Tree...")
    dt = DecisionTreeClassifier(max_depth=8, random_state=42, class_weight='balanced', min_samples_split=5)
    dt.fit(X_train, y_train)  # Decision tree on unscaled data
    models['decision_tree'] = dt
    y_pred_dt = dt.predict(X_test_global)
    acc_dt = accuracy_score(y_test_global, y_pred_dt)
    print(f"   ✓ Decision Tree trained - Accuracy: {acc_dt:.4f}")

    print("\n" + "=" * 70)
    print("ALL MODELS TRAINED SUCCESSFULLY!")
    print("=" * 70)

    # Save models
    print("\nSaving models to disk...")
    with open('models.pkl', 'wb') as f:
        pickle.dump({
            'models': models,
            'scaler': scaler,
            'feature_columns': feature_columns,
            'label_encoder': label_encoder
        }, f)
    print("✓ Models saved to 'models.pkl'")

    processed_data['models_trained'] = ['Logistic Regression', 'KNN', 'Decision Tree']
    processed_data['accuracies'] = {
        'logistic': float(acc_lr),
        'knn': float(acc_knn),
        'decision_tree': float(acc_dt)
    }

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/train', methods=['POST'])
def train():
    """Endpoint to train models"""
    try:
        print("\n" + "=" * 70)
        print("TRAINING REQUEST RECEIVED")
        print("=" * 70)
        train_models()
        return jsonify({
            'status': 'success',
            'message': 'Models trained successfully',
            'data': processed_data
        })
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print("\n" + "=" * 70)
        print("ERROR DURING TRAINING:")
        print("=" * 70)
        print(error_details)
        return jsonify({
            'status': 'error',
            'message': str(e),
            'details': error_details
        }), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    """Endpoint for single prediction"""
    try:
        data = request.json
        model_type = data.get('model', 'logistic')
        features = data.get('features', {})

        # Create feature vector with all zeros
        feature_vector = np.zeros(len(feature_columns))

        # Fill in provided features
        for i, col in enumerate(feature_columns):
            if col in features:
                feature_vector[i] = float(features[col])

        # Reshape for prediction
        feature_array = feature_vector.reshape(1, -1).astype(np.float64)

        # Scale features (except for decision tree)
        if model_type != 'decision_tree':
            feature_array = scaler.transform(feature_array)

        # Predict
        model = models[model_type]
        prediction = model.predict(feature_array)[0]

        # Get probability if available
        probability = None
        if hasattr(model, 'predict_proba'):
            probability = model.predict_proba(feature_array)[0].tolist()

        # Decode prediction
        if label_encoder:
            prediction_label = label_encoder.inverse_transform([int(prediction)])[0]
        else:
            prediction_label = str(int(prediction))

        return jsonify({
            'status': 'success',
            'prediction': int(prediction),
            'prediction_label': str(prediction_label),
            'probability': probability
        })
    except Exception as e:
        import traceback
        return jsonify({
            'status': 'error',
            'message': str(e),
            'details': traceback.format_exc()
        }), 500

@app.route('/api/evaluate', methods=['POST'])
def evaluate():
    """Endpoint to evaluate all models"""
    try:
        data = request.json
        model_type = data.get('model', 'all')

        results = {}

        models_to_eval = models.keys() if model_type == 'all' else [model_type]

        for m_type in models_to_eval:
            model = models[m_type]

            # Get predictions
            if m_type == 'decision_tree':
                y_pred = model.predict(X_test)
            else:
                X_test_scaled = scaler.transform(X_test)
                y_pred = model.predict(X_test_scaled)

            # Calculate metrics
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            cm = confusion_matrix(y_test, y_pred)

            results[m_type] = {
                'accuracy': float(acc),
                'precision': float(prec),
                'recall': float(rec),
                'f1_score': float(f1),
                'confusion_matrix': cm.tolist()
            }

        return jsonify({
            'status': 'success',
            'results': results
        })
    except Exception as e:
        import traceback
        return jsonify({
            'status': 'error',
            'message': str(e),
            'details': traceback.format_exc()
        }), 500

@app.route('/api/feature_importance', methods=['GET'])
def feature_importance():
    """Get feature importance from Decision Tree"""
    try:
        dt_model = models['decision_tree']
        importances = dt_model.feature_importances_

        feature_imp = [
            {'feature': str(col), 'importance': float(imp)}
            for col, imp in zip(feature_columns, importances)
        ]
        feature_imp = sorted(feature_imp, key=lambda x: x['importance'], reverse=True)[:20]

        return jsonify({
            'status': 'success',
            'feature_importance': feature_imp
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/data_stats', methods=['GET'])
def data_stats():
    """Get dataset statistics"""
    try:
        return jsonify({
            'status': 'success',
            'stats': processed_data
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("STARTING ROAD ACCIDENT PREDICTION SERVER")
    print("=" * 70)
    app.run(host="0.0.0.0", port=5000)
    # Check if data file exists
    if not os.path.exists(DATA_PATH):
        print(f"\n⚠️  WARNING: Dataset not found at: {DATA_PATH}")
        print("Please update DATA_PATH variable in the code!")
    else:
        print(f"✓ Dataset found: {DATA_PATH}")

    # Load or train models on startup
    if os.path.exists('models.pkl'):
        print("\n✓ Loading existing models from 'models.pkl'...")
        try:
            with open('models.pkl', 'rb') as f:
                saved = pickle.load(f)
                models = saved['models']
                scaler = saved['scaler']
                feature_columns = saved['feature_columns']
                label_encoder = saved.get('label_encoder')
            print("✓ Models loaded successfully!")
        except Exception as e:
            print(f"⚠️  Could not load models: {e}")
            print("Please train models using the 'Train Models' button in the web interface")
    else:
        print("\n⚠️  No saved models found.")
        print("Please train models using the 'Train Models' button in the web interface")

    print("\n" + "=" * 70)
    print("SERVER STARTING ON: http://localhost:5000")
    print("=" * 70)
    print("\nPress CTRL+C to stop the server\n")

    app.run(debug=True, port=5000, use_reloader=False)

