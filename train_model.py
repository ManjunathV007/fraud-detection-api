import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, classification_report, confusion_matrix,
    roc_auc_score, precision_recall_curve, roc_curve
)
from imblearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime
import os
from preprocess import preprocess_data

# Constants
MODEL_DIR = "models"
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

def load_data(file_path):
    """Load and preprocess data"""
    try:
        data = pd.read_csv(file_path)
        processed_data = preprocess_data(data)
        print(f"Data loaded and processed. Shape: {processed_data.shape}")
        return processed_data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def prepare_data(data, test_size=0.2):
    """Split data into train/test sets"""
    X = data.drop('Class', axis=1)
    y = data['Class']
    return train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

def create_model_pipeline():
    """Create pipeline with XGBoost"""
    return Pipeline([
        ('model', XGBClassifier(
            learning_rate=0.01,
            n_estimators=200,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        ))
    ])

def train_model(X_train, y_train):
    """Train model with GridSearchCV"""
    pipeline = create_model_pipeline()
    param_grid = {
        'model__max_depth': [3, 4, 5],
        'model__min_child_weight': [1, 3],
        'model__subsample': [0.8, 0.9],
        'model__colsample_bytree': [0.8, 0.9]
    }
    
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        scoring='precision',
        cv=3,
        n_jobs=-1,
        verbose=2
    )
    
    print("Starting model training...")
    grid_search.fit(X_train, y_train)
    print(f"Best parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'auc_roc': roc_auc_score(y_test, y_pred_proba)
    }
    
    print("\nModel Performance Metrics:")
    print("-" * 50)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    plot_results(y_test, y_pred, y_pred_proba)
    return metrics

def plot_results(y_test, y_pred, y_pred_proba):
    """Plot confusion matrix and ROC curve"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', ax=ax1)
    ax1.set_title('Confusion Matrix')
    ax1.set_ylabel('True Label')
    ax1.set_xlabel('Predicted Label')
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    ax2.plot(fpr, tpr)
    ax2.plot([0, 1], [0, 1], 'k--')
    ax2.set_title('ROC Curve')
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    
    plt.tight_layout()
    plt.show()

def save_model(model, metrics, prefix="fraud_detection"):
    """Save model and metrics"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(MODEL_DIR, f"{prefix}_model_{timestamp}.joblib")
    metrics_path = os.path.join(MODEL_DIR, f"{prefix}_metrics_{timestamp}.json")
    
    try:
        joblib.dump(model, model_path)
        pd.Series(metrics).to_json(metrics_path)
        print(f"Model saved: {model_path}")
        return model_path
    except Exception as e:
        print(f"Error saving model: {e}")
        return None

def main():
    # Load and prepare data
    data = load_data("creditcard.csv")
    if data is None:
        return
    
    # Split data
    X_train, X_test, y_train, y_test = prepare_data(data)
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test)
    
    # Save model
    save_model(model, metrics)

if __name__ == "__main__":
    main()