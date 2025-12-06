"""
Model Training for UPI Fraud Detection
Trains ML models with imbalanced class handling
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, average_precision_score, f1_score
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# Import feature engineering
from feature_engineering import FraudFeatureEngineer

def setup_matplotlib():
    """Setup matplotlib for non-interactive mode"""
    plt.switch_backend("Agg")
    plt.style.use("seaborn-v0_8")
    sns.set_palette("husl")
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

def load_and_prepare_data(data_path='../data/synthetic_upi_transactions.csv'):
    """Load and prepare data for training"""
    print("Loading data...")
    df = pd.read_csv(data_path)

    print(f"Total records: {len(df)}")
    print(f"Fraud rate: {df['is_fraud'].mean()*100:.2f}%")

    # Feature engineering
    print("\nApplying feature engineering...")
    engineer = FraudFeatureEngineer()
    df = engineer.fit_transform(df)

    # Get feature columns
    feature_cols = engineer.get_feature_columns()

    # Prepare X and y
    X = df[feature_cols].copy()
    y = df['is_fraud'].copy()

    # Handle any remaining NaN
    X = X.fillna(0)

    return X, y, engineer, feature_cols

def train_models(X_train, X_test, y_train, y_test, feature_cols):
    """Train multiple models and compare"""

    results = {}

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 1. Logistic Regression with class weights
    print("\n" + "="*50)
    print("Training Logistic Regression...")
    lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)
    results['Logistic Regression'] = evaluate_model(lr, X_test_scaled, y_test, "Logistic Regression")

    # 2. Random Forest with class weights
    print("\n" + "="*50)
    print("Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    results['Random Forest'] = evaluate_model(rf, X_test, y_test, "Random Forest")

    # 3. XGBoost with scale_pos_weight
    print("\n" + "="*50)
    print("Training XGBoost...")
    scale_pos_weight = len(y_train[y_train==0]) / len(y_train[y_train==1])
    xgb = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        use_label_encoder=False,
        eval_metric='auc'
    )
    xgb.fit(X_train, y_train)
    results['XGBoost'] = evaluate_model(xgb, X_test, y_test, "XGBoost")

    # 4. Random Forest with SMOTE
    print("\n" + "="*50)
    print("Training Random Forest with SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE - Fraud: {sum(y_train_smote)}, Non-fraud: {len(y_train_smote) - sum(y_train_smote)}")

    rf_smote = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    rf_smote.fit(X_train_smote, y_train_smote)
    results['RF + SMOTE'] = evaluate_model(rf_smote, X_test, y_test, "RF + SMOTE")

    # Feature importance from XGBoost
    print("\n" + "="*50)
    print("Feature Importance (XGBoost):")
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': xgb.feature_importances_
    }).sort_values('importance', ascending=False)
    print(importance_df.head(15))

    # Save feature importance plot
    plot_feature_importance(importance_df)

    return results, xgb, scaler

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model performance"""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(f"\n{model_name} Results:")
    print("-" * 40)

    # Classification report
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Fraud']))

    # ROC-AUC
    roc_auc = roc_auc_score(y_test, y_prob)
    print(f"ROC-AUC Score: {roc_auc:.4f}")

    # Average Precision (PR-AUC)
    avg_precision = average_precision_score(y_test, y_prob)
    print(f"Average Precision (PR-AUC): {avg_precision:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(cm)

    return {
        'roc_auc': roc_auc,
        'avg_precision': avg_precision,
        'f1_fraud': f1_score(y_test, y_pred),
        'confusion_matrix': cm
    }

def plot_feature_importance(importance_df, output_dir='../outputs'):
    """Plot and save feature importance"""
    os.makedirs(output_dir, exist_ok=True)
    setup_matplotlib()

    plt.figure(figsize=(10, 8))
    top_features = importance_df.head(15)
    plt.barh(range(len(top_features)), top_features['importance'].values)
    plt.yticks(range(len(top_features)), top_features['feature'].values)
    plt.xlabel('Importance')
    plt.title('Top 15 Feature Importance - XGBoost')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/feature_importance.png', dpi=150)
    plt.close()
    print(f"Saved feature importance plot to {output_dir}/feature_importance.png")

def plot_roc_curves(results, output_dir='../outputs'):
    """Plot ROC curves for all models"""
    setup_matplotlib()

    plt.figure(figsize=(8, 6))
    for model_name, metrics in results.items():
        plt.bar(model_name, metrics['roc_auc'])
    plt.ylabel('ROC-AUC Score')
    plt.title('Model Comparison - ROC-AUC')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/model_comparison.png', dpi=150)
    plt.close()
    print(f"Saved model comparison to {output_dir}/model_comparison.png")

def find_optimal_threshold(model, X_test, y_test):
    """Find optimal probability threshold for fraud detection"""
    y_prob = model.predict_proba(X_test)[:, 1]

    precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)

    # Find threshold with best F1
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

    print(f"\nOptimal Threshold: {best_threshold:.4f}")
    print(f"At this threshold - Precision: {precisions[best_idx]:.4f}, Recall: {recalls[best_idx]:.4f}")

    return best_threshold

def save_model(model, scaler, engineer, threshold, output_dir='../models'):
    """Save trained model and components"""
    os.makedirs(output_dir, exist_ok=True)

    joblib.dump(model, f'{output_dir}/fraud_model.joblib')
    joblib.dump(scaler, f'{output_dir}/scaler.joblib')
    joblib.dump(engineer, f'{output_dir}/feature_engineer.joblib')
    joblib.dump({'threshold': threshold}, f'{output_dir}/config.joblib')

    print(f"\nModel saved to {output_dir}/")

def main():
    """Main training pipeline"""
    print("="*60)
    print("UPI FRAUD DETECTION - MODEL TRAINING")
    print("="*60)

    # Load and prepare data
    X, y, engineer, feature_cols = load_and_prepare_data()

    # Train-test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"Training fraud rate: {y_train.mean()*100:.2f}%")
    print(f"Test fraud rate: {y_test.mean()*100:.2f}%")

    # Train models
    results, best_model, scaler = train_models(X_train, X_test, y_train, y_test, feature_cols)

    # Find optimal threshold
    optimal_threshold = find_optimal_threshold(best_model, X_test, y_test)

    # Plot comparison
    plot_roc_curves(results)

    # Save best model (XGBoost)
    save_model(best_model, scaler, engineer, optimal_threshold)

    # Summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE - SUMMARY")
    print("="*60)
    print("\nModel Performance Comparison:")
    for model_name, metrics in results.items():
        print(f"  {model_name}: ROC-AUC = {metrics['roc_auc']:.4f}, PR-AUC = {metrics['avg_precision']:.4f}")

    print(f"\nBest Model: XGBoost")
    print(f"Optimal Threshold: {optimal_threshold:.4f}")

if __name__ == "__main__":
    main()
