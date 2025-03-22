# Credit Card Fraud Detection System
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, auc
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

def load_data(train_path, test_path):
    # Load and combine training and test datasets
    print("Loading datasets...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    print(f"Train data: {train_df.shape[0]} rows, Test data: {test_df.shape[0]} rows")
    
    # Check if datasets have the same structure
    if set(train_df.columns) != set(test_df.columns):
        print("Warning: Train and test datasets have different columns!")
    
    return train_df, test_df

def explore_data(df):
    print("\n--- Data Exploration ---")
    fraud_count = df['is_fraud'].value_counts()
    print(f"Legitimate: {fraud_count[0]}, Fraudulent: {fraud_count[1]} ({fraud_count[1]/len(df)*100:.2f}%)")
    
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        print("\nMissing values:")
        print(missing_values[missing_values > 0])
    else:
        print("\nNo missing values found")
    
    return fraud_count[1]/len(df)

def preprocess_data(df):
    processed_df = df.copy()
    
    # Extract datetime features
    processed_df['trans_date_trans_time'] = pd.to_datetime(processed_df['trans_date_trans_time'])
    processed_df['hour'] = processed_df['trans_date_trans_time'].dt.hour
    processed_df['day'] = processed_df['trans_date_trans_time'].dt.day
    processed_df['month'] = processed_df['trans_date_trans_time'].dt.month
    processed_df['day_of_week'] = processed_df['trans_date_trans_time'].dt.dayofweek
    processed_df['is_weekend'] = processed_df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    
    # Age calculation
    processed_df['dob'] = pd.to_datetime(processed_df['dob'])
    reference_date = pd.to_datetime('2023-01-01')  
    processed_df['age'] = (reference_date - processed_df['dob']).dt.days // 365
    
    # Calculate distance between merchant and customer
    from math import radians, cos, sin, asin, sqrt
    
    def haversine(lat1, lon1, lat2, lon2):
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1 
        dlat = lat2 - lat1 
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a)) 
        r = 6371  # Radius of earth in kilometers
        return c * r
    
    processed_df['distance'] = processed_df.apply(
        lambda row: haversine(row['lat'], row['long'], row['merch_lat'], row['merch_long']), 
        axis=1
    )
    processed_df['amt_per_distance'] = processed_df['amt'] / (processed_df['distance'] + 1)
    
    # Transaction patterns
    processed_df = processed_df.sort_values(['cc_num', 'trans_date_trans_time'])
    processed_df['prev_tx_time'] = processed_df.groupby('cc_num')['unix_time'].shift(1)
    processed_df['time_since_prev_tx'] = processed_df['unix_time'] - processed_df['prev_tx_time']
    processed_df['time_since_prev_tx'].fillna(processed_df['time_since_prev_tx'].median(), inplace=True)
    
    # Transaction frequency
    tx_frequency = processed_df.groupby('cc_num').size().reset_index(name='tx_frequency')
    processed_df = processed_df.merge(tx_frequency, on='cc_num', how='left')
    
    # Risk factors
    merchant_fraud_rate = processed_df.groupby('merchant')['is_fraud'].mean().reset_index(name='merchant_fraud_rate')
    category_fraud_rate = processed_df.groupby('category')['is_fraud'].mean().reset_index(name='category_fraud_rate')
    processed_df = processed_df.merge(merchant_fraud_rate, on='merchant', how='left')
    processed_df = processed_df.merge(category_fraud_rate, on='category', how='left')
    
    # Get state statistics before one-hot encoding
    state_amt_stats = processed_df.groupby('state')['amt'].agg(['mean', 'std']).reset_index()
    state_amt_stats.columns = ['state', 'state_mean_amt', 'state_std_amt']
    processed_df = pd.merge(processed_df, state_amt_stats, on='state', how='left')

    # One-hot encoding
    categorical_cols = ['gender', 'state', 'job', 'category']
    processed_df = pd.get_dummies(processed_df, columns=categorical_cols, drop_first=True)
    
    # Customer average transaction amount
    customer_mean_amt = processed_df.groupby('cc_num')['amt'].mean().reset_index(name='mean_customer_amt')
    processed_df = processed_df.merge(customer_mean_amt, on='cc_num', how='left')
    processed_df['amt_deviation'] = processed_df['amt'] / processed_df['mean_customer_amt']
    
    # Drop unnecessary columns
    cols_to_drop = ['cc_num', 'first', 'last', 'trans_date_trans_time', 'merchant',
                    'street', 'city', 'zip', 'dob', 'trans_num', 'lat', 'long',
                    'merch_lat', 'merch_long', 'prev_tx_time', 'unix_time', 'index']
    processed_df = processed_df.drop([col for col in cols_to_drop if col in processed_df.columns], axis=1)
    
    # Fill missing values
    for col in processed_df.columns:
        if processed_df[col].dtype in [np.float64, np.int64]:
            processed_df[col].fillna(processed_df[col].median(), inplace=True)
        else:
            processed_df[col].fillna(processed_df[col].mode()[0], inplace=True)
    
    return processed_df

def engineer_features(df):
    df_engineered = df.copy()
       
    # City population features
    df_engineered['pop_density_factor'] = np.log1p(df_engineered['city_pop']) / 10
    
    # Transaction ratios
    df_engineered['amt_to_mean_ratio'] = df_engineered['amt'] / df_engineered['state_mean_amt']
    
    # Interaction features
    df_engineered['age_amt_interaction'] = df_engineered['age'] * df_engineered['amt'] / 1000
    df_engineered['weekend_amt_interaction'] = df_engineered['is_weekend'] * df_engineered['amt']
    df_engineered['transaction_velocity'] = df_engineered['amt'] / (df_engineered['time_since_prev_tx'] + 1)
    
    return df_engineered

def prepare_model_data(train_df, test_df, target_col='is_fraud'):
    # Separate features and target
    X_train = train_df.drop(target_col, axis=1)
    y_train = train_df[target_col]
    X_test = test_df.drop(target_col, axis=1)
    y_test = test_df[target_col]
    
    # Scale numerical features
    numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
    scaler = StandardScaler()
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
    
    print(f"Training set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test, scaler

def evaluate_model(model, X_test, y_test, model_name="Model"):
    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    print(f"\n--- {model_name} Evaluation ---")
    print(classification_report(y_test, y_pred))
    
    conf_matrix = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = conf_matrix.ravel()
    
    roc_auc = roc_auc_score(y_test, y_prob)
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall, precision)
    
    print(f"ROC AUC: {roc_auc:.4f}, PR-AUC: {pr_auc:.4f}")
    print(f"False Positive Rate: {fp/(fp+tn):.4f}, False Negative Rate: {fn/(fn+tp):.4f}")
    
    return {
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'confusion_matrix': conf_matrix,
        'y_prob': y_prob,
        'y_pred': y_pred
    }

def analyze_misclassifications(df, y_test, y_pred, y_prob):
    # Create DataFrame with test data and predictions
    results_df = df.copy()
    results_df['actual'] = y_test.values
    results_df['predicted'] = y_pred
    results_df['prob_fraud'] = y_prob
    
    # Identify false positives and false negatives
    false_positives = results_df[(results_df['actual'] == 0) & (results_df['predicted'] == 1)]
    false_negatives = results_df[(results_df['actual'] == 1) & (results_df['predicted'] == 0)]
    
    print(f"\nFalse Positives: {len(false_positives)}, False Negatives: {len(false_negatives)}")
    
    if len(false_positives) > 0:
        print(f"FP Avg: Amount ${false_positives['amt'].mean():.2f}, Age {false_positives['age'].mean():.1f} yrs")
    
    if len(false_negatives) > 0:
        print(f"FN Avg: Amount ${false_negatives['amt'].mean():.2f}, Age {false_negatives['age'].mean():.1f} yrs")
    
    return false_positives, false_negatives

def plot_feature_importance(model, X, n_features=10):
    # Extract feature importances
    if hasattr(model, 'named_steps') and 'classifier' in model.named_steps:
        features = model.named_steps['classifier'].feature_importances_
    elif hasattr(model, 'feature_importances_'):
        features = model.feature_importances_
    elif hasattr(model, 'coef_'):
        features = np.abs(model.coef_[0])
    else:
        print("Model does not provide feature importances")
        return
    
    # Plot top features
    importances = pd.DataFrame({'feature': X.columns, 'importance': features})
    importances = importances.sort_values('importance', ascending=False).head(n_features)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=importances)
    plt.title(f'Top {n_features} Feature Importances')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

def build_voting_ensemble(X_train, y_train):
    # Define models for ensemble
    log_reg = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, max_depth=15, class_weight='balanced', random_state=42)
    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    
    # Create voting classifier
    voting_clf = VotingClassifier(
        estimators=[
            ('logistic', log_reg),
            ('random_forest', rf),
            ('gradient_boosting', gb),
            ('xgboost', xgb)
        ],
        voting='soft'
    )
    
    # Handle class imbalance with SMOTE
    smote = SMOTE(random_state=42)
    pipeline = ImbPipeline([
        ('smote', smote),
        ('classifier', voting_clf)
    ])
    
    # Train the model
    print("Training ensemble model...")
    pipeline.fit(X_train, y_train)
    
    return pipeline

def save_model(model, scaler, filename='fraud_detection_model.pkl'):
    joblib.dump({
        'model': model,
        'scaler': scaler
    }, filename)
    print(f"Model saved to {filename}")

def main():
    print("=== Credit Card Fraud Detection System ===")
    
    # Load data
    train_df, test_df = load_data("fraudTrain.csv", "fraudTest.csv") #add ur data path
    
    # Explore and preprocess training data
    explore_data(train_df)
    train_preprocessed = preprocess_data(train_df)
    train_final = engineer_features(train_preprocessed)
    
    # Preprocess test data
    test_preprocessed = preprocess_data(test_df)
    test_final = engineer_features(test_preprocessed)
    
    # Prepare data for modeling
    X_train, X_test, y_train, y_test, scaler = prepare_model_data(train_final, test_final)
    
    # Build model
    model = build_voting_ensemble(X_train, y_train)
    
    # Evaluate model
    results = evaluate_model(model, X_test, y_test, "Ensemble Model")
    
    # Analyze misclassifications
    analyze_misclassifications(test_final, y_test, results['y_pred'], results['y_prob'])
    
    # Plot feature importance
    plot_feature_importance(model, X_train)
    
    # Save model
    save_model(model, scaler)
    
    print("\n=== Fraud Detection System Complete ===")
    print(f"Model achieved ROC-AUC: {results['roc_auc']:.4f} and PR-AUC: {results['pr_auc']:.4f}")

if __name__ == "__main__":
    main()