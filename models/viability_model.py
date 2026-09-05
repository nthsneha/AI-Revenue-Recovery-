import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

def train_and_save_model(
    events_path="dataset/implicit_failures.csv",
    history_path="dataset/user_history.csv",
    model_path="models/viability_model.pkl"
):
    """
    Train the viability model on COMBINED features:
      - Current event state (failure type, amount, intent, device, etc.)
      - User history (total orders, payment reliability, churn risk, etc.)

    The model outputs a recovery probability, not just a binary label.
    """
    # --- Load event data ---
    print(f"Loading event data from {events_path}...")
    try:
        events_df = pd.read_csv(events_path)
    except FileNotFoundError:
        print(f"Error: Could not find {events_path}. Run generate_implicit_failures.py first.")
        return

    # --- Load user history ---
    print(f"Loading user history from {history_path}...")
    try:
        history_df = pd.read_csv(history_path)
    except FileNotFoundError:
        print(f"Error: Could not find {history_path}. Run generate_user_history.py first.")
        return

    # --- Derive history features ---
    history_df['payment_reliability'] = (
        history_df['successful_payments'] / history_df['total_orders'].clip(lower=1)
    )

    history_features = history_df[[
        'user_id',
        'total_orders',
        'successful_payments',
        'failed_payments',
        'avg_order_value',
        'last_successful_payment_days',
        'churn_risk_score',
        'payment_reliability',
        'preferred_payment_method',
        'preferred_channel',
        'is_subscribed'
    ]]

    # --- Join event data with user history ---
    print("Joining event features with user history...")
    df = events_df.merge(history_features, on='user_id', how='left')

    # Fill NaN for users not found in history (neutral defaults)
    df['total_orders'] = df['total_orders'].fillna(0)
    df['successful_payments'] = df['successful_payments'].fillna(0)
    df['failed_payments'] = df['failed_payments'].fillna(0)
    df['avg_order_value'] = df['avg_order_value'].fillna(500.0)
    df['last_successful_payment_days'] = df['last_successful_payment_days'].fillna(180)
    df['churn_risk_score'] = df['churn_risk_score'].fillna(0.5)
    df['payment_reliability'] = df['payment_reliability'].fillna(0.5)
    df['preferred_payment_method'] = df['preferred_payment_method'].fillna('Unknown')
    df['preferred_channel'] = df['preferred_channel'].fillna('Unknown')
    df['is_subscribed'] = df['is_subscribed'].fillna(False)

    print(f"Combined dataset: {len(df)} rows, {len(df.columns)} columns")

    # --- Features (X) and Target (y) ---
    drop_cols = ['event_id', 'user_id', 'target_recovered']
    X = df.drop(columns=drop_cols)
    y = df['target_recovered']

    # One-Hot Encode categorical columns
    categorical_cols = [
        'revenue_loss_mode', 'device_type', 'time_of_day',
        'preferred_payment_method', 'preferred_channel'
    ]
    X = pd.get_dummies(X, columns=categorical_cols)

    # Ensure boolean columns are numeric
    bool_cols = X.select_dtypes(include=['bool']).columns
    X[bool_cols] = X[bool_cols].astype(int)

    # --- Train/Test Split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- Train XGBoost ---
    print("\nTraining XGBoost on combined event + history features...")
    model = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss',
        scale_pos_weight=len(y_train[y_train == 0]) / max(len(y_train[y_train == 1]), 1)
    )

    model.fit(X_train, y_train)

    # --- Evaluate ---
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # --- Feature Importance ---
    importance = pd.Series(
        model.feature_importances_, index=X.columns
    ).sort_values(ascending=False)
    print("\nTop 10 Features:")
    print(importance.head(10))

    # --- Save Model ---
    os.makedirs(os.path.dirname(model_path) if os.path.dirname(model_path) else '.', exist_ok=True)

    model_data = {
        'model': model,
        'feature_columns': list(X.columns),
        'history_feature_names': list(history_features.columns)
    }

    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)

    print(f"\nModel saved to {model_path}")
    print(f"Feature count: {len(X.columns)} (event + history combined)")

if __name__ == "__main__":
    train_and_save_model(
        events_path="dataset/implicit_failures.csv",
        history_path="dataset/user_history.csv",
        model_path="models/viability_model.pkl"
    )
