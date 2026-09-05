import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

def train_and_save_model(data_path="../dataset/implicit_failures.csv", model_path="viability_model.pkl"):
    print(f"Loading data from {data_path}...")
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: Could not find {data_path}. Please run generate_implicit_failures.py first.")
        return
        
    # Features (X) and Target (y)
    # Dropping non-predictive or target columns
    X = df.drop(columns=['event_id', 'user_id', 'target_recovered'])
    y = df['target_recovered']
    
    # Convert categorical columns to dummy variables (One-Hot Encoding)
    categorical_cols = ['revenue_loss_mode', 'device_type', 'time_of_day']
    X = pd.get_dummies(X, columns=categorical_cols)
    
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training XGBoost Classifier...")
    # Initialize and train the XGBoost model
    model = xgb.XGBClassifier(
        n_estimators=100, 
        max_depth=4, 
        learning_rate=0.1, 
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save the model and the expected feature columns
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    model_data = {
        'model': model,
        'feature_columns': list(X.columns)
    }
    
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
        
    print(f"\nModel saved successfully to {model_path}")

if __name__ == "__main__":
    # Assuming this is run from the 'models' directory
    train_and_save_model(data_path="../dataset/implicit_failures.csv", model_path="viability_model.pkl")
