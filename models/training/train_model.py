import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

def train_and_save_model(data_path, model_save_path, feature_names_save_path, encoder_save_path):
    print(f"Loading processed data from: {data_path}")
    
    try:
        df = pd.read_csv(data_path, low_memory=False)
    except FileNotFoundError:
        print(f"❌ ERROR: Could not find {data_path}. Run data_preprocessing.py first.")
        return

    # --- THE FIX: Remove corrupted disease labels ---
    print("Cleaning up corrupted target labels...")
    initial_count = len(df)
    
    # Drop rows where target_disease is literally '0', '0.0', 'nan', or empty
    df = df[~df['target_disease'].astype(str).str.strip().isin(['0', '0.0', 'nan', 'NaN', ''])]
    
    print(f"Dropped {initial_count - len(df)} rows containing invalid '0' disease names.")
    # ------------------------------------------------

    print("Splitting data into features and target...")
    
    X = df.drop(columns=['target_disease'])
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)
    y = df['target_disease'].astype(str)

    print("Encoding disease labels...")
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    feature_columns = list(X.columns)

    print("Splitting into training and testing sets (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    print("🧠 Training the Random Forest Classifier. This might take a moment...")
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42, n_jobs=-1)
    
    rf_model.fit(X_train, y_train)

    print("Evaluating model performance on test data...")
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\n" + "="*50)
    print(f"✅ Model Accuracy: {accuracy * 100:.2f}%")
    print("="*50 + "\n")

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    print("Saving the model, features, and label encoder...")
    joblib.dump(rf_model, model_save_path)
    joblib.dump(feature_columns, feature_names_save_path)
    joblib.dump(le, encoder_save_path) 
    
    print(f"💾 Model successfully saved to: {model_save_path}")
    print("System is ready for inference!")

if __name__ == "__main__":
    DATA_PATH = 'data/processed/processed_symptoms.csv'
    MODEL_SAVE_PATH = 'models/saved_models/health_model.pkl'
    FEATURES_SAVE_PATH = 'models/saved_models/feature_names.pkl'
    ENCODER_SAVE_PATH = 'models/saved_models/label_encoder.pkl'
    
    train_and_save_model(DATA_PATH, MODEL_SAVE_PATH, FEATURES_SAVE_PATH, ENCODER_SAVE_PATH)