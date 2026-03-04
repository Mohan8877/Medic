import pandas as pd
import logging
import joblib
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def engineer_features(input_path, output_path, target_col, numerical_cols=None, categorical_cols=None, encoder_save_dir=None):
    """
    Applies feature engineering techniques to the dataset.
    
    Args:
        input_path (str): Path to the cleaned data CSV.
        output_path (str): Path to save the fully engineered CSV.
        target_col (str): The column name of the target variable to predict.
        numerical_cols (list): List of numerical column names to scale.
        categorical_cols (list): List of categorical column names to encode.
        encoder_save_dir (str): Directory to save the fitted encoders for later inference.
    """
    logging.info(f"Loading data from {input_path}...")
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        logging.error(f"❌ File not found: {input_path}")
        return

    # 1. Separate Features and Target
    if target_col not in df.columns:
        logging.error(f"❌ Target column '{target_col}' not found in dataset.")
        return
        
    y = df[target_col]
    X = df.drop(columns=[target_col])

    # Optional: Save encoders for inference later
    if encoder_save_dir:
        os.makedirs(encoder_save_dir, exist_ok=True)

    # 2. Process Numerical Features
    if numerical_cols:
        logging.info("Processing numerical features (Imputing & Scaling)...")
        # Impute missing values with the median
        num_imputer = SimpleImputer(strategy='median')
        X[numerical_cols] = num_imputer.fit_transform(X[numerical_cols])
        
        # Scale values to have a mean of 0 and variance of 1
        scaler = StandardScaler()
        X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
        
        if encoder_save_dir:
            joblib.dump(scaler, os.path.join(encoder_save_dir, 'scaler.pkl'))

    # 3. Process Categorical Features (One-Hot Encoding)
    if categorical_cols:
        logging.info("Processing categorical features (One-Hot Encoding)...")
        # Fill missing categories with 'Unknown'
        cat_imputer = SimpleImputer(strategy='constant', fill_value='Unknown')
        X[categorical_cols] = cat_imputer.fit_transform(X[categorical_cols])
        
        # One-Hot Encode
        ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded_cats = ohe.fit_transform(X[categorical_cols])
        
        # Create a DataFrame with the new encoded columns
        encoded_cols = ohe.get_feature_names_out(categorical_cols)
        df_encoded = pd.DataFrame(encoded_cats, columns=encoded_cols, index=X.index)
        
        # Drop the original categorical columns and concatenate the new ones
        X = X.drop(columns=categorical_cols)
        X = pd.concat([X, df_encoded], axis=1)
        
        if encoder_save_dir:
            joblib.dump(ohe, os.path.join(encoder_save_dir, 'one_hot_encoder.pkl'))

    # 4. Process Target Variable (if it's text)
    if y.dtype == 'object':
        logging.info("Encoding target variable...")
        le = LabelEncoder()
        y = le.fit_transform(y)
        if encoder_save_dir:
            joblib.dump(le, os.path.join(encoder_save_dir, 'target_encoder.pkl'))

    # 5. Recombine and Save
    logging.info("Recombining features and target...")
    df_final = X.copy()
    df_final[target_col] = y

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df_final.to_csv(output_path, index=False)
    logging.info(f"✅ Feature engineering complete! Saved {df_final.shape[1]} columns to {output_path}")

if __name__ == "__main__":
    INPUT_DATA = 'data/processed/processed_symptoms.csv'
    OUTPUT_DATA = 'data/processed/engineered_data.csv' 
    TARGET = 'itching' # Make sure this matches your actual target column name!
    ENCODER_DIR = 'models/saved_models/'
    
    # FIX: Leave these as empty lists [] so it doesn't look for fake columns
    NUMERICAL_FEATURES = []     
    CATEGORICAL_FEATURES = []   

    engineer_features(
        input_path=INPUT_DATA,
        output_path=OUTPUT_DATA,
        target_col=TARGET,
        numerical_cols=NUMERICAL_FEATURES,
        categorical_cols=CATEGORICAL_FEATURES,
        encoder_save_dir=ENCODER_DIR
    )