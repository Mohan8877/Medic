import pandas as pd
import joblib
import logging
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def evaluate_model(model_path, test_data_path, target_column_name, encoder_path=None):
    """
    Loads a trained model and evaluates it against a test dataset.
    
    Args:
        model_path (str): Path to the saved .pkl model file.
        test_data_path (str): Path to the CSV file containing the test data.
        target_column_name (str): The name of the column containing the true labels.
        encoder_path (str, optional): Path to the saved label encoder, if used during training.
    """
    logging.info("Starting Model Evaluation...")
    
    try:
        # 1. Load the Model and Data
        logging.info(f"Loading model from {model_path}...")
        model = joblib.load(model_path)
        
        logging.info(f"Loading test data from {test_data_path}...")
        df_test = pd.read_csv(test_data_path)
        
        # 2. Separate Features (X) and Target (y)
        if target_column_name not in df_test.columns:
            logging.error(f"Target column '{target_column_name}' not found in the test dataset.")
            return

        X_test = df_test.drop(columns=[target_column_name])
        y_true = df_test[target_column_name]
        
        # Ensure input data is numeric (handling any accidental text)
        X_test = X_test.apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)

        # 3. Encode the true labels if an encoder was used during training
        if encoder_path:
            logging.info(f"Loading label encoder from {encoder_path}...")
            le = joblib.load(encoder_path)
            # Only transform labels that the encoder knows; handle unseen labels safely
            known_classes = set(le.classes_)
            # Filter out rows with target labels the model has never seen
            valid_indices = y_true.isin(known_classes)
            if not valid_indices.all():
                logging.warning("Some labels in the test set were not seen during training. Dropping them for evaluation.")
                X_test = X_test[valid_indices]
                y_true = y_true[valid_indices]
                
            y_true_encoded = le.transform(y_true)
        else:
            y_true_encoded = y_true

        # 4. Make Predictions
        logging.info("Generating predictions on the test set...")
        y_pred = model.predict(X_test)

        # 5. Calculate Metrics
        accuracy = accuracy_score(y_true_encoded, y_pred)
        
        print("\n" + "="*50)
        print("📊 MODEL EVALUATION REPORT")
        print("="*50)
        print(f"✅ Overall Accuracy: {accuracy * 100:.2f}%\n")
        
        print("🔍 Detailed Classification Report:")
        # If we have an encoder, use the actual text names for the report
        target_names = le.classes_ if encoder_path else None
        print(classification_report(y_true_encoded, y_pred, target_names=target_names))
        
        print("-" * 50)
        print("🧮 Confusion Matrix:")
        print(confusion_matrix(y_true_encoded, y_pred))
        print("="*50 + "\n")

    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
    except Exception as e:
        logging.error(f"An error occurred during evaluation: {e}")

if __name__ == "__main__":
    # ==========================================
    # Example Usage / Paths
    # ==========================================
    
    # Update these paths to match your actual files
    MODEL_PATH = 'models/saved_models/health_model.pkl'
    TEST_DATA_PATH = 'data/processed/processed_symptoms.csv' # Ideally, use a separate test file here!
    TARGET_COLUMN = 'target_disease'                         # The name of your output column
    ENCODER_PATH = 'models/saved_models/label_encoder.pkl'   # Uncomment/update if you used an encoder
    
    evaluate_model(
        model_path=MODEL_PATH,
        test_data_path=TEST_DATA_PATH,
        target_column_name=TARGET_COLUMN,
        encoder_path=ENCODER_PATH
    )