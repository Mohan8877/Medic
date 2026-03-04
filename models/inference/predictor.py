import joblib
import numpy as np
import pandas as pd
import logging

# Configure basic logging for debugging and tracking
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ModelPredictor:
    def __init__(self, model_path, feature_names_path=None, encoder_path=None):
        """
        Initializes the predictor by loading the trained model and any optional preprocessing assets.
        
        Args:
            model_path (str): Path to the saved .pkl model file.
            feature_names_path (str, optional): Path to a saved list of exact feature column names.
            encoder_path (str, optional): Path to a saved label encoder for decoding targets.
        """
        logging.info("Initializing model predictor...")
        try:
            self.model = joblib.load(model_path)
            logging.info(f"✅ Model loaded successfully from {model_path}")
            
            # Optional: Load feature names to ensure incoming data columns match training data exactly
            self.feature_names = joblib.load(feature_names_path) if feature_names_path else None
            
            # Optional: Load a label encoder to translate numerical predictions back to text labels
            self.label_encoder = joblib.load(encoder_path) if encoder_path else None
            
        except FileNotFoundError as e:
            logging.error(f"❌ Failed to load files: {e}")
            raise
        except Exception as e:
            logging.error(f"❌ Unexpected error during initialization: {e}")
            raise

    def _preprocess_input(self, input_data):
        """
        Internal method to convert raw input data into the exact format expected by the model.
        """
        if self.feature_names:
            # If feature names were provided, map the dictionary to a DataFrame to guarantee column order
            df = pd.DataFrame([input_data], columns=self.feature_names)
            
            # Fill any missing features with 0 (or NaN depending on your model's requirement)
            if df.isnull().values.any():
                logging.warning("Missing values detected in input. Filling with 0.")
                df = df.fillna(0)
                
            return df
        else:
            # If no feature names exist, assume input_data is a flat list/array in the correct order
            return np.array([input_data])

    def predict(self, input_data):
        """
        Makes a prediction based on the provided input data.
        """
        if not hasattr(self, 'model'):
            return {"error": "Model is not loaded."}

        try:
            # 1. Format the data
            processed_data = self._preprocess_input(input_data)
            
            # 2. Generate the raw prediction
            raw_prediction = self.model.predict(processed_data)[0]
            
            # 3. Calculate confidence/probability
            confidence = None
            if hasattr(self.model, "predict_proba"):
                probabilities = self.model.predict_proba(processed_data)[0]
                confidence = float(max(probabilities))
            
            # 4. Decode the label (if an encoder was provided)
            if self.label_encoder:
                # The inverse_transform expects an array, so we wrap raw_prediction in brackets
                decoded_label = self.label_encoder.inverse_transform([raw_prediction])[0]
                # CRITICAL FIX: Force the numpy output into a standard Python string!
                final_label = str(decoded_label) 
            else:
                # If no encoder, just cast the raw output to string so JSON handles it properly
                final_label = str(raw_prediction)
                
            return {
                "prediction": final_label,
                "confidence_score": confidence
            }
            
        except Exception as e:
            logging.error(f"Prediction failed: {e}")
            return {"error": str(e)}

if __name__ == "__main__":
    # ==========================================
    # Example Usage / Testing Block
    # ==========================================
    
    # Update these strings to your ACTUAL file paths
    MODEL_PATH = 'models/saved_models/health_model.pkl'           # <-- Change this
    # FEATURES_PATH = 'models/saved_models/feature_names.pkl'     # <-- Uncomment and change if using
    # ENCODER_PATH = 'models/saved_models/label_encoder.pkl'      # <-- Uncomment and change if using
    
    print("--- Testing Predictor ---")
    try:
        # Initialize the class
        predictor = ModelPredictor(
            model_path=MODEL_PATH, 
            # feature_names_path=FEATURES_PATH, 
            # encoder_path=ENCODER_PATH
        )
    except Exception as e:
        logging.error(f"Failed to initialize predictor: {e}")