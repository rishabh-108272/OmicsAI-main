import os
import pandas as pd
import numpy as np
import joblib
from django.conf import settings
import logging

logger = logging.getLogger(__name__)

class ColorectalCancerService:
    def __init__(self):
        self.model = None
        self.model_loaded = False
        try:
            self.load_model()
        except Exception as e:
            logger.error(f"Failed to initialize Colorectal Cancer service: {e}")

    def load_model(self):
        """Loads the trained logistic regression model from a file."""
        models_dir = os.path.join(settings.MEDIA_ROOT, 'models')
        model_path = os.path.join(models_dir, 'logistic_Regression_model.pkl')

        if not os.path.exists(model_path):
            logger.error(f"Model file not found at {model_path}")
            raise FileNotFoundError(f"Model file not found at {model_path}")

        try:
            self.model = joblib.load(model_path)
            logger.info("Colorectal logistic regression model loaded successfully")
            self.model_loaded = True
        except Exception as e:
            logger.error(f"Failed to load logistic regression model: {e}")
            raise

    def preprocess_patient_data(self, df):
        """
        Preprocesses single-patient data for prediction.
        Assumes the input DataFrame `df` contains one column of expression values.
        """
        if not self.model_loaded:
            raise RuntimeError("Model is not loaded")

        try:
            df_clean = df.dropna(axis=0, how='any')
            
            # Convert the single column of values into a 2D NumPy array
            # of shape (1, n_features) as required by scikit-learn.
            input_vector = df_clean.values.reshape(1, -1)
            
            logger.info(f"Patient input vector shape: {input_vector.shape}")
            return input_vector

        except Exception as e:
            logger.error(f"Error in preprocessing: {e}")
            raise

    def predict(self, patient_vector):
        """
        Predicts the class (0=Normal, 1=Cancer) for a preprocessed patient vector.
        """
        if not self.model_loaded:
            raise RuntimeError("Model not loaded")

        try:
            prediction = self.model.predict(patient_vector)
            prediction_prob = self.model.predict_proba(patient_vector)

            predicted_class = int(prediction[0])
            # Get the probability of the predicted class
            confidence = float(np.max(prediction_prob[0]))

            logger.info(f"Prediction class: {predicted_class}, confidence: {confidence}")
            return predicted_class, confidence

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise

    def format_results(self, predicted_class, confidence_prob):
        """Formats the prediction results into a user-friendly dictionary."""
        label_map = {0: 'Normal', 1: 'Cancer'}
        label = label_map.get(predicted_class, 'Unknown')

        return {
            'patient_prediction': {
                'label': label,
                'confidence': round(confidence_prob * 100, 2),
            }
        }

# Global instance for use in your Django views
colorectal_cancer_service = ColorectalCancerService()