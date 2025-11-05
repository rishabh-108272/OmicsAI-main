import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from django.conf import settings
import logging

logger = logging.getLogger(__name__)

class LiverCancerService:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.model_loaded = False
        try:
            self.load_model()
        except Exception as e:
            logger.error(f"Failed to initialize Liver Cancer service: {e}")

    def load_model(self):
        models_dir = os.path.join(settings.MEDIA_ROOT, 'models')
        model_path = os.path.join(models_dir, 'logistic_model.pkl')

        if not os.path.exists(model_path):
            logger.error(f"Liver cancer model file not found at {model_path}")
            raise FileNotFoundError(f"Model file not found at {model_path}")

        try:
            self.model = joblib.load(model_path)
            logger.info("Liver cancer ensemble model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load liver cancer ensemble model: {e}")
            raise

        self.model_loaded = True

    def preprocess_patient_data(self, df):
        """
        Preprocess single patient RNA-Seq CSV:
        - Drop rows with NaN
        - Transpose to (1, n_features)
        - Scale input (optional: scaler should be fit on training data and loaded)
        """
        if not self.model_loaded:
            raise RuntimeError("Model not loaded")

        try:
            df_clean = df.dropna(axis=0, how='any')

            input_vector = df_clean.values

            if input_vector.ndim == 1:
                input_vector = input_vector.reshape(1, -1)

            logger.info(f"Liver patient input vector shape: {input_vector.shape}")

            # Scale input - ideally load scaler fitted on training
            scaled_vector = self.scaler.fit_transform(input_vector)

            return scaled_vector

        except Exception as e:
            logger.error(f"Error in liver cancer data preprocessing: {e}")
            raise

    def predict(self, scaled_patient_vector):
        """
        Predict using ensemble model.
        Classes: 0 = Normal, 1 = Cancer
        """
        if not self.model_loaded:
            raise RuntimeError("Model not loaded")

        try:
            prediction = self.model.predict(scaled_patient_vector)
            prediction_proba = self.model.predict_proba(scaled_patient_vector)

            predicted_class = int(prediction[0])
            predicted_prob = float(np.max(prediction_proba[0]))

            logger.info(f"Liver cancer prediction class: {predicted_class}, probability: {predicted_prob}")

            return predicted_class, predicted_prob

        except Exception as e:
            logger.error(f"Liver cancer prediction failed: {e}")
            raise

    def format_results(self, predicted_class, predicted_prob):
        label_map = {0: 'Normal', 1: 'Cancer'}
        label = label_map.get(predicted_class, 'Unknown')

        return {
            'patient_prediction': {
                'label': label,
                'confidence': round(predicted_prob * 100, 2),
                'probability': round(predicted_prob * 100, 2)
            }
        }

# Global instance
liver_cancer_service = LiverCancerService()
