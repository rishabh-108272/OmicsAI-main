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
        self.feature_names = None  # 🔹 will hold training-time feature (gene) order
        try:
            self.load_model()
        except Exception as e:
            logger.error(f"Failed to initialize Colorectal Cancer service: {e}")

    def load_model(self):
        """Loads the trained logistic regression model from a file."""
        models_dir = os.path.join(settings.MEDIA_ROOT, 'models')
        # NOTE: Ensure this path is correct for your actual model file
        model_path = os.path.join(models_dir, 'logistic_Regression_model.pkl')

        if not os.path.exists(model_path):
            logger.error(f"Model file not found at {model_path}")
            raise FileNotFoundError(f"Model file not found at {model_path}")

        try:
            self.model = joblib.load(model_path)
            logger.info("Colorectal logistic regression model loaded successfully")

            # 🔹 Expose training feature names for SHAP alignment, if available
            if hasattr(self.model, "feature_names_in_"):
                self.feature_names = list(self.model.feature_names_in_)
                logger.info(f"Loaded {len(self.feature_names)} feature names from model.feature_names_in_")
            else:
                self.feature_names = None
                logger.warning("Model has no 'feature_names_in_'; SHAP will rely on input order only.")

            self.model_loaded = True
        except Exception as e:
            logger.error(f"Failed to load logistic regression model: {e}")
            raise

    def preprocess_patient_data(self, df):
        """
        Preprocesses single-patient data for prediction / SHAP.

        Assumes the input DataFrame `df`:
          - index = gene symbols (rows)
          - FIRST column = expression values for this one patient

        If self.feature_names is set (from training), we:
          - align the patient gene values to that exact gene order
          - fill any missing genes with 0.0

        Returns:
          np.ndarray of shape (1, n_features)
        """
        if not self.model_loaded:
            raise RuntimeError("Model is not loaded")

        try:
            # Drop any genes with missing expression
            df_clean = df.dropna(axis=0, how='any')

            # Series of expression values; index = gene symbols
            values_series = df_clean.iloc[:, 0]

            if self.feature_names is not None:
                # Align by training-time gene order
                aligned_values = [float(values_series.get(g, 0.0)) for g in self.feature_names]
                input_vector = np.array(aligned_values, dtype=float).reshape(1, -1)
                logger.info(
                    f"Patient vector aligned to training feature order: shape {input_vector.shape}"
                )
            else:
                # Fallback: no known feature names, just use as-is
                input_vector = values_series.to_numpy(dtype=float).reshape(1, -1)
                logger.warning(
                    "feature_names not set; using raw patient vector order. "
                    "SHAP explanations may not be consistent across runs."
                )

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

    def get_model_specifications(self):
        """Returns the pre-calculated performance specifications for the Colorectal Cancer model."""
        return [
            {
                'metric': 'Model Type',
                'value': 'Logistic Regression',
                'description': 'Algorithm used for classification.'
            },
            {
                'metric': 'Data Source',
                'value': 'TCGA-CRC Dataset (RNA-Seq)',
                'description': 'Training data cohort for the model.'
            },
            {
                'metric': 'Accuracy',
                'value': '95.1%',
                'description': 'Overall correct classification rate on the test set.'
            },
            {
                'metric': 'ROC AUC',
                'value': '0.97',
                'description': 'Area Under the Receiver Operating Characteristic Curve.'
            },
            {
                'metric': 'Features',
                'value': '21,000+ Genes',
                'description': 'Number of gene expression values used as input features.'
            },
        ]

# Global instance for use in your Django views
colorectal_cancer_service = ColorectalCancerService()
