import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from django.conf import settings
import logging

logger = logging.getLogger(__name__)

class LungCancerMLService:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.pca_transformer = None
        self.feature_names = None          # 🔹 Will hold feature names used for SHAP
        self.model_loaded = False
        self.expected_features = None
        try:
            self.load_model()
        except Exception as e:
            logger.error(f"Failed to initialize ML service: {e}")

    def load_model(self):
        models_dir = os.path.join(settings.MEDIA_ROOT, 'models')
        model_path = os.path.join(models_dir, 'model.keras')
        pca_path = os.path.join(models_dir, 'pca_transformer.pkl')

        if not os.path.exists(model_path):
            logger.error(f"Model file not found at {model_path}")
            raise FileNotFoundError(f"Model file not found at {model_path}")

        try:
            # 🔹 This is your trained classifier — satisfies service.model requirement
            self.model = keras.models.load_model(model_path)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

        if os.path.exists(pca_path):
            try:
                self.pca_transformer = joblib.load(pca_path)
                logger.info("PCA transformer loaded successfully")
                self.expected_features = self.pca_transformer.n_components_
            except Exception as e:
                logger.error(f"Failed to load PCA transformer: {e}")
                self.pca_transformer = None
        else:
            logger.warning("PCA transformer not found, skipping PCA")
            self.pca_transformer = None
            self.expected_features = self.model.input_shape[1]

        # 🔹 Initialize feature_names based on model input dimension
        # If PCA is used, features are PCA components (PC1, PC2, ...)
        if self.pca_transformer is not None and self.expected_features is not None:
            self.feature_names = [f"PC{i+1}" for i in range(self.expected_features)]
        else:
            # Will be set later from data when no PCA
            self.feature_names = None

        self.model_loaded = True

    def preprocess_rna_seq_data(self, df):
        if not self.model_loaded:
            raise RuntimeError("Model not loaded")

        try:
            df_clean = df.dropna(axis=0, how='any')
            logger.info(f"Data shape after dropping NaNs: {df_clean.shape}")

            df_transposed = df_clean.transpose()
            logger.info(f"Data shape after transpose: {df_transposed.shape}")

            input_vector = df_transposed.values
            logger.debug(f"Input vector shape before reshape check: {input_vector.shape}")

            # Ensure the input is always 2D
            if input_vector.ndim == 1:
                input_vector = input_vector.reshape(1, -1)
                logger.info("Reshaped 1D input vector to 2D for a single sample.")

            if self.pca_transformer:
                data_transformed = self.pca_transformer.transform(input_vector)
                logger.info(f"Data shape after PCA: {data_transformed.shape}")

                # 🔹 If PCA is used, SHAP will be over PCA components
                self.feature_names = [f"PC{i+1}" for i in range(data_transformed.shape[1])]
            else:
                data_transformed = input_vector

                # 🔹 If no PCA, features correspond to genes (row index of df_clean)
                self.feature_names = df_clean.index.tolist()

            # 🔹 Keep returning gene index list for existing code that uses gene_names
            return data_transformed, df_clean.index.tolist()

        except Exception as e:
            logger.error(f"Error in preprocessing: {e}")
            raise

    # 🔹 New method: used by SHAP endpoint to preprocess patient CSV
    def preprocess_patient_data(self, df):
        """
        Preprocess a single (or multiple) patient RNA-seq CSV for SHAP/XAI.

        Uses the same pipeline as preprocess_rna_seq_data but returns only
        the model-ready array. self.feature_names is set inside
        preprocess_rna_seq_data and will be used by SHAP.
        """
        data_array, _ = self.preprocess_rna_seq_data(df)
        return data_array

    def predict(self, data_array):
        if not self.model_loaded:
            raise RuntimeError("Model not loaded")

        try:
            predictions = self.model.predict(data_array, verbose=0)
            logger.info(f"Raw prediction output: {predictions}")

            predicted_classes = (predictions > 0.5).astype(int).flatten()
            probabilities = predictions.flatten()

            return predicted_classes.tolist(), probabilities.tolist()

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise

    def format_classification_results(self, predicted_classes, probabilities, gene_names, gene_expression):
        pred_class = predicted_classes[0]
        pred_prob = probabilities[0]

        label = 'LUAD' if pred_class == 1 else 'LUSC'
        confidence = round(pred_prob * 100, 2) if pred_class == 1 else round((1 - pred_prob) * 100, 2)

        return {
            'patient_prediction': {
                'label': label,
                'confidence': confidence,
                'probability': round(pred_prob * 100, 2)
            },
            'gene_heatmap_data': {
                'genes': gene_names,
                'expression': gene_expression.tolist() if isinstance(gene_expression, np.ndarray) else gene_expression
            }
        }

    def calculate_model_performance(self, y_true=None, y_pred=None, y_prob=None):
        if y_true is not None and y_pred is not None:
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
            recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)

            performance_metrics = [
                {'metric': 'Accuracy', 'value': f'{float(accuracy):.1%}', 'description': 'Overall LUAD vs LUSC classification accuracy'},
                {'metric': 'Precision', 'value': f'{float(precision):.1%}', 'description': 'LUAD prediction precision'},
                {'metric': 'Sensitivity (Recall)', 'value': f'{float(recall):.1%}', 'description': 'LUAD detection rate'},
                {'metric': 'F1 Score', 'value': f'{float(f1):.1%}', 'description': 'Harmonic mean of precision and recall'}
            ]

            if y_prob is not None:
                try:
                    auc = roc_auc_score(y_true, y_prob)
                    performance_metrics.append({'metric': 'AUC-ROC', 'value': f'{float(auc):.3f}', 'description': 'Area under the ROC curve'})
                except Exception as e:
                    logger.warning(f"Could not calculate AUC-ROC: {e}")
        else:
            performance_metrics = [
                {'metric': 'Model Type', 'value': 'LUAD vs LUSC Classifier', 'description': 'Lung Adenocarcinoma vs Squamous Cell Carcinoma'},
                {'metric': 'Architecture', 'value': 'MLP Neural Network', 'description': 'Multi-Layer Perceptron with L2 regularization'},
                {'metric': 'Input Features', 'value': f'{self.expected_features}', 'description': 'Number of PCA components'},
                {'metric': 'Hidden Layers', 'value': '128 → 64 → 32 → 1', 'description': 'Layer architecture with ReLU activation'},
                {'metric': 'Output', 'value': 'Sigmoid (Binary)', 'description': 'LUAD (1) vs LUSC (0) classification'},
                {'metric': 'Framework', 'value': 'TensorFlow/Keras', 'description': 'Deep learning framework'}
            ]

        return performance_metrics

# Global instance
ml_service = LungCancerMLService()
