import numpy as np
import tensorflow as tf
import json
from sklearn.preprocessing import StandardScaler
import os

from . import params


class InceptionClassifier():
    """
    Species classifier based on the Inception architecture for a single fold.
    """
    def __init__(self, fold):
        """Initialize the species classifier for a given fold.

        Parameters
        ----------
        fold : int
            Cross-validation fold index.

        Attributes
        ----------
        scaler : StandardScaler
        model : tf.keras.Model
        """
        self.fold = fold
        self.load_scaler()
        self.load_model()

    def load_scaler(self):
        """Load a pre-fitted StandardScaler from disk and attach it to the instance."""
        scaler_path = os.path.join(params.SCALER_DIR,
                                   f'clf-inception_fold-test-None_fold-val-{self.fold}_n-splits-val-5_random-state-0.json')
        scaler_specs = json.load(open(scaler_path))

        scaler = StandardScaler()
        for k, v in scaler_specs.items():
            setattr(scaler, k, v)
        self.scaler = scaler
        return

    def load_model(self):
        """Load the trained Inception-based Keras model for the given fold."""
        model_path = os.path.join(params.CLF_MODEL_DIR,
                                  f'clf-inception_fold-test-None_fold-val-{self.fold}_n-splits-val-5_random-state-0.keras')
        self.model = tf.keras.models.load_model(model_path)
        return

    def scale(self, X):
        """Apply standardization to the scalable features of each sequence."""
        X_scaled = []
        for x in X:
            x_scaled = np.atleast_2d(x.copy())
            x_scaled[:, -params.NUM_FEATURES_SCALE:] = self.scaler.transform(x[:, -params.NUM_FEATURES_SCALE:])
            X_scaled.append(x_scaled)
        return X_scaled

    @staticmethod
    def pad(X):
        """
        Zero-pad the sequences in X.

        Parameters
        ----------
        X : list of np.ndarray
            Each array has shape (num_features, sequence_length)
        MAX_SEQUENCE_LENGTH : int
            Target sequence length

        Returns
        -------
        X_padded : np.ndarray
            Array of shape (num_sequences, MAX_SEQUENCE_LENGTH, num_features)
        """
        num_sequences = len(X)
        num_features = len(params.FEATURES)

        # Initialize output with zeros
        X_padded = np.zeros((num_sequences, params.MAX_SEQUENCE_LENGTH, num_features))

        for i, x in enumerate(X):
            seq_len = x.shape[0]
            length = min(seq_len, params.MAX_SEQUENCE_LENGTH)
            X_padded[i, :length, :] = x[:length]

        return X_padded

    def preprocess(self, X):
        """Preprocess raw input sequences for model inference.

        Parameters
        ----------
        X : list of np.ndarray
            Each array has shape (T_i, F).

        Returns
        -------
        Z : tf.Tensor
            Tensor of shape (N, MAX_SEQUENCE_LENGTH, F, 1).
        """
        Z = self.scale(X)
        Z = self.pad(Z)
        Z = tf.convert_to_tensor(Z, dtype=tf.float32)
        Z = tf.expand_dims(Z, axis=-1)
        return Z

    def predict_proba(self, X):
        """Predict class probabilities.

        Parameters
        ----------
        X : list of np.ndarray
            Each array has shape (T_i, F).

        Returns
        -------
        p : np.ndarray
            Array of shape (N, C) with class probabilities.
        """
        Z = self.preprocess(X)
        logits = self.model.predict(Z)
        p = tf.nn.softmax(logits).numpy()
        return p

    def predict(self, X):
        """Predict class labels.

        Parameters
        ----------
        X : list of np.ndarray
            Each array has shape (T_i, F).

        Returns
        -------
        y : np.ndarray
            Array of shape (N,) with predicted class indices.
        """
        p = self.predict_proba(X)
        return p.argmax(axis=1)


class InceptionClassifierEnsemble():
    """
    Ensemble of species classifiers based on the Inception architecture.
    """
    def __init__(self):
        """Initialize an ensemble of classifiers across folds.

        Attributes
        ----------
        models : list of InceptionClassifier
            One model per fold.
        """
        models = []
        for fold in range(5):
            models.append(InceptionClassifier(fold))
        self.models = models

    def predict_proba(self, X):
        """Predict class probabilities for all ensemble members.

        Parameters
        ----------
        X : list of np.ndarray
            Each array has shape (T_i, F).

        Returns
        -------
        p : np.ndarray
            Array of shape (M, N, C), where M=5 is the number of models.
        """
        p = np.stack([m.predict_proba(X) for m in self.models], axis=0) # (num_models, num_samples, num_classes)
        return p

    def predict(self, X, return_probs=True):
        """Predict class labels for all ensemble members.

        Parameters
        ----------
        X : list of np.ndarray
            Each array has shape (T_i, F).

        Returns
        -------
        y : np.ndarray
            Array of shape (M, N), where M=5 is the number of models.
        """
        p = self.predict_proba(X)
        y = p.argmax(axis=2)
        if return_probs:
            return y, p
        else:
            return y
