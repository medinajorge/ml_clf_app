import numpy as np
import xgboost as xgb
from numba import njit
import os

from . import params


@njit
def compute_entropy(p):
    """Compute Shannon entropy for a batch of probability vectors.

    Parameters
    ----------
    p : np.ndarray
        Array of shape (N, C) with class probabilities.

    Returns
    -------
    entropy : np.ndarray
        Array of shape (N,) with entropy values.
    """
    return -(p * np.where(p > 0, np.log(p), 0)).sum(axis=1)


class XGB():
    """
    XGBoost confidence model for a single fold.
    """
    def __init__(self, fold, use_entropy=True):
        """Initialize an XGBoost confidence model for a given fold.

        Parameters
        ----------
        fold : int
            Cross-validation fold index.
        use_entropy : bool
            Whether to append entropy as an additional feature.
        """
        self.fold = fold
        self.use_entropy = use_entropy
        if self.use_entropy:
            self.get_features = self._get_features_with_entropy
        else:
            self.get_features = self._identity
        self.load_model()

    def load_model(self):
        model_path = os.path.join(params.CONF_MODEL_DIR,
                                  f'fold-test-None_fold-val-{self.fold}_n-splits-val-5_use-entropy-{self.use_entropy}.json')
        self.model = xgb.Booster()
        self.model.load_model(model_path)
        return

    @staticmethod
    def _get_features_with_entropy(p):
        return np.hstack((p, compute_entropy(p)[:, None]))

    @staticmethod
    def _identity(p):
        return p

    def preprocess(self, p):
        """Convert input features into an XGBoost DMatrix.

        Parameters
        ----------
        p : np.ndarray
            Array of shape (N, C) or (N, C + 1).

        Returns
        -------
        dtest : xgb.DMatrix
            DMatrix containing N samples.
        """
        p = self.get_features(p) # (num_samples, num_categories)
        dtest = xgb.DMatrix(p)
        return dtest

    def predict(self, p):
        """Predict confidence score using the XGBoost model.

        Parameters
        ----------
        p : np.ndarray
            Array of shape (N, C) with class probabilities.

        Returns
        -------
        c : np.ndarray
            Array of shape (N,).
        """
        dtest = self.preprocess(p)
        return self.model.predict(dtest)


class XGBEnsemble():
    def __init__(self, use_entropy=True):
        """Initialize an ensemble of XGBoost models across folds.

        Parameters
        ----------
        use_entropy : bool
            Whether all models use entropy-augmented features.
        """
        models = []
        for fold in range(5):
            models.append(XGB(fold=fold, use_entropy=use_entropy))
        self.models = models

    def predict(self, p):
        """Predict confidence score for each ensemble member.

        Parameters
        ----------
        p : np.ndarray
            Array of shape (M, N, C), where M=5 is the number of models.

        Returns
        -------
        c : np.ndarray
            Array of shape (M, N)
        """
        return np.stack([m.predict(p[i]) for i, m in enumerate(self.models)], axis=0)
