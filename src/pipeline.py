import numpy as np
import pandas as pd
from scipy.stats import mode as ss_mode
import os

from . import preprocessing, clf, confidence, params


class SpeciesClassifierPipeline():
    """
    End-to-end pipeline for species classification with confidence-based
    abstention.

    This pipeline performs the following steps:
    1. Preprocesses an input CSV file into model-ready features and metadata.
    2. Runs an ensemble of Inception-based classifiers to predict species
       categories across multiple folds.
    3. Uses an XGBoost-based confidence model to estimate prediction confidence
       for each fold.
    4. Computes an ensemble prediction by majority vote, considering only
       predictions whose confidence exceeds a configurable minimum threshold.
    5. Abstains from making a final prediction when insufficient confident
       votes are available.
    6. Saves the full set of per-fold predictions, confidence scores, and final
       ensemble outputs to a CSV file.

    Parameters
    ----------
    config : dict
        Configuration dictionary controlling pipeline behavior. Expected keys:
        - 'use_entropy' (bool): Whether to include entropy-based features in the
          confidence model.
        - 'c_min' (float): Minimum confidence threshold required for a fold's
          prediction to be considered valid.
        - 'ensemble_threshold' (float): Fraction of valid folds required to
          produce an ensemble prediction; otherwise the sample is marked as
          abstained.
        - 'overwrite' (bool): Whether to overwrite an existing output CSV file.

    Attributes
    ----------
    clf_model : InceptionClassifierEnsemble
        Ensemble classifier used to predict species labels.
    conf_model : XGBEnsemble
        Ensemble confidence model producing per-fold confidence scores.
    c_min : float
        Minimum confidence threshold for valid predictions.
    ensemble_threshold : float
        Threshold determining whether an ensemble prediction is accepted.
    overwrite : bool
        Whether existing output files should be overwritten.

    Methods
    -------
    compute_ensemble_pred(y_pred, valid)
        Computes the ensemble species prediction using confidence-filtered
        majority voting and determines abstentions.

    process_csv(input_path)
        Runs the full pipeline on an input CSV file and returns the output
        DataFrame containing predictions, confidence scores, and metadata.
    """
    def __init__(self, config: dict):
        self.config = config
        self.clf_model = clf.InceptionClassifierEnsemble()
        self.conf_model = confidence.XGBEnsemble(use_entropy=config['use_entropy'])

        self.c_min = config['c_min']
        self.ensemble_threshold = config['ensemble_threshold']
        self.overwrite = config['overwrite']

    def compute_ensemble_pred(self, y_pred, valid):
        """
        Compute ensemble species predictions using confidence-filtered majority voting.

        For each sample, predictions from multiple folds are considered valid only if
        their confidence scores meet or exceed the minimum confidence threshold.
        The ensemble prediction is obtained by taking the statistical mode across
        valid predictions. If all predictions for a sample are invalid, the method
        falls back to an unfiltered majority vote across all folds.

        An abstention flag is produced when the proportion of valid predictions does
        not exceed the ensemble threshold.

        Parameters
        ----------
        y_pred : np.ndarray
            Array of shape (n_folds, n_samples) containing predicted class indices
            from each fold.
        valid : np.ndarray of bool
            Boolean array of shape (n_folds, n_samples) indicating whether each
            fold prediction is considered valid based on confidence.

        Returns
        -------
        y_pred_ensemble : np.ndarray
            Array of shape (n_samples,) containing final ensemble class predictions.
        abstained_ensemble : np.ndarray of bool
            Boolean array of shape (n_samples,) indicating whether the ensemble
            abstained from making a prediction.
        """
        valid_ensemble = valid.mean(axis=0) > self.ensemble_threshold
        y_pred_ensemble = y_pred.copy().astype(np.float64)
        y_pred_ensemble[~valid] = np.nan
        y_pred_ensemble = ss_mode(y_pred_ensemble, axis=0, nan_policy='omit').mode

        nan_ensemble = np.isnan(y_pred_ensemble)
        if nan_ensemble.any(): # fallback to prediction without confidence callibration
            y_pred_ensemble_raw = ss_mode(y_pred, axis=0).mode
            y_pred_ensemble[nan_ensemble] = y_pred_ensemble_raw[nan_ensemble]
        y_pred_ensemble = y_pred_ensemble.astype(np.int64)

        abstained_ensemble = ~valid_ensemble
        return y_pred_ensemble, abstained_ensemble

    def process_csv(self, input_path: str):
        """
        Run the full species classification pipeline on an input CSV file.

        This method loads and preprocesses the input data, generates per-fold species
        predictions and confidence scores, computes confidence-aware ensemble
        predictions with abstention handling, and saves the results to an output CSV
        file. If an output file already exists and overwriting is disabled, the
        existing file is loaded instead of recomputing predictions.

        The output CSV includes:
        - Original metadata
        - Per-fold species predictions
        - Per-fold confidence scores
        - Final ensemble species prediction
        - Abstention indicator

        Parameters
        ----------
        input_path : str
            Path to the input CSV file to be processed.

        Returns
        -------
        output : pandas.DataFrame
            DataFrame containing metadata, per-fold predictions, confidence scores,
            and final ensemble predictions.
        """
        output_path = input_path.replace('.csv', '_output.csv')
        if os.path.exists(output_path) and not self.overwrite:
            print(f"Output file {output_path} already exists. Loading...")
            output = pd.read_csv(output_path)
        else:
            X, metadata = preprocessing.preprocess(input_path)

            print("Predicting species...")
            y, p = self.clf_model.predict(X)
            df_y = pd.DataFrame(y.T, index=metadata.index, columns=[f'species_predicted_fold_{i+1}' for i in range(5)])
            df_y = df_y.replace(params.CATEGORY_TO_SPECIES)

            print("Predicting confidence scores...")
            c = self.conf_model.predict(p) # (5, N)
            df_c = pd.DataFrame(c.T, index=metadata.index, columns=[f'confidence_fold_{i+1}' for i in range(5)])

            # Record predictions for specified minimum confidence
            print(f"Computing ensemble predictions for minimum confidence: c_min={self.c_min:.4f}")
            valid = c >= self.c_min
            y_ensemble, abstained_ensemble = self.compute_ensemble_pred(y, valid)
            y_ensemble_labels = pd.Series(y_ensemble).map(params.CATEGORY_TO_SPECIES).values
            df_ensemble = pd.DataFrame({'species_predicted': y_ensemble_labels,
                                        'abstained': abstained_ensemble})

            output = pd.concat([metadata, df_y, df_c, df_ensemble], axis=1)

            print(f"Saving output to {output_path}...")
            output.to_csv(output_path, index=False)
        return output
