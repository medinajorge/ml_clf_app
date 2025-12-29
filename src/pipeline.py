import numpy as np
import pandas as pd
from scipy.stats import mode as ss_mode
import os

from . import preprocessing, clf, confidence, params


class SpeciesClassifierPipeline():
    def __init__(self, config: dict):
        self.config = config
        self.clf_model = clf.InceptionClassifierEnsemble()
        self.conf_model = confidence.XGBEnsemble(use_entropy=config['use_entropy'])

        self.c_min = config['c_min']
        self.ensemble_threshold = config['ensemble_threshold']
        self.overwrite = config['overwrite']

    def compute_ensemble_pred(self, y_pred, valid):
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
