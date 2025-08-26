from dataclasses import asdict
from typing import Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd

from dto.enums.model_kind import ModelKind
from dto.model_meta import ModelMeta
from service.ml.data_pair_builder import DataPairBuilder
from service.ml.metrics_calculator import MetricsCalculator
from service.ml.model_fitter import ModelFitter
from service.ml.model_predictor import ModelPredictor


class ContextualModelTrainer:
    """Trains and applies contextual heart rate calibration models."""

    MIN_SAMPLES_PER_CONTEXT = 30

    def __init__(self, model_kind: str = ModelKind.PCHIP.value, min_scan_coverage_s: int = 30, local_tz: str = "Europe/Rome"):
        self.model_kind = model_kind
        self.min_scan_coverage_s = min_scan_coverage_s
        self.local_tz = local_tz
        self.model_fitter = ModelFitter()
        self.predictor = ModelPredictor()
        self.metrics_calc = MetricsCalculator()

    def train_and_apply(self, scan_csv: str, sqlite_path: str, user_id: Optional[int],
                        bin_size: str = "1min") -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Train contextual models and apply them to the dataset."""
        # Build paired dataset
        pairs = DataPairBuilder.build_pairs(
            scan_csv, sqlite_path, user_id, bin_size, self.min_scan_coverage_s, self.local_tz
        )

        if pairs.empty:
            raise RuntimeError("No paired minutes after filtering.")

        # Split by context
        rest_data, active_data = self._split_by_context(pairs)

        # Train context-specific models
        results = {}
        context_models = {}

        rest_model = self._train_context_model(rest_data, 'rest', results)
        if rest_model:
            context_models['rest'] = rest_model

        active_model = self._train_context_model(active_data, 'active', results)
        if active_model:
            context_models['active'] = active_model

        # Apply models and generate predictions
        pairs = self._apply_contextual_predictions(pairs, context_models, results)

        # Prepare export data
        export_data = self._prepare_export_data(context_models, results)

        return pairs, export_data

    def _split_by_context(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into rest and active contexts."""
        rest_data = df[df['sleep_status'].fillna(0) > 0].copy()
        active_data = df[df['is_sport'] | (df['sleep_status'].fillna(0) == 0)].copy()
        return rest_data, active_data

    def _train_context_model(self, context_data: pd.DataFrame, context_name: str,
                             results: Dict[str, Any]) -> Optional[ModelMeta]:
        """Train a model for a specific context."""
        if len(context_data) < self.MIN_SAMPLES_PER_CONTEXT:
            return None

        x_values = context_data['scan_bpm'].to_numpy()
        y_values = context_data['t10_bpm'].to_numpy()

        # Fit model based on specified kind
        if self.model_kind == ModelKind.PCHIP.value:
            model_meta = self.model_fitter.fit_pchip_from_binned(x_values, y_values, context=context_name)
        elif self.model_kind == ModelKind.ISOTONIC.value:
            model_meta = self.model_fitter.fit_isotonic(x_values, y_values, context=context_name)
        else:
            raise ValueError(f"Unknown model kind: {self.model_kind}")

        # Generate predictions and calculate metrics
        y_predicted = self.predictor.apply_model(model_meta, x_values)
        context_data['y_hat'] = y_predicted

        # Store results
        results[f'{context_name}_metrics'] = self.metrics_calc.calculate_metrics(context_data).to_dict(orient='records')
        results[f'{context_name}_n'] = int(len(context_data))
        results[f'{context_name}_model'] = asdict(model_meta)

        return model_meta

    def _apply_contextual_predictions(self, pairs: pd.DataFrame,
                                      context_models: Dict[str, ModelMeta],
                                      results: Dict[str, Any]) -> pd.DataFrame:
        """Apply context-specific models to generate predictions."""
        if not context_models:
            # Fallback to global model
            return self._apply_global_fallback(pairs, results)

        def predict_for_row(row):
            """Predict heart rate for a single row based on context."""
            scan_bpm = row['scan_bpm']

            # Choose model based on context
            if ('rest' in context_models and
                    row['sleep_status'] and int(row['sleep_status']) > 0):
                return float(self.predictor.apply_model(context_models['rest'], np.array([scan_bpm]))[0])
            elif 'active' in context_models:
                return float(self.predictor.apply_model(context_models['active'], np.array([scan_bpm]))[0])
            elif 'rest' in context_models:
                # Fallback to rest model if active not available
                return float(self.predictor.apply_model(context_models['rest'], np.array([scan_bpm]))[0])
            else:
                raise ValueError("No suitable model available for prediction")

        pairs['y_hat'] = pairs.apply(predict_for_row, axis=1)
        pairs['calibrated_bpm'] = pairs['y_hat']

        # Calculate overall metrics
        results['all_metrics'] = self.metrics_calc.calculate_metrics(pairs).to_dict(orient='records')
        results['all_n'] = int(len(pairs))

        return pairs

    def _apply_global_fallback(self, pairs: pd.DataFrame,
                               results: Dict[str, Any]) -> pd.DataFrame:
        """Apply a single global model as fallback."""
        x_values = pairs['scan_bpm'].to_numpy()
        y_values = pairs['t10_bpm'].to_numpy()

        global_model = self.model_fitter.fit_pchip_from_binned(x_values, y_values, context='global')
        pairs['y_hat'] = self.predictor.apply_model(global_model, x_values)
        pairs['calibrated_bpm'] = pairs['y_hat']

        results['global_metrics'] = self.metrics_calc.calculate_metrics(pairs).to_dict(orient='records')
        results['global_n'] = int(len(pairs))
        results['global_model'] = asdict(global_model)

        return pairs

    def _prepare_export_data(self, context_models: Dict[str, ModelMeta],
                             results: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare the final export data structure."""
        chosen_models = {}
        for context, model in context_models.items():
            chosen_models[context] = asdict(model)

        return {
            'chosen_models': chosen_models,
            'metrics': results,
        }
