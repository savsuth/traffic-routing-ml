"""
ML Model Training
Train LightGBM models to predict traffic travel times.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
import pickle
import json
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrafficPredictor:
    """Train and manage traffic prediction models."""

    def __init__(self, data_dir: str = "data/processed", model_dir: str = "data/models"):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.models = {}

    def load_data(self):
        """Load preprocessed training data."""
        logger.info("Loading training data...")

        self.train_df = pd.read_csv(self.data_dir / "train_features.csv")
        self.val_df = pd.read_csv(self.data_dir / "val_features.csv")
        self.test_df = pd.read_csv(self.data_dir / "test_features.csv")

        # Load feature columns
        with open(self.data_dir / "feature_columns.txt", 'r') as f:
            self.feature_cols = [line.strip() for line in f.readlines()]

        logger.info(f"Train: {len(self.train_df):,} samples")
        logger.info(f"Val: {len(self.val_df):,} samples")
        logger.info(f"Test: {len(self.test_df):,} samples")
        logger.info(f"Features: {len(self.feature_cols)}")

    def train_all_models(self):
        """Train models for all major road types."""

        # Group road types
        road_groups = {
            'highway': ['motorway', 'trunk'],
            'major': ['primary', 'secondary'],
            'minor': ['tertiary', 'residential', 'unclassified']
        }

        results = {}

        for group_name, road_types in road_groups.items():
            # Combine road types in this group
            train_group = self.train_df[self.train_df['road_type'].isin(road_types)]
            val_group = self.val_df[self.val_df['road_type'].isin(road_types)]

            if len(train_group) < 100:
                continue

            logger.info(f"\nTraining model for: {group_name} roads")
            logger.info(f"Road types: {road_types}")

            X_train = train_group[self.feature_cols]
            y_train = train_group['travel_time_sec']
            X_val = val_group[self.feature_cols]
            y_val = val_group['travel_time_sec']

            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

            params = {
                'objective': 'regression',
                'metric': 'mae',
                'boosting_type': 'gbdt',
                'num_leaves': 100,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'max_depth': 10,
            }

            logger.info(f"Training on {len(X_train):,} samples...")

            model = lgb.train(
                params,
                train_data,
                num_boost_round=500,
                valid_sets=[train_data, val_data],
                valid_names=['train', 'val'],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50, verbose=False),
                    lgb.log_evaluation(period=100)
                ]
            )

            # Evaluate
            y_pred = model.predict(X_val)
            mae = mean_absolute_error(y_val, y_pred)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            r2 = r2_score(y_val, y_pred)

            logger.info(f"Best iteration: {model.best_iteration}")
            logger.info(f"Val MAE: {mae:.2f} seconds")
            logger.info(f"Val RMSE: {rmse:.2f} seconds")
            logger.info(f"Val R2: {r2:.4f}")

            # Save model
            model_path = self.model_dir / f"model_{group_name}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            logger.info(f"Saved to {model_path}")

            self.models[group_name] = model

            results[group_name] = {
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'n_samples': len(train_group),
                'road_types': road_types
            }

            # Plot feature importance
            self._plot_feature_importance(model, group_name)

        return results

    def _plot_feature_importance(self, model, model_name: str):
        """Plot and save feature importance."""
        importance = model.feature_importance()
        features = self.feature_cols

        # Sort by importance
        indices = np.argsort(importance)[::-1][:15]  # Top 15

        plt.figure(figsize=(10, 6))
        plt.barh(range(len(indices)), importance[indices])
        plt.yticks(range(len(indices)), [features[i] for i in indices])
        plt.xlabel('Feature Importance')
        plt.title(f'Top 15 Features - {model_name}')
        plt.tight_layout()

        plot_path = self.model_dir / f"feature_importance_{model_name}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Feature importance plot saved to {plot_path}")

    def evaluate_on_test(self):
        """Final evaluation on test set."""
        logger.info("\nFINAL TEST SET EVALUATION")

        all_preds = []
        all_actuals = []

        for group_name, model in self.models.items():
            # Get road types for this group
            if group_name == 'highway':
                road_types = ['motorway', 'trunk']
            elif group_name == 'major':
                road_types = ['primary', 'secondary']
            else:
                road_types = ['tertiary', 'residential', 'unclassified']

            test_group = self.test_df[self.test_df['road_type'].isin(road_types)]

            if len(test_group) == 0:
                continue

            X_test = test_group[self.feature_cols]
            y_test = test_group['travel_time_sec']

            y_pred = model.predict(X_test)

            all_preds.extend(y_pred)
            all_actuals.extend(y_test)

            mae = mean_absolute_error(y_test, y_pred)
            logger.info(f"{group_name}: MAE = {mae:.2f} seconds")

        # Overall metrics
        overall_mae = mean_absolute_error(all_actuals, all_preds)
        overall_rmse = np.sqrt(mean_squared_error(all_actuals, all_preds))
        overall_r2 = r2_score(all_actuals, all_preds)

        logger.info(f"\nOVERALL TEST PERFORMANCE")
        logger.info(f"MAE:  {overall_mae:.2f} seconds")
        logger.info(f"RMSE: {overall_rmse:.2f} seconds")
        logger.info(f"R2:   {overall_r2:.4f}")

        return {
            'mae': overall_mae,
            'rmse': overall_rmse,
            'r2': overall_r2
        }

    def save_report(self, train_results, test_results):
        """Save training report."""
        report = {
            'training_results': train_results,
            'test_results': test_results,
            'feature_columns': self.feature_cols
        }

        report_path = self.model_dir / "training_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Training report saved to {report_path}")


def main():
    """Train traffic prediction models."""

    predictor = TrafficPredictor()

    # Load data
    predictor.load_data()

    # Train models
    train_results = predictor.train_all_models()

    # Test evaluation
    test_results = predictor.evaluate_on_test()

    # Save report
    predictor.save_report(train_results, test_results)

    print("\nMODEL TRAINING COMPLETE")
    print(f"\nTest Set Performance:")
    print(f"  MAE:  {test_results['mae']:.2f} seconds")
    print(f"  RMSE: {test_results['rmse']:.2f} seconds")
    print(f"  R2:   {test_results['r2']:.4f}")
    print(f"\nModels saved to: data/models/")
    print("\nReady for Phase 5: A* Implementation!")


if __name__ == "__main__":
    main()