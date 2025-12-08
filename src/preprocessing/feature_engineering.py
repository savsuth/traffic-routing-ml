"""
Feature Engineering for Traffic Prediction
Prepares features for ML model training.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Prepare features for traffic prediction."""

    def __init__(self, input_file: str, output_dir: str = "data/processed"):
        self.input_file = Path(input_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self, sample_frac: float = 0.1) -> pd.DataFrame:
        """Load data (use sampling for faster development)."""
        logger.info(f"Loading data from {self.input_file}...")


        df = pd.read_csv(self.input_file)

        if sample_frac < 1.0:
            df = df.sample(frac=sample_frac, random_state=42)
            logger.info(f"Sampled {sample_frac * 100}% of data: {len(df):,} records")
        else:
            logger.info(f"Loaded {len(df):,} records")

        return df

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create additional features."""
        logger.info("Creating features...")

        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Cyclical encoding for hour (so hour 23 and 0 are close)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

        # Cyclical encoding for day of week
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

        # Month
        df['month'] = df['timestamp'].dt.month

        # Time of day categories
        df['is_morning_rush'] = ((df['hour'] >= 7) & (df['hour'] <= 9)).astype(int)
        df['is_evening_rush'] = ((df['hour'] >= 16) & (df['hour'] <= 19)).astype(int)
        df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)

        # Weather encoding
        df['is_rainy'] = (df['weather'] == 'rainy').astype(int)
        df['is_snowy'] = (df['weather'] == 'snowy').astype(int)

        # Road type encoding
        road_hierarchy = {
            'motorway': 5,
            'trunk': 4,
            'primary': 3,
            'secondary': 2,
            'tertiary': 1,
            'residential': 0,
            'service': 0,
            'unclassified': 1
        }
        df['road_hierarchy'] = df['road_type'].map(road_hierarchy).fillna(1)

        # Distance categories
        df['is_downtown'] = (df['distance_to_downtown'] < 2000).astype(int)
        df['is_suburbs'] = (df['distance_to_downtown'] > 5000).astype(int)

        logger.info(f"✓ Created features. Shape: {df.shape}")
        return df

    def prepare_for_training(self, df: pd.DataFrame):
        """Split data and prepare for model training."""
        logger.info("Preparing train/val/test splits...")

        # Features to use
        feature_cols = [
            'hour', 'hour_sin', 'hour_cos',
            'day_of_week', 'day_sin', 'day_cos',
            'month', 'is_weekend',
            'is_morning_rush', 'is_evening_rush', 'is_night', 'is_rush_hour',
            'temperature', 'is_rainy', 'is_snowy',
            'event_nearby', 'road_hierarchy', 'speed_limit',
            'distance_to_downtown', 'is_downtown', 'is_suburbs'
        ]

        target_col = 'travel_time_sec'

        # Ensure all features exist
        missing_cols = [col for col in feature_cols if col not in df.columns]
        if missing_cols:
            logger.warning(f"Missing columns: {missing_cols}")
            feature_cols = [col for col in feature_cols if col in df.columns]

        X = df[feature_cols]
        y = df[target_col]

        # Time-based split (more realistic than random split)
        df_sorted = df.sort_values('timestamp')
        n = len(df_sorted)

        train_end = int(n * 0.7)
        val_end = int(n * 0.85)

        train_idx = df_sorted.index[:train_end]
        val_idx = df_sorted.index[train_end:val_end]
        test_idx = df_sorted.index[val_end:]

        X_train, y_train = X.loc[train_idx], y.loc[train_idx]
        X_val, y_val = X.loc[val_idx], y.loc[val_idx]
        X_test, y_test = X.loc[test_idx], y.loc[test_idx]

        logger.info(f"Train: {len(X_train):,} samples")
        logger.info(f"Val:   {len(X_val):,} samples")
        logger.info(f"Test:  {len(X_test):,} samples")

        # Add road_type for model splitting
        train_data = pd.concat([X_train, y_train, df.loc[train_idx, 'road_type']], axis=1)
        val_data = pd.concat([X_val, y_val, df.loc[val_idx, 'road_type']], axis=1)
        test_data = pd.concat([X_test, y_test, df.loc[test_idx, 'road_type']], axis=1)

        return train_data, val_data, test_data, feature_cols

    def save_processed_data(self, train_data, val_data, test_data, feature_cols):
        """Save processed datasets."""
        logger.info("Saving processed data...")

        train_data.to_csv(self.output_dir / "train_features.csv", index=False)
        val_data.to_csv(self.output_dir / "val_features.csv", index=False)
        test_data.to_csv(self.output_dir / "test_features.csv", index=False)

        # Save feature list
        with open(self.output_dir / "feature_columns.txt", 'w') as f:
            f.write('\n'.join(feature_cols))

        logger.info("✓ Saved train/val/test datasets")


def main():
    """Run feature engineering pipeline."""

    input_file = "data/processed/traffic_training_data.csv"

    engineer = FeatureEngineer(input_file)

    # Loading data (sample 20% for faster processing during development)
    df = engineer.load_data(sample_frac=0.2)

    # Create features
    df = engineer.create_features(df)

    # Prepare splits
    train_data, val_data, test_data, feature_cols = engineer.prepare_for_training(df)

    # Save
    engineer.save_processed_data(train_data, val_data, test_data, feature_cols)

    print("\n=== Feature Engineering Complete ===")
    print(f"Features created: {len(feature_cols)}")
    print(f"Train size: {len(train_data):,}")
    print(f"Val size: {len(val_data):,}")
    print(f"Test size: {len(test_data):,}")
    print("\n✓ Ready for Phase 4: Model Training!")


if __name__ == "__main__":
    main()