"""
Data loading utilities for Smart Biochar Advisor
Handles loading training data, formulations, and preparing data for ML models
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import sys

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent))

from models.data_schema import Formulation, STANDARD_FORMULATIONS


class DataLoader:
    """Handles loading and preprocessing of training data."""

    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize DataLoader.

        Args:
            data_dir: Path to data directory. Defaults to code/data/samples
        """
        if data_dir is None:
            self.data_dir = Path(__file__).parent.parent / "data" / "samples"
        else:
            self.data_dir = Path(data_dir)

        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

    def load_formulations(self) -> pd.DataFrame:
        """
        Load standard formulations (S1-S4).

        Returns:
            DataFrame with formulation data
        """
        formulations_file = self.data_dir / "formulations.csv"

        if not formulations_file.exists():
            raise FileNotFoundError(f"Formulations file not found: {formulations_file}")

        df = pd.read_csv(formulations_file)
        return df

    def load_training_data(self) -> pd.DataFrame:
        """
        Load training dataset.

        Returns:
            DataFrame with training data
        """
        training_file = self.data_dir / "training_data.csv"

        if not training_file.exists():
            raise FileNotFoundError(f"Training data file not found: {training_file}")

        df = pd.read_csv(training_file)
        return df

    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Prepare feature matrix for ML models.

        Args:
            df: DataFrame with training data

        Returns:
            Tuple of (features DataFrame, feature names list)
        """
        feature_columns = ['ph', 'ec', 'temperature', 'humidity', 'biochar_g', 'npk_g']

        # One-hot encode soil type
        soil_type_dummies = pd.get_dummies(df['soil_type'], prefix='soil')

        # Combine features
        features = pd.concat([
            df[feature_columns],
            soil_type_dummies
        ], axis=1)

        feature_names = list(features.columns)

        return features, feature_names

    def prepare_targets(self, df: pd.DataFrame, target: str = 'plant_growth_index') -> pd.Series:
        """
        Extract target variable for ML models.

        Args:
            df: DataFrame with training data
            target: Target column name

        Returns:
            Series with target values
        """
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in data")

        return df[target]

    def split_data(self, X: pd.DataFrame, y: pd.Series,
                   test_size: float = 0.2,
                   random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into training and test sets.

        Args:
            X: Features DataFrame
            y: Target Series
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state
        )

        return X_train, X_test, y_train, y_test

    def get_feature_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Calculate statistics for features.

        Args:
            df: DataFrame with data

        Returns:
            Dictionary with statistics
        """
        stats = {
            'count': len(df),
            'soil_types': df['soil_type'].value_counts().to_dict(),
            'ph': {
                'min': df['ph'].min(),
                'max': df['ph'].max(),
                'mean': df['ph'].mean(),
                'std': df['ph'].std()
            },
            'ec': {
                'min': df['ec'].min(),
                'max': df['ec'].max(),
                'mean': df['ec'].mean(),
                'std': df['ec'].std()
            },
            'biochar': {
                'min': df['biochar_g'].min(),
                'max': df['biochar_g'].max(),
                'mean': df['biochar_g'].mean(),
                'std': df['biochar_g'].std()
            },
            'npk': {
                'min': df['npk_g'].min(),
                'max': df['npk_g'].max(),
                'mean': df['npk_g'].mean(),
                'std': df['npk_g'].std()
            }
        }

        if 'plant_growth_index' in df.columns:
            stats['pgi'] = {
                'min': df['plant_growth_index'].min(),
                'max': df['plant_growth_index'].max(),
                'mean': df['plant_growth_index'].mean(),
                'std': df['plant_growth_index'].std()
            }

        return stats


def load_all_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience function to load all data.

    Returns:
        Tuple of (training_data, formulations)
    """
    loader = DataLoader()
    training_data = loader.load_training_data()
    formulations = loader.load_formulations()

    return training_data, formulations


def prepare_soil_sample_features(soil_type: str, ph: float, ec: float,
                                 temperature: float, humidity: float,
                                 biochar_g: float, npk_g: float,
                                 feature_names: List[str]) -> np.ndarray:
    """
    Prepare feature vector for a single soil sample.

    Args:
        soil_type: Type of soil
        ph: pH value
        ec: EC value
        temperature: Temperature in Celsius
        humidity: Humidity percentage
        biochar_g: Biochar amount
        npk_g: NPK amount
        feature_names: List of expected feature names

    Returns:
        NumPy array with features in correct order
    """
    # Create feature dictionary
    features = {
        'ph': ph,
        'ec': ec,
        'temperature': temperature,
        'humidity': humidity,
        'biochar_g': biochar_g,
        'npk_g': npk_g,
        'soil_sandy': 1 if soil_type == 'sandy' else 0,
        'soil_clay': 1 if soil_type == 'clay' else 0,
        'soil_mixed': 1 if soil_type == 'mixed' else 0,
    }

    # Create feature array in correct order
    feature_array = np.array([features.get(name, 0) for name in feature_names])

    return feature_array.reshape(1, -1)


if __name__ == "__main__":
    # Test data loading
    print("📦 Testing Data Loader...")

    try:
        loader = DataLoader()

        print("\n✅ Loading formulations...")
        formulations = loader.load_formulations()
        print(f"   Loaded {len(formulations)} formulations")
        print(formulations.to_string(index=False))

        print("\n✅ Loading training data...")
        training_data = loader.load_training_data()
        print(f"   Loaded {len(training_data)} samples")

        print("\n✅ Preparing features...")
        X, feature_names = loader.prepare_features(training_data)
        y = loader.prepare_targets(training_data)
        print(f"   Features shape: {X.shape}")
        print(f"   Feature names: {feature_names}")
        print(f"   Target shape: {y.shape}")

        print("\n✅ Data statistics:")
        stats = loader.get_feature_statistics(training_data)
        print(f"   Total samples: {stats['count']}")
        print(f"   Soil types: {stats['soil_types']}")
        print(f"   PGI range: {stats['pgi']['min']:.2f} - {stats['pgi']['max']:.2f}")
        print(f"   PGI mean: {stats['pgi']['mean']:.2f}")

        print("\n✅ All tests passed!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
