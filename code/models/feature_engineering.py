"""
Feature Engineering for Smart Biochar Advisor
Handles feature preparation, encoding, and scaling for ML models
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, List, Dict, Optional
import joblib
from pathlib import Path


class FeatureEngineer:
    """Handles feature engineering for soil and amendment data."""

    def __init__(self):
        """Initialize feature engineer."""
        self.scaler = StandardScaler()
        self.soil_type_encoder = LabelEncoder()
        self.feature_names = []
        self.is_fitted = False

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create engineered features from raw data.

        Args:
            df: DataFrame with raw soil and amendment data

        Returns:
            DataFrame with engineered features
        """
        features_df = df.copy()

        # Interaction features
        if 'biochar_g' in df.columns and 'npk_g' in df.columns:
            # Total amendment
            features_df['total_amendment'] = df['biochar_g'] + df['npk_g']

            # Biochar to NPK ratio (avoid division by zero)
            features_df['biochar_npk_ratio'] = df['biochar_g'] / (df['npk_g'] + 0.1)

            # Amendment intensity (total per unit soil)
            features_df['amendment_intensity'] = features_df['total_amendment'] / 100.0

        # pH categories
        if 'ph' in df.columns:
            features_df['ph_acidic'] = (df['ph'] < 6.0).astype(int)
            features_df['ph_neutral'] = ((df['ph'] >= 6.0) & (df['ph'] <= 7.5)).astype(int)
            features_df['ph_alkaline'] = (df['ph'] > 7.5).astype(int)

        # EC/Salinity categories
        if 'ec' in df.columns:
            features_df['ec_low'] = (df['ec'] < 2.0).astype(int)
            features_df['ec_moderate'] = ((df['ec'] >= 2.0) & (df['ec'] < 4.0)).astype(int)
            features_df['ec_high'] = (df['ec'] >= 4.0).astype(int)

        # Temperature categories
        if 'temperature' in df.columns:
            features_df['temp_optimal'] = ((df['temperature'] >= 20) & (df['temperature'] <= 30)).astype(int)
            features_df['temp_low'] = (df['temperature'] < 20).astype(int)
            features_df['temp_high'] = (df['temperature'] > 30).astype(int)

        # Humidity categories
        if 'humidity' in df.columns:
            features_df['humidity_optimal'] = ((df['humidity'] >= 40) & (df['humidity'] <= 70)).astype(int)
            features_df['humidity_low'] = (df['humidity'] < 40).astype(int)
            features_df['humidity_high'] = (df['humidity'] > 70).astype(int)

        # Soil type interactions with amendments
        if 'soil_type' in df.columns and 'biochar_g' in df.columns:
            # Sandy soil benefits more from biochar
            features_df['sandy_biochar_interaction'] = (
                (df['soil_type'] == 'sandy').astype(int) * df['biochar_g']
            )

            # Clay soil water retention
            features_df['clay_humidity_interaction'] = (
                (df['soil_type'] == 'clay').astype(int) * df.get('humidity', 50)
            )

        # Biochar salinity mitigation
        if 'biochar_g' in df.columns and 'ec' in df.columns:
            features_df['biochar_salinity_mitigation'] = df['biochar_g'] * (df['ec'] > 2.0).astype(int)

        return features_df

    def prepare_features_for_training(self, df: pd.DataFrame,
                                      include_target: bool = False) -> Tuple[pd.DataFrame, List[str]]:
        """
        Prepare features for model training.

        Args:
            df: DataFrame with raw data
            include_target: Whether to include target columns

        Returns:
            Tuple of (features DataFrame, feature names)
        """
        # Create engineered features
        features_df = self.create_features(df)

        # Base features
        base_features = ['ph', 'ec', 'temperature', 'humidity', 'biochar_g', 'npk_g']

        # One-hot encode soil type
        if 'soil_type' in features_df.columns:
            soil_dummies = pd.get_dummies(features_df['soil_type'], prefix='soil')
            features_df = pd.concat([features_df, soil_dummies], axis=1)

        # Engineered features
        engineered_features = [
            'total_amendment', 'biochar_npk_ratio', 'amendment_intensity',
            'ph_acidic', 'ph_neutral', 'ph_alkaline',
            'ec_low', 'ec_moderate', 'ec_high',
            'temp_optimal', 'temp_low', 'temp_high',
            'humidity_optimal', 'humidity_low', 'humidity_high',
            'sandy_biochar_interaction', 'clay_humidity_interaction',
            'biochar_salinity_mitigation'
        ]

        # Collect all feature columns
        feature_cols = base_features.copy()

        # Add soil type dummies
        soil_cols = [col for col in features_df.columns if col.startswith('soil_')]
        feature_cols.extend(soil_cols)

        # Add engineered features that exist
        for feat in engineered_features:
            if feat in features_df.columns:
                feature_cols.append(feat)

        # Select only feature columns
        X = features_df[feature_cols].copy()

        # Handle any missing values
        X = X.fillna(0)

        # Ensure all columns are numeric (drop any non-numeric columns)
        X = X.select_dtypes(include=[np.number])

        self.feature_names = list(X.columns)

        return X, list(X.columns)

    def fit(self, X: pd.DataFrame) -> 'FeatureEngineer':
        """
        Fit the scaler on training data.

        Args:
            X: Feature DataFrame

        Returns:
            Self for chaining
        """
        self.scaler.fit(X)
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform features using fitted scaler.

        Args:
            X: Feature DataFrame

        Returns:
            Scaled feature array
        """
        if not self.is_fitted:
            raise ValueError("FeatureEngineer must be fitted before transform")

        return self.scaler.transform(X)

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Fit and transform in one step.

        Args:
            X: Feature DataFrame

        Returns:
            Scaled feature array
        """
        return self.fit(X).transform(X)

    def inverse_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        """
        Inverse transform scaled features.

        Args:
            X_scaled: Scaled feature array

        Returns:
            Original scale features
        """
        return self.scaler.inverse_transform(X_scaled)

    def save(self, filepath: Path):
        """
        Save feature engineer to disk.

        Args:
            filepath: Path to save file
        """
        joblib.dump({
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted
        }, filepath)

    @classmethod
    def load(cls, filepath: Path) -> 'FeatureEngineer':
        """
        Load feature engineer from disk.

        Args:
            filepath: Path to saved file

        Returns:
            Loaded FeatureEngineer
        """
        data = joblib.load(filepath)
        engineer = cls()
        engineer.scaler = data['scaler']
        engineer.feature_names = data['feature_names']
        engineer.is_fitted = data['is_fitted']
        return engineer


def prepare_single_sample(soil_type: str, ph: float, ec: float,
                          temperature: float, humidity: float,
                          biochar_g: float, npk_g: float,
                          feature_names: List[str]) -> pd.DataFrame:
    """
    Prepare a single soil sample for prediction.

    Args:
        soil_type: Type of soil
        ph: pH value
        ec: EC value
        temperature: Temperature
        humidity: Humidity percentage
        biochar_g: Biochar amount
        npk_g: NPK amount
        feature_names: Expected feature names

    Returns:
        DataFrame with features matching training format
    """
    # Create base data
    data = {
        'soil_type': soil_type,
        'ph': ph,
        'ec': ec,
        'temperature': temperature,
        'humidity': humidity,
        'biochar_g': biochar_g,
        'npk_g': npk_g
    }

    df = pd.DataFrame([data])

    # Create features
    engineer = FeatureEngineer()
    features_df = engineer.create_features(df)

    # One-hot encode soil type
    soil_dummies = pd.get_dummies(features_df['soil_type'], prefix='soil')
    features_df = pd.concat([features_df, soil_dummies], axis=1)

    # Ensure all expected features are present
    for feat in feature_names:
        if feat not in features_df.columns:
            features_df[feat] = 0

    # Select features in correct order
    return features_df[feature_names]


if __name__ == "__main__":
    # Test feature engineering
    print("🔧 Testing Feature Engineering...")

    # Create sample data
    sample_data = pd.DataFrame([
        {
            'soil_type': 'sandy',
            'ph': 6.5,
            'ec': 1.8,
            'temperature': 25,
            'humidity': 60,
            'biochar_g': 10,
            'npk_g': 10
        },
        {
            'soil_type': 'clay',
            'ph': 7.0,
            'ec': 2.5,
            'temperature': 22,
            'humidity': 70,
            'biochar_g': 15,
            'npk_g': 10
        }
    ])

    engineer = FeatureEngineer()
    X, feature_names = engineer.prepare_features_for_training(sample_data)

    print(f"\n✅ Created {len(feature_names)} features:")
    for i, name in enumerate(feature_names, 1):
        print(f"   {i}. {name}")

    print(f"\n✅ Feature matrix shape: {X.shape}")
    print("\n✅ Sample features:")
    print(X.head())

    print("\n✅ Feature engineering test passed!")
