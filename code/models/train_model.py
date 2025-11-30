"""
Model Training for Smart Biochar Advisor
Trains ML models for PGI prediction and biochar/NPK recommendations
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
import sys

# ML imports
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent))

from models.feature_engineering import FeatureEngineer
from utils.data_loader import DataLoader


class BiocharAdvisorModel:
    """Main model for biochar and NPK recommendations."""

    def __init__(self, model_type='random_forest'):
        """
        Initialize the model.

        Args:
            model_type: Type of model ('random_forest', 'gradient_boost', 'ridge')
        """
        self.model_type = model_type
        self.model = self._create_model()
        self.feature_engineer = FeatureEngineer()
        self.is_trained = False
        self.feature_names = []
        self.metrics = {}

    def _create_model(self):
        """Create the underlying ML model."""
        if self.model_type == 'random_forest':
            return RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'gradient_boost':
            return GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        elif self.model_type == 'ridge':
            return Ridge(alpha=1.0, random_state=42)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def train(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
        """
        Train the model.

        Args:
            X: Feature DataFrame
            y: Target Series
            test_size: Proportion for test set
        """
        print(f"\n🚀 Training {self.model_type} model...")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        print(f"   Training samples: {len(X_train)}")
        print(f"   Test samples: {len(X_test)}")

        # Fit feature engineer and transform
        X_train_scaled = self.feature_engineer.fit_transform(X_train)
        X_test_scaled = self.feature_engineer.transform(X_test)

        # Train model
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        self.feature_names = list(X.columns)

        # Evaluate
        train_pred = self.model.predict(X_train_scaled)
        test_pred = self.model.predict(X_test_scaled)

        # Calculate metrics
        self.metrics = {
            'train': {
                'rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
                'mae': mean_absolute_error(y_train, train_pred),
                'r2': r2_score(y_train, train_pred)
            },
            'test': {
                'rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
                'mae': mean_absolute_error(y_test, test_pred),
                'r2': r2_score(y_test, test_pred)
            }
        }

        # Cross-validation
        cv_scores = cross_val_score(
            self.model, X_train_scaled, y_train,
            cv=5, scoring='r2', n_jobs=-1
        )
        self.metrics['cv_r2_mean'] = cv_scores.mean()
        self.metrics['cv_r2_std'] = cv_scores.std()

        print(f"\n📊 Model Performance:")
        print(f"   Train RMSE: {self.metrics['train']['rmse']:.3f}")
        print(f"   Train R²: {self.metrics['train']['r2']:.3f}")
        print(f"   Test RMSE: {self.metrics['test']['rmse']:.3f}")
        print(f"   Test R²: {self.metrics['test']['r2']:.3f}")
        print(f"   CV R² (mean ± std): {self.metrics['cv_r2_mean']:.3f} ± {self.metrics['cv_r2_std']:.3f}")

        return self.metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Feature DataFrame

        Returns:
            Predictions array
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        # Ensure features match training
        for feat in self.feature_names:
            if feat not in X.columns:
                X[feat] = 0

        X = X[self.feature_names]

        # Scale and predict
        X_scaled = self.feature_engineer.transform(X)
        return self.model.predict(X_scaled)

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance (for tree-based models).

        Returns:
            DataFrame with feature importance
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")

        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            return importance_df
        else:
            return None

    def save(self, filepath: Path):
        """
        Save model to disk.

        Args:
            filepath: Path to save file
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")

        # Save feature engineer
        fe_path = filepath.parent / f"{filepath.stem}_feature_engineer.pkl"
        self.feature_engineer.save(fe_path)

        # Save model and metadata
        joblib.dump({
            'model': self.model,
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'metrics': self.metrics,
            'is_trained': self.is_trained
        }, filepath)

        print(f"✅ Model saved to {filepath}")

    @classmethod
    def load(cls, filepath: Path) -> 'BiocharAdvisorModel':
        """
        Load model from disk.

        Args:
            filepath: Path to saved model

        Returns:
            Loaded model
        """
        data = joblib.load(filepath)

        # Create instance
        instance = cls(model_type=data['model_type'])
        instance.model = data['model']
        instance.feature_names = data['feature_names']
        instance.metrics = data['metrics']
        instance.is_trained = data['is_trained']

        # Load feature engineer
        fe_path = filepath.parent / f"{filepath.stem}_feature_engineer.pkl"
        instance.feature_engineer = FeatureEngineer.load(fe_path)

        return instance


def train_all_models():
    """Train and save all models."""
    print("=" * 60)
    print("🌱 SMART BIOCHAR ADVISOR - MODEL TRAINING")
    print("=" * 60)

    # Create models directory
    models_dir = Path(__file__).parent / "trained"
    models_dir.mkdir(exist_ok=True)

    # Load data
    print("\n📦 Loading training data...")
    loader = DataLoader()
    training_data = loader.load_training_data()

    print(f"   Loaded {len(training_data)} samples")
    print(f"   Features: {list(training_data.columns)}")

    # Prepare features
    print("\n🔧 Preparing features...")
    engineer = FeatureEngineer()
    X, feature_names = engineer.prepare_features_for_training(training_data)
    print(f"   Created {len(feature_names)} features")

    # Train PGI prediction model
    print("\n" + "=" * 60)
    print("📈 Training Plant Growth Index (PGI) Prediction Model")
    print("=" * 60)

    y_pgi = training_data['plant_growth_index']

    # Try different model types
    model_types = ['random_forest', 'gradient_boost', 'ridge']
    best_model = None
    best_score = -float('inf')

    results = {}

    for model_type in model_types:
        print(f"\n→ Testing {model_type}...")
        model = BiocharAdvisorModel(model_type=model_type)
        metrics = model.train(X, y_pgi)

        results[model_type] = metrics

        if metrics['test']['r2'] > best_score:
            best_score = metrics['test']['r2']
            best_model = model
            best_model_type = model_type

    print(f"\n🏆 Best model: {best_model_type} (R² = {best_score:.3f})")

    # Save best model
    model_path = models_dir / "pgi_model.pkl"
    best_model.save(model_path)

    # Feature importance
    print("\n📊 Feature Importance (Top 10):")
    importance = best_model.get_feature_importance()
    if importance is not None:
        print(importance.head(10).to_string(index=False))

    # Save training report
    report = {
        'model_type': best_model_type,
        'training_date': pd.Timestamp.now().isoformat(),
        'training_samples': len(training_data),
        'num_features': len(feature_names),
        'feature_names': feature_names,
        'metrics': best_model.metrics,
        'all_results': results
    }

    report_path = models_dir / "training_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n✅ Training report saved to {report_path}")

    print("\n" + "=" * 60)
    print("✅ MODEL TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\nTrained models saved in: {models_dir}")
    print(f"   - pgi_model.pkl (PGI prediction)")
    print(f"   - pgi_model_feature_engineer.pkl (feature processor)")
    print(f"   - training_report.json (metrics and info)")

    return best_model, results


if __name__ == "__main__":
    train_all_models()
