"""
Model Inference for Smart Biochar Advisor
Provides predictions and recommendations for soil samples
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import sys

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent))

from models.train_model import BiocharAdvisorModel
from models.feature_engineering import FeatureEngineer, prepare_single_sample
from models.data_schema import (
    SoilSample, Recommendation, assess_soil_status,
    STANDARD_FORMULATIONS, get_formulation_by_id
)


class BiocharPredictor:
    """Handles predictions and recommendations for soil samples."""

    def __init__(self, model_path: Optional[Path] = None):
        """
        Initialize predictor.

        Args:
            model_path: Path to trained model. If None, uses default location.
        """
        if model_path is None:
            model_path = Path(__file__).parent / "trained" / "pgi_model.pkl"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                "Please train the model first using train_model.py"
            )

        print(f"📦 Loading model from {model_path}...")
        self.model = BiocharAdvisorModel.load(model_path)
        print(f"✅ Model loaded successfully")

    def predict_pgi(self, soil_type: str, ph: float, ec: float,
                    temperature: float, humidity: float,
                    biochar_g: float, npk_g: float) -> float:
        """
        Predict Plant Growth Index for given conditions.

        Args:
            soil_type: Type of soil
            ph: pH value
            ec: EC value
            temperature: Temperature in Celsius
            humidity: Humidity percentage
            biochar_g: Biochar amount (g per 100g soil)
            npk_g: NPK amount (g per 100g soil)

        Returns:
            Predicted PGI value
        """
        # Prepare features
        features_df = prepare_single_sample(
            soil_type, ph, ec, temperature, humidity,
            biochar_g, npk_g, self.model.feature_names
        )

        # Predict
        pgi = self.model.predict(features_df)[0]

        # Ensure within valid range
        pgi = max(0, min(100, pgi))

        return round(pgi, 2)

    def recommend_formulation(self, soil_sample: SoilSample,
                             target_pgi: float = 90.0) -> Recommendation:
        """
        Recommend optimal biochar and NPK amounts for a soil sample.

        Args:
            soil_sample: SoilSample object with soil properties
            target_pgi: Target Plant Growth Index (default: 90)

        Returns:
            Recommendation object
        """
        # Validate soil sample
        soil_sample.validate()

        # Try standard formulations and find best one
        best_formulation = None
        best_pgi = -float('inf')
        best_diff = float('inf')

        for formulation in STANDARD_FORMULATIONS:
            pgi = self.predict_pgi(
                soil_sample.soil_type,
                soil_sample.ph,
                soil_sample.ec,
                soil_sample.temperature,
                soil_sample.humidity,
                formulation.biochar_g,
                formulation.npk_g
            )

            # Find formulation closest to target
            diff = abs(pgi - target_pgi)
            if diff < best_diff:
                best_diff = diff
                best_pgi = pgi
                best_formulation = formulation

        # Assess soil status
        soil_status = assess_soil_status(soil_sample.ph, soil_sample.ec)

        # Create recommendation
        recommendation = Recommendation(
            soil_sample=soil_sample,
            biochar_g=best_formulation.biochar_g,
            npk_g=best_formulation.npk_g,
            predicted_pgi=best_pgi,
            soil_status=soil_status,
            confidence=self.model.metrics.get('test', {}).get('r2', 0.85)
        )

        return recommendation

    def optimize_formulation(self, soil_sample: SoilSample,
                            target_pgi: float = 90.0,
                            max_biochar: float = 20.0,
                            max_npk: float = 20.0) -> Recommendation:
        """
        Optimize biochar and NPK amounts to achieve target PGI.

        Uses grid search to find optimal amounts.

        Args:
            soil_sample: SoilSample object
            target_pgi: Target PGI to achieve
            max_biochar: Maximum biochar amount to test
            max_npk: Maximum NPK amount to test

        Returns:
            Optimized Recommendation
        """
        # Validate
        soil_sample.validate()

        # Grid search
        biochar_range = np.linspace(0, max_biochar, 20)
        npk_range = np.linspace(0, max_npk, 20)

        best_biochar = 0
        best_npk = 0
        best_pgi = -float('inf')
        best_diff = float('inf')

        for biochar in biochar_range:
            for npk in npk_range:
                pgi = self.predict_pgi(
                    soil_sample.soil_type,
                    soil_sample.ph,
                    soil_sample.ec,
                    soil_sample.temperature,
                    soil_sample.humidity,
                    biochar, npk
                )

                # Find closest to target
                diff = abs(pgi - target_pgi)
                if diff < best_diff:
                    best_diff = diff
                    best_pgi = pgi
                    best_biochar = biochar
                    best_npk = npk

        # Assess soil status
        soil_status = assess_soil_status(soil_sample.ph, soil_sample.ec)

        # Create recommendation
        recommendation = Recommendation(
            soil_sample=soil_sample,
            biochar_g=round(best_biochar, 2),
            npk_g=round(best_npk, 2),
            predicted_pgi=best_pgi,
            soil_status=soil_status,
            confidence=self.model.metrics.get('test', {}).get('r2', 0.85)
        )

        return recommendation

    def compare_formulations(self, soil_sample: SoilSample) -> pd.DataFrame:
        """
        Compare all standard formulations for a soil sample.

        Args:
            soil_sample: SoilSample object

        Returns:
            DataFrame with comparison results
        """
        results = []

        for formulation in STANDARD_FORMULATIONS:
            pgi = self.predict_pgi(
                soil_sample.soil_type,
                soil_sample.ph,
                soil_sample.ec,
                soil_sample.temperature,
                soil_sample.humidity,
                formulation.biochar_g,
                formulation.npk_g
            )

            results.append({
                'Formulation': formulation.formulation_id,
                'Biochar (g)': formulation.biochar_g,
                'NPK (g)': formulation.npk_g,
                'Total Amendment (g)': formulation.total_amendment,
                'Predicted PGI': round(pgi, 2),
                'Description': formulation.description
            })

        df = pd.DataFrame(results)
        df = df.sort_values('Predicted PGI', ascending=False)

        return df


def make_recommendation(soil_type: str, ph: float, ec: float,
                       temperature: float, humidity: float,
                       optimize: bool = False) -> Dict:
    """
    Convenience function to make a recommendation.

    Args:
        soil_type: Type of soil
        ph: pH value
        ec: EC value
        temperature: Temperature
        humidity: Humidity
        optimize: Whether to optimize or use standard formulations

    Returns:
        Dictionary with recommendation details
    """
    # Create soil sample
    soil_sample = SoilSample(
        soil_type=soil_type,
        ph=ph,
        ec=ec,
        temperature=temperature,
        humidity=humidity
    )

    # Get predictor
    predictor = BiocharPredictor()

    # Get recommendation
    if optimize:
        recommendation = predictor.optimize_formulation(soil_sample)
    else:
        recommendation = predictor.recommend_formulation(soil_sample)

    # Format result
    result = {
        'soil_status': recommendation.soil_status,
        'biochar_g': recommendation.biochar_g,
        'npk_g': recommendation.npk_g,
        'predicted_pgi': recommendation.predicted_pgi,
        'confidence': recommendation.confidence,
        'soil_sample': {
            'type': soil_sample.soil_type,
            'ph': soil_sample.ph,
            'ec': soil_sample.ec,
            'temperature': soil_sample.temperature,
            'humidity': soil_sample.humidity
        }
    }

    return result


if __name__ == "__main__":
    print("🌱 Testing Biochar Predictor...")

    # Example soil sample
    print("\n📋 Example: Sandy soil with moderate salinity")
    soil = SoilSample(
        soil_type='sandy',
        ph=6.5,
        ec=2.3,
        temperature=25,
        humidity=55
    )

    try:
        predictor = BiocharPredictor()

        # Get recommendation
        print("\n🎯 Getting recommendation...")
        recommendation = predictor.recommend_formulation(soil)

        print(f"\n✅ Recommendation:")
        print(f"   Biochar: {recommendation.biochar_g}g per 100g soil")
        print(f"   NPK: {recommendation.npk_g}g per 100g soil")
        print(f"   Expected PGI: {recommendation.predicted_pgi:.1f}")
        print(f"   Soil Status: {recommendation.soil_status}")
        print(f"   Confidence: {recommendation.confidence:.2%}")

        # Compare formulations
        print("\n📊 Comparing all standard formulations:")
        comparison = predictor.compare_formulations(soil)
        print(comparison.to_string(index=False))

    except FileNotFoundError as e:
        print(f"\n⚠️  {e}")
        print("   Run 'python models/train_model.py' first to train the model")
