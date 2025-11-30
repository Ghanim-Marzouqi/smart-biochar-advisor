"""
Generate synthetic training data for Smart Biochar Advisor
Based on S1-S4 formulations and soil science principles
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from models.data_schema import STANDARD_FORMULATIONS, TrainingDataPoint


def calculate_pgi(soil_type: str, ph: float, ec: float, temperature: float,
                  humidity: float, biochar_g: float, npk_g: float) -> float:
    """
    Calculate Plant Growth Index based on soil properties and amendments.

    This is a simplified model based on agronomic principles:
    - Optimal pH range: 6.0-7.5
    - Lower EC (salinity) is better
    - Biochar improves nutrient retention
    - NPK provides nutrients but excess can be harmful
    - Soil type affects water and nutrient retention
    """
    # Base score starts at 50
    pgi = 50.0

    # pH factor (optimal: 6.0-7.5)
    if 6.0 <= ph <= 7.5:
        pgi += 15
    elif 5.5 <= ph < 6.0 or 7.5 < ph <= 8.0:
        pgi += 10
    elif 5.0 <= ph < 5.5 or 8.0 < ph <= 8.5:
        pgi += 5
    else:
        pgi -= 5  # Too acidic or alkaline

    # EC/Salinity factor (lower is better)
    if ec < 1.5:
        pgi += 15
    elif ec < 2.5:
        pgi += 10
    elif ec < 4.0:
        pgi += 5
    else:
        pgi -= 10  # High salinity reduces growth

    # Temperature factor (optimal: 20-30°C)
    if 20 <= temperature <= 30:
        pgi += 10
    elif 15 <= temperature < 20 or 30 < temperature <= 35:
        pgi += 5
    else:
        pgi -= 5

    # Humidity factor (optimal: 40-70%)
    if 40 <= humidity <= 70:
        pgi += 5
    elif 30 <= humidity < 40 or 70 < humidity <= 80:
        pgi += 2
    else:
        pgi -= 3

    # Biochar benefits (improves retention, reduces salinity impact)
    biochar_benefit = min(biochar_g * 1.2, 15)  # Cap at 15 points
    pgi += biochar_benefit

    # High biochar especially helps with high salinity
    if ec > 2.0:
        pgi += min(biochar_g * 0.5, 8)

    # NPK benefits (provides nutrients)
    npk_benefit = min(npk_g * 0.8, 12)  # Cap at 12 points
    pgi += npk_benefit

    # Excess NPK penalty (diminishing returns, can cause toxicity)
    if npk_g > 15:
        pgi -= (npk_g - 15) * 0.5

    # Soil type modifiers
    if soil_type == 'sandy':
        # Sandy soils drain fast, benefit more from biochar
        pgi += biochar_g * 0.3
        # But lose nutrients easily without enough amendments
        if biochar_g + npk_g < 12:
            pgi -= 5
    elif soil_type == 'clay':
        # Clay soils retain water/nutrients well
        pgi += 5
        # But can become waterlogged
        if humidity > 70:
            pgi -= 5
        # Don't need as much amendment
        if biochar_g + npk_g > 20:
            pgi -= 3
    else:  # mixed
        # Balanced properties
        pgi += 3

    # Synergy bonus (balanced biochar:NPK ratio)
    biochar_npk_ratio = biochar_g / (npk_g + 0.1)  # Avoid div by zero
    if 0.5 <= biochar_npk_ratio <= 1.5:
        pgi += 5  # Good balance

    # Ensure PGI is within 0-100 range
    pgi = max(0, min(100, pgi))

    # Add small random variation for realism
    pgi += np.random.normal(0, 2)
    pgi = max(0, min(100, pgi))

    return round(pgi, 2)


def generate_training_dataset(n_samples: int = 500) -> pd.DataFrame:
    """
    Generate synthetic training dataset with various soil conditions.

    Args:
        n_samples: Number of samples to generate

    Returns:
        DataFrame with training data
    """
    np.random.seed(42)  # For reproducibility

    data_points = []

    # Soil types distribution
    soil_types = ['sandy', 'clay', 'mixed']

    # Generate diverse samples
    for _ in range(n_samples):
        # Random soil properties
        soil_type = np.random.choice(soil_types)
        ph = np.random.uniform(4.5, 9.0)
        ec = np.random.uniform(0.5, 6.0)
        temperature = np.random.uniform(10, 40)
        humidity = np.random.uniform(20, 90)

        # Random amendment amounts (with some bias toward known formulations)
        if np.random.random() < 0.3:  # 30% use standard formulations
            formulation = np.random.choice(STANDARD_FORMULATIONS)
            biochar_g = formulation.biochar_g
            npk_g = formulation.npk_g
        else:  # 70% random amounts
            biochar_g = np.random.uniform(0, 20)
            npk_g = np.random.uniform(0, 20)

        # Calculate PGI
        pgi = calculate_pgi(soil_type, ph, ec, temperature, humidity, biochar_g, npk_g)

        # Create data point
        data_point = TrainingDataPoint(
            soil_type=soil_type,
            ph=round(ph, 2),
            ec=round(ec, 2),
            temperature=round(temperature, 1),
            humidity=round(humidity, 1),
            biochar_g=round(biochar_g, 2),
            npk_g=round(npk_g, 2),
            plant_growth_index=pgi,
            yield_increase_pct=round((pgi - 50) * 0.8, 2),  # Simplified relationship
            nutrient_retention_score=round(min(100, 50 + biochar_g * 2 + npk_g * 0.5), 2)
        )

        data_points.append(data_point)

    # Convert to DataFrame
    df = pd.DataFrame([
        {
            **dp.to_feature_dict(),
            **dp.to_target_dict()
        }
        for dp in data_points
    ])

    return df


def generate_formulation_reference() -> pd.DataFrame:
    """Generate reference data for standard formulations (S1-S4)."""

    formulation_data = []

    for formulation in STANDARD_FORMULATIONS:
        formulation_data.append({
            'formulation_id': formulation.formulation_id,
            'biochar_g': formulation.biochar_g,
            'npk_g': formulation.npk_g,
            'total_amendment_g': formulation.total_amendment,
            'biochar_ratio': round(formulation.biochar_ratio, 3),
            'npk_ratio': round(formulation.npk_ratio, 3),
            'description': formulation.description
        })

    return pd.DataFrame(formulation_data)


if __name__ == "__main__":
    # Create data directory if it doesn't exist
    data_dir = Path(__file__).parent / "samples"
    data_dir.mkdir(exist_ok=True)

    print("🌱 Generating training dataset...")

    # Generate training data
    training_df = generate_training_dataset(n_samples=500)
    training_file = data_dir / "training_data.csv"
    training_df.to_csv(training_file, index=False)
    print(f"✅ Training data saved: {training_file}")
    print(f"   - Samples: {len(training_df)}")
    print(f"   - Features: {training_df.shape[1]}")

    # Generate formulation reference
    formulation_df = generate_formulation_reference()
    formulation_file = data_dir / "formulations.csv"
    formulation_df.to_csv(formulation_file, index=False)
    print(f"✅ Formulation reference saved: {formulation_file}")
    print(f"   - Formulations: {len(formulation_df)}")

    # Display statistics
    print("\n📊 Training Data Statistics:")
    print(training_df.describe())

    print("\n📋 Formulation Reference:")
    print(formulation_df.to_string(index=False))
