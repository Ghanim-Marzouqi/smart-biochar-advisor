"""
Data Schema and Models for Smart Biochar Advisor
Defines the structure for soil samples, formulations, and predictions.
"""

from dataclasses import dataclass
from typing import Optional, Literal
from datetime import datetime


@dataclass
class SoilSample:
    """Represents a soil sample with its properties."""

    soil_type: Literal['sandy', 'clay', 'mixed']
    ph: float  # 0-14 scale
    ec: float  # Electrical conductivity (dS/m)
    temperature: float  # Celsius
    humidity: float  # Percentage (0-100)

    # Optional metadata
    sample_id: Optional[str] = None
    location: Optional[str] = None
    date_collected: Optional[datetime] = None

    def validate(self) -> bool:
        """Validate soil sample data ranges."""
        if self.ph < 0 or self.ph > 14:
            raise ValueError(f"pH must be between 0-14, got {self.ph}")

        if self.ec < 0:
            raise ValueError(f"EC must be positive, got {self.ec}")

        if self.temperature < -50 or self.temperature > 60:
            raise ValueError(f"Temperature out of range: {self.temperature}°C")

        if self.humidity < 0 or self.humidity > 100:
            raise ValueError(f"Humidity must be 0-100%, got {self.humidity}")

        return True


@dataclass
class Formulation:
    """Represents a biochar-NPK formulation."""

    formulation_id: str  # e.g., "S1", "S2", etc.
    biochar_g: float  # grams per 100g soil
    npk_g: float  # grams per 100g soil
    description: str

    @property
    def total_amendment(self) -> float:
        """Total amendment in grams."""
        return self.biochar_g + self.npk_g

    @property
    def biochar_ratio(self) -> float:
        """Ratio of biochar to total amendment."""
        return self.biochar_g / self.total_amendment if self.total_amendment > 0 else 0

    @property
    def npk_ratio(self) -> float:
        """Ratio of NPK to total amendment."""
        return self.npk_g / self.total_amendment if self.total_amendment > 0 else 0


@dataclass
class Recommendation:
    """Represents a recommendation for a soil sample."""

    soil_sample: SoilSample
    biochar_g: float  # Recommended biochar (g per 100g soil)
    npk_g: float  # Recommended NPK (g per 100g soil)
    predicted_pgi: float  # Plant Growth Index (0-100)
    soil_status: str  # e.g., "Balanced", "Slightly Saline", etc.
    confidence: Optional[float] = None  # Model confidence (0-1)

    @property
    def formulation(self) -> Formulation:
        """Convert recommendation to formulation."""
        return Formulation(
            formulation_id="CUSTOM",
            biochar_g=self.biochar_g,
            npk_g=self.npk_g,
            description=f"Custom recommendation for {self.soil_sample.soil_type} soil"
        )


@dataclass
class TrainingDataPoint:
    """Represents a single data point for model training."""

    # Input features
    soil_type: Literal['sandy', 'clay', 'mixed']
    ph: float
    ec: float
    temperature: float
    humidity: float
    biochar_g: float
    npk_g: float

    # Target outputs
    plant_growth_index: float  # PGI (0-100)
    yield_increase_pct: Optional[float] = None  # Yield increase percentage
    nutrient_retention_score: Optional[float] = None  # 0-100 scale

    def to_feature_dict(self) -> dict:
        """Convert to dictionary of features for ML model."""
        return {
            'soil_type': self.soil_type,
            'ph': self.ph,
            'ec': self.ec,
            'temperature': self.temperature,
            'humidity': self.humidity,
            'biochar_g': self.biochar_g,
            'npk_g': self.npk_g
        }

    def to_target_dict(self) -> dict:
        """Convert to dictionary of target values."""
        return {
            'plant_growth_index': self.plant_growth_index,
            'yield_increase_pct': self.yield_increase_pct,
            'nutrient_retention_score': self.nutrient_retention_score
        }


# Standard formulations based on research (S1-S4)
STANDARD_FORMULATIONS = [
    Formulation(
        formulation_id="S1",
        biochar_g=5.0,
        npk_g=5.0,
        description="Baseline formulation - light, uniform mixture"
    ),
    Formulation(
        formulation_id="S2",
        biochar_g=10.0,
        npk_g=10.0,
        description="Balanced formulation - denser, cohesive"
    ),
    Formulation(
        formulation_id="S3",
        biochar_g=10.0,
        npk_g=15.0,
        description="High NPK formulation - nutrient-rich, granular"
    ),
    Formulation(
        formulation_id="S4",
        biochar_g=15.0,
        npk_g=10.0,
        description="High biochar formulation - darker, carbon-dense"
    ),
]


def get_formulation_by_id(formulation_id: str) -> Optional[Formulation]:
    """Get a standard formulation by ID."""
    for formulation in STANDARD_FORMULATIONS:
        if formulation.formulation_id == formulation_id:
            return formulation
    return None


def assess_soil_status(ph: float, ec: float) -> str:
    """Assess soil status based on pH and EC values."""
    status_parts = []

    # pH assessment
    if ph < 5.5:
        status_parts.append("Acidic")
    elif ph > 7.5:
        status_parts.append("Alkaline")
    else:
        status_parts.append("Neutral pH")

    # EC/Salinity assessment
    if ec < 2.0:
        status_parts.append("Low Salinity")
    elif ec < 4.0:
        status_parts.append("Moderate Salinity")
    else:
        status_parts.append("High Salinity")

    # Overall status
    if 6.0 <= ph <= 7.5 and ec < 2.0:
        return "Balanced"
    elif ec >= 4.0:
        return "Highly Saline - " + ", ".join(status_parts)
    elif ec >= 2.0:
        return "Slightly Saline - " + ", ".join(status_parts)
    else:
        return ", ".join(status_parts)
