"""
Data validation utilities for Smart Biochar Advisor
Validates soil samples, formulations, and user inputs
"""

from typing import Tuple, Optional, Dict, Any
import sys
from pathlib import Path

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent))

from models.data_schema import SoilSample, Formulation


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


class SoilDataValidator:
    """Validator for soil sample data."""

    # Validation ranges
    PH_MIN = 0.0
    PH_MAX = 14.0
    PH_OPTIMAL_MIN = 6.0
    PH_OPTIMAL_MAX = 7.5

    EC_MIN = 0.0
    EC_MAX = 20.0  # dS/m
    EC_LOW_SALINITY = 2.0
    EC_MODERATE_SALINITY = 4.0

    TEMP_MIN = -10.0  # °C
    TEMP_MAX = 50.0
    TEMP_OPTIMAL_MIN = 20.0
    TEMP_OPTIMAL_MAX = 30.0

    HUMIDITY_MIN = 0.0
    HUMIDITY_MAX = 100.0
    HUMIDITY_OPTIMAL_MIN = 40.0
    HUMIDITY_OPTIMAL_MAX = 70.0

    VALID_SOIL_TYPES = ['sandy', 'clay', 'mixed']

    @classmethod
    def validate_soil_type(cls, soil_type: str) -> Tuple[bool, Optional[str]]:
        """Validate soil type."""
        if not soil_type:
            return False, "Soil type is required"

        soil_type_lower = soil_type.lower()
        if soil_type_lower not in cls.VALID_SOIL_TYPES:
            return False, f"Invalid soil type. Must be one of: {', '.join(cls.VALID_SOIL_TYPES)}"

        return True, None

    @classmethod
    def validate_ph(cls, ph: float) -> Tuple[bool, Optional[str]]:
        """Validate pH value."""
        if ph is None:
            return False, "pH value is required"

        if not isinstance(ph, (int, float)):
            return False, "pH must be a number"

        if ph < cls.PH_MIN or ph > cls.PH_MAX:
            return False, f"pH must be between {cls.PH_MIN} and {cls.PH_MAX}"

        # Warning for non-optimal values
        if ph < cls.PH_OPTIMAL_MIN:
            return True, f"Warning: pH is acidic (optimal: {cls.PH_OPTIMAL_MIN}-{cls.PH_OPTIMAL_MAX})"
        elif ph > cls.PH_OPTIMAL_MAX:
            return True, f"Warning: pH is alkaline (optimal: {cls.PH_OPTIMAL_MIN}-{cls.PH_OPTIMAL_MAX})"

        return True, None

    @classmethod
    def validate_ec(cls, ec: float) -> Tuple[bool, Optional[str]]:
        """Validate EC (electrical conductivity) value."""
        if ec is None:
            return False, "EC value is required"

        if not isinstance(ec, (int, float)):
            return False, "EC must be a number"

        if ec < cls.EC_MIN:
            return False, f"EC must be positive (minimum: {cls.EC_MIN})"

        if ec > cls.EC_MAX:
            return False, f"EC value too high (maximum: {cls.EC_MAX} dS/m)"

        # Warning for salinity
        if ec >= cls.EC_MODERATE_SALINITY:
            return True, f"Warning: High salinity detected (EC: {ec} dS/m)"
        elif ec >= cls.EC_LOW_SALINITY:
            return True, f"Warning: Moderate salinity (EC: {ec} dS/m)"

        return True, None

    @classmethod
    def validate_temperature(cls, temperature: float) -> Tuple[bool, Optional[str]]:
        """Validate temperature value."""
        if temperature is None:
            return False, "Temperature is required"

        if not isinstance(temperature, (int, float)):
            return False, "Temperature must be a number"

        if temperature < cls.TEMP_MIN or temperature > cls.TEMP_MAX:
            return False, f"Temperature out of range ({cls.TEMP_MIN}°C to {cls.TEMP_MAX}°C)"

        # Warning for non-optimal values
        if temperature < cls.TEMP_OPTIMAL_MIN:
            return True, f"Warning: Temperature is low (optimal: {cls.TEMP_OPTIMAL_MIN}-{cls.TEMP_OPTIMAL_MAX}°C)"
        elif temperature > cls.TEMP_OPTIMAL_MAX:
            return True, f"Warning: Temperature is high (optimal: {cls.TEMP_OPTIMAL_MIN}-{cls.TEMP_OPTIMAL_MAX}°C)"

        return True, None

    @classmethod
    def validate_humidity(cls, humidity: float) -> Tuple[bool, Optional[str]]:
        """Validate humidity value."""
        if humidity is None:
            return False, "Humidity is required"

        if not isinstance(humidity, (int, float)):
            return False, "Humidity must be a number"

        if humidity < cls.HUMIDITY_MIN or humidity > cls.HUMIDITY_MAX:
            return False, f"Humidity must be between {cls.HUMIDITY_MIN}% and {cls.HUMIDITY_MAX}%"

        # Warning for non-optimal values
        if humidity < cls.HUMIDITY_OPTIMAL_MIN:
            return True, f"Warning: Low humidity (optimal: {cls.HUMIDITY_OPTIMAL_MIN}-{cls.HUMIDITY_OPTIMAL_MAX}%)"
        elif humidity > cls.HUMIDITY_OPTIMAL_MAX:
            return True, f"Warning: High humidity (optimal: {cls.HUMIDITY_OPTIMAL_MIN}-{cls.HUMIDITY_OPTIMAL_MAX}%)"

        return True, None

    @classmethod
    def validate_soil_sample(cls, soil_sample: Dict[str, Any]) -> Tuple[bool, Dict[str, str]]:
        """
        Validate complete soil sample.

        Args:
            soil_sample: Dictionary with soil properties

        Returns:
            Tuple of (is_valid, errors_dict)
        """
        errors = {}
        warnings = {}

        # Validate each field
        validations = [
            ('soil_type', cls.validate_soil_type(soil_sample.get('soil_type'))),
            ('ph', cls.validate_ph(soil_sample.get('ph'))),
            ('ec', cls.validate_ec(soil_sample.get('ec'))),
            ('temperature', cls.validate_temperature(soil_sample.get('temperature'))),
            ('humidity', cls.validate_humidity(soil_sample.get('humidity')))
        ]

        for field, (is_valid, message) in validations:
            if not is_valid:
                errors[field] = message
            elif message:  # Warning message
                warnings[field] = message

        is_valid = len(errors) == 0

        return is_valid, {'errors': errors, 'warnings': warnings}


class FormulationValidator:
    """Validator for biochar-NPK formulations."""

    BIOCHAR_MIN = 0.0
    BIOCHAR_MAX = 30.0  # grams per 100g soil
    NPK_MIN = 0.0
    NPK_MAX = 30.0
    TOTAL_AMENDMENT_MAX = 40.0

    @classmethod
    def validate_biochar(cls, biochar_g: float) -> Tuple[bool, Optional[str]]:
        """Validate biochar amount."""
        if biochar_g is None:
            return False, "Biochar amount is required"

        if not isinstance(biochar_g, (int, float)):
            return False, "Biochar amount must be a number"

        if biochar_g < cls.BIOCHAR_MIN:
            return False, f"Biochar amount must be non-negative"

        if biochar_g > cls.BIOCHAR_MAX:
            return False, f"Biochar amount too high (maximum: {cls.BIOCHAR_MAX}g per 100g soil)"

        return True, None

    @classmethod
    def validate_npk(cls, npk_g: float) -> Tuple[bool, Optional[str]]:
        """Validate NPK amount."""
        if npk_g is None:
            return False, "NPK amount is required"

        if not isinstance(npk_g, (int, float)):
            return False, "NPK amount must be a number"

        if npk_g < cls.NPK_MIN:
            return False, f"NPK amount must be non-negative"

        if npk_g > cls.NPK_MAX:
            return False, f"NPK amount too high (maximum: {cls.NPK_MAX}g per 100g soil)"

        return True, None

    @classmethod
    def validate_formulation(cls, biochar_g: float, npk_g: float) -> Tuple[bool, Dict[str, str]]:
        """
        Validate complete formulation.

        Args:
            biochar_g: Biochar amount in grams
            npk_g: NPK amount in grams

        Returns:
            Tuple of (is_valid, errors_dict)
        """
        errors = {}
        warnings = {}

        # Validate individual amounts
        biochar_valid, biochar_msg = cls.validate_biochar(biochar_g)
        npk_valid, npk_msg = cls.validate_npk(npk_g)

        if not biochar_valid:
            errors['biochar'] = biochar_msg
        if not npk_valid:
            errors['npk'] = npk_msg

        # Validate total amendment
        if biochar_valid and npk_valid:
            total = biochar_g + npk_g
            if total > cls.TOTAL_AMENDMENT_MAX:
                errors['total'] = f"Total amendment too high: {total}g (max: {cls.TOTAL_AMENDMENT_MAX}g)"
            elif total == 0:
                warnings['total'] = "No amendments specified"

            # Check balance
            if total > 0:
                ratio = biochar_g / (npk_g + 0.1)
                if ratio < 0.3:
                    warnings['balance'] = "Low biochar ratio - may have reduced retention benefits"
                elif ratio > 2.0:
                    warnings['balance'] = "High biochar ratio - may need more nutrients"

        is_valid = len(errors) == 0

        return is_valid, {'errors': errors, 'warnings': warnings}


def validate_input_data(data: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate all input data (soil sample + optional formulation).

    Args:
        data: Dictionary with all input fields

    Returns:
        Tuple of (is_valid, validation_results)
    """
    results = {
        'valid': True,
        'soil_errors': {},
        'soil_warnings': {},
        'formulation_errors': {},
        'formulation_warnings': {}
    }

    # Validate soil sample
    soil_valid, soil_validation = SoilDataValidator.validate_soil_sample(data)
    results['soil_errors'] = soil_validation['errors']
    results['soil_warnings'] = soil_validation['warnings']

    if not soil_valid:
        results['valid'] = False

    # Validate formulation if provided
    if 'biochar_g' in data or 'npk_g' in data:
        biochar = data.get('biochar_g', 0)
        npk = data.get('npk_g', 0)
        formulation_valid, formulation_validation = FormulationValidator.validate_formulation(biochar, npk)
        results['formulation_errors'] = formulation_validation['errors']
        results['formulation_warnings'] = formulation_validation['warnings']

        if not formulation_valid:
            results['valid'] = False

    return results['valid'], results
