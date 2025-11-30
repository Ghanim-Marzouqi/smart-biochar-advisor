# Training Data Samples

This directory contains training data for the Smart Biochar Advisor ML models.

## Files

- **formulations.csv**: Reference data for standard S1-S4 formulations
- **training_data.csv**: Synthetic training dataset with soil properties and outcomes

## Generating Data

To regenerate the training dataset:

```bash
# Using Docker (recommended)
docker-compose run biochar-advisor python data/generate_training_data.py

# Or with local Python environment
cd code
python3 data/generate_training_data.py
```

## Data Schema

### Formulations
- `formulation_id`: Unique identifier (S1-S4)
- `biochar_g`: Biochar amount (grams per 100g soil)
- `npk_g`: NPK fertilizer amount (grams per 100g soil)
- `total_amendment_g`: Total amendment amount
- `biochar_ratio`: Proportion of biochar in mixture
- `npk_ratio`: Proportion of NPK in mixture
- `description`: Human-readable description

### Training Data
Input features:
- `soil_type`: sandy, clay, or mixed
- `ph`: Soil pH (0-14 scale)
- `ec`: Electrical conductivity (dS/m)
- `temperature`: Soil temperature (°C)
- `humidity`: Soil moisture (%)
- `biochar_g`: Applied biochar amount
- `npk_g`: Applied NPK amount

Target outputs:
- `plant_growth_index`: PGI score (0-100)
- `yield_increase_pct`: Yield improvement percentage
- `nutrient_retention_score`: Nutrient retention score (0-100)
