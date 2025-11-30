# ML Models for Smart Biochar Advisor

This directory contains machine learning models and related code for biochar and NPK recommendations.

## Files

### Core Modules
- **data_schema.py**: Data models and schemas (SoilSample, Formulation, Recommendation)
- **feature_engineering.py**: Feature engineering pipeline
- **train_model.py**: Model training script
- **predictor.py**: Model inference and recommendations

### Model Training

To train the models:

```bash
# Using Docker (recommended)
cd code
docker-compose run biochar-advisor python models/train_model.py

# Or with local Python environment
cd code
python3 models/train_model.py
```

This will:
1. Load training data from `data/samples/`
2. Engineer features
3. Train multiple models (Random Forest, Gradient Boosting, Ridge)
4. Select best performing model
5. Save trained model to `models/trained/`

### Model Files

After training, the following files will be created in `models/trained/`:
- **pgi_model.pkl**: Trained PGI prediction model
- **pgi_model_feature_engineer.pkl**: Feature preprocessing pipeline
- **training_report.json**: Training metrics and information

## Usage

### Making Predictions

```python
from models.predictor import BiocharPredictor
from models.data_schema import SoilSample

# Create predictor
predictor = BiocharPredictor()

# Create soil sample
soil = SoilSample(
    soil_type='sandy',
    ph=6.5,
    ec=2.3,
    temperature=25,
    humidity=55
)

# Get recommendation
recommendation = predictor.recommend_formulation(soil)

print(f"Biochar: {recommendation.biochar_g}g")
print(f"NPK: {recommendation.npk_g}g")
print(f"Expected PGI: {recommendation.predicted_pgi}")
```

### Optimizing Formulation

```python
# Optimize for target PGI
recommendation = predictor.optimize_formulation(
    soil,
    target_pgi=95.0,
    max_biochar=20.0,
    max_npk=20.0
)
```

### Comparing Formulations

```python
# Compare all standard formulations (S1-S4)
comparison_df = predictor.compare_formulations(soil)
print(comparison_df)
```

## Model Architecture

### Features (20+ engineered features)
- **Base features**: pH, EC, temperature, humidity, biochar, NPK
- **Soil type**: One-hot encoded (sandy, clay, mixed)
- **Interactions**: biochar-NPK ratio, salinity mitigation
- **Categories**: pH ranges, salinity levels, optimal ranges

### Model Types
1. **Random Forest** (default): Ensemble of decision trees
2. **Gradient Boosting**: Sequential boosting algorithm
3. **Ridge Regression**: Linear model with regularization

### Performance Metrics
- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **R²**: Coefficient of determination
- **Cross-validation**: 5-fold CV scores

## Advanced Usage

### Custom Model Training

```python
from models.train_model import BiocharAdvisorModel
from utils.data_loader import DataLoader

# Load data
loader = DataLoader()
training_data = loader.load_training_data()

# Prepare features
from models.feature_engineering import FeatureEngineer
engineer = FeatureEngineer()
X, feature_names = engineer.prepare_features_for_training(training_data)
y = training_data['plant_growth_index']

# Train model
model = BiocharAdvisorModel(model_type='random_forest')
metrics = model.train(X, y)

# Save
model.save('models/trained/custom_model.pkl')
```

### Feature Importance

```python
# Get feature importance (tree-based models only)
importance = model.get_feature_importance()
print(importance.head(10))
```

## References

Based on S1-S4 formulation research:
- S1: 5g biochar + 5g NPK (baseline)
- S2: 10g biochar + 10g NPK (balanced)
- S3: 10g biochar + 15g NPK (high NPK)
- S4: 15g biochar + 10g NPK (high biochar)

See `../docs/results.md` for detailed formulation research.
