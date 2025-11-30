# Smart Biochar Advisor - Streamlit Application

Web interface for the Smart Biochar Advisor AI system.

## Features

- **Interactive Input Form**: Easy-to-use sliders and selectors for soil properties
- **AI-Powered Recommendations**: Real-time biochar and NPK suggestions
- **Plant Growth Index Predictions**: Expected PGI based on soil conditions
- **Soil Status Assessment**: Automatic soil health evaluation
- **Formulation Comparison**: Compare standard S1-S4 formulations
- **Insights & Tips**: Actionable recommendations for application

## Running the Application

### Using Docker (Recommended)

```bash
cd code
./start.sh
```

Then open your browser to: http://localhost:8501

### Manual Python Setup

```bash
cd code

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# IMPORTANT: Train the model first (one-time)
python models/train_model.py

# Run the app
streamlit run app/main.py
```

## Prerequisites

### Before First Run

The ML model must be trained before using the app:

```bash
cd code
python models/train_model.py
```

This will create:
- `models/trained/pgi_model.pkl`
- `models/trained/pgi_model_feature_engineer.pkl`
- `models/trained/training_report.json`

## Application Structure

```
app/
├── main.py              # Main Streamlit application
└── README.md            # This file

.streamlit/
└── config.toml          # Streamlit configuration
```

## User Guide

### Step 1: Input Soil Properties

**Basic Properties:**
- **Soil Type**: Select sandy, clay, or mixed
- **pH Value**: Soil acidity/alkalinity (4.0-9.0)
  - Optimal: 6.0-7.5
- **EC Value**: Electrical conductivity / salinity (0.0-8.0 dS/m)
  - Low salinity: < 2.0
  - Moderate: 2.0-4.0
  - High: > 4.0

**Environmental Conditions:**
- **Temperature**: Soil temperature in °C (10-40)
  - Optimal: 20-30°C
- **Humidity**: Soil moisture percentage (20-90%)
  - Optimal: 40-70%

### Step 2: Analyze

Click "🔬 Analyze Soil & Get Recommendations" to get AI-powered advice.

### Step 3: Review Results

**Recommendations:**
- Biochar amount (g per 100g soil)
- NPK fertilizer amount (g per 100g soil)
- Expected Plant Growth Index (0-100)
- Model confidence score

**Soil Status:**
- Overall soil health assessment
- pH and salinity classification

**Comparison:**
- See how standard formulations (S1-S4) would perform

**Insights:**
- Specific analysis for your soil conditions
- Best practices for application
- Expected benefits

## Customization

### Theme

Edit `.streamlit/config.toml` to customize colors:

```toml
[theme]
primaryColor = "#66BB6A"      # Green
backgroundColor = "#FFFFFF"    # White
secondaryBackgroundColor = "#F1F8E9"  # Light green
textColor = "#1B5E20"         # Dark green
```

### Model

To use a different trained model:

```python
# In main.py
predictor = BiocharPredictor(model_path=Path("path/to/your/model.pkl"))
```

## Troubleshooting

### "Model not found" Error

**Solution**: Train the model first:
```bash
python models/train_model.py
```

### "Module not found" Error

**Solution**: Install dependencies:
```bash
pip install -r requirements.txt
```

### Port Already in Use

**Solution**: Use a different port:
```bash
streamlit run app/main.py --server.port 8502
```

### Docker Issues

**Solution**: Rebuild the container:
```bash
docker-compose up -d --build
```

## Development

### Adding New Features

The app uses Streamlit components:
- `st.slider()` for numeric inputs
- `st.selectbox()` for dropdowns
- `st.metric()` for displaying metrics
- `st.dataframe()` for tables
- Custom CSS via `st.markdown()`

### Hot Reload

Streamlit auto-reloads when you save changes. Just edit `main.py` and the browser will update.

## Performance

- Initial load: ~2-3 seconds (model loading)
- Prediction time: < 1 second
- Supports multiple concurrent users

## Security

- No user data is stored
- All processing happens server-side
- XSRF protection enabled
- No external API calls

## Support

For issues or questions:
- Check `docs/` folder for detailed information
- Review training report: `models/trained/training_report.json`
- Check model metrics for confidence levels
