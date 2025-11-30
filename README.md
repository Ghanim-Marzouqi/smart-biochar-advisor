# Smart Biochar Advisor

An AI-powered soil analysis system that recommends optimal biochar and NPK fertilizer amounts for sustainable agriculture.

## Overview

The Smart Biochar Advisor analyzes soil properties and provides intelligent recommendations for:
- Nano-biochar application rates
- NPK fertilizer amounts
- Expected Plant Growth Index (PGI)
- Soil status assessment

## Features

- **Interactive Web Interface**: Built with Streamlit for easy access
- **AI-Powered Recommendations**: Machine learning models trained on real formulation data (S1-S4)
- **Real-time Analysis**: Instant feedback on soil conditions
- **Data-Driven**: Based on scientific research and experimental data
- **Export Capabilities**: Download recommendations and reports

## Quick Start

### Prerequisites
- Docker Desktop (recommended)
- OR Python 3.9+ and pip

### Option 1: Docker (Recommended)

```bash
# Navigate to code directory
cd code

# Start the application with Docker
./start.sh
```

Or manually:
```bash
# Navigate to code directory
cd code

# Build and start
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

The app will be available at `http://localhost:8501`

### Option 2: Manual Python Setup

```bash
# Navigate to code directory
cd code

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app/main.py
```

The app will open in your browser at `http://localhost:8501`

## Input Parameters

- **Soil Type**: Sandy, Clay, or Mixed
- **pH Value**: Soil acidity/alkalinity (0-14)
- **EC Value**: Electrical conductivity (salinity indicator)
- **Temperature**: Soil temperature in °C
- **Humidity**: Soil moisture percentage

## Technology Stack

- **Frontend**: Streamlit
- **ML Framework**: scikit-learn
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn, plotly
- **Database**: SQLite
- **Deployment**: Docker & Docker Compose

## Project Structure

```
code/
├── app/              # Streamlit application
├── models/           # ML models and training
├── data/             # Sample datasets
├── utils/            # Helper functions
├── tests/            # Unit tests
└── requirements.txt
```

## Development

See `docs/tasks.md` for development roadmap and task tracking.

## Scientific Background

Based on formulation research:
- **S1**: 5g biochar + 5g NPK (baseline)
- **S2**: 10g biochar + 10g NPK (balanced)
- **S3**: 10g biochar + 15g NPK (nutrient-rich)
- **S4**: 15g biochar + 10g NPK (carbon-dense)

Higher biochar content improves nutrient retention and reduces leaching. The AI model learns optimal combinations for different soil conditions.

## Contributing

1. Check `docs/tasks.md` for open tasks
2. Create a feature branch
3. Make changes with tests
4. Submit a pull request

## Deployment

### Free Deployment Options

#### Streamlit Community Cloud (Recommended)

1. Go to https://share.streamlit.io/
2. Sign in with GitHub
3. Click "New app"
4. Configure:
   - Repository: `Ghanim-Marzouqi/smart-biochar-advisor`
   - Branch: `main`
   - Main file path: `code/app/main.py`
5. Click "Deploy"

The model will be trained automatically on first run.

**Your app URL will be**: `https://[your-app-name].streamlit.app`

#### Alternative Options

- **Hugging Face Spaces**: https://huggingface.co/spaces
- **Render**: https://render.com (free tier)
- **Railway**: https://railway.app (free tier with limits)

## License

MIT License

## Support

For detailed information, see the `docs/` folder.
