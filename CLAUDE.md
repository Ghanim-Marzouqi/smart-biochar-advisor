# Smart Biochar Advisor - Claude Code Development Guide

## Project Overview
An AI-powered soil analysis system that recommends optimal biochar and NPK fertilizer amounts based on soil properties.

## Project Structure
```
smart-biochar-advisor/
├── code/                    # All application code and Docker setup
│   ├── app/                # Streamlit application
│   ├── models/             # ML models and training scripts
│   ├── data/               # Sample data and datasets
│   ├── utils/              # Helper functions
│   ├── tests/              # Unit tests
│   ├── Dockerfile          # Docker image definition
│   ├── docker-compose.yml  # Docker orchestration
│   ├── .dockerignore       # Docker build exclusions
│   ├── start.sh            # Docker startup script
│   └── requirements.txt    # Python dependencies
├── docs/                   # Documentation
│   ├── tasks.md            # Development task tracking
│   ├── smart_biochar_advisor.md
│   └── results.md
├── .git/                   # Git repository
├── .gitignore              # Git exclusions
├── README.md               # Project overview
└── CLAUDE.md               # This file
```

## Technology Stack
- **Framework**: Streamlit (Python web framework)
- **ML/AI**: scikit-learn
- **Data**: pandas, numpy
- **Database**: SQLite
- **Deployment**: Docker + Streamlit Cloud

## Development Phases
1. Project Setup
2. Data Preparation (S1-S4 formulations)
3. ML Model Development
4. Streamlit Application (MVP)
5. Enhanced Features
6. Testing & Validation
7. Documentation
8. Deployment

## Key Features
- Input soil properties (type, pH, EC, temp, humidity)
- AI-powered biochar and NPK recommendations
- Plant Growth Index (PGI) predictions
- Soil status indicators
- Data visualization and export

## Getting Started

### Option 1: Docker (Recommended)
```bash
cd code
./start.sh

# Or manually
docker-compose up -d
```

### Option 2: Manual Python Setup
```bash
cd code
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/main.py
```

Access the app at: `http://localhost:8501`

## Development Notes
- All code must be in the `/code` folder
- Update `/docs/tasks.md` after completing each task
- Follow PEP 8 style guidelines
- Write tests for critical functions
- Keep models lightweight and efficient

## Data Model
Based on S1-S4 formulations:
- S1: 5g biochar + 5g NPK (baseline)
- S2: 10g biochar + 10g NPK (balanced)
- S3: 10g biochar + 15g NPK (high NPK)
- S4: 15g biochar + 10g NPK (high biochar)

## Contact & Support
For questions or issues, refer to documentation in `/docs` folder.
