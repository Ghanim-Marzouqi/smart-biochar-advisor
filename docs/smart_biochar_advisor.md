# Smart Biochar Advisor – AI Soil Analysis System

## Overview
An intelligent system for analyzing soil properties and recommending optimal amounts of NPK fertilizer and biochar.

## How It Works

### 1. Input Soil Data
- Soil type (sandy / clay / mixed)
- pH value
- EC value (salinity)
- Temperature and humidity (sensor or manual)

### 2. AI Analysis
Trained on previous datasets (e.g., S1–S4) to understand the relationship:

**Soil type + pH + EC + Biochar + NPK → Optimal Growth (High PGI)**

### 3. Intelligent Recommendation
Examples:
- For sandy soil with pH 6.5 and EC 1.9 → `12 g Nano-Biochar + 8 g NPK per 100 g soil`
- For clay soil with EC 2.4 → `Reduce NPK by 20% and increase biochar to 15 g`

### 4. Interface Output
- Recommended ratios  
- Expected growth index  
- Soil status (Balanced / Slightly saline)

## Development Levels

| Level | Implementation | Result |
|------|----------------|--------|
| Basic | Excel with formulas | Fast recommendations |
| Intermediate | Streamlit/Python web app | Interactive interface |
| Advanced | Sensor integration | Real-time soil analysis |

## Purpose
Supports sustainable, data-driven agriculture using Biochar and AI.

