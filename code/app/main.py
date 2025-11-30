"""
Smart Biochar Advisor - Streamlit Application
Main application entry point
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent))

from models.data_schema import SoilSample, STANDARD_FORMULATIONS
from models.predictor import BiocharPredictor
from utils.validators import SoilDataValidator, validate_input_data

# Page configuration
st.set_page_config(
    page_title="Smart Biochar Advisor",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #2E7D32;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #558B2F;
        text-align: center;
        padding-bottom: 2rem;
    }
    .metric-card {
        background-color: #F1F8E9;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #66BB6A;
    }
    .warning-card {
        background-color: #FFF3E0;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #FFA726;
    }
    .info-card {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #42A5F5;
    }
    </style>
""", unsafe_allow_html=True)


def load_model():
    """Load the trained ML model."""
    try:
        predictor = BiocharPredictor()
        return predictor
    except FileNotFoundError:
        st.error("""
        ⚠️ **Model not found!**

        The ML model has not been trained yet. Please run the training script:

        ```bash
        cd code
        python models/train_model.py
        ```

        Or using Docker:
        ```bash
        cd code
        docker-compose run biochar-advisor python models/train_model.py
        ```
        """)
        st.stop()


def render_header():
    """Render application header."""
    st.markdown('<h1 class="main-header">🌱 Smart Biochar Advisor</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">AI-Powered Soil Analysis & Biochar Recommendations</p>',
        unsafe_allow_html=True
    )


def render_sidebar():
    """Render sidebar with information."""
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This intelligent system analyzes soil properties and recommends optimal amounts of:
        - **Nano-biochar** for nutrient retention
        - **NPK fertilizer** for plant nutrition

        Based on scientific research (S1-S4 formulations).
        """)

        st.divider()

        st.header("📊 Standard Formulations")
        for formulation in STANDARD_FORMULATIONS:
            with st.expander(f"{formulation.formulation_id} - {formulation.description}"):
                st.write(f"**Biochar:** {formulation.biochar_g}g per 100g soil")
                st.write(f"**NPK:** {formulation.npk_g}g per 100g soil")
                st.write(f"**Total:** {formulation.total_amendment}g")

        st.divider()

        st.header("🔬 How It Works")
        st.markdown("""
        1. **Input** your soil properties
        2. **AI analyzes** soil conditions
        3. **Get recommendations** for optimal growth
        4. **View predictions** for Plant Growth Index
        """)


def render_input_form():
    """Render input form for soil properties."""
    st.header("📝 Soil Properties Input")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Basic Properties")

        soil_type = st.selectbox(
            "Soil Type",
            options=["sandy", "clay", "mixed"],
            index=0,
            help="Select the predominant soil type"
        )

        ph = st.slider(
            "pH Value",
            min_value=4.0,
            max_value=9.0,
            value=6.5,
            step=0.1,
            help="Soil acidity/alkalinity (optimal: 6.0-7.5)"
        )

        ec = st.slider(
            "EC Value (dS/m)",
            min_value=0.0,
            max_value=8.0,
            value=1.5,
            step=0.1,
            help="Electrical Conductivity - salinity indicator (optimal: < 2.0)"
        )

    with col2:
        st.subheader("Environmental Conditions")

        temperature = st.slider(
            "Temperature (°C)",
            min_value=10.0,
            max_value=40.0,
            value=25.0,
            step=0.5,
            help="Soil temperature (optimal: 20-30°C)"
        )

        humidity = st.slider(
            "Humidity (%)",
            min_value=20.0,
            max_value=90.0,
            value=60.0,
            step=1.0,
            help="Soil moisture percentage (optimal: 40-70%)"
        )

    return {
        'soil_type': soil_type,
        'ph': ph,
        'ec': ec,
        'temperature': temperature,
        'humidity': humidity
    }


def validate_and_show_warnings(soil_data):
    """Validate soil data and show warnings."""
    is_valid, validation_results = validate_input_data(soil_data)

    # Show errors if any
    if validation_results['soil_errors']:
        st.error("❌ **Validation Errors:**")
        for field, error in validation_results['soil_errors'].items():
            st.error(f"- {field}: {error}")
        return False

    # Show warnings if any
    if validation_results['soil_warnings']:
        st.warning("⚠️ **Warnings:**")
        for field, warning in validation_results['soil_warnings'].items():
            st.warning(f"- {warning}")

    return True


def render_results(recommendation, predictor, soil_data):
    """Render recommendation results."""
    st.header("🎯 Recommendations")

    # Main metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="🌿 Biochar",
            value=f"{recommendation.biochar_g}g",
            help="Nano-biochar per 100g soil"
        )

    with col2:
        st.metric(
            label="🧪 NPK Fertilizer",
            value=f"{recommendation.npk_g}g",
            help="NPK fertilizer per 100g soil"
        )

    with col3:
        st.metric(
            label="📈 Expected PGI",
            value=f"{recommendation.predicted_pgi:.1f}",
            help="Plant Growth Index (0-100 scale)"
        )

    with col4:
        st.metric(
            label="🎯 Confidence",
            value=f"{recommendation.confidence:.1%}",
            help="Model prediction confidence"
        )

    st.divider()

    # Soil status
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("🌍 Soil Status")
        status_color = "#66BB6A" if "Balanced" in recommendation.soil_status else "#FFA726"
        st.markdown(
            f'<div style="background-color: {status_color}20; padding: 1rem; '
            f'border-radius: 0.5rem; border-left: 4px solid {status_color};">'
            f'<h3 style="margin:0; color: {status_color};">{recommendation.soil_status}</h3>'
            f'</div>',
            unsafe_allow_html=True
        )

    with col2:
        st.subheader("📊 Total Amendment")
        total = recommendation.biochar_g + recommendation.npk_g
        st.info(f"**{total}g** total amendment per 100g soil")
        ratio = recommendation.biochar_g / (recommendation.npk_g + 0.1)
        st.info(f"**Biochar:NPK ratio** = {ratio:.2f}:1")

    st.divider()

    # Comparison with standard formulations
    st.subheader("📋 Comparison with Standard Formulations")

    try:
        comparison_df = predictor.compare_formulations(recommendation.soil_sample)

        # Style the dataframe
        styled_df = comparison_df.style.background_gradient(
            subset=['Predicted PGI'],
            cmap='Greens'
        ).format({
            'Biochar (g)': '{:.1f}',
            'NPK (g)': '{:.1f}',
            'Total Amendment (g)': '{:.1f}',
            'Predicted PGI': '{:.2f}'
        })

        st.dataframe(styled_df, use_container_width=True)

    except Exception as e:
        st.warning(f"Could not generate comparison: {e}")


def render_insights(soil_data, recommendation):
    """Render insights and recommendations."""
    st.header("💡 Insights & Recommendations")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔍 Analysis")

        insights = []

        # pH insights
        if soil_data['ph'] < 6.0:
            insights.append("🔸 Soil is acidic. Biochar can help buffer pH.")
        elif soil_data['ph'] > 7.5:
            insights.append("🔸 Soil is alkaline. Consider pH management.")
        else:
            insights.append("✅ pH is in optimal range (6.0-7.5)")

        # EC insights
        if soil_data['ec'] >= 4.0:
            insights.append("🔸 High salinity detected. Biochar helps reduce salt stress.")
        elif soil_data['ec'] >= 2.0:
            insights.append("🔸 Moderate salinity. Biochar improves nutrient retention.")
        else:
            insights.append("✅ Low salinity - ideal conditions")

        # Temperature insights
        if 20 <= soil_data['temperature'] <= 30:
            insights.append("✅ Temperature is optimal for growth")
        elif soil_data['temperature'] < 20:
            insights.append("🔸 Temperature is low. Consider warming soil.")
        else:
            insights.append("🔸 Temperature is high. Ensure adequate moisture.")

        # Soil type insights
        if soil_data['soil_type'] == 'sandy':
            insights.append("🔸 Sandy soil: Higher biochar helps retain water and nutrients.")
        elif soil_data['soil_type'] == 'clay':
            insights.append("🔸 Clay soil: Good nutrient retention, watch for waterlogging.")
        else:
            insights.append("✅ Mixed soil provides balanced characteristics")

        for insight in insights:
            st.info(insight)

    with col2:
        st.subheader("📌 Best Practices")

        st.markdown("""
        **Application Tips:**
        - Mix biochar thoroughly with soil
        - Apply NPK evenly across the area
        - Water adequately after application
        - Monitor plant response over 2-4 weeks

        **Expected Benefits:**
        - Improved nutrient retention
        - Better water holding capacity
        - Reduced nutrient leaching
        - Enhanced plant growth
        - Increased yield potential
        """)


def main():
    """Main application logic."""
    # Render header and sidebar
    render_header()
    render_sidebar()

    # Load model
    with st.spinner("Loading AI model..."):
        predictor = load_model()

    st.success("✅ Model loaded and ready!")

    # Input form
    soil_data = render_input_form()

    st.divider()

    # Analyze button
    if st.button("🔬 Analyze Soil & Get Recommendations", type="primary", use_container_width=True):
        # Validate input
        if not validate_and_show_warnings(soil_data):
            st.stop()

        # Create soil sample
        try:
            soil_sample = SoilSample(
                soil_type=soil_data['soil_type'],
                ph=soil_data['ph'],
                ec=soil_data['ec'],
                temperature=soil_data['temperature'],
                humidity=soil_data['humidity']
            )

            # Get recommendation
            with st.spinner("🤖 AI is analyzing your soil..."):
                recommendation = predictor.recommend_formulation(soil_sample)

            st.success("✅ Analysis complete!")

            # Show results
            render_results(recommendation, predictor, soil_data)

            st.divider()

            # Show insights
            render_insights(soil_data, recommendation)

        except Exception as e:
            st.error(f"❌ Error during analysis: {str(e)}")
            st.exception(e)


if __name__ == "__main__":
    main()
