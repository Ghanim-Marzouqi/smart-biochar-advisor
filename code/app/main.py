"""
Smart Biochar Advisor - Streamlit Application
Main application entry point with enhanced UI/UX
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import io
import sys
from pathlib import Path
from datetime import datetime

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
    .preset-button {
        margin: 0.25rem;
    }
    </style>
""", unsafe_allow_html=True)

# Preset soil scenarios
PRESETS = {
    "Sandy Coastal Soil": {
        "soil_type": "sandy",
        "ph": 7.2,
        "ec": 2.8,
        "temperature": 28.0,
        "humidity": 45.0,
        "description": "Typical coastal sandy soil with moderate salinity"
    },
    "Clay Agricultural Land": {
        "soil_type": "clay",
        "ph": 6.8,
        "ec": 1.5,
        "temperature": 24.0,
        "humidity": 65.0,
        "description": "Rich clay soil suitable for agriculture"
    },
    "Saline Desert Soil": {
        "soil_type": "mixed",
        "ph": 8.2,
        "ec": 4.5,
        "temperature": 35.0,
        "humidity": 25.0,
        "description": "High salinity desert soil requiring remediation"
    },
    "Acidic Forest Soil": {
        "soil_type": "mixed",
        "ph": 5.2,
        "ec": 0.8,
        "temperature": 18.0,
        "humidity": 75.0,
        "description": "Acidic forest soil with low salinity"
    }
}


def create_pgi_gauge(pgi_value):
    """Create a gauge chart for PGI score."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=pgi_value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Plant Growth Index", 'font': {'size': 20}},
        delta={'reference': 70, 'increasing': {'color': "#66BB6A"}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#1B5E20"},
            'bar': {'color': "#2E7D32"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#1B5E20",
            'steps': [
                {'range': [0, 50], 'color': '#FFCDD2'},
                {'range': [50, 70], 'color': '#FFF9C4'},
                {'range': [70, 85], 'color': '#C8E6C9'},
                {'range': [85, 100], 'color': '#66BB6A'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))

    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': "#1B5E20", 'family': "Arial"}
    )

    return fig


def create_comparison_chart(comparison_df):
    """Create bar chart for formulation comparison."""
    fig = go.Figure()

    fig.add_trace(go.Bar(
        name='Biochar',
        x=comparison_df['Formulation'],
        y=comparison_df['Biochar (g)'],
        marker_color='#66BB6A'
    ))

    fig.add_trace(go.Bar(
        name='NPK',
        x=comparison_df['Formulation'],
        y=comparison_df['NPK (g)'],
        marker_color='#FFA726'
    ))

    fig.update_layout(
        title="Standard Formulations Comparison",
        xaxis_title="Formulation",
        yaxis_title="Amount (g per 100g soil)",
        barmode='group',
        height=400,
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig


def generate_pdf_report(recommendation, soil_data, plot_size=None, total_biochar=None, total_npk=None):
    """Generate a simple text report for download."""
    report = f"""
SMART BIOCHAR ADVISOR - SOIL ANALYSIS REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*60}

SOIL PROPERTIES
{'='*60}
Soil Type:    {soil_data['soil_type'].title()}
pH Value:     {soil_data['ph']}
EC Value:     {soil_data['ec']} dS/m
Temperature:  {soil_data['temperature']}°C
Humidity:     {soil_data['humidity']}%

SOIL STATUS
{'='*60}
{recommendation.soil_status}

RECOMMENDATIONS
{'='*60}
Biochar:      {recommendation.biochar_g}g per 100g soil
NPK:          {recommendation.npk_g}g per 100g soil
Total Amendment: {recommendation.biochar_g + recommendation.npk_g}g per 100g soil
Biochar:NPK Ratio: {recommendation.biochar_g / (recommendation.npk_g + 0.1):.2f}:1

EXPECTED RESULTS
{'='*60}
Plant Growth Index (PGI): {recommendation.predicted_pgi:.1f}/100
Model Confidence: {recommendation.confidence:.1%}
"""

    if plot_size and total_biochar and total_npk:
        report += f"""
APPLICATION CALCULATIONS (for {plot_size}m² plot)
{'='*60}
Total Biochar Needed:  {total_biochar:.2f} kg
Total NPK Needed:      {total_npk:.2f} kg
Total Amendment:       {total_biochar + total_npk:.2f} kg
"""

    report += f"""
INSIGHTS & RECOMMENDATIONS
{'='*60}
"""

    # Add insights
    if soil_data['ph'] < 6.0:
        report += "• Soil is acidic. Biochar will help buffer pH.\n"
    elif soil_data['ph'] > 7.5:
        report += "• Soil is alkaline. Consider pH management alongside biochar.\n"
    else:
        report += "• pH is in optimal range (6.0-7.5)\n"

    if soil_data['ec'] >= 4.0:
        report += "• High salinity detected. Biochar helps reduce salt stress.\n"
    elif soil_data['ec'] >= 2.0:
        report += "• Moderate salinity. Biochar improves nutrient retention.\n"

    if soil_data['soil_type'] == 'sandy':
        report += "• Sandy soil: Higher biochar helps retain water and nutrients.\n"
    elif soil_data['soil_type'] == 'clay':
        report += "• Clay soil: Good nutrient retention, watch for waterlogging.\n"

    report += f"""
APPLICATION INSTRUCTIONS
{'='*60}
1. Mix biochar thoroughly with soil before application
2. Apply NPK evenly across the area
3. Water adequately after application
4. Monitor plant response over 2-4 weeks

EXPECTED BENEFITS
{'='*60}
• Improved nutrient retention
• Better water holding capacity
• Reduced nutrient leaching
• Enhanced plant growth
• Increased yield potential

{'='*60}
Report generated by Smart Biochar Advisor
AI-Powered Soil Analysis System
{'='*60}
"""

    return report


def load_model():
    """Load the trained ML model."""
    try:
        predictor = BiocharPredictor()
        return predictor
    except FileNotFoundError:
        st.warning("⚠️ Model not found. Training model now (this will take ~1 minute)...")

        # Try to train the model automatically
        try:
            from models.train_model import train_all_models
            with st.spinner("🤖 Training ML model..."):
                train_all_models()
            st.success("✅ Model trained successfully! Reloading page...")
            st.rerun()
        except Exception as e:
            st.error(f"""
            ⚠️ **Could not auto-train model**

            Error: {str(e)}

            Please run the training script manually:

            ```bash
            cd code
            python models/train_model.py
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


def render_preset_selector():
    """Render quick preset selector."""
    st.subheader("🚀 Quick Start - Try a Preset")

    cols = st.columns(4)

    for idx, (preset_name, preset_data) in enumerate(PRESETS.items()):
        with cols[idx % 4]:
            if st.button(
                preset_name,
                key=f"preset_{idx}",
                help=preset_data['description'],
                use_container_width=True
            ):
                # Store preset in session state
                for key, value in preset_data.items():
                    if key != 'description':
                        st.session_state[f'preset_{key}'] = value
                st.rerun()


def render_input_form():
    """Render input form for soil properties."""
    st.header("📝 Soil Properties Input")

    # Check if preset was selected
    preset_selected = any(f'preset_{key}' in st.session_state for key in ['soil_type', 'ph', 'ec'])

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Basic Properties")

        soil_type = st.selectbox(
            "Soil Type",
            options=["sandy", "clay", "mixed"],
            index=["sandy", "clay", "mixed"].index(st.session_state.get('preset_soil_type', 'sandy')),
            help="Select the predominant soil type"
        )

        ph = st.slider(
            "pH Value",
            min_value=4.0,
            max_value=9.0,
            value=float(st.session_state.get('preset_ph', 6.5)),
            step=0.1,
            help="Soil acidity/alkalinity (optimal: 6.0-7.5)"
        )

        ec = st.slider(
            "EC Value (dS/m)",
            min_value=0.0,
            max_value=8.0,
            value=float(st.session_state.get('preset_ec', 1.5)),
            step=0.1,
            help="Electrical Conductivity - salinity indicator (optimal: < 2.0)"
        )

    with col2:
        st.subheader("Environmental Conditions")

        temperature = st.slider(
            "Temperature (°C)",
            min_value=10.0,
            max_value=40.0,
            value=float(st.session_state.get('preset_temperature', 25.0)),
            step=0.5,
            help="Soil temperature (optimal: 20-30°C)"
        )

        humidity = st.slider(
            "Humidity (%)",
            min_value=20.0,
            max_value=90.0,
            value=float(st.session_state.get('preset_humidity', 60.0)),
            step=1.0,
            help="Soil moisture percentage (optimal: 40-70%)"
        )

    # Clear preset from session state after use
    if preset_selected:
        for key in list(st.session_state.keys()):
            if key.startswith('preset_'):
                del st.session_state[key]

    return {
        'soil_type': soil_type,
        'ph': ph,
        'ec': ec,
        'temperature': temperature,
        'humidity': humidity
    }


def render_application_calculator(recommendation):
    """Render application rate calculator."""
    st.subheader("📏 Application Rate Calculator")

    st.info("Calculate total amounts needed for your plot size")

    col1, col2 = st.columns(2)

    with col1:
        plot_size = st.number_input(
            "Plot Size (m²)",
            min_value=1.0,
            max_value=10000.0,
            value=100.0,
            step=10.0,
            help="Enter your plot size in square meters"
        )

        # Assuming average soil depth of 20cm and soil density of 1.3 g/cm³
        soil_volume_m3 = plot_size * 0.2  # 20cm depth
        soil_mass_kg = soil_volume_m3 * 1300  # 1.3 g/cm³ = 1300 kg/m³

    with col2:
        soil_depth = st.slider(
            "Soil Depth (cm)",
            min_value=10,
            max_value=50,
            value=20,
            step=5,
            help="Depth of soil to treat"
        )

        soil_volume_m3 = plot_size * (soil_depth / 100)
        soil_mass_kg = soil_volume_m3 * 1300

    # Calculate total amendments
    total_biochar_kg = (soil_mass_kg / 100) * recommendation.biochar_g
    total_npk_kg = (soil_mass_kg / 100) * recommendation.npk_g
    total_amendment_kg = total_biochar_kg + total_npk_kg

    # Display results
    st.markdown("### Required Materials")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Soil Mass",
            f"{soil_mass_kg:.1f} kg",
            help="Estimated soil mass to treat"
        )

    with col2:
        st.metric(
            "🌿 Biochar Needed",
            f"{total_biochar_kg:.2f} kg",
            help="Total biochar required"
        )

    with col3:
        st.metric(
            "🧪 NPK Needed",
            f"{total_npk_kg:.2f} kg",
            help="Total NPK fertilizer required"
        )

    with col4:
        st.metric(
            "📦 Total Amendment",
            f"{total_amendment_kg:.2f} kg",
            help="Total materials needed"
        )

    return plot_size, total_biochar_kg, total_npk_kg


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

    # PGI Gauge and main metrics
    col1, col2 = st.columns([1, 2])

    with col1:
        # PGI Gauge Chart
        pgi_gauge = create_pgi_gauge(recommendation.predicted_pgi)
        st.plotly_chart(pgi_gauge, use_container_width=True)

    with col2:
        # Main metrics
        met_col1, met_col2, met_col3 = st.columns(3)

        with met_col1:
            st.metric(
                label="🌿 Biochar",
                value=f"{recommendation.biochar_g}g",
                help="Nano-biochar per 100g soil"
            )

        with met_col2:
            st.metric(
                label="🧪 NPK Fertilizer",
                value=f"{recommendation.npk_g}g",
                help="NPK fertilizer per 100g soil"
            )

        with met_col3:
            st.metric(
                label="🎯 Confidence",
                value=f"{recommendation.confidence:.1%}",
                help="Model prediction confidence"
            )

        # Additional info
        st.markdown("### 🌍 Soil Status")
        status_color = "#66BB6A" if "Balanced" in recommendation.soil_status else "#FFA726"
        st.markdown(
            f'<div style="background-color: {status_color}20; padding: 1rem; '
            f'border-radius: 0.5rem; border-left: 4px solid {status_color};">'
            f'<h3 style="margin:0; color: {status_color};">{recommendation.soil_status}</h3>'
            f'</div>',
            unsafe_allow_html=True
        )

    st.divider()

    # Application calculator
    plot_size, total_biochar, total_npk = render_application_calculator(recommendation)

    st.divider()

    # Comparison with standard formulations
    st.subheader("📋 Comparison with Standard Formulations")

    try:
        comparison_df = predictor.compare_formulations(recommendation.soil_sample)

        # Bar chart
        comparison_chart = create_comparison_chart(comparison_df)
        st.plotly_chart(comparison_chart, use_container_width=True)

        # Data table
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

    # Download report button
    st.divider()
    st.subheader("💾 Download Report")

    report_text = generate_pdf_report(
        recommendation, soil_data, plot_size, total_biochar, total_npk
    )

    col1, col2 = st.columns([1, 3])

    with col1:
        st.download_button(
            label="📄 Download Report (TXT)",
            data=report_text,
            file_name=f"biochar_recommendation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True
        )

    with col2:
        st.info("💡 Download your personalized soil analysis report for your records")


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

    # Preset selector
    render_preset_selector()

    st.divider()

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
