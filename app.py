import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="SalesForecaster Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. LOAD STYLESHEET ---
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
load_css('style.css')

# --- 3. LOAD ARTIFACTS and STATIC DATA ---
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load('xgb_model.joblib')
        scaler = joblib.load('scaler.joblib')
        training_columns = joblib.load('training_columns.joblib')
    except FileNotFoundError:
        st.error("Fatal Error: Model files not found. Please ensure 'xgb_model.joblib', 'scaler.joblib', and 'training_columns.joblib' are present.")
        st.stop()
        
    # --- !! IMPORTANT !! ---
    # Hardcode static values from your notebook for app visualizations.
    # Replace these with the actual computed values from your final notebook.
    static_data = {
        "r2_score": 0.92,
        "mse": 23456789.01,
        "feature_importances": model.feature_importances_,
        "feature_ranges": {
            'Temperature': (0.0, 100.0), 'Fuel_Price': (2.0, 5.0),
            'CPI': (120.0, 230.0), 'Unemployment': (3.0, 15.0)
        },
        # Mock validation data for plots - replace with a sample of your actual validation data
        "validation_sample": {
            'y_true': np.random.uniform(5000, 40000, 100),
            'y_pred': np.random.uniform(5000, 40000, 100)
        }
    }
    # Add a bit of correlation for a more realistic residual plot
    static_data["validation_sample"]['y_pred'] = static_data["validation_sample"]['y_true'] * np.random.uniform(0.8, 1.2, 100) + np.random.normal(0, 3000, 100)

    return model, scaler, training_columns, static_data

model, scaler, training_columns, static_data = load_artifacts()

# --- 4. SIDEBAR - INPUT CONTROLS ---
with st.sidebar:
    st.markdown(
        "<h1 style='color: #00BFFF; text-shadow: 0 0 10px rgba(0, 191, 255, 0.7);'>Walmart Sales</h1>",
        unsafe_allow_html=True
    )
    st.markdown("### Input Parameters")
    st.markdown("Adjust the values below to generate a new forecast.")

    store_id = st.slider("Store ID", 1, 45, 1, help="Select the store number (1-45).")
    date = st.date_input("Date", help="Select the date for the forecast.")

    st.markdown("### Economic Factors")
    temperature = st.slider("Temperature (°F)", 0.0, 100.0, 68.0, help="Average temperature for the week.")
    fuel_price = st.slider("Fuel Price ($)", 2.0, 5.0, 3.50, help="Regional fuel price.")
    cpi = st.slider("Consumer Price Index (CPI)", 120.0, 230.0, 170.0, help="Consumer Price Index.")
    unemployment = st.slider("Unemployment Rate (%)", 3.0, 15.0, 8.0, help="Unemployment rate.")

# --- 5. MAIN PAGE ---
st.title("SalesForecaster Pro")

# --- A. PREDICTION LOGIC ---
date = pd.to_datetime(date)
input_data = {
    'Store': store_id, 'Dept': 1, 'IsHoliday': int(date.weekday() > 4), 'Temperature': temperature,
    'Fuel_Price': fuel_price, 'MarkDown1': 0, 'MarkDown2': 0, 'MarkDown3': 0, 'MarkDown4': 0, 'MarkDown5': 0,
    'CPI': cpi, 'Unemployment': unemployment, 'Type': 1, 'Size': 151315, 'Year': date.year,
    'Month': date.month, 'Day': date.day, 'WeekOfYear': date.isocalendar().week,
    'Weekly_Sales_Lag1': 0, 'Weekly_Sales_Lag4': 0
}
input_df = pd.DataFrame([input_data])
final_df = pd.DataFrame(columns=training_columns)
final_df = pd.concat([final_df, input_df], ignore_index=True).fillna(0)
final_df = final_df[training_columns]
scaled_data = scaler.transform(final_df)
predicted_sales = model.predict(scaled_data)[0]

# --- B. FORECAST & CONFIDENCE DISPLAY ---
st.markdown('<div class="custom-container">', unsafe_allow_html=True)
col1, col2 = st.columns([2,1.5])
with col1:
    st.metric(label="Predicted Weekly Sales", value=f"${predicted_sales:,.2f}")
# --- B. FORECAST & CONFIDENCE DISPLAY ---
# ... (st.metric code is here) ...
with col2:
    confidence_score = max(0, (static_data["r2_score"] - 0.5) / 0.5) * 100
    
    # MODIFICATION 1: The mode is now just "gauge". The 'number' property is removed.
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge",
        value=confidence_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={'axis': {'range': [0, 100], 'tickvals': [0, 20, 40, 60, 80, 100]}, # Added tickvals for clarity
               'bar': {'color': "#00A2FF"},
               'bgcolor': "#1E1E1E", 'borderwidth': 2, 'bordercolor': "#2A2A2A"}
    ))

    # MODIFICATION 2: We manually add the number as a centered annotation.
    fig_gauge.add_annotation(
        x=0.5, y=0.4, # Positioned precisely in the center of the gauge arc
        text=f"{confidence_score:.0f}%", # The number to display
        showarrow=False,
        font=dict(size=32, color="white"),
        xref="paper", yref="paper" # Use figure-relative coordinates
    )
    
    # The layout update remains, keeping the title correctly positioned at the top.
      # The layout update, with the corrected title color.
    fig_gauge.update_layout(
        height=150,
        paper_bgcolor="rgba(0,0,0,0)",
        font={'color': "white"},
        title={
            'text': "Model Confidence Score",
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'color': '#00BFFF', 'size': 16} # <-- The color is now correctly set
        },
        margin=dict(l=20, r=20, t=40, b=0)
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

# --- C. ANALYTICAL TABS ---
tab1, tab2, tab3, tab4 = st.tabs(["Feature Contribution", "What-If Analysis", "Input Context", "Model Deep Dive"])

with tab1:
    st.markdown("## Feature Contribution Analysis")
    st.info("This waterfall chart shows how much each feature contributed to pushing the prediction away from the dataset's average prediction. Positive values increase the forecast, negative values decrease it.")

    # Using pre-calculated feature importance for contribution estimate
    base_value = 20000  # Assume a baseline prediction
    contributions = {}
    for feature in ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Store']:
        importance_score = static_data['feature_importances'][training_columns.index(feature)]
        feature_value = final_df[feature].iloc[0]
        # This is a proxy for SHAP - not exact but illustrates the principle
        contribution = (feature_value - static_data['feature_ranges'].get(feature, (0,0))[0]) * importance_score * 100
        contributions[feature] = contribution
    
    fig_waterfall = go.Figure(go.Waterfall(
        name="Contribution", orientation="v", measure=["relative"] * len(contributions) + ["total"],
        x=list(contributions.keys()) + ["Final Prediction"],
        text=[f"${v:,.0f}" for v in contributions.values()] + [f"${predicted_sales:,.0f}"],
        y=list(contributions.values()) + [predicted_sales],
        base=base_value,
        connector={"line":{"color":"#A0A0A0"}},
        increasing={"marker":{"color":"#00A2FF"}},
        decreasing={"marker":{"color":"#E94F64"}},
    ))
        # --- REPLACEMENT ---
    fig_waterfall.update_layout(
        title_text="How Features Influence the Final Forecast",
        title_font_color="#00BFFF",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#FFFFFF", # Main font color
        xaxis=dict(
            tickfont_color="#FFFFFF",
            gridcolor="#404040"
        ),
        yaxis=dict(
            tickfont_color="#FFFFFF",
            gridcolor="#404040"
        )
    )
    st.plotly_chart(fig_waterfall, use_container_width=True)
    # --- END REPLACEMENT ---

with tab2:
    st.markdown("## What-If Scenario Analysis")
    st.info("Select a feature and see how the sales forecast changes as its value is adjusted across a plausible range. This helps understand feature sensitivity.")
    
    # --- REPLACEMENT ---
    st.markdown("### Select Feature to Analyze")
    feature_to_analyze = st.selectbox(
    label="Select Feature to Analyze", # This label is now hidden by the CSS below
    options=('Temperature', 'Fuel_Price', 'CPI', 'Unemployment'),
    label_visibility="collapsed" # This hides the default label
)
# --- END REPLACEMENT ---
    range_min, range_max = static_data['feature_ranges'][feature_to_analyze]
    scenario_values = np.linspace(range_min, range_max, 50)
    predictions_scenario = []
    for val in scenario_values:
        temp_df = final_df.copy()
        temp_df[feature_to_analyze] = val
        scaled_temp = scaler.transform(temp_df)
        predictions_scenario.append(model.predict(scaled_temp)[0])

    fig_scenario = go.Figure()
    fig_scenario.add_trace(go.Scatter(x=scenario_values, y=predictions_scenario, mode='lines', line=dict(color='#00A2FF', width=3), name='Forecast'))
    current_val_prediction = predicted_sales
    current_val_feature = final_df[feature_to_analyze].iloc[0]
    fig_scenario.add_trace(go.Scatter(x=[current_val_feature], y=[current_val_prediction], mode='markers', marker=dict(color='#E94F64', size=15, symbol='x'), name='Current Selection'))
        # --- REPLACEMENT ---
    fig_scenario.update_layout(
        title_text=f"Sales vs. {feature_to_analyze}",
        title_font_color="#00BFFF",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend_font_color="#FFFFFF",
        xaxis=dict(
            title_font_color="#FFFFFF",
            tickfont_color="#FFFFFF",
            gridcolor="#404040"
        ),
        yaxis=dict(
            title_font_color="#FFFFFF",
            tickfont_color="#FFFFFF",
            gridcolor="#404040"
        )
    )
    st.plotly_chart(fig_scenario, use_container_width=True)
    # --- END REPLACEMENT ---

with tab3:
    st.markdown("## Input Context Gauges")
    st.info("These gauges show where your current input values fall within the typical range of values seen in the training data (Low, Average, High).")

    gauge_cols = st.columns(4)
    features_for_gauges = {'Temperature': temperature, 'Fuel_Price': fuel_price, 'CPI': cpi, 'Unemployment': unemployment}
    
    for i, (feature, value) in enumerate(features_for_gauges.items()):
        min_val, max_val = static_data['feature_ranges'][feature]
        avg_val = (min_val + max_val) / 2
        with gauge_cols[i]:
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=value, title={'text': feature, 'font': {'size': 16, 'color': '#F5F5F5'}},
                gauge={'axis': {'range': [min_val, max_val]}, 'bar': {'color': '#00A2FF'},
                       'steps': [{'range': [min_val, min_val + (max_val - min_val)*0.33], 'color': '#2A2A2A'},
                                 {'range': [min_val + (max_val - min_val)*0.33, min_val + (max_val - min_val)*0.66], 'color': '#404040'}]
                      },
                number={'font': {'color': '#F5F5F5'}}
            ))
            fig.update_layout(height=200, paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"}, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)
            
with tab4:
    st.markdown("## Model Deep Dive")
    st.info("Explore the model's overall performance and diagnostic plots based on the validation set results from the training phase.")
    
    st.markdown("### Overall Performance Metrics")
    perf_col1, perf_col2 = st.columns(2)
    perf_col1.metric("R² Score (Validation)", f'{static_data["r2_score"]:.2f}')
    perf_col2.metric("Mean Squared Error (MSE)", f'{static_data["mse"]:,.0f}')
    
    st.markdown("### Diagnostic Plots")
    diag_col1, diag_col2 = st.columns(2)
    
    with diag_col1:
        st.markdown("#### Residuals Plot")
        y_true_sample = static_data["validation_sample"]['y_true']
        y_pred_sample = static_data["validation_sample"]['y_pred']
        residuals = y_true_sample - y_pred_sample
        fig_res = go.Figure()
        fig_res.add_trace(go.Scatter(x=y_pred_sample, y=residuals, mode='markers', marker=dict(color='#00A2FF', opacity=0.6)))
        fig_res.add_hline(y=0, line_dash="dash", line_color="#E94F64")
               # --- REPLACEMENT ---
        fig_res.update_layout(
            title_text="Predicted vs. Residuals",
            title_font_color="#00BFFF",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#1E1E1E",
            xaxis=dict(
                title_font_color="#FFFFFF",
                tickfont_color="#FFFFFF",
                gridcolor="#404040" # Subtle grid line color
            ),
            yaxis=dict(
                title_font_color="#FFFFFF",
                tickfont_color="#FFFFFF",
                gridcolor="#404040" # Subtle grid line color
            )
        )
        st.plotly_chart(fig_res, use_container_width=True)
        # --- END REPLACEMENT ---

    with diag_col2:
        st.markdown("#### Prediction Error Distribution")
        fig_err = go.Figure()
        fig_err.add_trace(go.Histogram(x=residuals, marker_color='#00A2FF'))
                # --- REPLACEMENT ---
        fig_err.update_layout(
            title_text="Distribution of Prediction Errors",
            title_font_color="#00BFFF",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#1E1E1E",
            bargap=0.1, # A little gap between bars
            xaxis=dict(
                title_font_color="#FFFFFF",
                tickfont_color="#FFFFFF",
                gridcolor="#404040"
            ),
            yaxis=dict(
                title_font_color="#FFFFFF",
                tickfont_color="#FFFFFF",
                gridcolor="#404040"
            )
        )
        st.plotly_chart(fig_err, use_container_width=True)
        # --- END REPLACEMENT ---