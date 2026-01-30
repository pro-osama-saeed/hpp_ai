import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt # New library for nicer charts (built-in to Streamlit)
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor

# --- 1. Page Config (Browser Title & Icon) ---
st.set_page_config(page_title="California Housing AI", page_icon="🏡", layout="wide")

# --- 2. Title and Header ---
st.title("🏡 California Real Estate AI")
st.markdown("""
<style>
    .big-font { font-size:20px !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
Welcome! Adjust the sliders to design a house, and our **Random Forest AI** will estimate its market value based on 1990 California census data.
""")
st.divider()

# --- 3. Load and Train (Cached) ---
@st.cache_resource
def load_and_train():
    housing = fetch_california_housing()
    df = pd.DataFrame(housing.data, columns=housing.feature_names)
    df['PRICE'] = housing.target
    X = df.drop('PRICE', axis=1)
    y = df['PRICE']
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)
    return model, df['PRICE'].mean() # Return average price too

with st.spinner("Warming up the AI engine..."):
    model, avg_price_statewide = load_and_train()

# --- 4. The Layout (Columns) ---
# We create two columns: Left (Controls) and Right (Results)
col1, col2 = st.columns([1, 2], gap="medium")

with col1:
    st.subheader("⚙️ House Specs")
    
    # Tooltips add helpful context
    MedInc = st.slider('Median Income ($10k)', 0.5, 15.0, 3.0, help="Income of the neighborhood.")
    HouseAge = st.slider('House Age (Years)', 1, 52, 20)
    AveRooms = st.slider('Average Rooms', 1.0, 10.0, 5.0)
    AveBedrms = st.slider('Average Bedrooms', 0.5, 5.0, 1.0)
    Population = st.slider('Population', 200, 5000, 1000)
    AveOccup = st.slider('Average Occupancy', 1.0, 6.0, 3.0)
    
    st.markdown("### 📍 Location")
    Latitude = st.slider('Latitude', 32.0, 42.0, 34.0)
    Longitude = st.slider('Longitude', -124.0, -114.0, -118.0)

    # DataFrame for the model
    input_data = {'MedInc': MedInc, 'HouseAge': HouseAge, 'AveRooms': AveRooms, 
                  'AveBedrms': AveBedrms, 'Population': Population, 'AveOccup': AveOccup,
                  'Latitude': Latitude, 'Longitude': Longitude}
    input_df = pd.DataFrame(input_data, index=[0])

with col2:
    st.subheader("📊 Prediction Dashboard")
    
    # Auto-predict (No button needed, but we can keep it for effect)
    prediction = model.predict(input_df)[0]
    real_price = prediction * 100000
    avg_real_price = avg_price_statewide * 100000
    
    # Calculate difference
    diff = real_price - avg_real_price
    
    # Display Big Metric
    st.metric(
        label="Estimated Market Value", 
        value=f"${real_price:,.0f}", 
        delta=f"{diff:,.0f} vs State Avg"
    )
    
    # Chart: Compare to Average
    chart_data = pd.DataFrame({
        'Category': ['Your House', 'State Average'],
        'Price': [real_price, avg_real_price]
    })
    
    # Altair Chart (Looks modern)
    c = alt.Chart(chart_data).mark_bar().encode(
        x='Category',
        y='Price',
        color=alt.condition(
            alt.datum.Category == 'Your House',
            alt.value('green'),  # The user's house is Green
            alt.value('gray')   # Average is Gray
        )
    )
    st.altair_chart(c, use_container_width=True)

    # Map with Zoom
    st.markdown("### 🗺️ Location Preview")
    map_df = input_df.rename(columns={'Latitude': 'latitude', 'Longitude': 'longitude'})
    st.map(map_df, zoom=6) # Set zoom level

# --- 5. Hidden Details ---
with st.expander("ℹ️ See Raw Input Data"):
    st.write(input_df)
    st.caption("This data is fed directly into the Random Forest model.")
