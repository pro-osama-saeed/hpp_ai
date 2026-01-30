import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# --- 1. Title and Description ---
st.title("🏡 California House Price Predictor")
st.write("""
### Enter the details of a house to estimate its price.
This app uses a **Random Forest** model trained on 1990 census data.
""")

# --- 2. Load and Train the Model (Cached) ---
# We use @st.cache_resource so it only trains ONCE, not every time you click a button.
@st.cache_resource
def load_and_train():
    housing = fetch_california_housing()
    df = pd.DataFrame(housing.data, columns=housing.feature_names)
    df['PRICE'] = housing.target
    
    X = df.drop('PRICE', axis=1)
    y = df['PRICE']
    
    # Train the model
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)
    
    return model, housing.feature_names

# Show a loading spinner while training
with st.spinner("Training the AI... (This may take a moment)"):
    model, feature_names = load_and_train()

st.success("Model Trained Successfully!")

# --- 3. Create Sidebar for User Input ---
st.sidebar.header("Specify House Details")

def user_input_features():
    # We create a slider for every feature in the dataset
    MedInc = st.sidebar.slider('Median Income (in $10k)', 0.5, 15.0, 3.0)
    HouseAge = st.sidebar.slider('House Age (Years)', 1, 52, 20)
    AveRooms = st.sidebar.slider('Average Rooms', 1.0, 10.0, 5.0)
    AveBedrms = st.sidebar.slider('Average Bedrooms', 0.5, 5.0, 1.0)
    Population = st.sidebar.slider('Population in Area', 200, 5000, 1000)
    AveOccup = st.sidebar.slider('Average Occupancy', 1.0, 6.0, 3.0)
    Latitude = st.sidebar.slider('Latitude', 32.0, 42.0, 34.0)
    Longitude = st.sidebar.slider('Longitude', -124.0, -114.0, -118.0)
    
   # ... inside user_input_features() ...

    # OLD CODE (Causes Error) ❌
    # data = { ...
    #         'Latitude': Latitude,
    #         'Longitude': Longitude}

   def user_input_features():
    # ... sliders ...
    
    # CHANGE THESE BACK TO CAPITALIZED
    data = {'MedInc': MedInc,
            'HouseAge': HouseAge,
            'AveRooms': AveRooms,
            'AveBedrms': AveBedrms,
            'Population': Population,
            'AveOccup': AveOccup,
            'Latitude': Latitude,   # Capital L (For the AI)
            'Longitude': Longitude} # Capital L (For the AI)
    
    return pd.DataFrame(data, index=[0])

# Get the input from the user
input_df = user_input_features()

# --- 4. Show User Input ---
# ... inside the button logic ...

    st.subheader(f"💰 Estimated Price: ${real_price:,.2f}")
    
    # NEW CODE: Create a copy with lowercase names just for the map
    map_df = input_df.rename(columns={'Latitude': 'latitude', 'Longitude': 'longitude'})
    st.map(map_df)

# --- 5. Make Prediction ---
if st.button("Predict Price"):
    prediction = model.predict(input_df)
    
    # The dataset price is in $100,000s, so we multiply by 100k
    real_price = prediction[0] * 100000 
    
    st.subheader(f"💰 Estimated Price: ${real_price:,.2f}")
    
    # Bonus: Show where this house is on a map
    st.map(input_df)
