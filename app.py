import streamlit as st
import joblib
import pandas as pd
import numpy as np

# 1. Load both packages independently
@st.cache_resource
def load_ml_assets():
    preprocessor = joblib.load("prediction_app/preprocessor.pkl")
    model = joblib.load("prediction_app/house_model.pkl")
    return preprocessor, model

preprocessor, model = load_ml_assets()

st.set_page_config(page_title="Housing Prediction App", layout="wide")
st.title("🏡 Housing Close Price Predictor")
st.write("Fill in the property specifications below to generate an automated valuation for the closing price.")
st.divider()

# 2. Layout Inputs using Columns (Grouped by Category)
tab1, tab2, tab3 = st.tabs(["📏 Property Basics", "📍 Location & Style", "✨ Features & Details"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        living_area = st.number_input("Living Area (Sq Ft)", min_value=100, max_value=20000, value=1800)
        lot_size_area = st.number_input("Lot Size Area", min_value=0.0, max_value=500.0, value=0.25)
        lot_sqft = st.number_input("Lot Size (Sq Ft)", min_value=0, max_value=1000000, value=7500)
    with col2:
        beds = st.number_input("Bedrooms Total", min_value=0, max_value=10, value=3, step=1)
        baths = st.number_input("Bathrooms Total", min_value=0, max_value=10, value=2, step=1)
        stories = st.number_input("Stories", min_value=1, max_value=4, value=1, step=1)

with tab2:
    col3, col4 = st.columns(2)
    with col3:
        city = st.text_input("City", value="Los Angeles")
        state = st.text_input("State or Province", value="CA")
        postal_code = st.text_input("Postal Code", value="90001")
        subdivision = st.text_input("Subdivision Name", value="Main Street Sub")
    with col4:
        latitude = st.number_input("Latitude", value=34.0522, format="%.4f")
        longitude = st.number_input("Longitude", value=-118.2437, format="%.4f")
        county = st.text_input("County or Parish", value="Los Angeles County")
        mls_area = st.text_input("MLS Area Major", value="101 - Downtown")
        high_school_dist = st.text_input("High School District", value="LAUSD")
        district_name = st.text_input("District Name", value="Central")

with tab3:
    col5, col6 = st.columns(2)
    with col5:
        year_built = st.number_input("Year Built", min_value=1800, max_value=2026, value=2000)
        age_property = st.number_input("Age of Property (Years)", min_value=0, max_value=250, value=26)
        levels = st.text_input("Levels", value="One")
        assoc_fee = st.number_input("Association Fee ($)", min_value=0, value=0)
    with col6:
        parking_total = st.number_input("Total Parking Spaces", min_value=0, max_value=10, value=2)
        garage_spaces = st.number_input("Garage Spaces", min_value=0, max_value=10, value=2)
        
        # Checkboxes for Boolean (YN) fields
        view_yn = "Y" if st.checkbox("Has a View?") else "N"
        pool_yn = "Y" if st.checkbox("Private Pool?") else "N"
        garage_yn = "Y" if st.checkbox("Attached Garage?") else "N"
        fireplace_yn = "Y" if st.checkbox("Fireplace?") else "N"
        new_const_yn = "Y" if st.checkbox("New Construction?") else "N"

# 3. Handle Prediction Button
st.divider()
if st.button("Generate Valuation", type="primary", use_container_width=True):
    
    # Calculate synthetic column if required by model, otherwise calculate dummy values
    bed_bath_ratio = beds / baths if baths > 0 else beds
    
    # BUILD DATAFRAME WITH EXACT COLUMN NAME MATCHES AND PROPER TYPES
    input_data = {
        'ViewYN': view_yn, 'PoolPrivateYN': pool_yn, 'AttachedGarageYN': garage_yn, 
        'FireplaceYN': fireplace_yn, 'NewConstructionYN': new_const_yn,
        'City': city, 'StateOrProvince': state, 'PostalCode': postal_code, 
        'SubdivisionName': subdivision, 'CountyOrParish': county, 
        'MLSAreaMajor': mls_area, 'HighSchoolDistrict': high_school_dist, 
        'DistrictName': district_name, 'Levels': levels,
        'LivingArea': float(living_area), 'LotSizeArea': float(lot_size_area), 
        'LotSizeSquareFeet': float(lot_sqft), 'Latitude': float(latitude), 
        'Longitude': float(longitude), 'YearBuilt': int(year_built), 
        'AgeProperty': float(age_property), 'BathroomsTotalInteger': int(baths), 
        'BedroomsTotal': int(beds), 'Stories': int(stories), 
        'ParkingTotal': float(parking_total), 'GarageSpaces': float(garage_spaces), 
        'AssociationFee': float(assoc_fee), 'BedBathRatio': float(bed_bath_ratio),
        
        # DUMMY VALUES FOR TARGET VARIABLES (If pipeline expects them to be in the input shape)
        'CloseDate': pd.Timestamp.now().strftime('%Y-%m-%d'), # Dummy current date 
        'ClosePrice': 0.0                                     # Dummy placeholder target
    }
    
    # Convert into a 1-row Pandas DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Reorder columns to match your exact Index layout 
    exact_columns = [
        'ViewYN', 'PoolPrivateYN', 'CloseDate', 'ClosePrice', 'Latitude',
        'Longitude', 'LivingArea', 'MLSAreaMajor', 'CountyOrParish',
        'AttachedGarageYN', 'ParkingTotal', 'SubdivisionName', 'YearBuilt',
        'BathroomsTotalInteger', 'City', 'BedroomsTotal', 'StateOrProvince',
        'FireplaceYN', 'Stories', 'Levels', 'LotSizeArea', 'NewConstructionYN',
        'GarageSpaces', 'HighSchoolDistrict', 'PostalCode', 'AssociationFee',
        'LotSizeSquareFeet', 'DistrictName', 'BedBathRatio', 'AgeProperty'
    ]
    input_df = input_df[exact_columns]
    
    try:
        # Step A: Pass raw text/numbers through the preprocessing pipeline transforms
        processed_features = preprocessor.transform(input_df)
        
        # Step B: Pass transformed features into the model to predict
        predicted_price = model.predict(processed_features)
        
        # Step C: Output Result
        st.success(f"### Market Close Price Estimate: **${predicted_price[0]:,.2f}**")
        
    except Exception as e:
        st.error(f"Prediction failed. Verify pipeline inputs. Error trace: {e}")
