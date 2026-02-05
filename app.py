import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Page Configuration
st.set_page_config(
    page_title="Social Network Clustering",
    page_icon="🕸️",
    layout="wide"
)

# Load Model Artifacts
@st.cache_resource
def load_artifacts():
    try:
        with open('model_data.pkl', 'rb') as f:
            artifacts = pickle.load(f)
        return artifacts
    except FileNotFoundError:
        st.error("Model artifacts not found. Please run 'generate_model.py' first.")
        return None

artifacts = load_artifacts()

if artifacts:
    kmeans = artifacts['model']
    scaler = artifacts['scaler']
    le = artifacts['encoder']
    feature_names = artifacts['feature_names']
    interest_cols = artifacts['interest_cols']

    # --- Header ---
    st.title("🕸️ Social Network Profile Clustering")
    st.markdown("""
    This app predicts which social cluster a user belongs to based on their demographics and interests.
    Adjust the inputs below to see the prediction.
    """)
    st.divider()

    # --- Sidebar: Demographics ---
    with st.sidebar:
        st.header("Demographics")
        age = st.number_input("Age", min_value=13, max_value=100, value=18, step=1)
        
        # Gender - Inspect classes to ensure correct mapping
        # Assuming classes are strings like 'F', 'M' etc.
        gender_options = list(le.classes_) if hasattr(le, 'classes_') else ['F', 'M']
        gender = st.selectbox("Gender", options=gender_options)

    # --- Main Area: Interests ---
    st.header("Interests (Rate 0-10)")
    
    # Create input dictionary
    input_data = {}
    
    # Grid Layout for Sliders
    # We have ~36 interests. 4 columns works well.
    cols = st.columns(4)
    
    for i, interest in enumerate(interest_cols):
        col = cols[i % 4]
        with col:
            # Clean up interest name for display
            display_name = interest.replace('_', ' ').title()
            val = st.slider(display_name, min_value=0, max_value=10, value=0, key=interest)
            input_data[interest] = val

    st.divider()

    # --- Prediction ---
    if st.button("Predict Cluster", type="primary", use_container_width=True):
        try:
            # 1. Prepare Input Dataframe matching feature order
            # Features expected: [Interests] + [Age, Gender_Enc]
            
            # Encode Gender
            # Handle potential unseen labels gracefully, though selectbox prevents it mostly
            try:
                gender_enc = le.transform([gender])[0]
            except ValueError:
                # Fallback if something weird happens
                gender_enc = 0 
                st.warning("Unknown gender value, using default.")

            # Create dict for dataframe creation
            final_input = input_data.copy()
            final_input['age'] = age
            final_input['gender_enc'] = gender_enc
            
            # Construct DataFrame ensuring column order matches training
            input_df = pd.DataFrame([final_input])
            
            # Reorder columns to match scaler's expectation
            # Use 'feature_names' from artifacts if available, otherwise reconstruct
            if 'feature_names' in artifacts:
                input_df = input_df[artifacts['feature_names']]
            
            # 2. Scale Features
            input_scaled = scaler.transform(input_df)
            
            # 3. Predict
            cluster = kmeans.predict(input_scaled)[0]
            
            # 4. Display Result
            st.success(f"### Predicted Cluster: {cluster}")
            
            # Optional: Add interpretation if we knew what clusters meant
            st.info(f"The model has assigned this profile to Cluster group **{cluster}**.")
            
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

else:
    st.warning("Application cannot start without model artifacts.")
