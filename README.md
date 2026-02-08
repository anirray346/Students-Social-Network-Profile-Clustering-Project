# Social Network Profile Clustering

## Project Overview
This project applies **Unsupervised Machine Learning (K-Means Clustering)** to segment social network users into distinct groups based on their demographics (Age, Gender) and interests. The goal is to identify patterns in user behavior and preferences, which can be useful for targeted marketing or content recommendation.

The project includes a **Streamlit Web Application** that allows users to input their profile details and receive a real-time cluster prediction.

## Key Features
- **Data Preprocessing**: Handles missing values for age and gender; encodes categorical variables.
- **Machine Learning**: Uses K-Means clustering (K=3) to categorize users.
- **Interactive UI**: A user-friendly Streamlit app for real-time predictions.
- **Feature Scaling**: Standardizes inputs to ensure accurate distance calculations for clustering.

## Technologies Used
- **Python**: Core programming language.
- **Scikit-learn**: For K-Means clustering, Label Encoding, and StandardScaler.
- **Streamlit**: For the interactive web interface.
- **Pandas & NumPy**: For data manipulation and numerical operations.
- **Pickle**: For serializing and saving the trained model and artifacts.

## File Structure
- `app.py`: The main Streamlit application for user interaction and prediction.
- `generate_model.py`: Script to load data, train the K-Means model, and save artifacts (`model_data.pkl`).
- `requirements.txt`: List of Python dependencies.

## Installation and Usage

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the model:**
   Run the generation script to create the model artifacts (`model_data.pkl`).
   ```bash
   python generate_model.py
   ```

3. **Run the Streamlit App:**
   ```bash
   streamlit run app.py
   ```

## Workflow
1. The `generate_model.py` loads the dataset, cleans missing values, selects relevant features (Interest pillars + Age + Gender), scales the data, and fits a K-Means model.
2. The model and preprocessing objects (Scaler, Encoder) are saved to `model_data.pkl`.
3. The `app.py` loads these artifacts to make predictions on new user input provided via the web interface.

