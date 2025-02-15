import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import os
import warnings
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, classification_report
)
from streamlit_option_menu import option_menu
# Filter warnings
warnings.filterwarnings('ignore')

# Welcome message
st.title(''':orange[***Multiple Disease Prediction System***]''')


page_bg_img = """
<style>
[data-testid="stAppViewContainer"]{
background-image: url;
background-size: cover;
}

[data-testid="stHeader"]{
background-color: rgba(0, 0, 0, 0);
}   

[data-testid="stSidebarContent"]{
background-color: rgba(0, 0, 0, 0);
background-size: cover;
}
</style>
"""
# Display the background image
st.markdown(page_bg_img, unsafe_allow_html=True)


# Sidebar Menu
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',
                           ['Kidney Disease Prediction', 'Liver Disease Prediction', '''Parkinson's Disease Prediction'''],
                           menu_icon='hospital-fill',
                           icons=['activity', 'heart', 'person'],
                           default_index=0)

# Page Titles
st.title(f"{selected} using ML")
st.empty()

# Upload dataset through Streamlit
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Load dataset
    df = pd.read_csv(uploaded_file)
    
    # Show dataset preview
    st.subheader("Dataset Preview")
    st.write(df.head(10))
    
    # Show dataset info and description
    st.subheader("Dataset Information")
    st.write(f"Shape of the Data: {df.shape}")
    st.write("Summary Statistics:")
    st.write(df.describe())

    # Checking for missing values
    st.subheader("Missing Values Check")
    st.write(df.isnull().sum())

    # Fill missing values
    numeric_cols = df.select_dtypes(include=['number']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    df[numeric_cols] = df[numeric_cols].apply(lambda col: col.fillna(col.mean()))
    df[categorical_cols] = df[categorical_cols].apply(lambda col: col.fillna(col.mode()[0]))
    
    # Encode categorical columns
    for col in categorical_cols:
        df[col] = df[col].astype('category').cat.codes
    
    # Define feature and target variables properly
    if 'classification' in df.columns:
        x = df.drop(['id', 'classification'], axis=1)
        y = df['classification']
    elif 'Dataset' in df.columns:
        x = df.drop(['Gender'], axis=1)
        y = df['Gender']
    elif 'status' in df.columns:
        x = df.drop(['name', 'status'], axis=1)
        y = df['status']
    else:
        st.error("Target variable not found in the dataset. Please check your column names.")
        st.stop()
    
    # Split data into train and test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

    # Scaling data
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # Logistic Regression Model
    lr = LogisticRegression()
    lr.fit(x_train_scaled, y_train)
    y_pred_lr = lr.predict(x_test_scaled)

    # Random Forest Classifier Model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=0)
    rf_model.fit(x_train_scaled, y_train)
    y_pred_rf = rf_model.predict(x_test_scaled)

    # Evaluation Metrics
    st.subheader("Model Evaluation Metrics")
    
    # Cross-validation score
    cv_scores = cross_val_score(lr, x_test_scaled, y_test, cv=5)
    st.write(f"Mean CV Score: {np.mean(cv_scores):.2f}")
    
    # Logistic Regression Metrics
    st.subheader("Logistic Regression Metrics")
    st.write(f"Accuracy: {accuracy_score(y_test, y_pred_lr) * 100:.2f}%")
    st.write(f"Precision: {precision_score(y_test, y_pred_lr, average='weighted'):.2f}")
    st.write(f"Recall: {recall_score(y_test, y_pred_lr, average='weighted'):.2f}")
    st.write(f"F1 Score: {f1_score(y_test, y_pred_lr, average='weighted'):.2f}")
    st.write(f"Mean Absolute Error (MAE): {mean_absolute_error(y_test, y_pred_lr):.2f}")
    st.write(f"Root Mean Squared Error (RMSE): {np.sqrt(mean_squared_error(y_test, y_pred_lr)):.2f}")
    
    # Random Forest Metrics
    st.subheader("Random Forest Metrics")
    st.write(f"Accuracy: {accuracy_score(y_test, y_pred_rf) * 100:.2f}%")
    st.write(f"Precision: {precision_score(y_test, y_pred_rf, average='weighted'):.2f}")
    st.write(f"Recall: {recall_score(y_test, y_pred_rf, average='weighted'):.2f}")
    st.write(f"F1 Score: {f1_score(y_test, y_pred_rf, average='weighted'):.2f}")

    # Feature Distribution
    st.subheader("Feature Distribution in Test Set (Scaled)")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(x_test_scaled, bins=20)
    ax.set_title("Feature Distribution - Scaled Test Set")
    st.pyplot(fig)

    # Save Model Option
    save_model = st.checkbox("Save Model", key="save_model_selector")
    if save_model:
        filename = st.text_input("Enter filename to save the model (with .pkl extension):", "model.pkl")
        if filename:
            joblib.dump(lr, filename)
            st.write(f"Model saved as '{filename}'.")
        else:
            st.warning("Please enter a valid filename with the .pkl extension.")
else:
    st.info("Please upload a CSV file to start.")
