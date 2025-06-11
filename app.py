import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Set page config
st.set_page_config(page_title="Ad Conversion Predictor", layout="wide")

# Custom styling
st.markdown(
    """
    <style>
    .main { background-color: #f5f5f5; }
    .sidebar .sidebar-content {
        background-color: #ffffff;
        padding: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar navigation
menu = st.sidebar.radio(
    "Navigation",
    ["About", "Dataset", "Model Training", "Results"],
    index=0
)

# Load dataset
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/Omii2510/add/refs/heads/main/Dataset_Ads.csv"
    df = pd.read_csv(url)
    df['Click Time'] = pd.to_datetime(df['Click Time'])
    df['Click_Hour'] = df['Click Time'].dt.hour
    df['Click_DayOfWeek'] = df['Click Time'].dt.dayofweek
    df = df.drop('Click Time', axis=1)
    df['Converted'] = (df['Conversion Rate'] > 0.1).astype(int)
    df = df.drop(['Conversion Rate', 'CTR'], axis=1)
    df = pd.get_dummies(df, drop_first=True)
    return df

df = load_data()
X = df.drop('Converted', axis=1)
y = df['Converted']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)
y_probs = model.predict_proba(X_test_scaled)[:, 1]

# Section 1: About
if menu == "About":
    st.title("📊 Ad Conversion Prediction Web App")
    st.markdown("""
    Welcome to the **Ad Conversion Predictor**!  
    This app predicts whether a user will **convert** (take action on an ad) based on historical ad interaction data.

    ### 🔍 Key Features:
    - Load and explore ad click dataset
    - Train a logistic regression model
    - Adjust prediction threshold interactively
    - View classification metrics and visualizations

    Use the navigation panel on the left to explore different parts of the app.
    """)

# Section 2: Dataset
elif menu == "Dataset":
    st.title("🗂️ Dataset Overview")
    st.subheader("Preview of the Dataset")
    st.dataframe(df.head())
    st.markdown("### Class Distribution")
    st.bar_chart(y.value_counts())

# Section 3: Model Training & Threshold
elif menu == "Model Training":
    st.title("🛠️ Model Prediction Settings")
    st.markdown("Adjust the threshold to control prediction sensitivity.")

    threshold = st.slider("Prediction Threshold", 0.0, 1.0, 0.3, 0.01)
    y_pred = (y_probs >= threshold).astype(int)

    st.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2%}")

    st.markdown("### Prediction Outcome Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x=y_pred, ax=ax)
    ax.set_xticklabels(['Not Converted', 'Converted'])
    st.pyplot(fig)

# Section 4: Results & Reports
elif menu == "Results":
    st.title("📈 Model Evaluation")
    threshold = st.slider("Prediction Threshold", 0.0, 1.0, 0.3, 0.01, key="threshold_results")
    y_pred = (y_probs >= threshold).astype(int)

    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred, zero_division=1))

    cm = confusion_matrix(y_test, y_pred)
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted No', 'Predicted Yes'],
                yticklabels=['Actual No', 'Actual Yes'], ax=ax)
    st.pyplot(fig)

    st.markdown("### Notes:")
    st.info("If precisi
