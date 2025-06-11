import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Page Configuration
st.set_page_config(page_title="Ad Conversion Predictor", layout="wide")

# Sidebar Navigation
section = st.sidebar.radio("🔍 Navigate", ["📘 About", "📊 Data Overview", "⚙️ Model Training", "📈 Results"])

# Load Data
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/Omii2510/add/refs/heads/main/Dataset_Ads.csv"
    df = pd.read_csv(url)

    # Data Preprocessing
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

# --- Sections ---

if section == "📘 About":
    st.title("📘 Ad Conversion Prediction Web App")
    st.markdown("""
    Welcome to the **Ad Conversion Predictor** powered by logistic regression.  
    This app helps marketers understand if a user is likely to convert (i.e., take an action on an ad) based on interaction data.

    ### What You Can Do:
    - 🧭 Explore the dataset
    - ⚙️ Train and evaluate a logistic regression model
    - 🎯 Adjust prediction thresholds and analyze performance
    - 📈 View accuracy, classification metrics, and visual charts
    """)

elif section == "📊 Data Overview":
    st.title("📊 Dataset Overview")
    st.subheader("First Five Rows")
    st.dataframe(df.head())

    st.subheader("Class Distribution (Converted or Not)")
    st.bar_chart(y.value_counts())

    st.info("✅ 'Converted' is derived from Conversion Rate > 0.1")

elif section == "⚙️ Model Training":
    st.title("⚙️ Model & Prediction Settings")

    st.markdown("### 🎯 Set Prediction Threshold")
    threshold = st.slider("Choose a threshold to classify conversion:", 0.0, 1.0, 0.3, 0.01)
    y_pred = (y_probs >= threshold).astype(int)
    acc = accuracy_score(y_test, y_pred)

    st.metric(label="Model Accuracy", value=f"{acc:.2%}")
    
    st.markdown("### 📊 Prediction Distribution")
    fig_pred, ax_pred = plt.subplots()
    sns.countplot(x=y_pred, ax=ax_pred, palette="Set2")
    ax_pred.set_xticklabels(['Not Converted', 'Converted'])
    ax_pred.set_title("Prediction Outcome Count")
    st.pyplot(fig_pred)

elif section == "📈 Results":
    st.title("📈 Model Evaluation & Metrics")

    st.markdown("### Adjust Threshold Again (optional)")
    threshold = st.slider("Threshold for conversion prediction:", 0.0, 1.0, 0.3, 0.01, key="eval_threshold")
    y_pred = (y_probs >= threshold).astype(int)

    # Classification report
    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred, zero_division=1))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    st.subheader("Confusion Matrix")
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted No', 'Predicted Yes'],
                yticklabels=['Actual No', 'Actual Yes'], ax=ax_cm)
    ax_cm.set_title("Confusion Matrix")
    st.pyplot(fig_cm)

    st.info("ℹ️ If precision is 0.0, the model may not be predicting that class — try adjusting the threshold.")

