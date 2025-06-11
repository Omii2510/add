import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.title("Conversion Prediction Web App")

# File uploader
uploaded_file = st.file_uploader("Dataset_Ads.csv", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.dataframe(df.head())

    # Data preprocessing
    df['Click Time'] = pd.to_datetime(df['Click Time'])
    df['Click_Hour'] = df['Click Time'].dt.hour
    df['Click_DayOfWeek'] = df['Click Time'].dt.dayofweek
    df = df.drop('Click Time', axis=1)

    df['Converted'] = (df['Conversion Rate'] > 0.1).astype(int)
    df = df.drop(['Conversion Rate', 'CTR'], axis=1)
    df = pd.get_dummies(df, drop_first=True)

    X = df.drop('Converted', axis=1)
    y = df['Converted']

    st.write("Class distribution:")
    st.write(y.value_counts())

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    # Predict probabilities
    y_probs = model.predict_proba(X_test_scaled)[:, 1]

    # Threshold slider for user
    threshold = st.slider("Prediction Threshold", 0.0, 1.0, 0.3)

    y_pred = (y_probs >= threshold).astype(int)

    # Show accuracy and classification report
    acc = accuracy_score(y_test, y_pred)
    st.write(f"### Accuracy: {acc:.2f}")

    st.write("### Classification Report:")
    st.text(classification_report(y_test, y_pred, zero_division=1))

    # Plot conversion distribution
    st.write("### Conversion Distribution")
    fig1, ax1 = plt.subplots()
    sns.countplot(x=y, ax=ax1)
    st.pyplot(fig1)

    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    st.write("### Confusion Matrix")
    fig2, ax2 = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted No', 'Predicted Yes'],
                yticklabels=['Actual No', 'Actual Yes'], ax=ax2)
    st.pyplot(fig2)
