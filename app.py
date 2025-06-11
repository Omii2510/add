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
section = st.sidebar.radio("🔍 Navigate", ["📘 About", "📊 Data Overview", "⚙️ Model Training", "🧑‍💻 User Prediction", "📈 Results"])

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

# Train-test split and scaling
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train logistic regression model
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
    - 🧑‍💻 Input user info and get instant conversion prediction
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
    st.title("⚙️ Model Training & Metrics")

    acc = accuracy_score(y_test, (y_probs >= 0.3).astype(int))
    st.metric(label="Model Accuracy (threshold=0.3)", value=f"{acc:.2%}")
    
    st.markdown("### Prediction Distribution on Test Set (threshold=0.3)")
    y_pred = (y_probs >= 0.3).astype(int)
    fig_pred, ax_pred = plt.subplots()
    sns.countplot(x=y_pred, ax=ax_pred, palette="Set2")
    ax_pred.set_xticklabels(['Not Converted', 'Converted'])
    ax_pred.set_title("Prediction Outcome Count")
    st.pyplot(fig_pred)

elif section == "🧑‍💻 User Prediction":
    st.title("🧑‍💻 Predict if a User Will Convert")

    # User inputs for features (based on original dataset columns before get_dummies)
    age = st.number_input("Age", min_value=10, max_value=100, value=30)
    gender = st.selectbox("Gender", options=['Male', 'Female'])
    income = st.number_input("Income (USD)", min_value=0, max_value=1000000, value=50000, step=1000)
    location = st.selectbox("Location", options=['Urban', 'Suburban', 'Rural'])  # example locations
    ad_type = st.selectbox("Ad Type", options=['Video', 'Banner', 'Popup', 'Native'])
    ad_topic = st.selectbox("Ad Topic", options=['Technology', 'Fashion', 'Food', 'Automotive'])
    ad_placement = st.selectbox("Ad Placement", options=['Top', 'Middle', 'Bottom'])
    clicks = st.number_input("Number of Clicks", min_value=0, max_value=1000, value=10)
    click_time = st.time_input("Click Time", value=pd.Timestamp('12:00').time())

    # Convert click_time to features
    click_hour = click_time.hour
    click_dayofweek = pd.Timestamp.now().dayofweek  # You could let user choose day or just use today

    # For conversion rate and CTR, these were dropped from training,
    # so user cannot input them (the target depends on these originally)

    # Prepare a DataFrame with one row for the user's input
    user_dict = {
        'Age': age,
        'Income': income,
        'Clicks': clicks,
        'Click_Hour': click_hour,
        'Click_DayOfWeek': click_dayofweek,
    }

    # Add categorical variables as dummy vars — these must match training dummy columns!

    # For Gender ('Male' or 'Female'), assuming df.get_dummies(drop_first=True)
    user_dict['Gender_Male'] = 1 if gender == 'Male' else 0

    # Locations, Ad Type, Ad Topic, Ad Placement: we need to create dummy variables accordingly
    # Check all possible dummy column names from training set
    # The pattern is "<column>_<category>" for categories except first (drop_first=True)
    
    # Prepare categorical dummies
    # You must get list of categorical variables from training df:
    cat_columns = [col for col in X.columns if '_' in col]

    # Initialize all dummy variables to 0
    for col in cat_columns:
        user_dict[col] = 0

    # Set 1 for selected categories (except the dropped first category)
    # Location
    if f"Location_Suburban" in user_dict:
        user_dict[f"Location_Suburban"] = 0
    if f"Location_Urban" in user_dict:
        user_dict[f"Location_Urban"] = 0

    if location == 'Suburban' and "Location_Suburban" in user_dict:
        user_dict["Location_Suburban"] = 1
    elif location == 'Urban' and "Location_Urban" in user_dict:
        user_dict["Location_Urban"] = 1
    # else: Rural is base category (dropped)

    # Ad Type
    for ad_t in ['Banner', 'Native', 'Popup', 'Video']:
        dummy_col = f"Ad Type_{ad_t}"
        if dummy_col in user_dict:
            user_dict[dummy_col] = 1 if ad_type == ad_t else 0

    # Ad Topic
    for ad_top in ['Automotive', 'Fashion', 'Food', 'Technology']:
        dummy_col = f"Ad Topic_{ad_top}"
        if dummy_col in user_dict:
            user_dict[dummy_col] = 1 if ad_topic == ad_top else 0

    # Ad Placement
    for placement in ['Bottom', 'Middle', 'Top']:
        dummy_col = f"Ad Placement_{placement}"
        if dummy_col in user_dict:
            user_dict[dummy_col] = 1 if ad_placement == placement else 0

    # Create DataFrame from user_dict
    user_df = pd.DataFrame([user_dict])

    # Align user_df columns to training X columns (in case of missing columns)
    user_df = user_df.reindex(columns=X.columns, fill_value=0)

    # Scale features
    user_scaled = scaler.transform(user_df)

    # Predict conversion probability and class
    prob = model.predict_proba(user_scaled)[0][1]
    prediction = model.predict(user_scaled)[0]

    st.write(f"### Prediction Result:")
    st.write(f"Probability of Conversion: **{prob:.2%}**")
    st.write(f"Predicted Conversion: **{'Yes' if prediction == 1 else 'No'}**")

elif section == "📈 Results":
    st.title("📈 Model Evaluation & Metrics")

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
