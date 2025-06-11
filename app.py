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

# Tabs Navigation
tabs = st.tabs(["Add New Data", "Visualization", "Model Evaluation", "Preprocessing"])

# Load Data
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/Omii2510/add/refs/heads/main/Dataset_Ads.csv"
    df = pd.read_csv(url)
    return df

df_raw = load_data()

def preprocess_data(df):
    df = df.copy()
    df['Click Time'] = pd.to_datetime(df['Click Time'])
    df['Click_Hour'] = df['Click Time'].dt.hour
    df['Click_DayOfWeek'] = df['Click Time'].dt.dayofweek
    df = df.drop('Click Time', axis=1)
    df['Converted'] = (df['Conversion Rate'] > 0.1).astype(int)
    df = df.drop(['Conversion Rate', 'CTR'], axis=1)
    df = pd.get_dummies(df, drop_first=True)
    return df

# Prepare processed data upfront for multiple tabs
df_processed = preprocess_data(df_raw)
X = df_processed.drop('Converted', axis=1)
y = df_processed['Converted']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)
y_probs = model.predict_proba(X_test_scaled)[:, 1]

# --- Tab 1: Add New Data ---
with tabs[0]:
    st.title("Add New Data Entry and Predict Conversion")

    age = st.number_input("Age", min_value=10, max_value=100, value=30)
    gender = st.selectbox("Gender", options=['Male', 'Female'])
    income = st.number_input("Income (USD)", min_value=0, max_value=1000000, value=50000, step=1000)
    location = st.selectbox("Location", options=['Urban', 'Suburban', 'Rural'])
    ad_type = st.selectbox("Ad Type", options=['Video', 'Banner', 'Popup', 'Native'])
    ad_topic = st.selectbox("Ad Topic", options=['Technology', 'Fashion', 'Food', 'Automotive'])
    ad_placement = st.selectbox("Ad Placement", options=['Top', 'Middle', 'Bottom'])
    clicks = st.number_input("Number of Clicks", min_value=0, max_value=1000, value=10)
    click_time = st.time_input("Click Time", value=pd.Timestamp('12:00').time())

    click_hour = click_time.hour
    click_dayofweek = pd.Timestamp.now().dayofweek

    user_dict = {
        'Age': age,
        'Income': income,
        'Clicks': clicks,
        'Click_Hour': click_hour,
        'Click_DayOfWeek': click_dayofweek,
        'Gender_Male': 1 if gender == 'Male' else 0,
    }

    cat_columns = [col for col in X.columns if '_' in col]
    for col in cat_columns:
        user_dict[col] = 0

    if location == 'Suburban' and "Location_Suburban" in user_dict:
        user_dict["Location_Suburban"] = 1
    elif location == 'Urban' and "Location_Urban" in user_dict:
        user_dict["Location_Urban"] = 1

    for ad_t in ['Banner', 'Native', 'Popup', 'Video']:
        dummy_col = f"Ad Type_{ad_t}"
        if dummy_col in user_dict:
            user_dict[dummy_col] = 1 if ad_type == ad_t else 0

    for ad_top in ['Automotive', 'Fashion', 'Food', 'Technology']:
        dummy_col = f"Ad Topic_{ad_top}"
        if dummy_col in user_dict:
            user_dict[dummy_col] = 1 if ad_topic == ad_top else 0

    for placement in ['Bottom', 'Middle', 'Top']:
        dummy_col = f"Ad Placement_{placement}"
        if dummy_col in user_dict:
            user_dict[dummy_col] = 1 if ad_placement == placement else 0

    user_df = pd.DataFrame([user_dict])
    user_df = user_df.reindex(columns=X.columns, fill_value=0)
    user_scaled = scaler.transform(user_df)
    prob = model.predict_proba(user_scaled)[0][1]
    prediction = model.predict(user_scaled)[0]

    if st.button("Predict Conversion"):
        st.success(f"Prediction: {'Yes' if prediction == 1 else 'No'} (Probability: {prob:.2%})")

# --- Tab 2: Visualization ---
with tabs[1]:
    st.title("Data Visualization")
    st.write("### Class Distribution")
    st.bar_chart(df_processed['Converted'].value_counts())

    feature = st.selectbox("Select feature to visualize", options=df_raw.columns)
    fig, ax = plt.subplots()
    if np.issubdtype(df_raw[feature].dtype, np.number):
        sns.histplot(df_raw[feature], kde=True, ax=ax)
    else:
        sns.countplot(x=feature, data=df_raw, ax=ax)
    st.pyplot(fig)

# --- Tab 3: Model Evaluation ---
with tabs[2]:
    st.title("Model Training and Evaluation")

    threshold = st.slider("Choose prediction threshold:", 0.0, 1.0, 0.3, 0.01)

    y_pred = (y_probs >= threshold).astype(int)
    acc = accuracy_score(y_test, y_pred)
    st.metric("Accuracy", f"{acc:.2%}")

    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred, zero_division=1))

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted No', 'Predicted Yes'],
                yticklabels=['Actual No', 'Actual Yes'], ax=ax_cm)
    ax_cm.set_title("Confusion Matrix")
    st.pyplot(fig_cm)

# --- Tab 4: Preprocessing ---
with tabs[3]:
    st.title("Preprocessing Step")
    st.write("### Raw Data Sample")
    st.dataframe(df_raw.head())

    st.write("### Data After Preprocessing")
    st.dataframe(df_processed.head())
