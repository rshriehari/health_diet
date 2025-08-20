import streamlit as st
import numpy as np
import pickle
import pandas as pd
import base64
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize MultiLabelBinarizer
mlb = MultiLabelBinarizer(classes=["Acne", "Diabetes", "Heart Disease", "Hypertension", "Kidney Disease", "Weight Gain", "Weight Loss"])

# Fit the mlb on your dataset if you haven't done it already (assuming y_train['Disease'] is available)
disease_labels = ["Acne", "Diabetes", "Heart Disease", "Hypertension", "Kidney Disease", "Weight Gain", "Weight Loss"]
mlb.fit([disease_labels])

st.set_page_config(
    page_title="Health and Fitness Prediction",
    page_icon="🏋️"
)

# Function to encode image to base64
def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")
    except FileNotFoundError:
        return None

# Set the background image
background_image = "image.webp"  # Replace with your image path
encoded_image = get_base64_image(background_image)
if encoded_image:
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url('data:image/png;base64,{encoded_image}');
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            color: white;
        }}
        .block-container {{
            background: rgba(0, 0, 0, 0.6);  /* Adds a semi-transparent background to the content */
            padding: 2rem;
            border-radius: 15px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
# Load the models
def load_model(file_path):
    try:
        with open(file_path, "rb") as file:
            return pickle.load(file)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

regressor_model = load_model("models/xgb_regressor_model.pkl")
classifier_model = load_model("models/xgb_classifier_model.pkl")

# Prepare input for prediction
def prepare_input(data):
    try:
        df = pd.DataFrame(data)
        return df
    except Exception as e:
        st.error(f"Error preparing input: {e}")
        return None

# Input labels
dietary_map = {0: "Vegan", 1: "Vegetarian", 2: "Non-Vegetarian", 3: "Keto"}
activity_map = {0: "Sedentary", 1: "Lightly Active", 2: "Moderately Active", 3: "Very Active", 4: "Extra Active"}

# Initialize session state for login
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

# Login page
if not st.session_state["logged_in"]:
    st.title("🏋️ Health and Fitness Prediction - Login")
    st.write("Please log in to access the app.")

    with st.form("login_form"):
        username = st.text_input("Username")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        login_button = st.form_submit_button("Log In")
    
    if login_button:
        if username and email and password:
            st.session_state["logged_in"] = True
            st.success("Login successful! Redirecting...")
            st.rerun()

        else:
            st.error("Please fill in all fields.")
else:
    # Main app content
    st.title("🏋️ Health and Fitness Prediction App")
    st.write("Welcome to the Health and Fitness Prediction App!")

    with st.form("user_input"):
        st.subheader("Enter Your Details:")
        age = st.number_input("Age", min_value=0, max_value=120, value=0)
        gender = st.selectbox("Gender", options=["Male", "Female"], index=1)
        height = st.number_input("Height (cm)", min_value=0, max_value=250, value=0)
        weight = st.number_input("Weight (kg)", min_value=0, max_value=300, value=0)
        dietary_preference = st.selectbox("Dietary Preference", options=list(dietary_map.values()))
        protein = st.number_input("Protein Intake (grams)", min_value=0, max_value=500, value=0)
        sugar = st.number_input("Sugar Intake (grams)", min_value=0, max_value=300, value=0)
        sodium = st.number_input("Sodium Intake (mg)", min_value=0, max_value=5000, value=0)
        carbs = st.number_input("Carbohydrates Intake (grams)", min_value=0, max_value=500, value=0)
        fiber = st.number_input("Fiber Intake (grams)", min_value=0, max_value=100, value=0)
        fat = st.number_input("Fat Intake (grams)", min_value=0, max_value=200, value=0)
        calories = st.number_input("Calorie Intake (kcal)", min_value=0, max_value=5000, value=0)
        activity_level = st.selectbox("Activity Level", options=list(activity_map.values()))

        submit = st.form_submit_button("Predict")

    if submit:
        # Map user inputs to the encoded data
        new_data = {
            "Ages": [age],
            "Gender": [1 if gender == "Male" else 0],
            "Height": [height],
            "Weight": [weight],
            "Dietary Preference": [list(dietary_map.keys())[list(dietary_map.values()).index(dietary_preference)]],
            "Protein": [protein],
            "Sugar": [sugar],
            "Sodium": [sodium],
            "Carbohydrates": [carbs],
            "Fiber": [fiber],
            "Fat": [fat],
            "Calories": [calories],
            "Activity Level Encoded": [
                list(activity_map.keys())[list(activity_map.values()).index(activity_level)]
            ],
        }

        input_data = prepare_input(new_data)

        if input_data is not None and regressor_model and classifier_model:
            # Predict calorie target
            calorie_prediction = regressor_model.predict(input_data)

            # Predict disease risk
            disease_binary_predictions = classifier_model.predict(input_data)
            predicted_disease_labels = mlb.inverse_transform(disease_binary_predictions)

            # Display results
            st.subheader("Results:")
            st.write(f"### Predicted Daily Calorie Target: {calorie_prediction[0]:.2f} kcal")
            st.write(f"### Predicted Diseases: {', '.join(predicted_disease_labels[0])}")

            # Prepare data for the report
            report_data = {
                "Feature": ["Age", "Height", "Weight", "Protein", "Sugar", "Sodium", "Carbohydrates", "Fiber", "Fat", "Calories", "Predicted Calories"],
                "Value": [age, height, weight, protein, sugar, sodium, carbs, fiber, fat, calories, calorie_prediction[0]]
            }

            # Tabular Report
            report_df = pd.DataFrame(report_data)
            st.subheader("Tabular Report")
            st.write(report_df)

            # Plotting the bar chart
            plt.figure(figsize=(10, 6))
            sns.barplot(x="Feature", y="Value", data=report_df, palette="viridis")
            plt.xticks(rotation=45, ha="right")
            plt.title("Report: Input Features and Predicted Calorie Value")
            plt.ylabel("Value")
            plt.xlabel("Features")
            plt.tight_layout()
            st.pyplot(plt)

            # Line Chart
            plt.figure(figsize=(10, 6))
            plt.plot(report_df["Feature"], report_df["Value"], marker='o', label='Input Features')
            plt.axhline(y=calorie_prediction[0], color='r', linestyle='--', label='Predicted Calories')

            # Add annotations
            for i, value in enumerate(report_df["Value"]):
                plt.text(i, value + 50, f"{value}", ha="center", fontsize=9)
            
            plt.text(len(report_df["Feature"]), calorie_prediction[0] + 50, 
                     f"Prediction: {calorie_prediction[0]}", color='red', fontsize=10)

            plt.title("Factors Influencing Predicted Calorie Intake", fontsize=14)
            plt.xlabel("Features", fontsize=12)
            plt.ylabel("Values", fontsize=12)
            plt.xticks(rotation=45, ha="right")
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            st.pyplot(plt)

            # Reasoning for Prediction
            st.subheader("Reasoning:")
            st.write("The predicted calorie value is influenced by features such as Protein, Carbohydrates, Fat, and Calories.")
            st.write("Higher values in Protein and Carbohydrates suggest a higher caloric requirement.")
        else:
            st.error("Error: Could not make predictions. Ensure all inputs are valid and models are loaded.")
