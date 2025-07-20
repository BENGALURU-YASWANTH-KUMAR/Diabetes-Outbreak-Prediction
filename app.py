import streamlit as st
import pickle
import numpy as np
from streamlit_option_menu import option_menu
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
import pandas as pd

import os
import pickle
# Define the EnsembleClassifier class
class EnsembleClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, weights=None):
        self.rf = RandomForestClassifier(random_state=42)
        self.knn = KNeighborsClassifier(n_neighbors=5)
        self.xgb = XGBClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42
        )
        self.gb = GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42
        )

        self.weights = weights if weights is not None else [0.25, 0.3, 0.25, 0.2]
        self.models = [self.rf, self.knn, self.xgb, self.gb]

    def fit(self, X, y):
        for model in self.models:
            model.fit(X, y)
        return self

    def predict_proba(self, X):
        predictions = []
        for model, weight in zip(self.models, self.weights):
            if hasattr(model, "predict_proba"):
                pred = model.predict_proba(X) * weight
            else:
                pred_labels = model.predict(X)
                pred = np.zeros((len(pred_labels), 2))
                for i, label in enumerate(pred_labels):
                    pred[i, label] = 1 * weight
            predictions.append(pred)
        return np.sum(predictions, axis=0)

    def predict(self, X):
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)


# Page config
st.set_page_config(
    page_title="Health Disease Prediction System",
    layout="wide",
    page_icon="🏥",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
    }
    .stButton>button:hover {
        background-color: #FF6B6B;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 1rem;
        margin: 1rem 0;
    }
    .stAlert {
        border-radius: 0.5rem;
    }
    </style>
""",
    unsafe_allow_html=True,
)


# Load models with caching
@st.cache_resource
def load_models():
    try:
        heart_model = pickle.load(open(r"models/heart_model1.pkl", "rb"))
        diabetes_model = pickle.load(open(r"models/stacking_model.pkl", "rb"))
        parkinson_model = pickle.load(open(r"models/ensemble_model.pkl", "rb"))
        return heart_model, diabetes_model, parkinson_model
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None


heart_model, diabetes_model, parkinson_model = load_models()

# Sidebar navigation
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/heart-health.png", width=100)
    st.title("Health Prediction System")
    selected = option_menu(
        menu_title="Disease Prediction",
        options=["Home", "Heart Disease", "Diabetes", "Parkinson's"],
        icons=["house", "heart-pulse", "droplet", "brain"],
        default_index=0,
        styles={
            "container": {"padding": "1rem"},
            "icon": {"font-size": "1rem"},
            "nav-link": {
                "font-size": "0.9rem",
                "text-align": "left",
                "margin": "0.5rem",
            },
            "nav-link-selected": {"background-color": "#FF4B4B"},
        },
    )

# Home Page
if selected == "Home":
    st.title("Welcome to Health Disease Prediction System 🏥")
    st.markdown("""
    ### Your Smart Health Assistant
    
    This application uses advanced machine learning models to predict the risk of various diseases:
    
    1. **Heart Disease** 🫀
       - Analyzes 13 different health parameters
       - Provides instant risk assessment
    
    2. **Diabetes** 🩸
       - Evaluates based on key health indicators
       - Includes advanced metrics calculations
    
    3. **Parkinson's Disease** 🧠
       - Analyzes voice parameters
       - Provides early detection insights
    
    ### How to Use
    1. Select a disease from the sidebar
    2. Enter your health parameters
    3. Get instant prediction results
    
    ### Important Note
    This tool is for screening purposes only and should not replace professional medical advice.
    """)

# Heart Disease Page
# Modify the heart disease section in your code. Replace the existing heart disease section with this:

elif selected == "Heart Disease":
    st.title("Heart Disease Prediction 🫀")

    with st.expander("ℹ️ Information about Heart Disease Parameters", expanded=False):
        st.markdown("""
        ### Understanding the Parameters:
        - **Age**: Patient's age in years
        - **Sex**: Gender (1 = Male, 0 = Female)
        - **CP**: Chest Pain Type (0-3)
            - 0: Typical angina
            - 1: Atypical angina
            - 2: Non-anginal pain
            - 3: Asymptomatic
        - **Trestbps**: Resting Blood Pressure (mm/Hg)
        - **Chol**: Serum Cholesterol (mg/dl)
        - **Fbs**: Fasting Blood Sugar > 120 mg/dl (1 = true; 0 = false)
        - **Restecg**: Resting ECG Results (0-2)
        - **Thalach**: Maximum Heart Rate Achieved
        - **Exang**: Exercise Induced Angina (1 = yes; 0 = no)
        - **Oldpeak**: ST Depression Induced by Exercise
        - **Slope**: Slope of Peak Exercise ST Segment (0-2)
        - **Ca**: Number of Major Vessels (0-4)
        - **Thal**: Thalassemia Type (1-3)
        """)

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=0, max_value=120, value=50)
        sex = st.selectbox(
            "Sex", options=[1, 0], format_func=lambda x: "Male" if x == 1 else "Female"
        )
        cp = st.selectbox(
            "Chest Pain Type",
            options=[0, 1, 2, 3],
            help="0: Typical angina, 1: Atypical angina, 2: Non-anginal pain, 3: Asymptomatic",
        )
        trestbps = st.number_input(
            "Resting Blood Pressure (mm/Hg)", min_value=0, max_value=300, value=120
        )

    with col2:
        chol = st.number_input(
            "Serum Cholesterol (mg/dl)", min_value=0, max_value=600, value=200
        )
        fbs = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dl",
            options=[0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No",
        )
        restecg = st.selectbox(
            "Resting ECG Results",
            options=[0, 1, 2],
            help="0: Normal, 1: ST-T wave abnormality, 2: Left ventricular hypertrophy",
        )
        thalach = st.number_input(
            "Maximum Heart Rate", min_value=0, max_value=250, value=150
        )

    with col3:
        exang = st.selectbox(
            "Exercise Induced Angina",
            options=[0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No",
        )
        oldpeak = st.number_input(
            "ST Depression", min_value=0.0, max_value=10.0, value=0.0, step=0.1
        )
        slope = st.selectbox(
            "ST Slope", options=[0, 1, 2], help="0: Upsloping, 1: Flat, 2: Downsloping"
        )
        ca = st.number_input(
            "Number of Major Vessels", min_value=0, max_value=4, value=0
        )
        thal = st.selectbox(
            "Thalassemia",
            options=[1, 2, 3],
            help="1: Normal, 2: Fixed defect, 3: Reversible defect",
        )

    if st.button("Predict Heart Disease Risk"):
        try:
            # Prepare input features in the correct order
            features = [
                age,
                sex,
                cp,
                trestbps,
                chol,
                fbs,
                restecg,
                thalach,
                exang,
                oldpeak,
                slope,
                ca,
                thal,
            ]

            # Make prediction
            prediction = heart_model.predict([features])

            # Handle prediction probabilities
            try:
                proba = heart_model.predict_proba([features])
                confidence = proba[0][1] if prediction[0] == 1 else proba[0][0]
                confidence_text = f"(Confidence: {confidence:.2%})"
            except (AttributeError, Exception) as e:
                confidence_text = "(Confidence score unavailable)"

            # Display results
            if prediction[0] == 1:
                st.error(f"❗ High Risk of Heart Disease {confidence_text}")
                st.markdown("""
                ### Recommendations:
                1. 👨‍⚕️ Consult a cardiologist immediately
                2. 📊 Monitor blood pressure daily
                3. ❤️ Follow a heart-healthy diet:
                   - Reduce salt intake
                   - Limit saturated fats
                   - Increase fruits and vegetables
                4. 🏃‍♂️ Start gentle exercise (after doctor's approval)
                5. 💊 Review current medications with your doctor
                """)

                # Additional risk factors
                st.warning("""
                ### Key Risk Factors to Address:
                - Blood Pressure Management
                - Cholesterol Control
                - Regular Exercise
                - Stress Management
                - Smoking Cessation (if applicable)
                """)

            else:
                st.success(f"✅ Low Risk of Heart Disease {confidence_text}")
                st.markdown("""
                ### Keep up the good work:
                1. 🥗 Maintain a healthy diet
                2. 🚶‍♂️ Regular exercise (150 minutes/week)
                3. 😴 Adequate sleep (7-9 hours)
                4. 🩺 Regular check-ups
                5. 🧘‍♂️ Stress management
                """)

            # Display entered values summary
            with st.expander("View Your Entered Values Summary"):
                st.markdown(f"""
                - Age: {age}
                - Blood Pressure: {trestbps} mm/Hg
                - Cholesterol: {chol} mg/dl
                - Max Heart Rate: {thalach}
                - ST Depression: {oldpeak}
                """)

        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")
            st.info("Please ensure all values are entered correctly and try again.")

# Diabetes Page
elif selected == "Diabetes":
    st.title("Diabetes Prediction 🩸")

    with st.expander("ℹ️ Information about Diabetes Parameters", expanded=False):
        st.markdown("""
        ### Understanding the Parameters:
        - **Pregnancies**: Number of pregnancies
        - **Glucose**: Plasma glucose concentration
        - **Blood Pressure**: Diastolic blood pressure (mm Hg)
        - **Skin Thickness**: Triceps skinfold thickness (mm)
        - **BMI**: Body Mass Index
        - **Diabetes Pedigree Function**: Diabetes hereditary factor
        - **Age**: Age in years
        """)

    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0)
        glucose = st.number_input(
            "Glucose Level", min_value=0, max_value=300, value=120
        )
        blood_pressure = st.number_input(
            "Blood Pressure", min_value=0, max_value=150, value=70
        )
        skin_thickness = st.number_input(
            "Skin Thickness", min_value=0, max_value=100, value=20
        )

    with col2:
        bmi = st.number_input(
            "BMI", min_value=0.0, max_value=70.0, value=25.0, step=0.1
        )
        diabetes_pedigree = st.number_input(
            "Diabetes Pedigree Function",
            min_value=0.0,
            max_value=2.5,
            value=0.5,
            step=0.01,
        )
        age = st.number_input("Age (Diabetes)", min_value=0, max_value=120, value=30)

    # Calculate derived features
    bmi_age_interaction = bmi * age
    glucose_squared = glucose**2
    pregnancy_age_ratio = pregnancies / age if age > 0 else 0

    if st.button("Predict Diabetes Risk"):
        try:
            features = [
                pregnancies,
                glucose,
                blood_pressure,
                skin_thickness,
                bmi,
                diabetes_pedigree,
                age,
                bmi_age_interaction,
                glucose_squared,
                pregnancy_age_ratio,
            ]
            prediction = diabetes_model.predict([features])
            proba = diabetes_model.predict_proba([features])

            if prediction[0] == 1:
                st.error(f"❗ High Risk of Diabetes (Confidence: {proba[0][1]:.2%})")
                st.markdown("""
                ### Recommended Actions:
                1. 👨‍⚕️ Schedule a doctor's appointment
                2. 📊 Monitor blood sugar regularly
                3. 🥗 Follow a diabetes-friendly diet
                4. 🏃‍♂️ Increase physical activity
                """)
            else:
                st.success(f"✅ Low Risk of Diabetes (Confidence: {proba[0][0]:.2%})")
                st.markdown("""
                ### Maintain Your Health:
                1. 🥗 Continue healthy eating habits
                2. 🚶‍♂️ Regular exercise
                3. ⚖️ Maintain healthy weight
                4. 🩺 Regular check-ups
                """)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Parkinson's Disease Page
elif selected == "Parkinson's":
    st.title("Parkinson's Disease Prediction 🧠")

    with st.expander("ℹ️ Information about Voice Parameters", expanded=False):
        st.markdown("""
        ### Understanding Voice Parameters:
        - **MDVP:Fo**: Average vocal fundamental frequency
        - **MDVP:Fhi**: Maximum vocal fundamental frequency
        - **MDVP:Flo**: Minimum vocal fundamental frequency
        - **MDVP:Jitter/Shimmer**: Various measures of variation in fundamental frequency
        - **HNR**: Ratio of noise to tonal components
        - **RPDE/DFA**: Nonlinear dynamical complexity measures
        - **PPE**: Nonlinear measure of fundamental frequency variation
        """)

    col1, col2, col3 = st.columns(3)

    with col1:
        mdvp_fo = st.number_input("MDVP:Fo(Hz)", value=120.0, format="%.2f")
        mdvp_flo = st.number_input("MDVP:Flo(Hz)", value=80.0, format="%.2f")
        mdvp_jitter = st.number_input("MDVP:Jitter(%)", value=0.01, format="%.4f")

    with col2:
        mdvp_shimmer = st.number_input("MDVP:Shimmer", value=0.05, format="%.4f")
        mdvp_shimmer_db = st.number_input("MDVP:Shimmer(dB)", value=0.5, format="%.4f")
        shimmer_dda = st.number_input("Shimmer:DDA", value=0.07, format="%.4f")

    with col3:
        hnr = st.number_input("HNR", value=20.0, format="%.2f")
        rpde = st.number_input("RPDE", value=0.5, format="%.4f")
        dfa = st.number_input("DFA", value=0.7, format="%.4f")
        spread1 = st.number_input("spread1", value=-4.0, format="%.4f")
        spread2 = st.number_input("spread2", value=0.3, format="%.4f")

    if st.button("Predict Parkinson's Disease Risk"):
        try:
            features = [
                mdvp_fo,
                mdvp_flo,
                mdvp_jitter,
                mdvp_shimmer,
                mdvp_shimmer_db,
                shimmer_dda,
                hnr,
                rpde,
                dfa,
                spread1,
                spread2,
            ]
            prediction = parkinson_model.predict([features])
            proba = parkinson_model.predict_proba([features])

            if prediction[0] == 1:
                st.error(
                    f"❗ High Risk of Parkinson's Disease (Confidence: {proba[0][1]:.2%})"
                )
                st.markdown("""
                ### Recommended Actions:
                1. 👨‍⚕️ Consult a neurologist immediately
                2. 📋 Document all symptoms
                3. 🧠 Consider cognitive assessments
                4. 💪 Begin appropriate physical therapy
                """)
            else:
                st.success(
                    f"✅ Low Risk of Parkinson's Disease (Confidence: {proba[0][0]:.2%})"
                )
                st.markdown("""
                ### Maintain Your Health:
                1. 🧘‍♂️ Regular exercise
                2. 🧠 Keep mentally active
                3. 😴 Maintain good sleep habits
                4. 🩺 Regular check-ups
                """)

            # Additional Information
            st.info("""
            **Note:** Voice analysis is just one screening tool for Parkinson's disease. 
            A proper medical diagnosis requires comprehensive neurological examination.
            """)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Footer
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown(
        """
    <div style='text-align: center'>
        <h4>Health Disease Prediction System</h4>
        <p>Developed with ❤️ by Yaswanth</p>
        <p style='font-size: 0.8em'>Version 1.0.0 | © 2024</p>
        <p style='font-size: 0.7em; color: #666;'>
            Disclaimer: This tool is for screening purposes only and should not replace professional medical advice.
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )
