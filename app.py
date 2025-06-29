import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, precision_score, recall_score, classification_report

# Ensure scikit-learn is installed in your environment.
# If running locally, you might need: pip install scikit-learn
# In a hosted environment, ensure 'scikit-learn' is in your requirements.txt.


# Set Streamlit page configuration
st.set_page_config(
    page_title="Hospital Readmission Risk Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🏥 AI System for Hospital Readmission Risk Prediction")
st.markdown("---")

# --- 2.2 Data Strategy ---
# Propose Data Sources (Conceptual Data Generation for Demonstration):
# In a real scenario, this data would come from EHRs, demographics databases, etc.
# Ethical Concern 1: Patient Privacy (HIPAA Compliance) - PHI must be strictly protected, anonymized/pseudonymized.
# Ethical Concern 2: Algorithmic Bias - Historical care disparities in data could lead to biased predictions.

@st.cache_resource # Cache the data generation and model training to avoid re-running on every interaction
def load_and_train_model(num_patients=1000):
    """Generates hypothetical patient data and trains the model pipeline."""
    np.random.seed(42) # for reproducibility

    data = {
        'patient_id': range(1, num_patients + 1),
        'age': np.random.randint(18, 90, num_patients),
        'gender': np.random.choice(['Male', 'Female'], num_patients, p=[0.48, 0.52]),
        'ethnicity': np.random.choice(['Caucasian', 'African American', 'Hispanic', 'Asian', 'Other'], num_patients, p=[0.6, 0.2, 0.1, 0.05, 0.05]),
        'num_chronic_conditions': np.random.randint(0, 5, num_patients),
        'admission_type': np.random.choice(['Emergency', 'Elective', 'Urgent'], num_patients, p=[0.5, 0.3, 0.2]),
        'length_of_stay': np.random.randint(1, 30, num_patients), # in days
        'num_medications_discharge': np.random.randint(1, 15, num_patients),
        'previous_readmission': np.random.choice([0, 1], num_patients, p=[0.85, 0.15]), # 0=No, 1=Yes
        'lab_value_creatinine': np.random.uniform(0.6, 3.0, num_patients), # mg/dL
        'lab_value_hemoglobin': np.random.uniform(8.0, 18.0, num_patients), # g/dL
        'zip_code_income_proxy': np.random.normal(50000, 15000, num_patients), # Hypothetical income proxy
        'readmitted_30_days': np.random.choice([0, 1], num_patients, p=[0.9, 0.1]) # Target variable: 0=No, 1=Yes
    }

    # Introduce some synthetic correlation for 'readmitted_30_days'
    for i in range(num_patients):
        readmit_prob = 0.05
        if data['age'][i] > 65: readmit_prob += 0.05
        if data['num_chronic_conditions'][i] >= 3: readmit_prob += 0.07
        if data['admission_type'][i] == 'Emergency': readmit_prob += 0.04
        if data['length_of_stay'][i] > 10: readmit_prob += 0.03
        if data['previous_readmission'][i] == 1: readmit_prob += 0.10
        if data['zip_code_income_proxy'][i] < 35000: readmit_prob += 0.05
        
        if np.random.rand() < readmit_prob:
            data['readmitted_30_days'][i] = 1
        else:
            data['readmitted_30_days'][i] = 0

    df_patients = pd.DataFrame(data)

    # Design a preprocessing pipeline (include feature engineering steps)
    categorical_features = ['gender', 'ethnicity', 'admission_type']
    numerical_features = [
        'age', 'num_chronic_conditions', 'length_of_stay',
        'num_medications_discharge', 'lab_value_creatinine',
        'lab_value_hemoglobin', 'zip_code_income_proxy'
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )

    # --- 2.3 Model Development ---
    # Chosen Model: Logistic Regression
    # Justification: Interpretability, probability output, robust baseline, regulatory compliance.

    X = df_patients.drop(['patient_id', 'readmitted_30_days'], axis=1)
    y = df_patients['readmitted_30_days']

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(random_state=42, solver='liblinear', C=1.0)) # C=1.0 is default, represents inverse of regularization strength
    ])

    model_pipeline.fit(X_train, y_train)

    return model_pipeline, X_test, y_test

# Load and train the model (cached)
model_pipeline, X_test, y_test = load_and_train_model()

# --- Streamlit UI Sections ---

st.sidebar.header("Problem Scope & Context")
st.sidebar.markdown(
    """
    **Problem Definition:** To predict the risk of a patient being readmitted to the hospital within 30 days of their discharge.

    **Objectives:**
    1.  Improve Patient Outcomes
    2.  Reduce Healthcare Costs
    3.  Enhance Quality of Care

    **Stakeholders:** Hospital Administration, Clinicians, Patients & Families, Insurance Companies/Payers.
    """
)

st.sidebar.header("Data Strategy & Ethics")
st.sidebar.markdown(
    """
    **Data Sources:** Electronic Health Records (EHRs), Socioeconomic & Environmental Data (conceptual).
    **Ethical Concerns:**
    1.  **Patient Privacy (HIPAA):** Strict protection of Protected Health Information (PHI).
    2.  **Algorithmic Bias:** Risk of perpetuating historical care disparities.

    **Preprocessing (Conceptual):**
    * Data Cleaning (missing values, outliers)
    * Data Transformation (encoding, scaling)
    * Feature Engineering (Comorbidity Index, Readmission History, etc.)
    """
)

st.sidebar.header("Model Details")
st.sidebar.markdown(
    """
    **Chosen Model:** Logistic Regression.
    **Justification:** Highly interpretable, provides probability scores, robust, and aids regulatory compliance.
    """
)

st.header("✨ Predict Patient Readmission Risk")

with st.expander("Model Performance Metrics (on Test Set)"):
    y_pred = model_pipeline.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Confusion Matrix")
        st.table(pd.DataFrame(cm, index=['Actual No Readmission', 'Actual Readmission'], columns=['Predicted No Readmission', 'Predicted Readmission']))
    
    with col2:
        st.subheader("Key Metrics")
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        st.write(f"**Precision:** `{precision:.2f}`")
        st.write(f"**Recall:** `{recall:.2f}`")
        st.write("---")
        st.text("Classification Report:")
        st.code(classification_report(y_test, y_pred, zero_division=0))


st.markdown("---")
st.subheader("Patient Information Input")

# --- Input Widgets for a New Patient ---
col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider("Age", min_value=18, max_value=90, value=60, step=1)
    gender = st.selectbox("Gender", ['Male', 'Female'])
    ethnicity = st.selectbox("Ethnicity", ['Caucasian', 'African American', 'Hispanic', 'Asian', 'Other'])
    num_chronic_conditions = st.number_input("Number of Chronic Conditions", min_value=0, max_value=10, value=2, step=1)

with col2:
    admission_type = st.selectbox("Admission Type", ['Emergency', 'Elective', 'Urgent'])
    length_of_stay = st.number_input("Length of Stay (days)", min_value=1, max_value=60, value=5, step=1)
    num_medications_discharge = st.number_input("Medications at Discharge", min_value=1, max_value=30, value=5, step=1)
    previous_readmission = st.selectbox("Previous Readmission?", ['No', 'Yes'])
    previous_readmission_val = 1 if previous_readmission == 'Yes' else 0

with col3:
    lab_value_creatinine = st.number_input("Creatinine (mg/dL)", min_value=0.5, max_value=5.0, value=1.0, step=0.1, format="%.1f")
    lab_value_hemoglobin = st.number_input("Hemoglobin (g/dL)", min_value=5.0, max_value=20.0, value=12.0, step=0.1, format="%.1f")
    zip_code_income_proxy = st.number_input("Zip Code Income Proxy ($)", min_value=10000, max_value=150000, value=50000, step=1000)

# --- Prediction Button ---
if st.button("Predict Readmission Risk", use_container_width=True, type="primary"):
    new_patient_data = {
        'age': age,
        'gender': gender,
        'ethnicity': ethnicity,
        'num_chronic_conditions': num_chronic_conditions,
        'admission_type': admission_type,
        'length_of_stay': length_of_stay,
        'num_medications_discharge': num_medications_discharge,
        'previous_readmission': previous_readmission_val,
        'lab_value_creatinine': lab_value_creatinine,
        'lab_value_hemoglobin': lab_value_hemoglobin,
        'zip_code_income_proxy': zip_code_income_proxy
    }
    
    # Create DataFrame for prediction
    new_patient_df = pd.DataFrame([new_patient_data])
    
    # Get prediction and probability
    predicted_class = model_pipeline.predict(new_patient_df)[0]
    predicted_prob = model_pipeline.predict_proba(new_patient_df)[:, 1][0]

    st.subheader("Prediction Result:")
    if predicted_class == 1:
        st.error(f"**HIGH RISK of Readmission!** (Probability: {predicted_prob:.2%}) 🚨")
        st.write("Consider immediate post-discharge interventions such as follow-up calls, home health visits, or medication reconciliation.")
    else:
        st.success(f"**LOW RISK of Readmission.** (Probability: {predicted_prob:.2%}) ✅")
        st.write("Continue with standard discharge protocols. Monitor for changes in condition.")

    st.markdown("---")

st.markdown("---")
st.header("Deployment & Optimization Considerations")

st.subheader("2.4 Deployment Outline:")
st.markdown("""
1.  **API Endpoint Creation:** Deploy the trained model as a RESTful API service (e.g., using Flask/Django or cloud services like Google Cloud Endpoints).
2.  **EHR System Integration:** Develop connectors for the Electronic Health Record (EHR) system to send relevant patient data to the model's API upon discharge.
3.  **Real-time Scoring:** Data is automatically fed to the API, generating instant risk scores.
4.  **Alert System/Dashboard:** High-risk scores trigger alerts or display on dashboards for care coordinators to prioritize follow-up actions.
5.  **Clinical Decision Support:** Risk scores are integrated directly into the EHR for clinicians to consider during discharge planning.
""")

st.subheader("Compliance with Healthcare Regulations (e.g., HIPAA):")
st.markdown("""
* **Data Anonymization/Pseudonymization:** Strict protocols for Protected Health Information (PHI).
* **Access Controls (RBAC):** Robust role-based access to the model, data, and API.
* **Data Encryption:** Encrypt all patient data at rest and in transit.
* **Audit Trails:** Maintain comprehensive logs of all model/data access and predictions.
* **Secure Hosting Environment:** Deploy on HIPAA-compliant cloud infrastructure.
* **Consent and Patient Rights:** Align with patient consent policies.
""")

st.subheader("2.5 Optimization: Addressing Overfitting")
st.markdown("""
* **Method:** **Regularization (L1/L2)** for Logistic Regression.
    * This involves adding a penalty term to the model's loss function during training, which discourages overly complex models by penalizing large coefficients.
    * In scikit-learn's `LogisticRegression`, this is controlled by the `C` parameter (inverse of regularization strength) and `penalty` parameter (`'l1'` or `'l2'`). A smaller `C` value implies stronger regularization.
""")

st.markdown("---")
st.header("Reflections & Workflow")

st.subheader("Part 4.1 Reflection:")
st.markdown("""
* **Most Challenging Part:** **Data Collection & Preprocessing** (especially ethical/HIPAA compliance, data quality from EHRs, and bias mitigation).
* **Improvements with more time/resources:**
    * Robust Data Governance & Automated Pipelines.
    * Extensive Bias Auditing & Mitigation Research.
    * Controlled Clinical Pilots & A/B Testing.
    * Full MLOps (Monitoring, Retraining, Versioning).
    * Deeper Multi-Disciplinary Team Collaboration.
""")

# The st.graphviz part has been removed entirely.
st.subheader("Part 4.2 AI Development Workflow Diagram (Textual Representation):")
st.markdown("""
    * **Problem Definition**
    * **Data Collection**
    * **Preprocessing & Feature Engineering**
    * **Model Development**
    * **Model Evaluation**
    * **Deployment**
    * **Monitoring & Maintenance**

    *(Feedback loops exist from Monitoring to Data Collection/Preprocessing, and from Evaluation/Preprocessing to Problem Definition)*
    """)
