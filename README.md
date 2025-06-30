# AI Project Workflow: Churn & Readmission Risk
This document summarizes an AI project assessment, covering a hypothetical customer churn prediction problem and a case study on hospital patient readmission risk, including critical thinking and workflow reflection.

## Table of Contents
- Part 1: Short Answer Questions

- Part 2: Case Study Application (Hospital Readmission Risk)

- Part 3: Critical Thinking

- Part 4: Reflection & Workflow Diagram

### Part 1: Short Answer Questions
#### 1.1 Problem Definition (Customer Churn)
Problem: Predicting telecom customer churn.

Objectives: Customer retention, revenue protection, targeted interventions.

Stakeholders: Marketing, Customer Service.

KPI: Churn Rate Reduction (%).

#### 1.2 Data Collection & Preprocessing
Data Sources: Call records, billing history.

Potential Bias: Selection bias (e.g., focus on high-value customers).

Preprocessing: Missing data imputation, feature scaling, categorical encoding.

#### 1.3 Model Development
Chosen Model: Gradient Boosting Classifier (e.g., XGBoost).

Justification: High accuracy, handles non-linearity, provides feature importance.

Data Split: Stratified sampling: Train (70%), Validation (15%), Test (15%).

Hyperparameters: n_estimators, learning_rate (for complexity and robustness).

#### 1.4 Evaluation & Deployment
Evaluation Metrics: Precision (correctly flagged churners), Recall (identified actual churners).

Concept Drift: Changes in feature-target relationship over time.

Monitoring: Performance tracking, data drift detection, regular retraining.

Technical Challenge: Real-time inference latency.

### Part 2: Case Study Application (Hospital Readmission Risk)
#### 2.1 Problem Scope
Problem: Predict patient readmission risk within 30 days of discharge.

Objectives: Improve patient outcomes, reduce costs, enhance care quality.

Stakeholders: Hospital Admin, Clinicians, Patients, Insurers.

#### 2.2 Data Strategy
Data Sources: Electronic Health Records (EHRs), Socioeconomic data.

Ethical Concerns: Patient Privacy (HIPAA), Algorithmic Bias.

Preprocessing & Feature Engineering: Cleaning (missing values), Transformation (encoding, scaling), Engineering (Length of Stay, Comorbidity Index).

#### 2.3 Model Development
Chosen Model: Logistic Regression.

Justification: Interpretability, probability output, robust baseline, regulatory compliance.

Confusion Matrix (Hypothetical):

TP=60, FN=40, FP=90, TN=870 (out of 1000 patients)

Precision: 

60+90
60
​
 =0.40
    * Recall:

60+40
60
​
 =0.60
#### 2.4 Deployment
Integration: API endpoint, EHR integration, real-time scoring, alerts/dashboard.

Compliance (HIPAA): Anonymization, access controls, encryption, audit trails, secure hosting, consent.

#### 2.5 Optimization
Overfitting Method: Regularization (L1/L2) for Logistic Regression.

### Part 3: Critical Thinking
#### 3.1 Ethics & Bias
Effect of Biased Data: Under/over-prediction for certain groups, perpetuating disparities.

Mitigation Strategy: Fairness-Aware Data Collection & Feature Engineering (diverse data, social determinants, bias auditing).

#### 3.2 Trade-offs
Interpretability vs. Accuracy: Interpretability often prioritized in healthcare for trust, actionability, and accountability.

Limited Computational Resources Impact: Favors simpler, less resource-intensive models (e.g., Logistic Regression); avoids Deep Learning; simpler feature engineering.

### Part 4: Reflection & Workflow Diagram
#### 4.1 Reflection
Most Challenging Part: Data Collection & Preprocessing (ethical/HIPAA, data quality, bias mitigation).

Improvements: Robust Data Governance, Bias Auditing, Clinical Pilots, MLOps, Multi-Disciplinary Collaboration.

#### 4.2 AI Development Workflow Diagram
graph TD
    A[Problem Definition] --> B(Data Collection)
    B --> C(Preprocessing)
    C --> D(Model Development)
    D --> E(Evaluation)
    E --> F(Deployment)
    F --> G(Monitoring)
    G -- Drift / New Data --> B
    C -- Feature Insights --> A
    E -- Insights --> A

##### Stages Labeled: Problem Definition, Data Collection, Preprocessing, Model Development, Evaluation, Deployment, Monitoring.

##### Feedback Loops: Monitoring to Data Collection/Preprocessing, and Insights from various stages back to Problem Definition.
