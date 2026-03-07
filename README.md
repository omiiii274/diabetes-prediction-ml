
# 🩺 Diabetes Prediction — Healthcare Machine Learning

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)

## 📋 About This Project

This project builds machine learning models to predict whether a patient will develop Type 2 diabetes based on 8 routine clinical measurements. It uses the Pima Indians Diabetes dataset — one of the most widely used datasets in healthcare machine learning research — containing records from 768 female patients aged 21 and above.

Diabetes affects over 4.9 million people in the UK and costs the NHS approximately £10 billion per year. Type 2 diabetes can often be prevented or delayed through early lifestyle changes — but only if high-risk individuals are identified before the disease develops. Machine learning models trained on routine clinical data can support targeted screening programmes.

## ❓ Problem This Solves

GPs currently use simple risk calculators (like the QDiabetes tool) to assess diabetes risk. These tools use basic rules. Machine learning models can capture complex, non-linear relationships between risk factors that simple rules miss — potentially identifying at-risk patients earlier.

## 🔬 How It Works

**Step 1: The Data (8 Features)**

| Feature | What It Measures | Why It Matters |
|---------|-----------------|---------------|
| Pregnancies | Number of pregnancies | Gestational diabetes increases future risk |
| Glucose | Blood glucose level (mg/dL) | Directly measures blood sugar control |
| Blood Pressure | Diastolic BP (mm Hg) | Hypertension is a diabetes comorbidity |
| Skin Thickness | Triceps skin fold (mm) | Proxy for body fat percentage |
| Insulin | 2-hour serum insulin (μU/mL) | Measures insulin production capacity |
| BMI | Body Mass Index (kg/m²) | Obesity is the strongest modifiable risk factor |
| Diabetes Pedigree | Family history score | Genetic predisposition to diabetes |
| Age | Patient age (years) | Risk increases significantly after age 45 |

**Step 2: Exploratory Data Analysis**

Before building models, I explored the data to understand:
- How each feature differs between diabetic and healthy patients
- Which features correlate with each other (to check for multicollinearity)
- The overall distribution of the target variable (268 diabetic, 500 healthy = 35% prevalence)

**Step 3: Three Models Compared**

| Model | AUC-ROC |
|-------|---------|
| Logistic Regression | 0.83 |
| SVM | 0.84 |
| **Random Forest** | **0.85** |

**Step 4: Most Important Features**

Random Forest identified the top predictors:
1. **Glucose** — strongest predictor (aligns with clinical diagnostic criteria: fasting glucose ≥ 126 mg/dL = diabetes)
2. **BMI** — obesity is the primary modifiable risk factor for Type 2 diabetes
3. **Age** — risk doubles every decade after age 45
4. **Diabetes Pedigree** — family history contributes significant genetic risk
5. **Pregnancies** — gestational diabetes history increases lifelong risk

## 📊 Key Numbers

| Metric | Value |
|--------|-------|
| Total patients | 768 |
| Diabetic patients | 268 (35%) |
| Healthy patients | 500 (65%) |
| Best model | Random Forest |
| Best AUC-ROC | 0.85 |
| Top predictor | Glucose level |

## 🛠️ Tools Used

| What | Tool |
|------|------|
| Models | Logistic Regression, Random Forest, SVM |
| Framework | scikit-learn |
| Data scaling | StandardScaler |
| Data source | Pima Indians Diabetes Dataset |
| Plotting | Matplotlib |

## 📁 Files In This Project
