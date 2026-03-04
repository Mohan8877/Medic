# 🩺 AI-Based Early Risk Detection & Intelligent Medical Triage System

## 📌 Overview

The **AI-Based Early Risk Detection & Intelligent Medical Triage System** is an intelligent healthcare assistant designed to analyze user symptoms, predict possible diseases, estimate health risks, and provide medical recommendations.

This system acts as a **digital triage assistant**, helping users understand the seriousness of their symptoms before consulting a doctor.

The platform uses **Machine Learning, NLP, and rule-based risk analysis** to generate predictions and provide actionable healthcare guidance.

---

# 🎯 Objectives

The primary goals of this project are:

- Detect health risks early using AI
- Provide accessible medical guidance
- Support voice and text symptom input
- Assist users in medical triage decisions
- Improve healthcare awareness
- Provide explainable AI predictions

---

# 🚀 Key Features

## 1️⃣ Symptom Input System

Users can describe their symptoms through:

- Text Input
- Voice Input (Speech Recognition)

Example input:

```
I have fever, cough and chest pain
```

---

## 2️⃣ Automatic Symptom Detection

The system detects symptoms from user input.

Example:

Input:

```
I have fever and cough
```

Detected symptoms:

```
fever
cough
```

---

## 3️⃣ Disease Prediction (Machine Learning)

The system predicts possible diseases using a trained ML model.

Example output:

```
Predicted Disease: Pneumonia
Confidence: 78%
```

---

## 4️⃣ Health Risk Scoring

A **risk score (0–100)** is calculated based on predicted disease severity.

Example:

```
Risk Score: 72/100
```

---

## 5️⃣ Risk Classification

| Score | Risk Level |
|------|------------|
| 0 – 25 | Low |
| 26 – 50 | Moderate |
| 51 – 75 | High |
| 76 – 100 | Critical |

Example:

```
Risk Level: High
```

---

## 6️⃣ Intelligent Medical Recommendations

The system provides recommendations based on risk level.

### Low Risk
- Self care
- Drink fluids
- Take rest
- Monitor symptoms

### Moderate Risk
- Teleconsultation recommended
- Monitor symptoms for 24 hours

### High Risk
- Visit nearest hospital
- Medical checkup required

### Critical Risk
- Emergency medical attention required
- Call ambulance immediately

---

## 7️⃣ Explainable AI

The system explains why a prediction occurred.

Example:

```
Prediction based on:
• Fever detected
• Chest pain detected
• Breathing difficulty detected
```

---

## 8️⃣ Emergency Detection

If critical symptoms are detected the system triggers emergency alerts.

Example:

```
CRITICAL CONDITION DETECTED
Seek emergency medical care immediately
```

---

# 🧠 System Workflow

```
User Input (Text / Voice)
        ↓
Symptom Detection
        ↓
Feature Vector Creation
        ↓
Machine Learning Prediction
        ↓
Risk Score Calculation
        ↓
Risk Classification
        ↓
Recommendation Engine
        ↓
Emergency / Hospital Suggestion
```

---

# 🏗 System Architecture

The project follows a **modular scalable architecture**.

Layers include:

1. Interface Layer
2. API Layer
3. Core AI Logic Layer
4. Machine Learning Layer
5. Knowledge Base Layer
6. Integration Layer

---

# 📂 Project Structure

```
health-triage-ai/

├── config/
│   ├── settings.py
│   ├── constants.py
│   └── logging_config.py
│
├── data/
│   ├── raw/
│   │   └── symptom_dataset.csv
│   ├── processed/
│   │   └── processed_symptoms.csv
│   └── knowledge_base/
│       ├── disease_severity.json
│       └── symptom_dictionary.json
│
├── models/
│   ├── training/
│   │   ├── train_model.py
│   │   ├── data_preprocessing.py
│   │   └── feature_engineering.py
│   │
│   ├── inference/
│   │   ├── predictor.py
│   │   └── model_loader.py
│   │
│   └── saved_models/
│       └── health_model.pkl
│
├── core/
│   ├── symptom_detection/
│   │   ├── symptom_extractor.py
│   │   └── symptom_mapper.py
│   │
│   ├── disease_prediction/
│   │   └── disease_classifier.py
│   │
│   ├── risk_assessment/
│   │   ├── risk_engine.py
│   │   └── severity_classifier.py
│   │
│   ├── triage/
│   │   ├── triage_engine.py
│   │   └── emergency_detector.py
│   │
│   └── explainability/
│       └── explanation_generator.py
│
├── services/
│   ├── recommendation_service.py
│   ├── hospital_locator_service.py
│   ├── language_service.py
│   └── notification_service.py
│
├── api/
│   ├── routes/
│   │   ├── health_route.py
│   │   ├── prediction_route.py
│   │   └── emergency_route.py
│   │
│   └── app.py
│
├── frontend/
│   ├── public/
│   │   └── index.html
│   │
│   ├── css/
│   │   └── styles.css
│   │
│   ├── js/
│   │   ├── main.js
│   │   ├── voice_input.js
│   │   └── visualization.js
│   │
│   └── assets/
│       └── icons/
│
├── integrations/
│   ├── speech/
│   │   ├── speech_to_text.py
│   │   └── text_to_speech.py
│   │
│   ├── translation/
│   │   └── translator.py
│   │
│   └── maps/
│       └── hospital_locator.py
│
├── database/
│   ├── models.py
│   └── db_manager.py
│
├── tests/
│   ├── test_prediction.py
│   └── test_symptom_detection.py
│
├── scripts/
│   └── initialize_system.py
│
├── requirements.txt
└── README.md
```

---

# ⚙️ Technology Stack

### Frontend
- HTML
- CSS
- JavaScript
- Chart.js

### Backend
- Python
- Flask

### Machine Learning
- Scikit-learn
- Pandas
- NumPy

### Integrations
- Web Speech API
- Google Maps API
- Translation APIs

---

# 📊 Dataset

Example dataset structure:

| Fever | Cough | Chest Pain | Breathlessness | Disease |
|------|------|------|------|------|
|1|1|0|0|Flu|
|1|1|1|1|Pneumonia|
|0|0|1|1|Heart Disease|

---

# 🧪 Model Training

Training workflow:

```
Load dataset
Preprocess symptoms
Train ML model
Evaluate accuracy
Save trained model
```

Possible models:

- Random Forest
- Decision Tree
- Logistic Regression

---

# ▶️ Running the Project

Install dependencies:

```
pip install -r requirements.txt
```

Train model:

```
python models/training/train_model.py
```

Run API:

```
python api/app.py
```

Open frontend:

```
frontend/public/index.html
```

---

# 🔮 Future Extensions

- AI health chatbot
- Multi-language support
- Wearable device integration
- Personalized health recommendations
- Telemedicine integration

---

# 📈 Expected Impact

This system can help with:

- Early disease detection
- Faster healthcare decision-making
- Improved healthcare awareness
- Assistance for rural populations

---

# 👨‍💻 Author

AI-Based Health Triage System  
Developed as an AI healthcare innovation project.# Medic
# Medic
# Medic
# Medic
# Medic
