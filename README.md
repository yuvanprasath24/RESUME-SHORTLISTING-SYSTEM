# 🚀 AI Resume Screening System

An end-to-end Machine Learning + API-based Resume Screening system that classifies resumes as **Relevant / Not Relevant** based on job description and role matching.

Built using **Scikit-learn, Flask, NLP processing, and React**.

---

## 📌 Features

- Resume classification (Hire / Reject → Relevant / Not Relevant)
- TF-IDF + Logistic Regression model
- Role-based keyword matching
- Optional semantic similarity scoring
- Resume upload support:
  - `.txt`
  - `.pdf`
  - `.docx`
- REST API with confidence score
- Model metrics endpoint
- Production-ready model artifact export

---

# 🏗 Project Architecture

## Model Flow: From Data to API

```
Dataset (AI_Resume_Screening.csv)
        ↓
Data Ingestion & Preprocessing (train.py)
        ↓
Training Pipeline (TF-IDF + LogisticRegression)
        ↓
Evaluation & Artifact Export (.pkl + metrics.json)
        ↓
API Startup (app.py loads artifacts)
        ↓
POST /api/predict → Inference + Role Relevance Layer
```

---

# 📊 Training Pipeline (train.py)

## 1️⃣ Data Ingestion

- Reads dataset from:
  - `AI_Resume_Screening.csv`
  - or structured Category/Resume format
- Cleans text using `clean_text()` function
- Builds resume text if dataset contains structured columns

---

## 2️⃣ Model Training

- Encodes labels using `LabelEncoder`
  - Hire → 1
  - Reject → 0
- Stratified train-test split using `train_test_split`
- Converts text to vectors using:

```
TfidfVectorizer
```

- Trains classifier:

```
LogisticRegression
```

---

## 3️⃣ Evaluation

Generates:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

Exports metrics to:

```
metrics.json
```

---

## 4️⃣ Saved Artifacts

After training:

```
model.pkl
vectorizer.pkl
label_encoder.pkl
metrics.json
```

These are loaded by the API during startup.

---

# 🌐 API (app.py)

On startup:

```python
joblib.load("model.pkl")
joblib.load("vectorizer.pkl")
joblib.load("label_encoder.pkl")
```

---

## 📌 API Endpoints

### ✅ Health Check

```
GET /api/health
```

Returns API status.

---

### 📊 Metrics

```
GET /api/metrics
```

Returns saved training metrics from `metrics.json`.

---

### 🔍 Prediction

```
POST /api/predict
```

### Request:
- Resume file (txt/pdf/docx)
- job_description (string)

---

# 🔍 Inference Flow (POST /api/predict)

1. Extract resume text from file
2. Clean text
3. Transform using:
   ```
   vectorizer.transform()
   ```
4. Run prediction:
   ```
   model.predict()
   model.predict_proba()
   ```
5. Apply Role Relevance Layer:
   - Keyword match scoring
   - Optional semantic similarity scoring
6. Return structured JSON response

---

# 📤 Sample Response

```json
{
  "prediction": "Relevant",
  "predicted_category": "Data Scientist",
  "confidence": 0.91,
  "semantic_score": 0.87,
  "matched_skills": ["Python", "Machine Learning", "Pandas"]
}
```

---

# 🛠 Tech Stack

## Backend
- Python
- Flask
- Flask-CORS

## ML / Data
- scikit-learn
- pandas
- numpy
- joblib

## NLP / Semantic Layer
- spaCy
- sentence-transformers

## File Parsing
- PyPDF2
- python-docx

## Frontend
- React 18
- Vite

## Deployment
- GitHub
- Render

---

# 🖥 Installation Guide

## 1️⃣ Clone Repository

```bash
git clone https://github.com/yourusername/ai-resume-screening.git
cd ai-resume-screening
```

---

## 2️⃣ Create Virtual Environment

```bash
python -m venv venv
```

Activate:

**Windows**
```
venv\Scripts\activate
```

**Mac/Linux**
```
source venv/bin/activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4️⃣ Train Model

```bash
python train.py
```

This will generate the `.pkl` artifacts.

---

## 5️⃣ Run API

```bash
python app.py
```

Server runs at:

```
http://localhost:5000
```

---

# 📂 Project Structure

```
├── train.py
├── app.py
├── AI_Resume_Screening.csv
├── model.pkl
├── vectorizer.pkl
├── label_encoder.pkl
├── metrics.json
├── requirements.txt
└── frontend/
```

---

# 📈 Why Logistic Regression?

- Fast training
- Lightweight
- Interpretable
- Works effectively with TF-IDF features
- Easy to deploy in production APIs

---

# 🔮 Future Improvements

- Replace TF-IDF with Transformer embeddings
- Add resume ranking system
- Add recruiter dashboard
- Dockerize application
- Add authentication & role-based access

---

# 👨‍💻 Author

Built as a complete ML-to-API deployment project demonstrating:

- Data preprocessing
- Model training
- Model evaluation
- Artifact export
- REST API development
- Full inference pipeline

---

⭐ If you found this useful, consider giving the repository a star.
