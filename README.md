# Customer Segmentation & Fraud Detection Pipeline

An end-to-end Machine Learning project combining unsupervised behavioral clustering and supervised classification models to analyze customer transaction patterns and support financial fraud detection.

---

## 📌 Project Overview
- **Objective:** Segment banking customers based on behavioral transaction features and detect potential high-risk anomalous activities.
- **Workflow:**
  1. Exploratory Data Analysis (EDA) and data preprocessing.
  2. Feature engineering and scaling.
  3. Customer segmentation using **K-Means Clustering** (validated via Elbow Method & Silhouette Score).
  4. Supervised predictive modeling using **Decision Tree** and **Random Forest Classifier** with hyperparameter tuning.

---

## 📂 Repository Contents
- `01_customer_clustering.ipynb`: Data exploration, preprocessing, scaling, and clustering analysis.
- `02_fraud_classification.ipynb`: Model training, hyperparameter tuning (GridSearchCV), and evaluation for segment classification.

---

## 🛠️ Tech Stack & Libraries
- **Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn, Yellowbrick, Joblib

---

## 📊 Key Results
- Optimal cluster evaluation identified behavioral groups with distinct spending and frequency distributions.
- Random Forest Classifier achieved high precision, recall, and F1-score across all segment classifications.
