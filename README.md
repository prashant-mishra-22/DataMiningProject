# 🧬 Disease Classification Model Comparison

This project was developed as part of an academic course on **B.Sc. (Hons.) Computer Science Semester VI BHCS 17B: Data Mining**. It explores and compares the performance of multiple classification algorithms on medical datasets to analyze their effectiveness in disease prediction and diagnosis.

## 🧾 Objective

The primary objective is to apply supervised machine learning models on disease-related datasets and compare their performance based on evaluation metrics such as **Accuracy**, **Precision**, **Recall**, **F1-score**, and **ROC-AUC**. This comparative analysis helps understand which models are most reliable for different types of medical classification problems.

---

## ⚙️ Tools & Technologies

- **Programming Language**: Python  
- **Libraries**: 
  - Data Processing: `pandas`, `numpy`
  - Visualization: `matplotlib`, `seaborn`
  - Modeling: `scikit-learn`
- **Environment**: Jupyter Notebook

---

## 🧪 Models Implemented

We compared the following classification models:
- **Logistic Regression**
- **K-Nearest Neighbors (KNN)**
- **Support Vector Machine (SVM)**
- **Decision Tree**
- **Random Forest**

---

## 📊 Evaluation Metrics

Each model was evaluated based on:
- **Confusion Matrix**
- **Classification Report (Precision, Recall, F1-Score)**
- **ROC Curve and AUC**
- **Accuracy Score**

---

## 📁 Dataset Overview

Multiple disease-related datasets (e.g., heart disease, diabetes) were used. Each dataset underwent:
- Data cleaning and preprocessing
- Feature scaling (Standardization/Normalization)
- Train-test split
- Model training and evaluation

---

## 📈 Key Findings

- **Random Forest** and **SVM** generally offered higher classification performance on imbalanced medical datasets.
- Simpler models like **Logistic Regression** performed well when feature separation was linear.
- **KNN** was sensitive to feature scaling and showed variance based on the value of 'k'.
- Visualization helped uncover patterns in model performance that raw metrics did not fully capture.

---

## 👥 Team Members

- **Prashant**
- **Prateek**
- **Sparsh**
- **Raj**

---

## 📎 File

- `classification-model-comparison-diseases.ipynb`: Main Jupyter Notebook containing code, analysis, visualizations, and results.

---

## 📌 Future Work

- Integrate more advanced techniques like **XGBoost** or **Neural Networks**
- Use larger, real-world medical datasets (e.g., from Kaggle or UCI)
- Address class imbalance using **SMOTE** or other resampling techniques
- Deploy the best-performing model in a web interface for demonstration

---

