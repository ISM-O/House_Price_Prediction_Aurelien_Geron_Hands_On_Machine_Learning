# 🏠 California Housing Price Prediction

> A complete machine learning project inspired by the book *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* by Aurélien Géron.

## 📘 Project Overview

This project was conducted as part of my **personal and academic learning journey** while studying the concepts from Aurélien Géron's book. The goal is to predict median house prices in California districts using various features such as income, population, and proximity to the ocean.

---

## 🎯 Objectives

- Apply end-to-end data science and machine learning workflows.
- Build and evaluate multiple regression models.
- Understand the impact of feature engineering.
- Deploy data pipelines and perform model optimization.

---

## 🧰 Technologies & Libraries

- Python 3.10
- Scikit-learn
- Pandas / NumPy
- Matplotlib / Seaborn
- XGBoost
- Jupyter Notebook

---

## 📊 Dataset

- Source: California housing dataset (available via Scikit-learn or Géron's book).
- Features include: `longitude`, `latitude`, `housing_median_age`, `total_rooms`, `total_bedrooms`, `population`, `households`, `median_income`, `ocean_proximity`, etc.

---

## 📌 Project Workflow

### 1. Data Preparation
- Handling missing values.
- Encoding categorical variable: `ocean_proximity`.
- Creating new features:
  - `rooms_per_household`
  - `bedrooms_per_room`
  - `population_per_household`

### 2. Data Exploration & Visualization
- Correlation analysis.
- Histograms, scatter plots, and geospatial plots.

### 3. Model Training & Evaluation

| Model                 | RMSE (Test) | Notes                              |
|----------------------|-------------|-------------------------------------|
| Linear Regression     | ~68.6       | Underfitting                        |
| Decision Tree         | ~36.9       | Overfitting                         |
| Random Forest         | ~35.7       | Good performance, more stable       |
| SVR (RBF Kernel)      | ~72.7       | Underfitting                        |
| **XGBoost**           | **43.1**    | **Best performing model overall**   |
| Voting Regressor      | 45.6        | Slightly worse than XGBoost alone   |

### 4. Optimization & Pipelines
- Implemented full data processing pipelines using `Pipeline` from Scikit-learn.
- Hyperparameter tuning with `GridSearchCV` and `RandomizedSearchCV`.

### 5. Feature Importance
- Used XGBoost to compute feature importances.
- `median_income` and engineered features like `bedrooms_per_room` were the most significant.

---

## 🧾 Deliverables

- ✅ Jupyter Notebook with complete project steps.
- ✅ Machine learning pipeline (`housing_preprocessing_pipeline.pkl`).
- ✅ Trained model saved with `joblib` (`xgboost_model.pkl`).
- ✅ Visualizations for data understanding and performance comparison.

---

## 🚀 Results & Insights

- The feature `median_income` had the strongest correlation with housing prices.
- Engineered features significantly improved model performance.
- XGBoost outperformed all other models with an RMSE of ~43.1.
- The model shows promising generalization on test data.

---

## 🧠 Skills Demonstrated

- Data cleaning and transformation
- Feature engineering
- Regression modeling
- Model selection and validation
- Pipeline automation
- Performance tuning

---

## 🔮 Future Improvements

- Deploy the model with a Flask or FastAPI backend for inference.
- Build an interactive dashboard with Streamlit or Dash.
- Extend the project to work with real-world datasets.

---

## 📚 Reference

- *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* – Aurélien Géron

---

## 👨‍💻 Author

Ismael — Data Science & Cloud Computing Engineering Student  

