# 🏠 House Price Prediction – End-to-End Machine Learning Project

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-Model-green)
![Status](https://img.shields.io/badge/Project-Completed-brightgreen)
![License](https://img.shields.io/badge/License-Academic-lightgrey)

---

## 📌 Project Overview

This project is an **end-to-end Machine Learning system** designed to predict **house prices** based on real estate features such as:

* Surface area (m²)
* Number of bedrooms
* Floors
* Year of construction
* Location type (Urban / Rural),...

The system compares multiple models and produces a **final robust price estimation** using ensemble prediction.

---

## 🎯 Objective

The main goal is to build a **reliable predictive system for real estate pricing** using Machine Learning techniques.

✔ Improve prediction accuracy
✔ Compare multiple ML models
✔ Build a robust ensemble system
✔ Provide a final market-consistent price

---

## 🧠 Machine Learning Pipeline

### 🔄 Workflow Diagram

```
DATA COLLECTION
      ↓
DATA CLEANING
      ↓
EDA (Exploratory Data Analysis)
      ↓
FEATURE ENGINEERING
      ↓
TRAIN / TEST SPLIT
      ↓
MODEL TRAINING
      ↓
MODEL EVALUATION
      ↓
ENSEMBLE PREDICTION
      ↓
FINAL PRICE ESTIMATION
```

---

## 📊 Models Used

* Linear Regression
* Ridge Regression
* Random Forest
* Gradient Boosting
* XGBoost
* Stacking Regressor *(optional)*

---

## 🔍 Exploratory Data Analysis (EDA)

In this stage, we analyze:

* Distribution of house prices
* Correlation between features
* Missing values
* Outliers detection
* Feature importance

📌 Key insight:
Surface area and location are the strongest predictors of house price.

---

## ⚙️ Feature Engineering

Important transformations applied:

* Encoding categorical variables (Location)
* Handling missing values
* Feature scaling (if required)
* Creating derived features (e.g., age of building)

Example:

```
Building Age = Current Year - Year of Construction
```

---

## 🏗️ Model Training

Each model is trained using:

```python
model.fit(X_train, y_train)
```

Training strategy:

* Train/Test Split
* Cross-validation (for robust evaluation)
* Hyperparameter tuning (GridSearch for best performance)

---

## 📈 Model Evaluation

Models are evaluated using:

* R² Score
* RMSE (Root Mean Squared Error)
* MAE (Mean Absolute Error)

### 🔬 Best Model Performance (Example)

| Model             | R²   | RMSE | MAE |
| ----------------- | ---- | ---- | --- |
| Random Forest     | 0.99 | 4.1  | 2.8 |
| XGBoost           | 0.98 | 5.0  | 3.2 |
| Gradient Boosting | 0.97 | 6.0  | 3.8 |

---

## 🧪 Final Prediction System

The final price is computed using an **ensemble average**:

```
Final Price = (XGBoost + RandomForest + GradientBoosting) / 3
```

### 📌 Example Output

```
🏠 House Prediction Result

Predictions:
- XGBoost → 166.77M Ar
- RandomForest → 166.21M Ar
- GradientBoosting → 167.23M Ar

🎯 Final Price: 166.74 Million Ar
📊 Confidence: High (low variance between models)
```

---

## 📉 Model Insight

* High R² (>0.99) → excellent predictive power
* Low RMSE → very small prediction error
* Low variance → models are consistent

⚠️ Note: Slight risk of overfitting should be validated using cross-validation.

---

## 🚀 Project Structure

```
house_price_prediction/
│
├── data/
├── models/
├── notebooks/
│   └── EDA.ipynb
├── src/
├── test/
│   └── test_local.py
├── README.md
```

---

## 🔮 Future Improvements

* 🌐 Deploy model with Flask / FastAPI
* 📱 Build web app interface
* ☁️ Deploy on cloud (AWS / Render / Heroku)
* 📊 Add real-time dashboards
* 🧠 Add Deep Learning model comparison

---

## 👨‍💻 Author

**Fanantenana Manaosoa**
🎓 Master 2 – Artificial Intelligence
📍 ENI Fianarantsoa – Madagascar
📌 Domain: Data Science & Machine Learning (Gouvernance et Ingenerie de Données GID)

---

## ⭐ Conclusion

This project demonstrates a complete **Machine Learning pipeline** from data preprocessing to final prediction, using multiple models and ensemble learning to ensure **high accuracy and robustness**.
