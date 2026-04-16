import streamlit as st
import numpy as np
import joblib
import pandas as pd

import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="🏠 House Price Dashboard",
    layout="wide"
)

st.title("🏠 House Price Prediction - DATA SCIENTIST DASHBOARD")
st.write("Comparaison de modèles + visualisation des prédictions")

# =========================
# LOAD MODELS
# =========================
model_rf = joblib.load("models/model_randomforest.pkl")
model_xgb = joblib.load("models/model_xgboost.pkl")
model_gb = joblib.load("models/model_gradientboosting.pkl")

models = {
    "Random Forest": model_rf,
    "XGBoost": model_xgb,
    "Gradient Boosting": model_gb
}

# =========================
# SIDEBAR INPUTS
# =========================
st.sidebar.header("📋 Features Maison")

superficie_m2 = st.sidebar.number_input("Superficie (m²)", 10.0, 1000.0, 100.0)
nb_chambres = st.sidebar.number_input("Chambres", 1, 10, 3)
nb_etages = st.sidebar.number_input("Étages", 1, 5, 1)

acces_route = st.sidebar.selectbox("Accès route", [0, 1])
eau_electricite = st.sidebar.selectbox("Eau + électricité", [0, 1])
parking = st.sidebar.selectbox("Parking", [0, 1])

annee = st.sidebar.number_input("Année construction", 1900, 2025, 2000)

# localisation
loc_periurbain = st.sidebar.selectbox("Périurbain", [0, 1])
loc_rural = st.sidebar.selectbox("Rural", [0, 1])
loc_urbain = st.sidebar.selectbox("Urbain", [0, 1])

# connexion
conn_aucune = st.sidebar.selectbox("Connexion aucune", [0, 1])
conn_fibre = st.sidebar.selectbox("Fibre", [0, 1])
conn_starlink = st.sidebar.selectbox("Starlink", [0, 1])

# sol
sol_brut = st.sidebar.selectbox("Sol brut", [0, 1])
sol_carrelage = st.sidebar.selectbox("Carrelage", [0, 1])
sol_ciment = st.sidebar.selectbox("Ciment", [0, 1])

# etat
etat_reno = st.sidebar.selectbox("À rénover", [0, 1])
etat_bon = st.sidebar.selectbox("Bon état", [0, 1])
etat_neuf = st.sidebar.selectbox("Neuf", [0, 1])

# =========================
# INPUT VECTOR
# =========================
input_data = np.array([[
    superficie_m2,
    nb_chambres,
    nb_etages,
    acces_route,
    eau_electricite,
    parking,
    annee,
    loc_periurbain,
    loc_rural,
    loc_urbain,
    conn_aucune,
    conn_fibre,
    conn_starlink,
    sol_brut,
    sol_carrelage,
    sol_ciment,
    etat_reno,
    etat_bon,
    etat_neuf
]])

# =========================
# PREDICTIONS
# =========================
predictions = {}

if st.button("🔮 Predict Price"):

    st.subheader("📊 Model Predictions")

    for name, model in models.items():
        pred = model.predict(input_data)[0]
        predictions[name] = pred

    df = pd.DataFrame({
        "Model": list(predictions.keys()),
        "Price (Millions Ar)": list(predictions.values())
    })

    st.dataframe(df)

    # =========================
    # AVERAGE (ENSEMBLE)
    # =========================
    avg_price = np.mean(list(predictions.values()))

    st.success(f"🎯 Final Price (Ensemble): {avg_price:.2f} Millions Ar")

    # =========================
    # GRAPH
    # =========================
    st.subheader("📈 Model Comparison")

    fig, ax = plt.subplots()
    ax.bar(df["Model"], df["Price (Millions Ar)"])
    ax.set_ylabel("Millions Ar")
    ax.set_title("Comparison of ML Models")

    st.pyplot(fig)
