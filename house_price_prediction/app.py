

import streamlit as st
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# =========================
# CONFIG PAGE
# =========================
st.set_page_config(
    page_title="🏠 House Price Predictor Pro",
    page_icon="🏠",
    layout="wide"
)

# =========================
#    CUSTOM CSS 
# =========================
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .price-card {
        background-color: #1e3c72;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .recommendation-good {
        background-color: #2ecc71;
        padding: 1rem;
        border-radius: 10px;
        color: white;
    }
    .recommendation-warning {
        background-color: #e74c3c;
        padding: 1rem;
        border-radius: 10px;
        color: white;
    }
    .insight-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1e3c72;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# =========================
#        HEADER
# =========================
st.markdown("""
<div class="main-header">
    <h1>🏠 House Price Prediction Dashboard</h1>
    <p>Modèles: Random Forest | XGBoost | Gradient Boosting</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# =========================
#     LOAD MODELS
# =========================
@st.cache_resource
def load_models():
    try:
        model_rf = joblib.load("models/model_randomforest.pkl")
        model_xgb = joblib.load("models/model_xgboost.pkl")
        model_gb = joblib.load("models/model_gradientboosting.pkl")
        models = {
            "🌲 Random Forest": model_rf,
            "⚡ XGBoost": model_xgb,
            "📈 Gradient Boosting": model_gb
        }
        return models, True
    except Exception as e:
        st.error(f"Erreur chargement modèles: {e}")
        return {}, False

models, models_loaded = load_models()

# =========================
# SIDEBAR - INPUTS CLIENTS
# =========================
st.sidebar.header("🏡 Caractéristiques de la maison")
st.sidebar.markdown("---")

# Colonnes dans le sidebar
superficie_m2 = st.sidebar.slider("📐 Superficie (m²)", 20.0, 500.0, 120.0, 5.0)
nb_chambres = st.sidebar.selectbox("🛏️ Nombre de chambres", [1, 2, 3, 4, 5, 6, 7, 8,9,10,11,12,13,14,15])
nb_etages = st.sidebar.selectbox("🏢 Nombre d'étages", [1, 2, 3, 4, 5,6,7,8])

st.sidebar.markdown("---")
st.sidebar.subheader("📍 Localisation")
localisation = st.sidebar.selectbox("Type de zone", ["Urbain", "Périurbain", "Rural"])

st.sidebar.markdown("---")
st.sidebar.subheader("🛣️ Accès et commodités")
acces_route = st.sidebar.radio("Accès route", ["Oui", "Non"], horizontal=True)
eau_electricite = st.sidebar.radio("Eau + Électricité", ["Oui", "Non"], horizontal=True)
parking = st.sidebar.radio("Parking", ["Oui", "Non"], horizontal=True)

st.sidebar.markdown("---")
st.sidebar.subheader("🏗️ Construction")
annee = st.sidebar.slider("Année de construction", 1950, 2026, 2010, 5)

st.sidebar.markdown("---")
st.sidebar.subheader("🖼️ Type de sol")
type_sol = st.sidebar.selectbox("Revêtement", ["Carrelage", "Ciment", "Brut"])

st.sidebar.markdown("---")
st.sidebar.subheader("🔌 Connexion internet")
connexion = st.sidebar.selectbox("Type", ["Fibre", "Starlink", "Aucune"])

st.sidebar.markdown("---")
st.sidebar.subheader("🔨 État du bien")
etat = st.sidebar.selectbox("État", ["Neuf", "Bon état", "À rénover"])

# =========================
# CONVERSION DES INPUTS
# =========================
# Localisation
loc_periurbain, loc_rural, loc_urbain = 0, 0, 0
if localisation == "Urbain":
    loc_urbain = 1
elif localisation == "Périurbain":
    loc_periurbain = 1
else:
    loc_rural = 1

# Accès route
acces_route_val = 1 if acces_route == "Oui" else 0
eau_electricite_val = 1 if eau_electricite == "Oui" else 0
parking_val = 1 if parking == "Oui" else 0

# Type de sol
sol_brut, sol_carrelage, sol_ciment = 0, 0, 0
if type_sol == "Carrelage":
    sol_carrelage = 1
elif type_sol == "Ciment":
    sol_ciment = 1
else:
    sol_brut = 1

# Connexion
conn_aucune, conn_fibre, conn_starlink = 0, 0, 0
if connexion == "Fibre":
    conn_fibre = 1
elif connexion == "Starlink":
    conn_starlink = 1
else:
    conn_aucune = 1

# État
etat_reno, etat_bon, etat_neuf = 0, 0, 0
if etat == "Neuf":
    etat_neuf = 1
elif etat == "Bon état":
    etat_bon = 1
else:
    etat_reno = 1

# =========================
# INPUT VECTOR
# =========================
input_data = np.array([[
    superficie_m2,
    nb_chambres,
    nb_etages,
    acces_route_val,
    eau_electricite_val,
    parking_val,
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
# BOUTON PRÉDICTION
# =========================
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_btn = st.button("🔮 PRÉDIRE LE PRIX", type="primary", use_container_width=True)

# =========================
# PRÉDICTION ET AFFICHAGE
# =========================
if predict_btn and models_loaded:
    
    predictions = {}
    for name, model in models.items():
        pred = model.predict(input_data)[0]
        predictions[name] = pred
    
    # Calcul des stats
    avg_price = np.mean(list(predictions.values()))
    std_price = np.std(list(predictions.values()))
    min_price = min(predictions.values())
    max_price = max(predictions.values())
    cv = (std_price / avg_price) * 100  # Coefficient de variation
    
    # =========================
    # AFFICHAGE PRIX PRINCIPAL
    # =========================
    st.markdown("---")
    
    col_prix, col_stats = st.columns([1, 1])
    
    with col_prix:
        st.markdown(f"""
        <div class="price-card">
            <h2>🎯 Prix Recommandé</h2>
            <h1 style="font-size: 3rem;">{avg_price:.2f} M Ariary</h1>
            <p>(Moyenne des 3 modèles)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_stats:
        st.subheader("📊 Statistiques des prédictions")
        st.metric("Moyenne", f"{avg_price:.2f} M Ariary")
        st.metric("Écart-type", f"{std_price:.2f} M Ariary", delta=f"±{std_price:.2f}")
        st.metric("Intervalle", f"[{min_price:.2f} - {max_price:.2f}] M Ariary")
        st.metric("Coefficient de variation", f"{cv:.1f}%", 
                  delta="Très fiable" if cv < 5 else "Peu fiable",
                  delta_color="normal" if cv < 5 else "inverse")
    
    # =========================
    # GRAPHIQUE 1: BAR CHART (Plotly)
    # =========================
    st.markdown("---")
    st.subheader("📊 Comparaison des modèles")
    
    df_models = pd.DataFrame({
        "Modèle": list(predictions.keys()),
        "Prix (Millions Ar)": list(predictions.values())
    })
    
    fig1 = px.bar(df_models, x="Modèle", y="Prix (Millions Ar)", 
                  text="Prix (Millions Ar)", color="Modèle",
                  color_discrete_sequence=px.colors.qualitative.Set2,
                  title="Prédiction par modèle")
    fig1.update_traces(texttemplate='%{text:.2f} M Ar', textposition='outside')
    fig1.update_layout(showlegend=False, height=500)
    st.plotly_chart(fig1, use_container_width=True)
    
    # =========================
    # GRAPHIQUE 2: GAUGE (jauge) - Prix vs Marché
    # =========================
    st.subheader("📈 Position sur le marché")
    
    # Simulation de prix de marché selon superficie
    prix_marche_bas = superficie_m2 * 0.5
    prix_marche_haut = superficie_m2 * 1.2
    
    fig2 = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = avg_price,
        title = {"text": "Prix estimé vs Marché"},
        delta = {"reference": superficie_m2 * 0.85, "increasing": {"color": "red"}},
        gauge = {
            "axis": {"range": [None, superficie_m2 * 1.5]},
            "bar": {"color": "#1e3c72"},
            "steps": [
                {"range": [0, prix_marche_bas], "color": "#2ecc71", "name": "Bon marché"},
                {"range": [prix_marche_bas, prix_marche_haut], "color": "#f39c12", "name": "Prix moyen"},
                {"range": [prix_marche_haut, superficie_m2 * 1.5], "color": "#e74c3c", "name": "Cher"}
            ],
            "threshold": {
                "line": {"color": "red", "width": 4},
                "thickness": 0.75,
                "value": prix_marche_haut
            }
        }
    ))
    fig2.update_layout(height=400)
    st.plotly_chart(fig2, use_container_width=True)
    
    # =========================
    # GRAPHIQUE 3: RADAR CHART (comparaison des features)
    # =========================
    st.subheader("⭐ Score des caractéristiques")
    
    # Calcul des scores simulés
    features_scores = {
        "Superficie": min(100, superficie_m2 / 5),
        "Chambres": (nb_chambres / 8) * 100,
        "Localisation": 80 if localisation == "Urbain" else 60 if localisation == "Périurbain" else 40,
        "Accès/Commodités": (acces_route_val + eau_electricite_val + parking_val) * 33,
        "État": 90 if etat == "Neuf" else 70 if etat == "Bon état" else 40,
        "Connexion": 90 if connexion == "Fibre" else 70 if connexion == "Starlink" else 30
    }
    
    df_radar = pd.DataFrame({
        "Caractéristique": list(features_scores.keys()),
        "Score": list(features_scores.values())
    })
    
    fig3 = px.line_polar(df_radar, r="Score", theta="Caractéristique", 
                          line_close=True, markers=True,
                          color_discrete_sequence=["#1e3c72"],
                          title="Qualité des caractéristiques")
    fig3.update_traces(fill='toself', fillcolor='rgba(30,60,114,0.3)')
    fig3.update_layout(height=500)
    st.plotly_chart(fig3, use_container_width=True)
    
    # =========================
    # RECOMMANDATION ET INSIGHTS
    # =========================
    st.markdown("---")
    st.subheader("💡 Recommandation & Analyse")
    
    col_rec1, col_rec2 = st.columns(2)
    
    with col_rec1:
        # Recommandation sur le prix
        if cv < 5:
            st.markdown("""
            <div class="recommendation-good">
                <h3>✅ Prix Très Fiable</h3>
                <p>Les 3 modèles sont en forte concordance (CV < 5%).<br>
                Le prix recommandé est <strong>fiable et précis</strong>.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="recommendation-warning">
                <h3>⚠️ Prix à Confirmer</h3>
                <p>Les modèles montrent des écarts importants.<br>
                Une expertise complémentaire est recommandée.</p>
            </div>
            """, unsafe_allow_html=True)
    
    
    # =========================
    # CONSEILS PERSONNALISÉS
    # =========================
    st.markdown("---")
    st.subheader("🏡 Conseils personnalisés pour maximiser la valeur")
    
    conseils = []
    
    if parking_val == 0:
        conseils.append("🚗 **Ajouter un parking** peut augmenter la valeur de +5 à +15%")
    if connexion == "Aucune":
        conseils.append("🔌 **Installer la fibre ou Starlink** → plus-value significative")
    if etat == "À rénover":
        conseils.append("🔨 **Rénover le bien** peut augmenter le prix de +20 à +40%")
    if type_sol == "Brut":
        conseils.append("🖼️ **Changer le revêtement de sol** (carrelage/ciment) améliore la valeur")
    if acces_route_val == 0:
        conseils.append("🛣️ **Améliorer l'accès routier** est un levier de valorisation")
    
    if conseils:
        for c in conseils:
            st.markdown(f"- {c}")
    else:
        st.markdown("✅ **Félicitations!** Votre maison a d'excellentes caractéristiques. Le prix estimé est cohérent avec le marché.")
    
    # =========================
    #         FOOTER
    # =========================
    st.markdown("---")
    st.caption("  Projet Data Science @2026 - House Price Prediction | Modèles entraînés sur données réelles à Madagascar")

else:
    if not models_loaded:
        st.error("❌ Impossible de charger les modèles. Vérifiez les fichiers .pkl")
    else:
        st.info("👈 Remplissez les caractéristiques dans la barre latérale et cliquez sur **Prédire le prix**")