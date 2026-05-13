"""
Sonar Signal Classification - Mines vs Rocks
Auteur: Fanantenana Manaosoa
Description: Classification des signaux sonar pour détecter mines (M) ou roches (R)
"""

import streamlit as st
import numpy as np
import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

# =========================
# CONFIG PAGE
# =========================
st.set_page_config(
    page_title="Sonar Classifier - Mines vs Rocks",
    page_icon="🔊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# CUSTOM CSS - DARK THEME
# =========================
st.markdown("""
<style>
    /* Style global */
    .stApp {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
    }
    
    /* Header principal */
    .main-header {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        border: 1px solid rgba(255,255,255,0.1);
    }
    .main-header h1 {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    .main-header p {
        font-size: 1rem;
        opacity: 0.9;
    }
    
    /* Carte résultat MINE */
    .result-card-mine {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
        padding: 2rem;
        border-radius: 25px;
        text-align: center;
        color: white;
        box-shadow: 0 15px 35px rgba(0,0,0,0.3);
        animation: fadeIn 0.5s ease-in;
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    /* Carte résultat ROCK */
    .result-card-rock {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 2rem;
        border-radius: 25px;
        text-align: center;
        color: white;
        box-shadow: 0 15px 35px rgba(0,0,0,0.3);
        animation: fadeIn 0.5s ease-in;
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    /* Animation */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Boîte d'information - Dark theme */
    .info-box {
        background: linear-gradient(135deg, #1e3a4d 0%, #0f2c3d 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #38ef7d;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        color: #ffffff;
    }
    .info-box h4 {
        color: #38ef7d;
        margin-bottom: 0.8rem;
    }
    .info-box p, .info-box li {
        color: #e0e0e0;
    }
    .info-box ul {
        color: #e0e0e0;
    }
    
    /* Carte métrique - Dark theme */
    .metric-card {
        background: linear-gradient(135deg, #1a3a4f 0%, #0f2c3d 100%);
        padding: 1.2rem;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        text-align: center;
        transition: transform 0.3s, box-shadow 0.3s;
        color: white;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(0,0,0,0.3);
    }
    .metric-card h3 {
        color: #c3cfe2;
        margin-bottom: 0.5rem;
        font-size: 1rem;
    }
    .metric-card h2 {
        color: #38ef7d;
        font-size: 2rem;
        margin: 0;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 1.5rem;
        margin-top: 2rem;
        color: #c3cfe2;
        border-top: 1px solid #2c5364;
        background: linear-gradient(135deg, #1a3a4f 0%, #0f2027 100%);
        border-radius: 15px;
    }
    
    /* Section title */
    .section-title {
        background: linear-gradient(135deg, #2c5364, #203a43);
        padding: 0.8rem 1.5rem;
        border-radius: 40px;
        color: white;
        display: inline-block;
        margin-bottom: 1.5rem;
        font-size: 1.2rem;
        font-weight: bold;
    }
    
    /* Sidebar style */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f2027 0%, #1a3a4f 100%);
        border-right: 1px solid #2c5364;
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: white;
    }
    
    /* Style pour les sliders */
    .stSlider label {
        color: white !important;
    }
    
    /* Style pour les radio buttons */
    .stRadio label {
        color: white !important;
    }
    
    /* Style pour les info/warning/success */
    .stAlert {
        background-color: #1a3a4f !important;
        color: white !important;
    }
    
    /* Style pour les boutons */
    .stButton button {
        background: linear-gradient(135deg, #ff416c, #ff4b2b);
        color: white;
        border: none;
        border-radius: 30px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: transform 0.2s;
    }
    .stButton button:hover {
        transform: scale(1.02);
        background: linear-gradient(135deg, #ff4b2b, #ff416c);
    }
    
    /* Style pour les métriques Streamlit */
    [data-testid="stMetricValue"] {
        color: #38ef7d !important;
    }
    [data-testid="stMetricLabel"] {
        color: white !important;
    }
    
    /* Style pour les expanders */
    .streamlit-expanderHeader {
        color: white !important;
        background-color: #1a3a4f !important;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.markdown("""
<div class="main-header">
    <h1>🔊 Sonar Signal Classification</h1>
    <p>🤖 Intelligence Artificielle pour la Détection de Mines sous-marines</p>
    <p style="font-size: 0.85rem;">⚡ 60 bandes de fréquence | 🌲 Random Forest | ✅ Accuracy: ~88%</p>
</div>
""", unsafe_allow_html=True)

# =========================
# SIDEBAR - INFORMATIONS
# =========================
with st.sidebar:
    st.markdown("### 🧠 À propos")
    st.markdown("""
    <div style="background: #1a3a4f; padding: 1rem; border-radius: 15px; color: white; border-left: 3px solid #38ef7d;">
        <p>Ce modèle classifie les signaux sonar pour distinguer :</p>
        <ul>
            <li>🧨 <strong>Mine (Métal)</strong> - Cylindre métallique</li>
            <li>🪨 <strong>Rock (Roche)</strong> - Roche naturelle</li>
        </ul>
        <hr style="border-color: #2c5364;">
        <p><strong>📊 Données :</strong> 208 échantillons<br>
        <strong>🎯 Modèle :</strong> Random Forest<br>
        <strong>✅ Accuracy :</strong> ~88% sur test set</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 📚 Référence scientifique")
    st.markdown("""
    <div style="background: #1a3a4f; padding: 1rem; border-radius: 15px; color: #c3cfe2;">
        <small>
        Gorman, R. P., & Sejnowski, T. J. (1988).<br>
        <em>Analysis of Hidden Units in a Layered Network Trained to Classify Sonar Targets</em><br>
        Neural Networks, Vol. 1, pp. 75-89.
        </small>
    </div>
    """, unsafe_allow_html=True)
    

# =========================
# LOAD MODELS
# =========================
@st.cache_resource
def load_model():
    try:
        # Manandrana mamaky modèle RandomForest
        model_path = "notebooks/model/RandomForest.pkl"
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            return model, "Random Forest"
        else:
            st.warning(f"⚠️ Fichier non trouvé: {model_path}")
            return None, None
    except Exception as e:
        st.error(f"❌ Erreur chargement modèle: {e}")
        return None, None

model, model_name = load_model()

# =========================
# INPUT METHODS
# =========================
st.markdown('<h2 class="section-title">📊 Saisie des 60 fréquences sonar</h2>', unsafe_allow_html=True)

input_method = st.radio(
    "📌 Choisissez votre méthode de saisie :",
    ["🎛️ Sliders individuels", "📝 Saisie rapide (JSON)", "🎲 Aléatoire + Visualisation"],
    horizontal=True,
    help="Sélectionnez la méthode la plus adaptée à votre besoin"
)

frequency_values = []

if input_method == "🎛️ Sliders individuels":
    st.info("🔧 Ajustez chaque fréquence avec les curseurs ci-dessous (valeurs entre 0 et 1)")
    
    # Affichage des sliders en 5 colonnes
    cols = st.columns(5)
    for i in range(60):
        col_idx = i % 5
        with cols[col_idx]:
            val = st.slider(
                f"{i+1}", 
                min_value=0.0, 
                max_value=1.0, 
                value=0.5, 
                step=0.01,
                key=f"freq_{i}",
                label_visibility="collapsed"
            )
            frequency_values.append(val)

elif input_method == "📝 Saisie rapide (JSON)":
    st.info("💡 Entrez les 60 valeurs sous forme de liste JSON (copiez-collez)")
    
    json_input = st.text_area(
        "Valeurs JSON:",
        value="[0.5] * 60",
        height=100,
        help="Exemple: [0.1, 0.2, 0.3, ...] ou [0.5]*60"
    )
    
    col_val1, col_val2 = st.columns(2)
    with col_val1:
        if st.button("✅ Valider les valeurs", use_container_width=True):
            try:
                frequency_values = eval(json_input)
                if len(frequency_values) != 60:
                    st.error(f"❌ Veuillez entrer exactement 60 valeurs (vous avez {len(frequency_values)})")
                    frequency_values = [0.5] * 60
                else:
                    st.success("✅ 60 valeurs chargées avec succès!")
            except:
                st.error("❌ Format JSON invalide. Utilisez le format: [0.1, 0.2, 0.3, ...]")
                frequency_values = [0.5] * 60
    with col_val2:
        if st.button("🔄 Réinitialiser", use_container_width=True):
            frequency_values = [0.5] * 60
            st.success("✅ Valeurs réinitialisées!")
    
    if not frequency_values:
        frequency_values = [0.5] * 60

else:  # Génération aléatoire
    st.info("🎲 Générez un signal aléatoire pour tester le modèle")
    
    col_gen1, col_gen2 = st.columns(2)
    with col_gen1:
        if st.button("🎲 Générer un signal aléatoire", use_container_width=True, type="primary"):
            frequency_values = list(np.random.uniform(0, 1, 60))
            st.success("✅ Signal aléatoire généré! Cliquez sur 'Classifier' ci-dessous.")
    
    with col_gen2:
        if st.button("📊 Visualiser le signal", use_container_width=True):
            frequency_values = list(np.random.uniform(0, 1, 60))
            # Afficher le graphique immédiatement
            df_signal = pd.DataFrame({
                "Bande": [f"{i+1}" for i in range(60)],
                "Intensité": frequency_values
            })
            fig = px.line(df_signal, x="Bande", y="Intensité", 
                         title="📈 Aperçu du signal aléatoire",
                         color_discrete_sequence=["#38ef7d"])
            fig.update_layout(height=400, xaxis_title="Bande de fréquence", yaxis_title="Intensité",
                            plot_bgcolor="#1a3a4f", paper_bgcolor="#1a3a4f",
                            font_color="white")
            st.plotly_chart(fig, use_container_width=True)
    
    if not frequency_values:
        frequency_values = [0.5] * 60

# =========================
# VISUALISATION DU SIGNAL (aperçu avant classification)
# =========================
if frequency_values and frequency_values != [0.5]*60:
    st.markdown("---")
    st.markdown("#### 📈 Aperçu du signal saisi")
    
    df_preview = pd.DataFrame({
        "Bande": [f"{i+1}" for i in range(60)],
        "Intensité": frequency_values
    })
    
    fig_preview = px.area(
        df_preview, 
        x="Bande", 
        y="Intensité",
        title="Visualisation du signal sonar",
        color_discrete_sequence=["#38ef7d"],
        line_shape="spline"
    )
    fig_preview.update_layout(height=350, xaxis_title="Bande de fréquence", yaxis_title="Intensité",
                              plot_bgcolor="#1a3a4f", paper_bgcolor="#1a3a4f",
                              font_color="white")
    st.plotly_chart(fig_preview, use_container_width=True)

# =========================
# BOUTON DE PRÉDICTION
# =========================
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_btn = st.button(
        "🔍 CLASSIFIER LE SIGNAL SONAR", 
        type="primary", 
        use_container_width=True
    )

# =========================
# PRÉDICTION
# =========================
if predict_btn and model is not None:
    
    # Préparation des données
    input_array = np.array(frequency_values).reshape(1, -1)
    
    # Prédiction
    try:
        prediction = model.predict(input_array)[0]
        if hasattr(model, 'predict_proba'):
            probability = model.predict_proba(input_array)[0]
        else:
            probability = [0.5, 0.5]
    except Exception as e:
        st.error(f"Erreur lors de la prédiction: {e}")
        prediction = None
        probability = [0.5, 0.5]
    
    if prediction is not None:
        # =========================
        # AFFICHAGE RÉSULTAT
        # =========================
        st.markdown("---")
        st.markdown('<h2 class="section-title">🎯 Résultat de la classification</h2>', unsafe_allow_html=True)
        
        col_result, col_metrics = st.columns([1, 1])
        
        with col_result:
            if prediction == 1:
                st.markdown("""
                <div class="result-card-mine">
                    <h1 style="font-size: 3.5rem;">🧨 MINE</h1>
                    <p style="font-size: 1.3rem;">Objet métallique détecté</p>
                    <p style="font-size: 0.9rem;">⚠️ Alerte: Présence potentielle de mine sous-marine</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="result-card-rock">
                    <h1 style="font-size: 3.5rem;">🪨 ROCK</h1>
                    <p style="font-size: 1.3rem;">Roche naturelle détectée</p>
                    <p style="font-size: 0.9rem;">✅ Zone sécurisée - Aucune menace détectée</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col_metrics:
            prob_mine = probability[1] if len(probability) > 1 else 0.5
            prob_rock = probability[0] if len(probability) > 0 else 0.5
            
            st.markdown("### 📊 Niveau de confiance")
            
            # Barre pour Mine
            st.markdown(f"""
            <div style="margin-bottom: 1.5rem;">
                <p><strong>🧨 Mine (Métal)</strong></p>
                <div style="background: #2c3e50; border-radius: 20px; height: 35px; overflow: hidden;">
                    <div style="background: linear-gradient(90deg, #ff416c, #ff4b2b); width: {prob_mine*100}%; height: 35px; border-radius: 20px; text-align: center; color: white; line-height: 35px; font-weight: bold;">
                        {prob_mine:.1%}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Barre pour Rock
            st.markdown(f"""
            <div style="margin-bottom: 1.5rem;">
                <p><strong>🪨 Rock (Roche)</strong></p>
                <div style="background: #2c3e50; border-radius: 20px; height: 35px; overflow: hidden;">
                    <div style="background: linear-gradient(90deg, #11998e, #38ef7d); width: {prob_rock*100}%; height: 35px; border-radius: 20px; text-align: center; color: white; line-height: 35px; font-weight: bold;">
                        {prob_rock:.1%}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Niveau de confiance
            confidence = max(prob_mine, prob_rock)
            if confidence > 0.8:
                st.success(f"✅ Confiance ÉLEVÉE : {confidence:.1%}")
            elif confidence > 0.6:
                st.warning(f"⚠️ Confiance MOYENNE : {confidence:.1%}")
            else:
                st.error(f"❌ Confiance FAIBLE : {confidence:.1%} - Vérifiez les valeurs saisies")
    
    # =========================
    # STATISTIQUES DU SIGNAL
    # =========================
    st.markdown("---")
    st.markdown('<h2 class="section-title">📊 Statistiques du signal</h2>', unsafe_allow_html=True)
    
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    
    with col_stat1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📏 Moyenne</h3>
            <h2>{np.mean(frequency_values):.3f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col_stat2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📊 Écart-type</h3>
            <h2>{np.std(frequency_values):.3f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col_stat3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📈 Maximum</h3>
            <h2>{np.max(frequency_values):.3f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col_stat4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📉 Minimum</h3>
            <h2>{np.min(frequency_values):.3f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # =========================
    # INTERPRÉTATION
    # =========================
    st.markdown("---")
    st.markdown('<h2 class="section-title">💡 Interprétation & Recommandation</h2>', unsafe_allow_html=True)
    
    col_interpret, col_advice = st.columns(2)
    
    with col_interpret:
        st.markdown("""
        <div class="info-box">
            <h4>🔍 Analyse du signal</h4>
        """, unsafe_allow_html=True)
        
        if prediction == 1:
            st.markdown("""
            - **Motif détecté** : Réflexion forte et cohérente
            - **Caractéristique** : Typique d'un objet métallique
            - **Application** : ⚠️ Zone à risque - Procédure de sécurité requise
            """)
        else:
            st.markdown("""
            - **Motif détecté** : Réflexion diffuse et irrégulière
            - **Caractéristique** : Typique d'une roche naturelle
            - **Application** : ✅ Zone sécurisée - Navigation normale
            """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col_advice:
        st.markdown("""
        <div class="info-box">
            <h4>🎯 Recommandation</h4>
        """, unsafe_allow_html=True)
        
        if prediction == 1:
            st.markdown("""
            - ⚠️ **Action immédiate** : Signaler la présence suspecte
            - 🔍 **Vérification** : Inspection visuelle recommandée
            - 🛡️ **Sécurité** : Suivre les protocoles standards
            """)
        else:
            st.markdown("""
            - ✅ **Action** : Procéder normalement
            - 📊 **Confiance** : Modèle fiable
            - 🔄 **Surveillance** : Continuer la routine standard
            """)
        st.markdown("</div>", unsafe_allow_html=True)

elif predict_btn and model is None:
    st.error("❌ Modèle non chargé. Vérifiez que le fichier 'RandomForest.pkl' existe dans 'notebooks/model/'")

# =========================
# FOOTER
# =========================
st.markdown("""
<div class="footer">
    <p>🚀 Projet Data Science - Classification Sonar | API développée par <strong>Manaosoa Randria</strong></p>
    <p style="font-size: 0.7rem;">© 2026 - Tous droits réservés | Modèle Random Forest | Accuracy: ~88%</p>
</div>
""", unsafe_allow_html=True)