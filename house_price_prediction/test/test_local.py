import joblib
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime

print("📁 Vérification des fichiers...")
print(f"Chemin actuel: {os.getcwd()}")

# ========== 1. CHARGEMENT CORRECT DES MODÈLES ==========
# Utiliser des chemins relatifs au lieu de chemins absolus
models_path = "models/models"  # Chemin relatif

# Vérifier si le dossier existe
if not os.path.exists(models_path):
    print(f"❌ Dossier non trouvé: {models_path}")
    print("📂 Création du dossier...")
    os.makedirs(models_path, exist_ok=True)

# Charger les modèles avec gestion d'erreurs
def load_model_safely(filepath):
    """Charger un modèle avec gestion d'erreur"""
    try:
        if os.path.exists(filepath):
            return joblib.load(filepath)
        else:
            print(f"⚠️ Fichier non trouvé: {filepath}")
            return None
    except Exception as e:
        print(f"❌ Erreur chargement {filepath}: {e}")
        return None

# Charger chaque modèle
model_xgb = load_model_safely(f"{models_path}/model_xgboost.pkl")
model_rf = load_model_safely(f"{models_path}/model_randomforest.pkl")
model_gbr = load_model_safely(f"{models_path}/model_gradientboosting.pkl")
stack_model_loaded = load_model_safely(f"{models_path}/stacking.pkl")
RFCV_model_loaded = load_model_safely(f"{models_path}/randomforestCrossValidation.pkl")

# Vérifier quels modèles sont chargés
models_loaded = {
    "XGBoost": model_xgb,
    "RandomForest": model_rf,
    "GradientBoosting": model_gbr,
    "Stacking": stack_model_loaded,
    "RF_CV": RFCV_model_loaded
}

available_models = {name: model for name, model in models_loaded.items() if model is not None}

print(f"\n✅ Modèles chargés: {len(available_models)}/{len(models_loaded)}")
for name in available_models:
    print(f"   ✓ {name}")

# ========== 2. FONCTION DE VALIDATION ==========
def is_valid_house(features):
    """Vérifier si la maison a des données valides"""
    errors = []
    
    # Règle 1: Superficie valide (10-1000 m²)
    if features["superficie_m2"] < 10 or features["superficie_m2"] > 1000:
        errors.append(f"Superficie invalide: {features['superficie_m2']} m² (doit être 10-1000)")
    
    # Règle 2: Nombre de chambres valide (1-20)
    if features["nb_chambres"] < 1 or features["nb_chambres"] > 20:
        errors.append(f"Nombre de chambres invalide: {features['nb_chambres']} (doit être 1-20)")
    
    # Règle 3: Année construction valide (1800-2026)
    if features["annee_construction"] < 1800 or features["annee_construction"] > 2026:
        errors.append(f"Année invalide: {features['annee_construction']} (doit être 1800-2026)")
    
    # Règle 4: Une seule localisation
    loc_sum = features["localisation_periurbain"] + features["localisation_rural"] + features["localisation_urbain"]
    if loc_sum != 1:
        errors.append("Choisir EXACTEMENT une localisation (1 = oui, 0 = non)")
    
    if errors:
        return False, errors
    return True, "OK"

# ========== 3. TEST AVEC EXEMPLE ==========
house_example = {
    "superficie_m2": 200,
    "nb_chambres": 8,  # Changé de 8 à 3 pour être plus réaliste
    "nb_etages": 1,
    "acces_route": 1,
    "eau_electricite": 1,  # Changé à 1
    "parking": 1,  # Changé à 1
    "annee_construction": 2025,
    "localisation_periurbain": 0,
    "localisation_rural": 1,
    "localisation_urbain": 0,
    "type_connexion_aucune": 0,
    "type_connexion_fibre": 0,
    "type_connexion_starlink": 1,
    "type_sol_brut": 0,
    "type_sol_carrelage": 0,  # Changé pour plus réaliste
    "type_sol_ciment": 1,
    "etat_maison_a_renover": 0,
    "etat_maison_bon": 1,  # Changé à 1
    "etat_maison_neuf": 0
}

# ========== 4. PRÉDICTION AVEC VISUALISATION ==========
valid, message = is_valid_house(house_example)

if not valid:
    print(f"\n❌ DONNÉES INVALIDES:")
    for err in message:
        print(f"   • {err}")
    print("\n💾 Prix estimé: 0 millions Ar")
    print("\n💡 Conseil: Veuillez corriger les données!")
else:
    # Créer DataFrame
    df = pd.DataFrame([house_example])
    
    # Faire prédictions avec modèles disponibles
    predictions = {}
    for name, model in available_models.items():
        try:
            pred = model.predict(df)[0]
            predictions[name] = pred
        except Exception as e:
            print(f"⚠️ Erreur avec {name}: {e}")
    
    if not predictions:
        print("❌ Aucun modèle n'a pu faire de prédiction!")
        exit()
    
    # ========== 5. AFFICHAGE DES RÉSULTATS ==========
    print("\n" + "="*60)
    print("🏠 ESTIMATION DU PRIX IMMOBILIER")
    print("="*60)
    
    print("\n📋 CARACTÉRISTIQUES DE LA MAISON:")
    print(f"   • Superficie: {house_example['superficie_m2']} m²")
    print(f"   • Chambres: {house_example['nb_chambres']}")
    print(f"   • Étages: {house_example['nb_etages']}")
    print(f"   • Construction: {house_example['annee_construction']}")
    print(f"   • Localisation: ", end="")
    if house_example['localisation_urbain']: print("Urbain")
    elif house_example['localisation_periurbain']: print("Péri-urbain")
    else: print("Rural")
    
    print("\n📊 PRÉDICTIONS PAR MODÈLE:")
    for name, pred in predictions.items():
        print(f"   • {name:20} → {pred:>10.2f} millions Ar")
    
    # Statistiques
    pred_values = list(predictions.values())
    mean_pred = np.mean(pred_values)
    std_pred = np.std(pred_values)
    min_pred = min(pred_values)
    max_pred = max(pred_values)
    
    print("\n" + "-"*60)
    print(f"📈 STATISTIQUES:")
    print(f"   • Moyenne:     {mean_pred:.2f} millions Ar")
    print(f"   • Écart-type:  {std_pred:.2f} millions Ar")
    print(f"   • Intervalle:  [{min_pred:.2f} - {max_pred:.2f}] millions Ar")
    
    # Modèle le plus confiant (écart-type le plus petit parmi les modèles)
    # Pour simplifier, on prend la moyenne
    print(f"\n🎯 PRIX RECOMMANDÉ: {mean_pred:.2f} millions Ar")
    print(f"   (Moyenne des {len(predictions)} modèles disponibles)")
    
    
    # ========== 7. ANALYSE SUPPLÉMENTAIRE ==========
    print("\n" + "="*60)
    print("💡 INTERPRÉTATION ET CONSEILS")
    print("="*60)
    
    # Comparer avec la moyenne du marché (à ajuster selon vos données)
    prix_moyen_region = 150  # Exemple: 150 millions Ar
    if mean_pred > prix_moyen_region * 1.2:
        print("⚠️  Prix estimé ÉLEVÉ par rapport à la moyenne régionale")
        print("   → Vérifiez les caractéristiques premium (localisation, finitions)")
    elif mean_pred < prix_moyen_region * 0.8:
        print("⚠️  Prix estimé BAS par rapport à la moyenne régionale")
        print("   → Potentielle bonne affaire ou vérifier l'état du bien")
    else:
        print("✅ Prix estimé dans la moyenne du marché")
    
    # Analyse de la dispersion des modèles
    cv = (std_pred / mean_pred) * 100  # Coefficient de variation
    if cv > 20:
        print(f"\n⚠️  Forte dispersion entre modèles (CV={cv:.1f}%)")
        print("   → Les modèles ne sont pas d'accord sur le prix")
        print("   → Conseil: Considérez l'intervalle complet [{:.1f} - {:.1f}]".format(min_pred, max_pred))
    elif cv > 10:
        print(f"\n📊 Dispersion modérée entre modèles (CV={cv:.1f}%)")
        print(f"   → Prix probable: {mean_pred:.1f} ± {std_pred:.1f} millions Ar")
    else:
        print(f"\n✅ Excellente concordance entre modèles (CV={cv:.1f}%)")
        print(f"   → Prix très fiable: {mean_pred:.1f} millions Ar")
    
    # Conseils personnalisés selon les caractéristiques
    print("\n🏡 CONSEILS PERSONNALISÉS:")
    if house_example["etat_maison_a_renover"]:
        print("   • Maison à rénover → Prévoir 20-30% du prix en travaux")
    if not house_example["eau_electricite"]:
        print("   • Sans eau/électricité → Impact négatif sur le prix (-15 à -25%)")
    if house_example["type_connexion_fibre"]:
        print("   • Connexion fibre → Plus-value de +5 à +10%")
    if house_example["parking"]:
        print("   • Parking disponible → Plus-value de +5 à +15%")

print("\n✅ Analyse terminée!")





# import joblib
# import pandas as pd
# import numpy as np
# import os
# from datetime import datetime

# print("📁 Vérification des fichiers...")
# print(f"Chemin actuel: {os.getcwd()}")

# # ========== 1. CHARGEMENT CORRECT DES MODÈLES ==========
# models_path = "models/models"

# if not os.path.exists(models_path):
#     print(f"❌ Dossier non trouvé: {models_path}")
#     print("📂 Création du dossier...")
#     os.makedirs(models_path, exist_ok=True)

# def load_model_safely(filepath):
#     try:
#         if os.path.exists(filepath):
#             return joblib.load(filepath)
#         else:
#             print(f"⚠️ Fichier non trouvé: {filepath}")
#             return None
#     except Exception as e:
#         print(f"❌ Erreur chargement {filepath}: {e}")
#         return None

# model_xgb = load_model_safely(f"{models_path}/model_xgboost.pkl")
# model_rf = load_model_safely(f"{models_path}/model_randomforest.pkl")
# model_gbr = load_model_safely(f"{models_path}/model_gradientboosting.pkl")
# stack_model_loaded = load_model_safely(f"{models_path}/stacking.pkl")
# RFCV_model_loaded = load_model_safely(f"{models_path}/randomforestCrossValidation.pkl")

# models_loaded = {
#     "XGBoost": model_xgb,
#     "RandomForest": model_rf,
#     "GradientBoosting": model_gbr,
#     "Stacking": stack_model_loaded,
#     "RF_CV": RFCV_model_loaded
# }

# available_models = {name: model for name, model in models_loaded.items() if model is not None}

# print(f"\n✅ Modèles chargés: {len(available_models)}/{len(models_loaded)}")

# # ========== 2. FONCTION DE VALIDATION ==========
# def is_valid_house(features):
#     errors = []
    
#     if features["superficie_m2"] < 10 or features["superficie_m2"] > 1000:
#         errors.append(f"Superficie invalide: {features['superficie_m2']} m² (doit être 10-1000)")
    
#     if features["nb_chambres"] < 1 or features["nb_chambres"] > 20:
#         errors.append(f"Nombre de chambres invalide: {features['nb_chambres']} (doit être 1-20)")
    
#     if features["annee_construction"] < 1800 or features["annee_construction"] > 2026:
#         errors.append(f"Année invalide: {features['annee_construction']} (doit être 1800-2026)")
    
#     loc_sum = features["localisation_periurbain"] + features["localisation_rural"] + features["localisation_urbain"]
#     if loc_sum != 1:
#         errors.append("Choisir EXACTEMENT une localisation")
    
#     if errors:
#         return False, errors
#     return True, "OK"

# # ========== 3. FONCTION DE CONSEILS CONVAINQUANTS ==========
# def generate_persuasive_advice(features, predicted_price, mean_pred, min_pred, max_pred):
#     """Génère des conseils personnalisés pour convaincre l'acheteur"""
    
#     advice = []
#     selling_points = []
#     negotiation_points = []
#     roi_analysis = []
    
#     # === POINTS FORTS À METTRE EN AVANT ===
    
#     # 1. Analyse de la superficie
#     if features["superficie_m2"] >= 150:
#         selling_points.append(f"✓ GRAND ESPACE: {features['superficie_m2']} m² - Idéal pour famille nombreuse ou investissement locatif")
#         roi_analysis.append(f"  → Potentiel locatif: Jusqu'à {int(features['superficie_m2'] * 0.015)} millions Ar/mois en location")
#     elif features["superficie_m2"] >= 100:
#         selling_points.append(f"✓ SUPERFICIE CONFORTABLE: {features['superficie_m2']} m² - Parfait pour une famille")
#         roi_analysis.append(f"  → Rapport qualité/prix excellent pour cette surface")
#     else:
#         negotiation_points.append(f"✓ Petite surface ({features['superficie_m2']} m²) = Entretien facile et charges réduites")
    
#     # 2. Nombre de chambres
#     if features["nb_chambres"] >= 4:
#         selling_points.append(f"✓ NOMBREUSES CHAMBRES: {features['nb_chambres']} chambres - Idéal pour famille nombreuse ou location à des étudiants")
#         roi_analysis.append(f"  → Revenu locatif potentiel: {features['nb_chambres'] * 0.4:.1f} - {features['nb_chambres'] * 0.6:.1f} millions Ar/mois")
#     elif features["nb_chambres"] == 3:
#         selling_points.append(f"✓ CONFIGURATION IDÉALE: {features['nb_chambres']} chambres - Standard familial parfait")
    
#     # 3. Localisation
#     if features["localisation_urbain"] == 1:
#         selling_points.append("✓ LOCALISATION PRESTIGE: En plein centre-ville - Proximité commerces, écoles et transports")
#         roi_analysis.append("  → Plus-value annuelle estimée: +8 à +12% (zone urbaine dynamique)")
#     elif features["localisation_periurbain"] == 1:
#         selling_points.append("✓ CADRE RECHERCHÉ: Zone péri-urbaine calme - Proche de la ville sans le bruit")
#         roi_analysis.append("  → Potentiel de plus-value: +5 à +8%/an (zone en développement)")
#     else:
#         selling_points.append("✓ CHARME RURAL: Cadre paisible et nature - Idéal pour résidence principale ou gîte rural")
#         roi_analysis.append("  → Opportunité unique: Les prix ruraux augmentent de +3 à +5%/an")
    
#     # 4. État de la maison
#     if features["etat_maison_neuf"] == 1:
#         selling_points.append("✓ CONSTRUCTION NEUVE: Zéro travaux à prévoir - Garantie décennale incluse")
#         roi_analysis.append("  → Économies immédiates: Évite 30-50 millions Ar de travaux")
#     elif features["etat_maison_bon"] == 1:
#         selling_points.append("✓ BON ÉTAT: Entretenue régulièrement - Prête à habiter")
#         roi_analysis.append("  → Aucun investissement immédiat nécessaire")
#     else:
#         negotiation_points.append("💰 OPPORTUNITÉ: Maison à rénover - Prix négociable, potentiel de plus-value important après travaux")
#         roi_analysis.append("  → Après rénovation, valeur estimée: +40% par rapport au prix d'achat")
    
#     # 5. Connexion internet
#     if features["type_connexion_fibre"] == 1:
#         selling_points.append("✓ FIBRE OPTIQUE: Internet ultra-rapide - Idéal pour télétravail et vie connectée")
#         roi_analysis.append("  → Atout majeur: +15% de valeur locative par rapport aux zones sans fibre")
#     elif features["type_connexion_starlink"] == 1:
#         selling_points.append("✓ STARLINK DISPONIBLE: Connexion satellite haute vitesse - Parfait pour télétravail")
    
#     # 6. Équipements
#     if features["parking"] == 1:
#         selling_points.append("✓ PARKING PRIVÉ: Stationnement sécurisé - Rare et très recherché")
#         roi_analysis.append("  → Plus-value estimée: +5 à +10 millions Ar")
    
#     if features["eau_electricite"] == 1:
#         selling_points.append("✓ EAU & ÉLECTRICITÉ: Raccordements officiels - Conformité légale totale")
    
#     if features["acces_route"] == 1:
#         selling_points.append("✓ ACCÈS FACILE: Route bitumée jusqu'à la propriété - Accessible par tous temps")
    
#     # 7. Année de construction
#     age = 2026 - features["annee_construction"]
#     if age <= 5:
#         selling_points.append(f"✓ CONSTRUCTION RÉCENTE ({features['annee_construction']}) - Aux normes actuelles, isolation performante")
#         roi_analysis.append("  → Économies d'énergie: Factures réduites de 30-40%")
#     elif age <= 15:
#         selling_points.append(f"✓ BONNE ANCIENNETÉ ({features['annee_construction']}) - Construction solide, déjà bien décotée")
    
#     # 8. Type de sol
#     if features["type_sol_carrelage"] == 1:
#         selling_points.append("✓ SOL CARRELÉ DE QUALITÉ - Facile d'entretien, durable et esthétique")
    
#     # === CALCUL DU RETOUR SUR INVESTISSEMENT (ROI) ===
    
#     # Estimation du loyer mensuel selon les caractéristiques
#     base_rent = mean_pred * 0.008  # Règle générale: loyer = 0.8% du prix
#     if features["localisation_urbain"] == 1:
#         base_rent *= 1.3
#     elif features["localisation_periurbain"] == 1:
#         base_rent *= 1.1
    
#     if features["type_connexion_fibre"] == 1:
#         base_rent *= 1.15
#     if features["parking"] == 1:
#         base_rent *= 1.1
#     if features["nb_chambres"] >= 4:
#         base_rent *= 1.2
    
#     annual_rent = base_rent * 12
#     gross_roi = (annual_rent / mean_pred) * 100
    
#     # === GÉNÉRATION DU MESSAGE CONVAINCANT ===
    
#     print("\n" + "="*70)
#     print("🏡 RAPPORT D'ANALYSE DÉTAILLÉ - OPPORTUNITÉ IMMOBILIÈRE")
#     print("="*70)
    
#     print("\n📊 PRIX ESTIMÉ PAR NOS MODÈLES:")
#     print(f"   • Prix recommandé: {mean_pred:.2f} millions Ar")
#     print(f"   • Fourchette de prix: [{min_pred:.2f} - {max_pred:.2f}] millions Ar")
#     print(f"   • Basé sur l'analyse de {len(available_models)} modèles experts")
    
#     print("\n" + "="*70)
#     print("✨ POINTS FORTS DE CETTE PROPRIÉTÉ:")
#     print("="*70)
#     for point in selling_points[:5]:  # Top 5 arguments de vente
#         print(point)
    
#     if negotiation_points:
#         print("\n" + "="*70)
#         print("💡 OPPORTUNITÉS DE NÉGOCIATION:")
#         print("="*70)
#         for point in negotiation_points:
#             print(point)
    
#     print("\n" + "="*70)
#     print("💰 ANALYSE FINANCIÈRE - RETOUR SUR INVESTISSEMENT")
#     print("="*70)
    
#     for analysis in roi_analysis[:4]:
#         print(analysis)
    
#     print(f"\n📈 PROJECTIONS FINANCIÈRES:")
#     print(f"   • Estimation loyer mensuel: {base_rent:.2f} millions Ar/mois")
#     print(f"   • Revenu locatif annuel: {annual_rent:.2f} millions Ar/an")
#     print(f"   • Rendement brut estimé: {gross_roi:.1f}%/an")
#     print(f"   • Seuil de rentabilité: {mean_pred / annual_rent:.1f} années (hors vacance locative)")
    
#     # Comparaison avec investissements alternatifs
#     print(f"\n📊 COMPARAISON AVEC AUTRES INVESTISSEMENTS:")
#     print(f"   • Immobilier (ce bien): {gross_roi:.1f}%/an + plus-value potentielle")
#     print(f"   • Compte épargne: 3-5%/an")
#     print(f"   • Actions/Bourse: 8-12%/an (avec risques élevés)")
    
#     if gross_roi > 12:
#         print(f"\n🎯 EXCELLENT RENDEMENT! Ce bien surperforme le marché immobilier!")
#     elif gross_roi > 8:
#         print(f"\n👍 BON RENDEMENT! Investissement solide et sécurisé")
    
#     print("\n" + "="*70)
#     print("🎯 ARGUMENTS DÉCISIFS POUR L'ACHAT:")
#     print("="*70)
    
#     # Arguments personnalisés selon le profil
#     if features["etat_maison_neuf"] == 1:
#         print("✓ MAISON NEUVE: Zéro travaux, économies immédiates de 30-50 millions Ar")
#         print("✓ Conforme aux dernières normes (isolation, électricité, plomberie)")
#         print("✓ Tranquillité d'esprit avec garanties constructeur")
    
#     if features["localisation_urbain"] == 1 and features["parking"] == 1:
#         print("✓ COMBO GAGNANT: Centre-ville + parking privé - Rare et très recherché")
#         print("✓ Revente facile et rapide avec forte plus-value")
    
#     if features["nb_chambres"] >= 4:
#         print(f"✓ POTENTIEL LOCATIF MAXIMUM: {features['nb_chambres']} chambres")
#         print("✓ Idéal pour colocation étudiante ou professionnelle")
#         print(f"✓ Revenus locatifs potentiels: {features['nb_chambres'] * 0.4:.1f} - {features['nb_chambres'] * 0.6:.1f} millions Ar/mois")
    
#     # Message final percutant
#     print("\n" + "="*70)
#     print("💎 CONCLUSION - POURQUOI INVESTIR MAINTENANT?")
#     print("="*70)
    
#     if mean_pred < 100:
#         print("🏠 PRIX ACCESSIBLE - Excellente entrée sur le marché immobilier")
#         print("📈 Potentiel de plus-value important dans les 3-5 ans")
#         print("💰 Investissement sûr avec rendement locatif attractif")
#     elif mean_pred < 200:
#         print("🏡 PRIX COMPÉTITIF - Rapport qualité-prix exceptionnel")
#         print("📈 Zone en développement avec forte demande locative")
#         print("💰 Équilibre parfait entre prix d'achat et potentiel de revenus")
#     else:
#         print("🏰 PROPRIÉTÉ D'EXCEPTION - Investissement patrimonial")
#         print("📈 Valeur refuge qui ne fera que s'apprécier")
#         print("💰 Pour portefeuille d'investisseur averti")
    
#     print("\n⚠️ RAPPEL: Les prix immobiliers augmentent en moyenne de 5-10%/an")
#     print("   Attendre 1 an = Risque de payer 10-20 millions Ar de plus")
    
#     print("\n✅ RECOMMANDATION FINALE: INVESTISSEMENT RECOMMANDÉ")
#     print("   Contactez-nous pour une visite et une analyse personnalisée!")
    
#     return selling_points, roi_analysis

# # ========== 4. TEST AVEC EXEMPLE ==========
# house_example = {
#     "superficie_m2": 120,
#     "nb_chambres": 3,
#     "nb_etages": 1,
#     "acces_route": 1,
#     "eau_electricite": 1,
#     "parking": 1,
#     "annee_construction": 2020,
#     "localisation_periurbain": 1,
#     "localisation_rural": 0,
#     "localisation_urbain": 0,
#     "type_connexion_aucune": 0,
#     "type_connexion_fibre": 0,
#     "type_connexion_starlink": 1,
#     "type_sol_brut": 0,
#     "type_sol_carrelage": 1,
#     "type_sol_ciment": 0,
#     "etat_maison_a_renover": 0,
#     "etat_maison_bon": 1,
#     "etat_maison_neuf": 0
# }

# # Validation et prédiction
# valid, message = is_valid_house(house_example)

# if not valid:
#     print(f"\n❌ DONNÉES INVALIDES:")
#     for err in message:
#         print(f"   • {err}")
#     print("\n💾 Prix estimé: 0 millions Ar")
# else:
#     df = pd.DataFrame([house_example])
    
#     predictions = {}
#     for name, model in available_models.items():
#         try:
#             pred = model.predict(df)[0]
#             predictions[name] = pred
#         except Exception as e:
#             print(f"⚠️ Erreur avec {name}: {e}")
    
#     if not predictions:
#         print("❌ Aucun modèle n'a pu faire de prédiction!")
#         exit()
    
#     pred_values = list(predictions.values())
#     mean_pred = np.mean(pred_values)
#     std_pred = np.std(pred_values)
#     min_pred = min(pred_values)
#     max_pred = max(pred_values)
    
#     # Générer les conseils convaincants
#     selling_points, roi_analysis = generate_persuasive_advice(
#         house_example, pred_values, mean_pred, min_pred, max_pred
#     )
    
#     # Sauvegarder le rapport
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     filename = f"rapport_immobilier_{timestamp}.txt"
    
#     print(f"\n💾 Rapport sauvegardé: {filename}")
#     print("\n✅ Analyse terminée - Ce bien mérite votre attention!")