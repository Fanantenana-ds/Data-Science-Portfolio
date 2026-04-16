"""
model_loader.py
Mampiditra (load) ny modèles .pkl voatahiry
"""

import joblib
import os

# Mahazo ny lalana mankany amin'ny folder models
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

def load_model(model_name):
    """
    Mampiditra modèle iray amin'ny anarana omena
    model_name: 'model_xgboost', 'model_randomforest', ou 'model_gradientboosting'
    """
    # Fanitsiana ny anarana mifanaraka amin'ny fichier anao
    model_filename = f'model_{model_name}.pkl'
    model_path = os.path.join(MODELS_DIR, model_filename)
    
    try:
        model = joblib.load(model_path)
        print(f"✅ Modèle {model_name} chargé avec succès")
        return model
    except FileNotFoundError:
        print(f"❌ Fichier non trouvé: {model_path}")
        return None

def load_all_models():
    """
    Mampiditra ny modèles rehetra
    """
    models = {
        'XGBoost': load_model('xgboost'),
        'Random Forest': load_model('randomforest'),
        'Gradient Boosting': load_model('gradientboosting')
    }
    
    # Manala ny modèles tsy hita (None)
    models = {name: model for name, model in models.items() if model is not None}
    
    return models