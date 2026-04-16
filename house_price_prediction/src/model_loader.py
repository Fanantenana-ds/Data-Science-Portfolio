import joblib

def load_models():

    models = {
        "XGBoost": joblib.load("models/xgboost.pkl"),
        "RandomForest": joblib.load("models/randomforest.pkl"),
        "GradientBoosting": joblib.load("models/gradientboosting.pkl")
    }

    return models