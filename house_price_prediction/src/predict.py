def predict_all_models(models, input_data):

    results = {}

    for name, model in models.items():
        pred = model.predict(input_data)[0]
        results[name] = pred

    return results