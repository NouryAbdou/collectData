# Dans views.py

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .models import TrainingData
from sklearn.linear_model import LinearRegression
import numpy as np
import json
import joblib


@csrf_exempt
def train_linear_regression(request):
    if request.method == 'POST':
        # Charger les données du corps de la requête POST
        data = json.loads(request.body)

        # Extraire les caractéristiques d'entrée et de sortie des données
        X_train = [[item['x']] for item in data]
        y_train = [item['y'] for item in data]

        # Entraîner le modèle de régression linéaire
        #Positive = true signifie que les coeficients attendus doivent etre positifs
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Enregistrer le modèle entraîné
        joblib.dump(model, 'linear_regression_model.pkl')

        model_coef_list = model.coef_.tolist()

        return JsonResponse({'message': 'Model is trained successfully', 'model coef_' : model_coef_list, 'intercept' : model.intercept_}, status=200)
    else:
        return JsonResponse({'error': 'Cette méthode n\'est pas autorisée'}, status=405)

@csrf_exempt
def predict_regression(request):
    if request.method == 'POST':
        # Charger les données du corps de la requête POST
        data = json.loads(request.body)

        # Charger le modèle entraîné
        model = joblib.load('linear_regression_model.pkl')

        # Extraire les caractéristiques d'entrée des données
        X_predict = [[item['x']] for item in data]

        # Faire des prédictions avec le modèle chargé
        predictions = model.predict(X_predict)

        # Formater les prédictions sous forme de dictionnaire
        results = [{'x': data[i]['x'], 'predicted_y': predictions[i]} for i in range(len(X_predict))]

        # Répondre avec les résultats des prédictions
        return JsonResponse({'predictions': results}, status=200)
    else:
        return JsonResponse({'error': 'Cette méthode n\'est pas autorisée'}, status=405)