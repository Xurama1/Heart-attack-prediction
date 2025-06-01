#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  1 18:37:35 2025

@author: thibaultjames
"""

# heart_attack_pipeline.py

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# 1. Chargement des données
df = pd.read_csv("Medicaldataset.csv")

# 2. Séparer X et y
X = df.drop("Result", axis=1)
y = df["Result"].map({"negative": 0, "positive": 1})

# 3. Définir les types de colonnes
numeric_features = ['Age', 'Heart rate', 'Systolic blood pressure',
                    'Diastolic blood pressure', 'Blood sugar', 'CK-MB', 'Troponin']
categorical_features = ['Gender']

# 4. Prétraitements
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

#  5. Pipeline complet
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=11))
])

# 6. Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=11)

#  7. Entraînement
pipeline.fit(X_train, y_train)

# 8. Évaluation
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

# 9. Sauvegarde du modèle
joblib.dump(pipeline, "heart_attack_model.joblib")
print("Modèle sauvegardé sous 'heart_attack_model.joblib'")
