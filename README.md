Dashboard Néo-Banque – Scoring Crédit

📌 Description
Ce projet est un Proof of Concept (POC) visant à déployer une solution d’intelligence artificielle pour le scoring de crédit dans une néo-banque.
L’objectif est de fournir aux conseillers clientèle un dashboard interactif leur permettant de :

Visualiser le score de probabilité de défaut d’un client.

Comprendre les facteurs explicatifs grâce à SHAP.

Accéder aux informations descriptives d’un client via des filtres simples et ergonomiques.

Le projet repose sur deux composants principaux :

Une API FastAPI servant de moteur de prédiction (scoring).

Une application web Streamlit (dashboard) consommant l’API.


⚙️ Architecture du projet

api/ → contient l’API FastAPI (main.py).

model/ → contient le script d’entraînement, la génération de données synthétiques et les artefacts du modèle (model.pkl, scaler.pkl).

streamlit_app.py → application Dashboard développée avec Streamlit.

requirements.txt → liste des dépendances Python.

README.md → documentation du projet.

🚀 Fonctionnalités

Prédiction en temps réel d’un score de solvabilité via l’API.

Interprétation explicable avec SHAP (facteurs positifs/négatifs).

Visualisation du score avec une jauge colorée intuitive (vert = faible risque, rouge = fort risque).

Interface utilisateur claire (conçue sur Figma puis implémentée en Streamlit).

Conformité RGPD : données anonymisées, non stockées, utilisées uniquement pour le calcul du score.

Cybersécurité : API protégée (HTTPS, authentification JWT prévue en production, contrôle des accès et audit).

🔧 Installation et utilisation

Cloner le projet depuis GitHub.

Créer un environnement virtuel Python et l’activer.

Installer les dépendances avec le fichier requirements.txt.

Lancer l’API avec FastAPI (via Uvicorn). L’API sera disponible sur http://localhost:8000
, avec la documentation interactive sur http://localhost:8000/docs

Lancer le Dashboard avec Streamlit. Le dashboard sera disponible sur http://localhost:8501

🌐 Déploiement

Le projet est déployé sur Render avec deux services distincts :

API FastAPI : disponible via une URL publique dédiée.

Dashboard Streamlit : disponible via une autre URL publique.

Cette séparation permet de garantir la robustesse et l’indépendance des composants.

📊 Exemple d’utilisation de l’API

Un appel POST à l’endpoint /predict avec une liste de features numériques renvoie en sortie :

Un score de probabilité de défaut (compris entre 0 et 1).

Une explication des contributions de chaque variable via SHAP.

🔒 RGPD et Cybersécurité

Données anonymisées et pseudonymisées.

Données non stockées : uniquement utilisées pour le calcul du score en temps réel.

Sécurité prévue en production : protocole HTTPS, authentification JWT, journalisation des accès, infrastructure conforme aux standards bancaires.
