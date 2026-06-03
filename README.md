# Automatic Fraud Detection

## 1. Présentation du projet

Ce projet met en place un système automatisé de détection de fraude sur des transactions de paiement.

L’objectif est de construire une chaîne complète capable de :

- récupérer des transactions en temps réel depuis une API externe ;
- appliquer un modèle de machine learning pour estimer le risque de fraude ;
- stocker les transactions, prédictions et alertes dans une base PostgreSQL ;
- orchestrer les traitements avec Apache Airflow ;
- suivre les expériences et modèles avec MLflow ;
- exporter des rapports quotidiens dans un datalake S3 ;
- notifier automatiquement une alerte métier lorsqu’une transaction est considérée comme frauduleuse.

---

## 2. Objectifs métier

Le système répond à trois besoins principaux :

1. **Détection rapide des transactions suspectes**  
   Chaque transaction récupérée depuis l’API source est évaluée par un modèle de prédiction.

2. **Traçabilité des décisions**  
   Les transactions, scores de fraude, modèles utilisés et alertes sont historisés dans NeonDB.

3. **Monitoring quotidien**  
   Un rapport journalier synthétise le volume de transactions, le nombre de fraudes détectées, le taux de fraude et les montants associés.

---

## 3. Architecture générale

L’architecture repose sur les composants suivants :

```text
API organisme de paiement
        ↓
Airflow - DAG temps réel
        ↓
Pipeline de normalisation
        ↓
API de prédiction FastAPI
        ↓
NeonDB PostgreSQL
        ↓
Alerting Discord si fraude détectée

En parallèle :

Dataset historique
        ↓
Pipeline d'entraînement
        ↓
MLflow Tracking Server
        ↓
Modèle de production exporté dans S3

Et chaque matin :

Airflow - DAG reporting quotidien
        ↓
Agrégation NeonDB
        ↓
Table daily_reports
        ↓
Export CSV dans S3
```
---

## 4. Composants techniques

Le projet s’appuie sur les composants suivants :

- **Apache Airflow** : orchestration des workflows de collecte, prédiction et reporting ;
- **FastAPI** : exposition du modèle de prédiction sous forme d’API ;
- **MLflow** : suivi des expériences, métriques et artefacts modèles ;
- **S3** : stockage du dataset, du modèle de production, des métadonnées et des rapports ;
- **NeonDB PostgreSQL** : stockage applicatif des transactions, prédictions, alertes et rapports ;
- **Discord Webhook** : notification automatique des alertes de fraude ;
- **Docker Compose** : exécution locale d’Airflow pour la démonstration.

---

## 5. Pipeline de machine learning

Le modèle est entraîné sur un dataset historique de transactions labellisées.

Les principales étapes sont :

1. chargement du dataset depuis S3 ;
2. nettoyage et préparation des variables ;
3. feature engineering ;
4. entraînement de plusieurs modèles candidats ;
5. comparaison des performances dans MLflow ;
6. sélection du modèle de production ;
7. export du modèle et de ses métadonnées dans S3.

Le modèle retenu est un modèle **XGBoost pondéré**, adapté à un problème de classification fortement déséquilibré.

Le seuil de décision de production est stocké dans les métadonnées du modèle afin que l’API de prédiction utilise la même règle de décision que celle validée lors de l’entraînement.

---

## 6. Feature engineering

Les variables suivantes sont dérivées avant entraînement ou prédiction :

- heure de transaction ;
- jour de la semaine ;
- mois ;
- indicateur week-end ;
- âge du client ;
- distance client ↔ marchand en kilomètres.

La distance client ↔ marchand est calculée à partir des coordonnées GPS avec la formule de Haversine.

---

## 7. Orchestration Airflow

Deux DAGs principaux sont définis.

### `fraud_realtime_ingestion_dag`

Ce DAG orchestre le traitement temps réel :

1. récupération d’une transaction depuis l’API externe ;
2. normalisation des données ;
3. insertion de la transaction dans NeonDB ;
4. appel de l’API de prédiction ;
5. insertion du score et de la décision ;
6. création d’une alerte si la transaction dépasse le seuil de fraude ;
7. notification Discord si une nouvelle alerte est créée.

### `fraud_daily_report_dag`

Ce DAG génère un rapport quotidien :

1. extraction des transactions, prédictions et alertes de la veille ;
2. calcul des KPI ;
3. insertion ou mise à jour de la table `daily_reports` ;
4. export d’un rapport CSV vers S3.

---

## 8. Modèle de données NeonDB

La base applicative contient quatre tables principales :

### `transactions`

Stocke les transactions normalisées et le payload source.

### `predictions`

Stocke les prédictions produites par le modèle.

### `alerts`

Stocke les alertes créées lorsqu’une transaction est considérée comme frauduleuse.

### `daily_reports`

Stocke les indicateurs quotidiens de monitoring.

Le schéma SQL est disponible dans :

```text
sql/init_fraud_app_db.sql
```

---

## 9. Idempotence

Le pipeline est conçu pour éviter les doublons lors des retries Airflow ou des tests manuels.

Les protections principales sont :

transactions.transaction_id comme clé primaire ;
index unique sur predictions.transaction_id ;
index unique sur alerts.transaction_id ;
usage de ON CONFLICT DO NOTHING.

Pour les alertes, la notification Discord est envoyée uniquement si une nouvelle ligne est réellement créée dans la table alerts.

---

## 10. Alerting Discord

Lorsqu’une transaction est prédite comme frauduleuse, une notification Discord est envoyée.

Le message contient :

identifiant de transaction ;
probabilité de fraude ;
seuil de décision ;
montant ;
marchand ;
catégorie ;
client pseudonymisé ;
localisation client ;
coordonnées client et marchand ;
distance client ↔ marchand ;
date et heure de transaction.

Les données personnelles directes ne sont pas envoyées dans Discord. Le client est identifié par un identifiant pseudonymisé.

---

## 11. Reporting quotidien

Le rapport quotidien calcule notamment :

nombre total de transactions ;
nombre de fraudes détectées ;
taux de fraude ;
montant total des transactions ;
montant associé aux fraudes détectées ;
probabilité moyenne de fraude ;
probabilité maximale de fraude ;
nombre d’alertes créées.

Les rapports sont stockés dans S3 sous le préfixe :

```
reports/daily/
```

---

## 12. Configuration

Le projet utilise un fichier `.env` pour centraliser les paramètres d’exécution et les secrets.

Les principales familles de variables sont :

- accès AWS / S3 ;
- configuration MLflow ;
- URL de la base NeonDB ;
- URL de l’API source de transactions ;
- URL de l’API de prédiction ;
- configuration du webhook Discord ;
- plannings Airflow.

Un fichier `.env.example` fournit la structure attendue sans exposer de secrets.

Le fichier `.env` ne doit pas être versionné.

---

## 13. Lancement Airflow

Airflow est exécuté localement via Docker Compose.

Initialisation :

```
docker compose up airflow-init
```

Lancement :

```
docker compose up
```

Interface Airflow :

http://localhost:8080

Identifiants de développement :

admin / admin

Tester un DAG :
```
docker compose exec airflow-webserver airflow dags test fraud_realtime_ingestion_dag 2026-06-02
```
```
docker compose exec airflow-webserver airflow dags test fraud_daily_report_dag 2026-06-03T08:00:00
```

---

## 14. Limites

Le projet présente plusieurs limites :

l’API source fournit une transaction à la fois ;
les nouvelles transactions ne sont pas utilisées automatiquement pour réentraîner le modèle ;
l’alerting Discord sert de démonstrateur et non de canal de production ;
- Airflow est exécuté localement via Docker Compose, avec une exposition publique ponctuelle possible via ngrok.