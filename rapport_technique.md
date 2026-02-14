# Rapport Technique — Pipeline NLP de Classification de Tickets Support

## 1. Contexte du Projet

Ce projet vise à industrialiser un pipeline NLP batch pour la classification automatique de tickets de support IT. L'entreprise reçoit un volume important d'emails clients, et la catégorisation manuelle est coûteuse et hétérogène.

**Objectifs :**
- Automatiser la classification des tickets (prédiction du type)
- Générer des représentations sémantiques (embeddings) pour la recherche vectorielle
- Superviser la qualité du modèle et la dérive des données
- Déployer le pipeline dans un environnement containerisé et orchestré

## 2. Architecture du Pipeline

```
dataset.csv
    │
    ▼
[Étape 1] preprocess_advanced.py
    │   → Nettoyage NLP (lowercase, stopwords, lemmatisation via spaCy)
    │   → Fusion subject + body → full_text
    ▼
cleaned_dataset_advanced.csv
    │
    ├──► [Étape 2] generate_embeddings.py
    │       → Encodage avec all-MiniLM-L6-v2 (Hugging Face)
    │       → Stockage dans ChromaDB
    │
    ├──► [Étape 3] train_classifier.py
    │       → TF-IDF (10k features, 1-3 n-grams)
    │       → Logistic Regression (scikit-learn)
    │       → Sauvegarde models/classifier.joblib
    │
    └──► [Étape 5] monitoring/evidently_report.py
            → Rapport Data Drift + Classification Preset
            → Fichier HTML interactif
```

## 3. Étape 1 — Analyse Exploratoire & Préparation NLP

### Fichiers
- `eda.ipynb` : notebook d'analyse exploratoire (distribution des types, longueur des emails)
- `preprocess_advanced.py` : pipeline de nettoyage NLP

### Traitement appliqué
| Opération | Outil |
|-----------|-------|
| Lowercase | Python `str.lower()` |
| Suppression HTML | `re.sub` |
| Tokenisation | spaCy `en_core_web_sm` |
| Suppression stopwords | spaCy `.is_stop` |
| Suppression ponctuation | spaCy `.is_punct` |
| Lemmatisation | spaCy `.lemma_` |

### Résultat
- Fichier `data/cleaned_dataset_advanced.csv` avec colonnes `clean_subject`, `clean_body`, `full_text`

## 4. Étape 2 — Embeddings & ChromaDB

### Modèle choisi
- **all-MiniLM-L6-v2** (sentence-transformers) — modèle léger, performant pour le calcul de similarité sémantique

### Processus
1. Chargement des textes nettoyés (`full_text`)
2. Encodage en vecteurs de dimension 384
3. Indexation dans ChromaDB (base vectorielle persistante)
4. Métadonnées stockées : `subject`, `type`, `priority`, `language`

### Vérification
- Script `verify_chroma.py` : confirme le stockage et teste des requêtes sémantiques

## 5. Étape 3 — Entraînement du Modèle de Classification

### Comparaison de modèles (`model_comparison.py`)

| Modèle | Accuracy | F1 Macro | Temps (s) |
|--------|----------|----------|-----------|
| **Logistic Regression** | **0.7885** | **0.7663** | 9.0 |
| DistilBERT | 0.7390 | 0.6353 | 150.9 |
| spaCy | 0.7068 | 0.7151 | 188.5 |

### Modèle sélectionné : Logistic Regression
- **Justification** : meilleur rapport performance/efficacité en environnement CPU
- **Vectorisation** : TF-IDF avec 10 000 features, n-grams 1-3
- **Persistance** : `models/classifier.joblib` + `models/tfidf_vectorizer.joblib`

## 6. Étape 5 — Monitoring ML avec Evidently AI

### Approche
- Division du dataset en deux parties temporelles (référence vs courant)
- Ajout des prédictions du modèle aux deux jeux de données
- Analyse automatique via Evidently :
  - **DataDriftPreset** : détection de dérive des features
  - **ClassificationPreset** : suivi des métriques de classification

### Livrable
- Rapport HTML interactif : `monitoring/reports/drift_report.html`

## 7. Étape 6 — Conteneurisation & Orchestration

### Docker
- **Image** : `python:3.11-slim`
- **Pipeline** : préprocessing → entraînement → embeddings → monitoring
- **Entrypoint** : `run_pipeline.sh` (orchestration séquentielle)

### Kubernetes (Minikube)
- `k8s/pipeline-job.yaml` : Job pour exécution one-shot
- `k8s/pipeline-cronjob.yaml` : CronJob hebdomadaire (dimanche 02:00)

### CI/CD — GitHub Actions
- **Trigger** : push sur `main` ou `develop`
- **Job 1** : Lint Python avec `flake8`
- **Job 2** : Build de l'image Docker

## 8. Étape 7 — Monitoring Infrastructure

### Stack de monitoring (`monitoring/docker-compose.yml`)
| Service | Port | Rôle |
|---------|------|------|
| Prometheus | 9090 | Collecte et stockage des métriques |
| Grafana | 3000 | Visualisation des dashboards |
| cAdvisor | 8080 | Métriques des containers Docker |
| Node Exporter | 9100 | Métriques CPU/RAM/disque de la machine hôte |

### Configuration Prometheus (`monitoring/prometheus.yml`)
- Scrape interval : 5 secondes
- Targets : cAdvisor + Node Exporter

## 9. Structure du Projet

```
nlp-ticket-classifier/
├── .github/workflows/ci.yml      # CI/CD GitHub Actions
├── data/                          # Datasets (gitignored)
├── models/                        # Modèles sauvegardés (gitignored)
├── k8s/
│   ├── pipeline-job.yaml          # K8s Job
│   └── pipeline-cronjob.yaml      # K8s CronJob
├── monitoring/
│   ├── docker-compose.yml         # Stack Prometheus/Grafana
│   ├── prometheus.yml             # Config Prometheus
│   ├── evidently_report.py        # Rapport de drift
│   └── reports/                   # Rapports HTML (gitignored)
├── eda.ipynb                      # Analyse exploratoire
├── preprocess_advanced.py         # Nettoyage NLP
├── generate_embeddings.py         # Embeddings + ChromaDB
├── model_comparison.py            # Comparaison de modèles
├── train_classifier.py            # Entraînement modèle final
├── verify_chroma.py               # Vérification ChromaDB
├── run_pipeline.sh                # Orchestration pipeline
├── Dockerfile                     # Image Docker
├── requirements.txt               # Dépendances Python
├── model_selection_report.md      # Rapport de sélection
└── rapport_technique.md           # Ce rapport
```

## 10. Conclusion

Ce projet implémente un pipeline NLP end-to-end industrialisé :
- **Classification** : Logistic Regression atteignant ~79% d'accuracy et 0.77 de F1 macro
- **Reproductibilité** : Docker + Kubernetes pour une exécution portable
- **Supervision** : Evidently AI pour le drift ML, Prometheus/Grafana pour l'infrastructure
- **Automatisation** : CI/CD via GitHub Actions, CronJob K8s pour la re-exécution périodique
