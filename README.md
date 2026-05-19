# Suicide Rate Prediction — ML PoC

Proof of concept de Machine Learning pour la prediction du taux de suicide pour 100 000 habitants, a partir de donnees socio-economiques mondiales (1985-2016).

## Objectif

Predire le **taux de suicide pour 100 000 habitants** d'un groupe demographique defini par pays, annee, sexe et tranche d'age. Ce modele peut aider les organisations de sante publique a identifier les populations a risque et a allouer les ressources de prevention.

## Donnees

Le dataset combine trois sources :

- **OMS** — Donnees de mortalite par suicide
- **Banque Mondiale** — PIB et PIB par habitant
- **PNUD** — Indice de Developpement Humain (IDH)

Fichier source : `data/master.csv` (~27 000 observations, 101 pays)

## Pipeline ML

```
Chargement CSV → Nettoyage & Encodage → StandardScaler → Entrainement → Evaluation
```

### Preprocessing

| Feature | Encodage |
|---|---|
| `year` | Numerique brut |
| `sex` | Binaire (male=1, female=0) |
| `age` | Ordinal (0-5) |
| `gdp_per_capita` | Numerique brut |
| `HDI_for_year` | Numerique brut |
| `population` | Numerique brut |
| `generation` | Label encoding |
| `country_encoded` | Target encoding (moyenne du taux par pays) |

### Modeles

| Modele | MAE | RMSE | R² | MAPE |
|---|---|---|---|---|
| Linear Regression | 7.86 | 11.80 | 0.549 | 380.3% |
| **Random Forest** | **2.54** | **5.51** | **0.902** | **33.6%** |
| Gradient Boosting | 3.96 | 6.98 | 0.842 | 81.3% |

Le **Random Forest** (200 arbres, random_state=42) obtient les meilleures performances avec un R² de 0.902.

## Structure du projet

```
ml-poc-project/
├── data/                  # Dataset CSV
├── models/                # Modeles serialises (.joblib)
├── notebooks/             # Jupyter notebook d'entrainement
│   └── train_models.ipynb
├── plots/                 # Visualisations EDA (9 graphiques)
├── results/               # Metriques d'evaluation (CSV)
├── scripts/
│   └── main.py            # Point d'entree principal
├── src/
│   ├── config.py          # Configuration et chemins
│   ├── data.py            # Chargement et preprocessing
│   ├── metrics.py         # Calcul des metriques (MAE, RMSE, R², MAPE)
│   ├── app.py             # Application Streamlit
│   ├── model_io.py        # Utilitaires de chargement des modeles
│   └── results.py         # Ecriture des resultats
├── .streamlit/
│   └── config.toml        # Theme Streamlit
├── .env                   # Variables d'environnement
├── requirements.txt       # Dependances Python
└── README.md
```

## Installation

```bash
# Cloner le repo
git clone https://github.com/KytK4t/ml-poc-project.git
cd ml-poc-project

# Creer un environnement virtuel
python3 -m venv .venv
source .venv/bin/activate

# Installer les dependances
pip install -r requirements.txt
```

## Utilisation

### Lancer le projet complet

```bash
python3 scripts/main.py
```

Cette commande :
1. Valide la presence de l'app et des modeles
2. Charge le dataset et effectue le preprocessing
3. Evalue chaque modele sur le split test (20%)
4. Sauvegarde les metriques dans `results/model_metrics.csv`
5. Lance l'application Streamlit sur `http://localhost:8501`

### Lancer uniquement le Streamlit

```bash
python3 -m streamlit run src/app.py
```

## Application Streamlit

L'application propose 4 pages accessibles via la sidebar :

- **Dashboard** — Vue d'ensemble du projet, pipeline ML, sources de donnees, features
- **Analyse** — Carte du monde, taux par sexe/age, evolution temporelle, top 15 pays, correlation IDH
- **Modeles** — Comparaison des 3 modeles (tableau, radar chart, bar charts)
- **Prediction** — Demo interactive avec sliders pour predire un taux de suicide

## Technologies

- Python 3.10+
- scikit-learn (Pipeline, StandardScaler, modeles)
- Streamlit (interface web)
- Plotly (graphiques interactifs)
- pandas / numpy (manipulation de donnees)
- joblib (serialisation des modeles)
