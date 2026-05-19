from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
PLOTS_DIR = PROJECT_ROOT / "plots"
RESULTS_DIR = PROJECT_ROOT / "results"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
TESTS_DIR = PROJECT_ROOT / "tests"
BASE_DIR = PROJECT_ROOT

for d in [
    DATA_DIR,
    LOGS_DIR,
    MODELS_DIR,
    NOTEBOOKS_DIR,
    PLOTS_DIR,
    RESULTS_DIR,
    SCRIPTS_DIR,
    TESTS_DIR,
]:
    d.mkdir(exist_ok=True)

ENV_FILE = PROJECT_ROOT / ".env"
APP_ENTRYPOINT = PROJECT_ROOT / "src" / "app.py"
MODEL_METRICS_FILE = RESULTS_DIR / "model_metrics.csv"

STREAMLIT_HOST = "localhost"
STREAMLIT_PORT = 8501

STREAMLIT_CONFIG = {
    "page_title": "Suicide Rate Prediction - ML PoC",
    "layout": "wide",
    "page_icon": "📊",
}

MODELS = {
    "linear_reg": {
        "name": "Linear Regression",
        "description": "Baseline linear model with standardized features.",
        "path": MODELS_DIR / "linear_reg.joblib",
    },
    "random_forest": {
        "name": "Random Forest",
        "description": "Ensemble of 200 decision trees with random state 42.",
        "path": MODELS_DIR / "random_forest.joblib",
    },
    "gradient_boosting": {
        "name": "Gradient Boosting",
        "description": "Boosted trees with 200 estimators, learning rate 0.05.",
        "path": MODELS_DIR / "gradient_boosting.joblib",
    },
}
