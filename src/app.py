"""Streamlit application for suicide rate prediction PoC."""

from __future__ import annotations

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from pathlib import Path

from config import (
    DATA_DIR,
    MODELS,
    MODELS_DIR,
    PLOTS_DIR,
    RESULTS_DIR,
    MODEL_METRICS_FILE,
    STREAMLIT_CONFIG,
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .main .block-container {
        padding-top: 2rem;
        max-width: 1200px;
    }

    h1 {
        background: linear-gradient(120deg, #1B4332, #2D6A4F, #40916C);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700 !important;
        font-size: 2.2rem !important;
        margin-bottom: 0.3rem !important;
    }

    h2 {
        color: #1B4332 !important;
        font-weight: 600 !important;
        border-bottom: 2px solid #D8F3DC;
        padding-bottom: 0.4rem;
    }

    h3 {
        color: #2D6A4F !important;
        font-weight: 500 !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #F0FFF4;
        border-radius: 12px;
        padding: 6px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 500;
    }

    .stTabs [aria-selected="true"] {
        background-color: #2D6A4F !important;
        color: white !important;
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #F0FFF4, #D8F3DC);
        border: 1px solid #B7E4C7;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
    }

    div[data-testid="stMetric"] label {
        color: #52796F !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #1B4332 !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }

    .success-box {
        background: linear-gradient(135deg, #D8F3DC, #B7E4C7);
        border-left: 4px solid #2D6A4F;
        border-radius: 8px;
        padding: 20px;
        margin: 16px 0;
    }

    .info-card {
        background: white;
        border: 1px solid #E8F5E9;
        border-radius: 12px;
        padding: 24px;
        margin: 12px 0;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.03);
    }

    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #2D6A4F, #40916C) !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.6rem 2rem !important;
        font-weight: 600 !important;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid #E8F5E9;
        border-radius: 8px;
        overflow: hidden;
    }

    .subtitle {
        color: #52796F;
        font-size: 1.1rem;
        margin-top: -0.5rem;
        margin-bottom: 1.5rem;
    }
</style>
"""

# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------
COLORS = {
    "primary": "#2D6A4F",
    "secondary": "#40916C",
    "accent": "#52B788",
    "light": "#D8F3DC",
    "dark": "#1B4332",
    "male": "#3A86FF",
    "female": "#FF006E",
    "palette": ["#2D6A4F", "#40916C", "#52B788", "#74C69D", "#95D5B2", "#B7E4C7"],
    "models": ["#3A86FF", "#FF006E", "#FFBE0B"],
}


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def _load_raw_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "master.csv")
    df.rename(
        columns={
            "suicides/100k pop": "suicides_per_100k",
            "HDI for year": "HDI_for_year",
            " gdp_for_year ($) ": "gdp_for_year",
            "gdp_per_capita ($)": "gdp_per_capita",
        },
        inplace=True,
    )
    return df


# ---------------------------------------------------------------------------
# Tab 1 : Context
# ---------------------------------------------------------------------------
def _tab_context() -> None:
    st.markdown('<p class="subtitle">Comprendre et predire les taux de suicide a l\'echelle mondiale</p>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown("### Objectif metier")
        st.markdown(
            """
            Predire le **taux de suicide pour 100 000 habitants** d'un groupe
            defini par pays, annee, sexe et tranche d'age, a partir de ses
            caracteristiques socio-economiques.

            Ce modele peut aider les organisations de sante publique a
            **identifier les populations a risque** et a **allouer les
            ressources de prevention** de maniere plus efficace.
            """
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown("### Approche ML")
        st.markdown(
            """
            **Type** : Regression supervisee

            **Pipeline** :
            1. Nettoyage et encodage des variables
            2. StandardScaler pour la normalisation
            3. Comparaison de 3 algorithmes
            4. Evaluation sur split test 20%

            **Metriques** : MAE, RMSE, R², MAPE
            """
        )
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Source des donnees")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown("**Organisation Mondiale de la Sante (OMS)**")
        st.caption("Donnees de mortalite par suicide")
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown("**Banque Mondiale**")
        st.caption("PIB, PIB par habitant")
        st.markdown('</div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown("**PNUD**")
        st.caption("Indice de Developpement Humain (IDH)")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Features du modele")

    features_df = pd.DataFrame({
        "Feature": ["year", "sex", "age", "gdp_per_capita", "HDI_for_year", "population", "generation", "country_encoded"],
        "Description": [
            "Annee d'observation",
            "Sexe (male=1, female=0)",
            "Tranche d'age encodee en ordinal (0-5)",
            "PIB par habitant en dollars",
            "Indice de Developpement Humain",
            "Population du groupe demographique",
            "Generation (label encoding)",
            "Target encoding du pays (moyenne du taux)",
        ],
        "Type d'encodage": [
            "Numerique brut",
            "Binaire",
            "Ordinal",
            "Numerique brut",
            "Numerique brut",
            "Numerique brut",
            "Label encoding",
            "Target encoding",
        ],
    })
    st.dataframe(features_df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Tab 2 : Data
# ---------------------------------------------------------------------------
def _tab_data() -> None:
    st.markdown('<p class="subtitle">Exploration et statistiques cles du dataset</p>', unsafe_allow_html=True)

    df = _load_raw_data()

    # KPI row
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Observations", f"{len(df):,}")
    c2.metric("Pays", df["country"].nunique())
    c3.metric("Periode", f"{df['year'].min()}-{df['year'].max()}")
    c4.metric("Taux moyen", f"{df['suicides_per_100k'].mean():.1f}")
    c5.metric("Taux median", f"{df['suicides_per_100k'].median():.1f}")

    st.markdown("---")

    # Row 1: distribution + by sex
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Distribution du taux de suicide")
        fig = px.histogram(
            df,
            x="suicides_per_100k",
            nbins=60,
            color_discrete_sequence=[COLORS["primary"]],
            opacity=0.85,
        )
        fig.update_layout(
            xaxis_title="Suicides pour 100k habitants",
            yaxis_title="Frequence",
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(t=20, b=40, l=40, r=20),
            height=350,
        )
        fig.update_xaxes(gridcolor="#E8F5E9")
        fig.update_yaxes(gridcolor="#E8F5E9")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Taux moyen par sexe et tranche d'age")
        age_order = ["5-14 years", "15-24 years", "25-34 years", "35-54 years", "55-74 years", "75+ years"]
        sex_age = df.groupby(["sex", "age"])["suicides_per_100k"].mean().reset_index()
        sex_age["age"] = pd.Categorical(sex_age["age"], categories=age_order, ordered=True)
        sex_age = sex_age.sort_values("age")

        fig = px.bar(
            sex_age,
            x="age",
            y="suicides_per_100k",
            color="sex",
            barmode="group",
            color_discrete_map={"male": COLORS["male"], "female": COLORS["female"]},
        )
        fig.update_layout(
            xaxis_title="Tranche d'age",
            yaxis_title="Suicides pour 100k",
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(t=20, b=40, l=40, r=20),
            height=350,
            legend=dict(title="Sexe", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        fig.update_xaxes(gridcolor="#E8F5E9")
        fig.update_yaxes(gridcolor="#E8F5E9")
        st.plotly_chart(fig, use_container_width=True)

    # Row 2: temporal + top countries
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("### Evolution temporelle")
        yearly_sex = df.groupby(["year", "sex"])["suicides_per_100k"].mean().reset_index()

        fig = px.line(
            yearly_sex,
            x="year",
            y="suicides_per_100k",
            color="sex",
            color_discrete_map={"male": COLORS["male"], "female": COLORS["female"]},
            markers=True,
        )
        fig.update_layout(
            xaxis_title="Annee",
            yaxis_title="Suicides pour 100k",
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(t=20, b=40, l=40, r=20),
            height=350,
            legend=dict(title="Sexe", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        fig.update_xaxes(gridcolor="#E8F5E9")
        fig.update_yaxes(gridcolor="#E8F5E9")
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.markdown("### Top 15 pays")
        top = (
            df.groupby("country")["suicides_per_100k"]
            .mean()
            .sort_values(ascending=True)
            .tail(15)
            .reset_index()
        )
        fig = px.bar(
            top,
            x="suicides_per_100k",
            y="country",
            orientation="h",
            color="suicides_per_100k",
            color_continuous_scale=["#B7E4C7", "#2D6A4F", "#1B4332"],
        )
        fig.update_layout(
            xaxis_title="Suicides pour 100k",
            yaxis_title="",
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(t=20, b=40, l=120, r=20),
            height=350,
            coloraxis_showscale=False,
        )
        fig.update_xaxes(gridcolor="#E8F5E9")
        st.plotly_chart(fig, use_container_width=True)

    # Row 3: HDI correlation
    st.markdown("### Correlation IDH vs Taux de suicide")
    df_hdi = df.dropna(subset=["HDI_for_year"])
    hdi_country = df_hdi.groupby("country").agg(
        HDI=("HDI_for_year", "mean"),
        suicide_rate=("suicides_per_100k", "mean"),
        population=("population", "sum"),
    ).reset_index()

    fig = px.scatter(
        hdi_country,
        x="HDI",
        y="suicide_rate",
        size="population",
        hover_name="country",
        color="suicide_rate",
        color_continuous_scale=["#D8F3DC", "#2D6A4F", "#1B4332"],
        size_max=40,
    )
    fig.update_layout(
        xaxis_title="Indice de Developpement Humain (IDH)",
        yaxis_title="Taux moyen de suicide pour 100k",
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(t=20, b=40, l=40, r=20),
        height=400,
        coloraxis_showscale=False,
    )
    fig.update_xaxes(gridcolor="#E8F5E9")
    fig.update_yaxes(gridcolor="#E8F5E9")
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 3 : Models
# ---------------------------------------------------------------------------
def _tab_models() -> None:
    st.markdown('<p class="subtitle">Comparaison des performances des 3 modeles</p>', unsafe_allow_html=True)

    if not MODEL_METRICS_FILE.exists():
        st.warning("Lancez `python scripts/main.py` pour generer les metriques.")
        return

    metrics_df = pd.read_csv(MODEL_METRICS_FILE)
    metric_cols = [c for c in metrics_df.columns if c not in ("model_key", "model_name", "model_path")]

    # Best model highlight
    best_idx = metrics_df["r2"].idxmax()
    best_model = metrics_df.loc[best_idx, "model_name"]
    best_r2 = metrics_df.loc[best_idx, "r2"]

    st.markdown(
        f'<div class="success-box">'
        f'<strong>Meilleur modele :</strong> {best_model} avec un R² de <strong>{best_r2:.4f}</strong>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # KPI cards for best model
    c1, c2, c3, c4 = st.columns(4)
    best_row = metrics_df.loc[best_idx]
    c1.metric("MAE", f"{best_row['mae']:.2f}")
    c2.metric("RMSE", f"{best_row['rmse']:.2f}")
    c3.metric("R²", f"{best_row['r2']:.4f}")
    c4.metric("MAPE", f"{best_row['mape']:.1f}%")

    st.markdown("---")

    # Metrics table with styling
    st.markdown("### Tableau comparatif")

    display_df = metrics_df[["model_name"] + metric_cols].copy()
    display_df.columns = ["Modele"] + metric_cols

    def _highlight_best(s: pd.Series) -> list[str]:
        if s.name == "r2":
            is_best = s == s.max()
        else:
            is_best = s == s.min()
        return ["background-color: #D8F3DC; font-weight: bold" if v else "" for v in is_best]

    styled = display_df.style.apply(_highlight_best, subset=metric_cols)
    styled = styled.format({c: "{:.4f}" for c in metric_cols})
    st.dataframe(styled, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Radar chart comparing models
    st.markdown("### Vue radar des performances")

    # Normalize metrics for radar (invert those where lower is better)
    radar_df = metrics_df[["model_name", "mae", "rmse", "r2", "mape"]].copy()

    # For radar: normalize 0-1, invert mae/rmse/mape so higher = better
    for col in ["mae", "rmse", "mape"]:
        max_val = radar_df[col].max()
        if max_val > 0:
            radar_df[col + "_norm"] = 1 - (radar_df[col] / max_val)
        else:
            radar_df[col + "_norm"] = 1.0
    radar_df["r2_norm"] = radar_df["r2"]

    categories = ["MAE (inv.)", "RMSE (inv.)", "R²", "MAPE (inv.)"]

    fig = go.Figure()
    for i, row in radar_df.iterrows():
        values = [row["mae_norm"], row["rmse_norm"], row["r2_norm"], row["mape_norm"]]
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill="toself",
            name=row["model_name"],
            line_color=COLORS["models"][i],
            opacity=0.7,
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], gridcolor="#E8F5E9"),
            bgcolor="white",
        ),
        showlegend=True,
        height=400,
        paper_bgcolor="white",
        margin=dict(t=40, b=40, l=60, r=60),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Bar chart comparison
    st.markdown("### Comparaison par metrique")
    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(
            metrics_df,
            x="model_name",
            y="r2",
            color="model_name",
            color_discrete_sequence=COLORS["models"],
            text_auto=".4f",
        )
        fig.update_layout(
            title="R² (plus haut = meilleur)",
            xaxis_title="",
            yaxis_title="R²",
            plot_bgcolor="white",
            paper_bgcolor="white",
            showlegend=False,
            height=300,
            margin=dict(t=40, b=20),
        )
        fig.update_xaxes(gridcolor="#E8F5E9")
        fig.update_yaxes(gridcolor="#E8F5E9")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(
            metrics_df,
            x="model_name",
            y="mae",
            color="model_name",
            color_discrete_sequence=COLORS["models"],
            text_auto=".2f",
        )
        fig.update_layout(
            title="MAE (plus bas = meilleur)",
            xaxis_title="",
            yaxis_title="MAE",
            plot_bgcolor="white",
            paper_bgcolor="white",
            showlegend=False,
            height=300,
            margin=dict(t=40, b=20),
        )
        fig.update_xaxes(gridcolor="#E8F5E9")
        fig.update_yaxes(gridcolor="#E8F5E9")
        st.plotly_chart(fig, use_container_width=True)

    # Model descriptions
    st.markdown("---")
    st.markdown("### Description des modeles")
    for key, cfg in MODELS.items():
        with st.expander(f"**{cfg['name']}**", expanded=False):
            st.markdown(cfg["description"])
            st.code(f"Path: {cfg['path']}", language="text")


# ---------------------------------------------------------------------------
# Tab 4 : Demo
# ---------------------------------------------------------------------------
def _tab_demo() -> None:
    st.markdown('<p class="subtitle">Predire le taux de suicide en ajustant les parametres</p>', unsafe_allow_html=True)

    # Model selector
    model_options = {cfg["name"]: key for key, cfg in MODELS.items()}
    selected_name = st.selectbox("Choisir un modele", list(model_options.keys()), index=1)
    selected_key = model_options[selected_name]

    model_path = MODELS[selected_key]["path"]
    if not Path(model_path).exists():
        st.error(f"Modele introuvable : {model_path}")
        return

    model = joblib.load(model_path)

    st.markdown("---")
    st.markdown("### Parametres d'entree")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        year = st.slider("Annee", 1985, 2016, 2010)
        sex = st.radio("Sexe", ["Homme", "Femme"], horizontal=True)
        sex_val = 1 if sex == "Homme" else 0
        age = st.select_slider(
            "Tranche d'age",
            options=["5-14 ans", "15-24 ans", "25-34 ans", "35-54 ans", "55-74 ans", "75+ ans"],
            value="35-54 ans",
        )
        age_mapping = {
            "5-14 ans": 0, "15-24 ans": 1, "25-34 ans": 2,
            "35-54 ans": 3, "55-74 ans": 4, "75+ ans": 5,
        }
        age_val = age_mapping[age]
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        gdp_per_capita = st.slider("PIB par habitant ($)", 100, 120_000, 15_000, step=500)
        hdi = st.slider("Indice de Dev. Humain (IDH)", 0.30, 1.00, 0.75, step=0.01)
        population = st.number_input(
            "Population du groupe",
            min_value=100, max_value=50_000_000, value=500_000, step=10_000,
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        generation = st.slider("Generation (encodee)", 0, 5, 2, help="0=Boomers, 1=Gen Alpha, 2=Gen X, 3=Gen Z, 4=Millenials, 5=Silent")
        country_encoded = st.slider(
            "Encodage pays (taux moyen du pays)",
            0.0, 50.0, 12.0, step=0.5,
            help="Taux moyen de suicide du pays d'origine",
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # Feature vector
    features = pd.DataFrame(
        [[year, sex_val, age_val, gdp_per_capita, hdi, population, generation, country_encoded]],
        columns=["year", "sex", "age", "gdp_per_capita", "HDI_for_year", "population", "generation", "country_encoded"],
    )

    st.markdown("---")

    col_btn, col_space = st.columns([1, 3])
    with col_btn:
        predict = st.button("Lancer la prediction", type="primary", use_container_width=True)

    if predict:
        prediction = model.predict(features)[0]
        prediction = max(0, prediction)  # Clamp to 0

        st.markdown("---")

        # Result display
        res_col1, res_col2, res_col3 = st.columns([1, 2, 1])
        with res_col2:
            st.markdown(
                f'<div style="text-align:center; padding:30px; background:linear-gradient(135deg, #D8F3DC, #B7E4C7); '
                f'border-radius:16px; border:2px solid #52B788;">'
                f'<p style="color:#52796F; font-size:1rem; margin-bottom:8px;">Taux de suicide predit</p>'
                f'<p style="color:#1B4332; font-size:3rem; font-weight:700; margin:0;">{prediction:.2f}</p>'
                f'<p style="color:#52796F; font-size:0.9rem; margin-top:4px;">pour 100 000 habitants</p>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown("")

        # Feature summary
        with st.expander("Voir le vecteur de features"):
            st.dataframe(features, use_container_width=True, hide_index=True)

        # Context gauge
        st.markdown("### Contextualisation")
        df = _load_raw_data()
        avg_rate = df["suicides_per_100k"].mean()
        med_rate = df["suicides_per_100k"].median()

        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=prediction,
            delta={"reference": avg_rate, "suffix": " vs moyenne"},
            gauge={
                "axis": {"range": [0, max(60, prediction * 1.2)]},
                "bar": {"color": COLORS["primary"]},
                "steps": [
                    {"range": [0, med_rate], "color": "#D8F3DC"},
                    {"range": [med_rate, avg_rate * 1.5], "color": "#FFF3CD"},
                    {"range": [avg_rate * 1.5, max(60, prediction * 1.2)], "color": "#F8D7DA"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 2},
                    "thickness": 0.75,
                    "value": avg_rate,
                },
            },
            title={"text": f"Moyenne globale: {avg_rate:.1f} | Mediane: {med_rate:.1f}"},
        ))
        fig.update_layout(height=300, margin=dict(t=60, b=20), paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def build_app() -> None:
    """Main Streamlit application with 4 tabs."""

    st.set_page_config(
        page_title=STREAMLIT_CONFIG["page_title"],
        layout=STREAMLIT_CONFIG["layout"],
        page_icon=STREAMLIT_CONFIG["page_icon"],
    )

    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    st.title("Prediction du taux de suicide")
    st.markdown('<p class="subtitle">Analyse et modelisation des taux de suicide mondiaux (1985-2016)</p>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📋 Contexte", "📊 Donnees", "🤖 Modeles", "🎯 Demo"]
    )

    with tab1:
        _tab_context()
    with tab2:
        _tab_data()
    with tab3:
        _tab_models()
    with tab4:
        _tab_demo()


if __name__ == "__main__":
    build_app()
