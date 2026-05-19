"""Streamlit application for suicide rate prediction PoC."""

from __future__ import annotations

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import joblib
from pathlib import Path

from config import (
    DATA_DIR,
    MODELS,
    MODELS_DIR,
    RESULTS_DIR,
    MODEL_METRICS_FILE,
    STREAMLIT_CONFIG,
)

# ═══════════════════════════════════════════════════════════════════════════════
# CSS — Styles globaux
# ═══════════════════════════════════════════════════════════════════════════════
STYLES_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ---------- Hide Streamlit chrome ---------- */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header[data-testid="stHeader"] {background: transparent;}
    [data-testid="stDeployButton"] {display: none;}

    /* ---------- Layout ---------- */
    .main .block-container {
        padding-top: 1.5rem;
        max-width: 1200px;
    }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1B4332 0%, #2D6A4F 100%);
        min-width: 260px;
    }
    section[data-testid="stSidebar"] .stRadio label {
        color: #D8F3DC !important;
        font-weight: 500;
    }
    section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover {
        background-color: rgba(255,255,255,0.08);
        border-radius: 8px;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #FFFFFF !important;
    }
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label {
        color: #D8F3DC !important;
    }

    /* ---------- Page header ---------- */
    .entete-page {
        background: linear-gradient(135deg, #1B4332, #2D6A4F, #40916C);
        color: white;
        padding: 28px 32px;
        border-radius: 16px;
        margin-bottom: 24px;
    }
    .entete-page h1 {
        color: white !important;
        background: none !important;
        -webkit-text-fill-color: white !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        margin: 0 0 4px 0 !important;
    }
    .entete-page p {
        color: #B7E4C7 !important;
        margin: 0;
        font-size: 1rem;
    }

    /* ---------- Card component ---------- */
    .carte {
        background: white;
        border: 1px solid #E8F5E9;
        border-radius: 14px;
        padding: 22px 24px;
        margin-bottom: 16px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.04);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .carte:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.08);
    }
    .carte h3 {
        color: #1B4332;
        font-size: 1.05rem;
        font-weight: 600;
        margin: 0 0 8px 0;
    }
    .carte p {
        color: #52796F;
        font-size: 0.92rem;
        margin: 0;
        line-height: 1.5;
    }

    /* ---------- KPI card ---------- */
    .carte-kpi {
        background: linear-gradient(135deg, #F0FFF4, #D8F3DC);
        border: 1px solid #B7E4C7;
        border-radius: 14px;
        padding: 18px 22px;
        text-align: center;
        transition: transform 0.2s ease;
    }
    .carte-kpi:hover {
        transform: translateY(-2px);
    }
    .carte-kpi .kpi-label {
        color: #52796F;
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 4px;
    }
    .carte-kpi .kpi-value {
        color: #1B4332;
        font-size: 1.7rem;
        font-weight: 700;
        line-height: 1.2;
    }

    /* ---------- Success box ---------- */
    .boite-succes {
        background: linear-gradient(135deg, #D8F3DC, #B7E4C7);
        border-left: 5px solid #2D6A4F;
        border-radius: 10px;
        padding: 18px 24px;
        margin: 16px 0;
    }
    .boite-succes strong { color: #1B4332; }

    /* ---------- Section title ---------- */
    .titre-section {
        color: #1B4332;
        font-size: 1.15rem;
        font-weight: 600;
        padding-bottom: 8px;
        border-bottom: 2px solid #D8F3DC;
        margin: 24px 0 16px 0;
    }

    /* ---------- Pipeline stepper ---------- */
    .pipeline-etapes {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0;
        margin: 20px 0;
        flex-wrap: wrap;
    }
    .pipeline-etape {
        background: linear-gradient(135deg, #2D6A4F, #40916C);
        color: white;
        padding: 10px 22px;
        border-radius: 24px;
        font-size: 0.85rem;
        font-weight: 600;
        white-space: nowrap;
    }
    .pipeline-fleche {
        color: #95D5B2;
        font-size: 1.4rem;
        margin: 0 6px;
    }

    /* ---------- Metric table ---------- */
    div[data-testid="stDataFrame"] {
        border: 1px solid #E8F5E9;
        border-radius: 10px;
        overflow: hidden;
    }

    /* ---------- Plotly chart rounding ---------- */
    .js-plotly-plot {
        border-radius: 12px;
        overflow: hidden;
    }

    /* ---------- Prediction result ---------- */
    .resultat-prediction {
        text-align: center;
        padding: 32px;
        background: linear-gradient(135deg, #D8F3DC, #B7E4C7);
        border-radius: 16px;
        border: 2px solid #52B788;
    }
    .resultat-prediction .valeur {
        color: #1B4332;
        font-size: 3rem;
        font-weight: 700;
        margin: 8px 0;
    }
    .resultat-prediction .label {
        color: #52796F;
        font-size: 0.95rem;
    }

    /* ---------- Badge / tag ---------- */
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .badge-vert { background: #D8F3DC; color: #1B4332; }
    .badge-bleu { background: #DBEAFE; color: #1E40AF; }
    .badge-jaune { background: #FEF3C7; color: #92400E; }
</style>
"""

# ═══════════════════════════════════════════════════════════════════════════════
# Palette de couleurs
# ═══════════════════════════════════════════════════════════════════════════════
COULEURS = {
    "primaire": "#2D6A4F",
    "secondaire": "#40916C",
    "accent": "#52B788",
    "clair": "#D8F3DC",
    "fonce": "#1B4332",
    "homme": "#3A86FF",
    "femme": "#FF006E",
    "palette": ["#2D6A4F", "#40916C", "#52B788", "#74C69D", "#95D5B2", "#B7E4C7"],
    "modeles": ["#3A86FF", "#FF006E", "#FFBE0B"],
}


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers d'affichage
# ═══════════════════════════════════════════════════════════════════════════════

def afficher_entete_page(titre: str, description: str) -> None:
    """Affiche l'en-tete de page avec style gradient."""
    st.markdown(
        f'<div class="entete-page">'
        f'<h1>{titre}</h1>'
        f'<p>{description}</p>'
        f'</div>',
        unsafe_allow_html=True,
    )


def afficher_titre_section(titre: str) -> None:
    """Affiche un titre de section avec bordure."""
    st.markdown(f'<div class="titre-section">{titre}</div>', unsafe_allow_html=True)


def html_carte(titre: str, contenu: str) -> str:
    """Retourne le HTML d'une carte."""
    return (
        f'<div class="carte">'
        f'<h3>{titre}</h3>'
        f'<p>{contenu}</p>'
        f'</div>'
    )


def html_carte_kpi(label: str, valeur: str) -> str:
    """Retourne le HTML d'une carte KPI."""
    return (
        f'<div class="carte-kpi">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{valeur}</div>'
        f'</div>'
    )


def html_pipeline_ml() -> str:
    """Retourne le HTML du pipeline ML en etapes."""
    etapes = [
        "Chargement CSV",
        "Nettoyage & Encodage",
        "StandardScaler",
        "Entrainement",
        "Evaluation",
    ]
    html = '<div class="pipeline-etapes">'
    for i, etape in enumerate(etapes):
        html += f'<div class="pipeline-etape">{etape}</div>'
        if i < len(etapes) - 1:
            html += '<span class="pipeline-fleche">&#10140;</span>'
    html += '</div>'
    return html


def appliquer_theme_graphique(fig: go.Figure, hauteur: int = 350) -> go.Figure:
    """Applique le theme visuel coherent aux graphiques Plotly."""
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(t=30, b=40, l=50, r=20),
        height=hauteur,
        font=dict(family="Inter, sans-serif", color="#0F172A"),
    )
    fig.update_xaxes(gridcolor="#E8F5E9", linecolor="#CBD5E1")
    fig.update_yaxes(gridcolor="#E8F5E9", linecolor="#CBD5E1")
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# Chargement des donnees & modeles (avec cache)
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def charger_donnees_brutes() -> pd.DataFrame:
    """Charge et renomme les colonnes du CSV brut."""
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


@st.cache_data
def charger_metriques() -> pd.DataFrame | None:
    """Charge les metriques des modeles si disponibles."""
    if MODEL_METRICS_FILE.exists():
        return pd.read_csv(MODEL_METRICS_FILE)
    return None


@st.cache_resource
def charger_modele(chemin: str):
    """Charge un modele serialise avec joblib."""
    p = Path(chemin)
    if p.exists():
        return joblib.load(p)
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# Page 1 : Dashboard
# ═══════════════════════════════════════════════════════════════════════════════

def page_dashboard() -> None:
    """Page d'accueil avec contexte du projet et pipeline ML."""

    afficher_entete_page(
        "Dashboard",
        "Vue d'ensemble du projet de prediction du taux de suicide",
    )

    # --- Contexte ---
    afficher_titre_section("Contexte du projet")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            html_carte(
                "Objectif",
                "Predire le <strong>taux de suicide pour 100 000 habitants</strong> "
                "d'un groupe demographique a partir de ses caracteristiques "
                "socio-economiques. Ce modele aide a identifier les populations "
                "a risque et a allouer les ressources de prevention.",
            ),
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            html_carte(
                "Approche",
                "<strong>Type :</strong> Regression supervisee<br>"
                "<strong>Target :</strong> suicides_per_100k<br>"
                "<strong>Split :</strong> 80% train / 20% test<br>"
                "<strong>Metriques :</strong> MAE, RMSE, R², MAPE",
            ),
            unsafe_allow_html=True,
        )

    # --- Pipeline ---
    afficher_titre_section("Pipeline ML")
    st.markdown(html_pipeline_ml(), unsafe_allow_html=True)

    # --- Sources de donnees ---
    afficher_titre_section("Sources de donnees")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            html_carte(
                "OMS",
                "Organisation Mondiale de la Sante — donnees de mortalite par suicide (1985-2016).",
            ),
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            html_carte(
                "Banque Mondiale",
                "PIB total et PIB par habitant pour chaque pays et annee.",
            ),
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            html_carte(
                "PNUD",
                "Indice de Developpement Humain (IDH) par pays et annee.",
            ),
            unsafe_allow_html=True,
        )

    # --- KPI rapides ---
    df = charger_donnees_brutes()
    afficher_titre_section("Apercu du dataset")

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.markdown(html_carte_kpi("Observations", f"{len(df):,}"), unsafe_allow_html=True)
    with k2:
        st.markdown(html_carte_kpi("Pays", str(df["country"].nunique())), unsafe_allow_html=True)
    with k3:
        st.markdown(html_carte_kpi("Periode", f"{df['year'].min()}-{df['year'].max()}"), unsafe_allow_html=True)
    with k4:
        st.markdown(html_carte_kpi("Taux moyen", f"{df['suicides_per_100k'].mean():.1f}"), unsafe_allow_html=True)
    with k5:
        st.markdown(html_carte_kpi("Taux median", f"{df['suicides_per_100k'].median():.1f}"), unsafe_allow_html=True)

    # --- Features ---
    afficher_titre_section("Features du modele")

    features_df = pd.DataFrame({
        "Feature": [
            "year", "sex", "age", "gdp_per_capita",
            "HDI_for_year", "population", "generation", "country_encoded",
        ],
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
        "Encodage": [
            "Numerique brut", "Binaire", "Ordinal", "Numerique brut",
            "Numerique brut", "Numerique brut", "Label encoding", "Target encoding",
        ],
    })
    st.dataframe(features_df, use_container_width=True, hide_index=True)

    # --- Description des modeles ---
    afficher_titre_section("Modeles utilises")

    m1, m2, m3 = st.columns(3)
    modeles_info = [
        ("Linear Regression", "Modele lineaire de base avec StandardScaler.", "badge-bleu"),
        ("Random Forest", "Ensemble de 200 arbres de decision (random_state=42).", "badge-vert"),
        ("Gradient Boosting", "200 estimateurs, learning rate 0.05, random_state=42.", "badge-jaune"),
    ]
    for col, (nom, desc, badge) in zip([m1, m2, m3], modeles_info):
        with col:
            st.markdown(
                f'<div class="carte">'
                f'<span class="badge {badge}">{nom}</span>'
                f'<p style="margin-top:10px;">{desc}</p>'
                f'</div>',
                unsafe_allow_html=True,
            )


# ═══════════════════════════════════════════════════════════════════════════════
# Page 2 : Analyse des donnees
# ═══════════════════════════════════════════════════════════════════════════════

def page_analyse() -> None:
    """Page d'exploration et visualisation des donnees."""

    afficher_entete_page(
        "Analyse des donnees",
        "Exploration et visualisations du dataset mondial de suicides",
    )

    df = charger_donnees_brutes()

    # --- Carte du monde ---
    afficher_titre_section("Carte mondiale du taux de suicide")

    pays_taux = df.groupby("country")["suicides_per_100k"].mean().reset_index()
    pays_taux.columns = ["country", "taux_moyen"]

    fig = px.choropleth(
        pays_taux,
        locations="country",
        locationmode="country names",
        color="taux_moyen",
        color_continuous_scale=["#D8F3DC", "#52B788", "#2D6A4F", "#1B4332"],
        hover_name="country",
        labels={"taux_moyen": "Suicides / 100k"},
    )
    fig.update_layout(
        geo=dict(
            showframe=False,
            showcoastlines=True,
            coastlinecolor="#CBD5E1",
            projection_type="natural earth",
            bgcolor="white",
            landcolor="#F1F5F9",
        ),
        paper_bgcolor="white",
        margin=dict(t=10, b=10, l=10, r=10),
        height=450,
        coloraxis_colorbar=dict(
            title="Taux moyen",
            thickness=15,
            len=0.6,
        ),
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Par sexe & age ---
    afficher_titre_section("Taux moyen par sexe et tranche d'age")

    age_order = [
        "5-14 years", "15-24 years", "25-34 years",
        "35-54 years", "55-74 years", "75+ years",
    ]
    sex_age = df.groupby(["sex", "age"])["suicides_per_100k"].mean().reset_index()
    sex_age["age"] = pd.Categorical(sex_age["age"], categories=age_order, ordered=True)
    sex_age = sex_age.sort_values("age")

    fig = px.bar(
        sex_age, x="age", y="suicides_per_100k",
        color="sex", barmode="group",
        color_discrete_map={"male": COULEURS["homme"], "female": COULEURS["femme"]},
    )
    fig.update_layout(
        xaxis_title="Tranche d'age",
        yaxis_title="Suicides pour 100k",
        legend=dict(title="Sexe", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(appliquer_theme_graphique(fig, 380), use_container_width=True)

    # --- Evolution temporelle ---
    afficher_titre_section("Evolution temporelle")

    col1, col2 = st.columns(2)

    with col1:
        yearly_sex = df.groupby(["year", "sex"])["suicides_per_100k"].mean().reset_index()
        fig = px.line(
            yearly_sex, x="year", y="suicides_per_100k",
            color="sex",
            color_discrete_map={"male": COULEURS["homme"], "female": COULEURS["femme"]},
            markers=True,
        )
        fig.update_layout(
            xaxis_title="Annee",
            yaxis_title="Suicides pour 100k",
            legend=dict(title="Sexe", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(appliquer_theme_graphique(fig, 370), use_container_width=True)

    with col2:
        yearly_total = df.groupby("year")["suicides_per_100k"].mean().reset_index()
        fig = px.area(
            yearly_total, x="year", y="suicides_per_100k",
            color_discrete_sequence=[COULEURS["accent"]],
        )
        fig.update_layout(
            xaxis_title="Annee",
            yaxis_title="Taux moyen global",
        )
        fig.update_traces(line=dict(width=2.5), fillcolor="rgba(82,183,136,0.15)")
        st.plotly_chart(appliquer_theme_graphique(fig, 370), use_container_width=True)

    # --- Top 15 pays ---
    afficher_titre_section("Top 15 pays par taux moyen")

    top = (
        df.groupby("country")["suicides_per_100k"]
        .mean()
        .sort_values(ascending=True)
        .tail(15)
        .reset_index()
    )
    fig = px.bar(
        top, x="suicides_per_100k", y="country",
        orientation="h",
        color="suicides_per_100k",
        color_continuous_scale=["#B7E4C7", "#2D6A4F", "#1B4332"],
    )
    fig.update_layout(
        xaxis_title="Suicides pour 100k",
        yaxis_title="",
        coloraxis_showscale=False,
        margin=dict(t=30, b=40, l=130, r=20),
    )
    st.plotly_chart(appliquer_theme_graphique(fig, 420), use_container_width=True)

    # --- Correlation HDI ---
    afficher_titre_section("Correlation IDH vs Taux de suicide")

    df_hdi = df.dropna(subset=["HDI_for_year"])
    hdi_country = df_hdi.groupby("country").agg(
        HDI=("HDI_for_year", "mean"),
        suicide_rate=("suicides_per_100k", "mean"),
        population=("population", "sum"),
    ).reset_index()

    fig = px.scatter(
        hdi_country, x="HDI", y="suicide_rate",
        size="population", hover_name="country",
        color="suicide_rate",
        color_continuous_scale=["#D8F3DC", "#2D6A4F", "#1B4332"],
        size_max=40,
    )
    fig.update_layout(
        xaxis_title="Indice de Developpement Humain (IDH)",
        yaxis_title="Taux moyen de suicide pour 100k",
        coloraxis_showscale=False,
    )
    st.plotly_chart(appliquer_theme_graphique(fig, 420), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Page 3 : Modeles
# ═══════════════════════════════════════════════════════════════════════════════

def page_modeles() -> None:
    """Page de comparaison des performances des modeles."""

    afficher_entete_page(
        "Performances des modeles",
        "Comparaison detaillee des 3 algorithmes de regression",
    )

    metrics_df = charger_metriques()
    if metrics_df is None:
        st.warning("Lancez `python scripts/main.py` pour generer les metriques.")
        return

    metric_cols = [c for c in metrics_df.columns if c not in ("model_key", "model_name", "model_path")]

    # --- Meilleur modele ---
    best_idx = metrics_df["r2"].idxmax()
    best_model = metrics_df.loc[best_idx, "model_name"]
    best_r2 = metrics_df.loc[best_idx, "r2"]
    best_row = metrics_df.loc[best_idx]

    st.markdown(
        f'<div class="boite-succes">'
        f'<strong>Meilleur modele :</strong> {best_model} &mdash; '
        f'R² = <strong>{best_r2:.4f}</strong>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # --- KPI cards ---
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(html_carte_kpi("MAE", f"{best_row['mae']:.2f}"), unsafe_allow_html=True)
    with k2:
        st.markdown(html_carte_kpi("RMSE", f"{best_row['rmse']:.2f}"), unsafe_allow_html=True)
    with k3:
        st.markdown(html_carte_kpi("R²", f"{best_row['r2']:.4f}"), unsafe_allow_html=True)
    with k4:
        st.markdown(html_carte_kpi("MAPE", f"{best_row['mape']:.1f}%"), unsafe_allow_html=True)

    # --- Tableau comparatif ---
    afficher_titre_section("Tableau comparatif")

    display_df = metrics_df[["model_name"] + metric_cols].copy()
    display_df.columns = ["Modele"] + metric_cols

    def _highlight_best(s: pd.Series) -> list[str]:
        if s.name == "r2":
            is_best = s == s.max()
        else:
            is_best = s == s.min()
        return [
            "background-color: #D8F3DC; font-weight: bold" if v else ""
            for v in is_best
        ]

    styled = display_df.style.apply(_highlight_best, subset=metric_cols)
    styled = styled.format({c: "{:.4f}" for c in metric_cols})
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # --- Radar chart ---
    afficher_titre_section("Vue radar des performances")

    radar_df = metrics_df[["model_name", "mae", "rmse", "r2", "mape"]].copy()
    for col in ["mae", "rmse", "mape"]:
        max_val = radar_df[col].max()
        radar_df[col + "_n"] = 1 - (radar_df[col] / max_val) if max_val > 0 else 1.0
    radar_df["r2_n"] = radar_df["r2"]

    categories = ["MAE (inv.)", "RMSE (inv.)", "R²", "MAPE (inv.)"]

    fig = go.Figure()
    for i, row in radar_df.iterrows():
        vals = [row["mae_n"], row["rmse_n"], row["r2_n"], row["mape_n"]]
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=categories + [categories[0]],
            fill="toself",
            name=row["model_name"],
            line_color=COULEURS["modeles"][i],
            opacity=0.7,
        ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], gridcolor="#E8F5E9"),
            bgcolor="white",
        ),
        showlegend=True,
        height=420,
        paper_bgcolor="white",
        margin=dict(t=40, b=40, l=60, r=60),
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Bar charts ---
    afficher_titre_section("Comparaison par metrique")

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(
            metrics_df, x="model_name", y="r2",
            color="model_name",
            color_discrete_sequence=COULEURS["modeles"],
            text_auto=".4f",
        )
        fig.update_layout(
            title="R² (plus haut = meilleur)",
            xaxis_title="", yaxis_title="R²",
            showlegend=False,
        )
        st.plotly_chart(appliquer_theme_graphique(fig, 320), use_container_width=True)

    with col2:
        fig = px.bar(
            metrics_df, x="model_name", y="mae",
            color="model_name",
            color_discrete_sequence=COULEURS["modeles"],
            text_auto=".2f",
        )
        fig.update_layout(
            title="MAE (plus bas = meilleur)",
            xaxis_title="", yaxis_title="MAE",
            showlegend=False,
        )
        st.plotly_chart(appliquer_theme_graphique(fig, 320), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Page 4 : Prediction
# ═══════════════════════════════════════════════════════════════════════════════

def page_prediction() -> None:
    """Page de demonstration interactive de prediction."""

    afficher_entete_page(
        "Prediction",
        "Predire le taux de suicide en ajustant les parametres socio-economiques",
    )

    # --- Selection du modele ---
    afficher_titre_section("Choix du modele")

    model_options = {cfg["name"]: key for key, cfg in MODELS.items()}
    selected_name = st.selectbox(
        "Modele", list(model_options.keys()), index=1, label_visibility="collapsed",
    )
    selected_key = model_options[selected_name]
    model = charger_modele(str(MODELS[selected_key]["path"]))

    if model is None:
        st.error(f"Modele introuvable : {MODELS[selected_key]['path']}")
        return

    # --- Parametres ---
    afficher_titre_section("Parametres d'entree")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(html_carte("Demographique", ""), unsafe_allow_html=True)
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

    with col2:
        st.markdown(html_carte("Economique", ""), unsafe_allow_html=True)
        gdp_per_capita = st.slider("PIB par habitant ($)", 100, 120_000, 15_000, step=500)
        hdi = st.slider("IDH", 0.30, 1.00, 0.75, step=0.01)
        population = st.number_input(
            "Population du groupe",
            min_value=100, max_value=50_000_000, value=500_000, step=10_000,
        )

    with col3:
        st.markdown(html_carte("Contextuel", ""), unsafe_allow_html=True)
        generation = st.slider(
            "Generation (encodee)", 0, 5, 2,
            help="0=Boomers, 1=Gen Alpha, 2=Gen X, 3=Gen Z, 4=Millenials, 5=Silent",
        )
        country_encoded = st.slider(
            "Encodage pays",
            0.0, 50.0, 12.0, step=0.5,
            help="Taux moyen de suicide du pays d'origine",
        )

    # --- Feature vector ---
    features = pd.DataFrame(
        [[year, sex_val, age_val, gdp_per_capita, hdi, population, generation, country_encoded]],
        columns=[
            "year", "sex", "age", "gdp_per_capita",
            "HDI_for_year", "population", "generation", "country_encoded",
        ],
    )

    # --- Prediction ---
    st.markdown("")
    col_btn, _ = st.columns([1, 3])
    with col_btn:
        predict = st.button("Lancer la prediction", type="primary", use_container_width=True)

    if predict:
        prediction = max(0.0, float(model.predict(features)[0]))

        st.markdown("---")

        # Resultat
        _, res_col, _ = st.columns([1, 2, 1])
        with res_col:
            st.markdown(
                f'<div class="resultat-prediction">'
                f'<div class="label">Taux de suicide predit</div>'
                f'<div class="valeur">{prediction:.2f}</div>'
                f'<div class="label">pour 100 000 habitants</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown("")

        with st.expander("Voir le vecteur de features"):
            st.dataframe(features, use_container_width=True, hide_index=True)

        # Gauge contextuelle
        afficher_titre_section("Contextualisation")

        df = charger_donnees_brutes()
        avg_rate = df["suicides_per_100k"].mean()
        med_rate = df["suicides_per_100k"].median()

        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=prediction,
            delta={"reference": avg_rate, "suffix": " vs moyenne"},
            gauge={
                "axis": {"range": [0, max(60, prediction * 1.2)]},
                "bar": {"color": COULEURS["primaire"]},
                "steps": [
                    {"range": [0, med_rate], "color": "#D8F3DC"},
                    {"range": [med_rate, avg_rate * 1.5], "color": "#FEF3C7"},
                    {"range": [avg_rate * 1.5, max(60, prediction * 1.2)], "color": "#FEE2E2"},
                ],
                "threshold": {
                    "line": {"color": "#EF4444", "width": 2},
                    "thickness": 0.75,
                    "value": avg_rate,
                },
            },
            title={"text": f"Moyenne: {avg_rate:.1f} | Mediane: {med_rate:.1f}"},
        ))
        fig.update_layout(height=300, margin=dict(t=60, b=20), paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Point d'entree — Navigation sidebar
# ═══════════════════════════════════════════════════════════════════════════════

def build_app() -> None:
    """Application Streamlit avec navigation sidebar."""

    st.set_page_config(
        page_title=STREAMLIT_CONFIG["page_title"],
        layout=STREAMLIT_CONFIG["layout"],
        page_icon=STREAMLIT_CONFIG["page_icon"],
    )

    st.markdown(STYLES_CSS, unsafe_allow_html=True)

    # --- Sidebar ---
    with st.sidebar:
        st.markdown("## Suicide Rate Prediction")
        st.markdown("*ML PoC — Regression*")
        st.markdown("---")

        page = st.radio(
            "Navigation",
            ["Dashboard", "Analyse", "Modeles", "Prediction"],
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown(
            '<p style="font-size:0.75rem; opacity:0.7;">'
            'Donnees OMS / Banque Mondiale<br>1985 – 2016</p>',
            unsafe_allow_html=True,
        )

    # --- Routing ---
    if page == "Dashboard":
        page_dashboard()
    elif page == "Analyse":
        page_analyse()
    elif page == "Modeles":
        page_modeles()
    elif page == "Prediction":
        page_prediction()


if __name__ == "__main__":
    build_app()
