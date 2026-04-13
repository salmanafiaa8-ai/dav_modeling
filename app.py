"""
╔══════════════════════════════════════════════════════════════════╗
║         RACHAT ANTICIPÉ — APPLICATION STREAMLIT                 ║
║         Modélisation par Régression Logistique                  ║
╚══════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score,
    accuracy_score, f1_score, precision_score, recall_score,
    brier_score_loss, log_loss
)
from sklearn.calibration import calibration_curve
import io

# ─────────────────────────────────────────────────────────────────
# CONFIG PAGE
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Rachat Anticipé — ML Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────
# LOGO CIH BANK EN BASE64 (intégré directement — aucun fichier externe requis)
# ─────────────────────────────────────────────────────────────────
CIH_LOGO_B64 = "/9j/4QAYRXhpZgAASUkqAAgAAAAAAAAAAAAAAP/sABFEdWNreQABAAQAAABQAAD/4QMtaHR0cDovL25zLmFkb2JlLmNvbS94YXAvMS4wLwA8P3hwYWNrZXQgYmVnaW49Iu+7vyIgaWQ9Ilc1TTBNcENlaGlIenJlU3pOVGN6a2M5ZCI/PiA8eDp4bXBtZXRhIHhtbG5zOng9ImFkb2JlOm5zOm1ldGEvIiB4OnhtcHRrPSJBZG9iZSBYTVAgQ29yZSA1LjMtYzAxMSA2Ni4xNDU2NjEsIDIwMTIvMDIvMDYtMTQ6NTY6MjcgICAgICAgICI+IDxyZGY6UkRGIHhtbG5zOnJkZj0iaHR0cDovL3d3dy53My5vcmcvMTk5OS8wMi8yMi1yZGYtc3ludGF4LW5zIyI+IDxyZGY6RGVzY3JpcHRpb24gcmRmOmFib3V0PSIiIHhtbG5zOnhtcD0iaHR0cDovL25zLmFkb2JlLmNvbS94YXAvMS4wLyIgeG1sbnM6eG1wTU09Imh0dHA6Ly9ucy5hZG9iZS5jb20veGFwLzEuMC9tbS8iIHhtbG5zOnN0UmVmPSJodHRwOi8vbnMuYWRvYmUuY29tL3hhcC8xLjAvc1R5cGUvUmVzb3VyY2VSZWYjIiB4bXA6Q3JlYXRvclRvb2w9IkFkb2JlIFBob3Rvc2hvcCBDUzYgKE1hY2ludG9zaCkiIHhtcE1NOkluc3RhbmNlSUQ9InhtcC5paWQ6NEI3RjcyM0VDRUM4MTFFODlCMDE4NjBDMDAwRjVGOUYiIHhtcE1NOkRvY3VtZW50SUQ9InhtcC5kaWQ6NEI3RjcyM0ZDRUM4MTFFODlCMDE4NjBDMDAwRjVGOUYiPiA8eG1wTU06RGVyaXZlZEZyb20gc3RSZWY6aW5zdGFuY2VJRD0ieG1wLmlpZDo0QjdGNzIzQ0NFQzgxMUU4OUIwMTg2MEMwMDBGNUY5RiIgc3RSZWY6ZG9jdW1lbnRJRD0ieG1wLmRpZDo0QjdGNzIzRENFQzgxMUU4OUIwMTg2MEMwMDBGNUY5RiIvPiA8L3JkZjpEZXNjcmlwdGlvbj4gPC9yZGY6UkRGPiA8L3g6eG1wbWV0YT4gPD94cGFja2V0IGVuZD0iciI/Pv/uAA5BZG9iZQBkwAAAAAH/2wCEAAICAgICAgICAgIDAgICAwQD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AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA//2Q=="

# ─────────────────────────────────────────────────────────────────
# CSS CUSTOM
# ─────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

  html, body, [class*="css"] {{ font-family: 'IBM Plex Sans', sans-serif; }}

  /* ── BACKGROUND : logo CIH Bank centré, plein écran, filigrane ── */
  .stApp {{
    background-color: #0d1117;
    color: #e6edf3;
  }}
  .stApp::before {{
    content: '';
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    background-image: url("data:image/jpeg;base64,{CIH_LOGO_B64}");
    background-repeat: no-repeat;
    background-position: center center;
    background-size: 70vw;
    opacity: 0.06;
    pointer-events: none;
    z-index: 0;
  }}

  /* Header principal */
  .hero-header {{
    background: linear-gradient(135deg, #161b22 0%, #1c2128 50%, #161b22 100%);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
  }}
  .hero-header::before {{
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #1f6feb, #58a6ff, #3fb950, #58a6ff, #1f6feb);
    background-size: 200% 100%;
    animation: shimmer 3s linear infinite;
  }}
  @keyframes shimmer {{ 0%{{background-position:200% 0}} 100%{{background-position:-200% 0}} }}
  .hero-title {{
    font-size: 2rem; font-weight: 700; color: #58a6ff;
    font-family: 'IBM Plex Mono', monospace; margin: 0;
    letter-spacing: -0.5px;
  }}
  .hero-sub {{ color: #8b949e; font-size: 0.95rem; margin-top: 0.3rem; }}

  /* Metric cards */
  .metric-card {{
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    transition: border-color 0.2s;
  }}
  .metric-card:hover {{ border-color: #58a6ff; }}
  .metric-value {{ font-size: 2rem; font-weight: 700; font-family: 'IBM Plex Mono', monospace; }}
  .metric-label {{ font-size: 0.78rem; color: #8b949e; margin-top: 0.2rem; text-transform: uppercase; letter-spacing: 0.05em; }}
  .metric-blue  {{ color: #58a6ff; }}
  .metric-green {{ color: #3fb950; }}
  .metric-orange{{ color: #f0883e; }}
  .metric-red   {{ color: #f85149; }}

  /* Prediction card */
  .pred-card-rachat {{
    background: linear-gradient(135deg, #1f1009, #2d1a0e);
    border: 2px solid #f85149;
    border-radius: 12px; padding: 1.8rem; text-align: center;
  }}
  .pred-card-stable {{
    background: linear-gradient(135deg, #0d1f0d, #0f2a0f);
    border: 2px solid #3fb950;
    border-radius: 12px; padding: 1.8rem; text-align: center;
  }}
  .pred-title {{ font-size: 1.5rem; font-weight: 700; margin-bottom: 0.3rem; }}
  .pred-prob  {{ font-family: 'IBM Plex Mono', monospace; font-size: 3rem; font-weight: 700; }}
  .pred-sub   {{ color: #8b949e; font-size: 0.85rem; }}

  /* Section titles */
  .section-title {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem; font-weight: 600;
    color: #58a6ff; text-transform: uppercase; letter-spacing: 0.1em;
    border-bottom: 1px solid #30363d;
    padding-bottom: 0.5rem; margin-bottom: 1rem;
  }}

  /* Sidebar */
  [data-testid="stSidebar"] {{
    background: #161b22 !important;
    border-right: 1px solid #30363d;
  }}

  /* Input labels */
  label {{ color: #c9d1d9 !important; font-size: 0.85rem !important; }}

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] {{ background: #161b22; border-radius: 8px; padding: 4px; }}
  .stTabs [data-baseweb="tab"] {{ color: #8b949e !important; border-radius: 6px; }}
  .stTabs [aria-selected="true"] {{ background: #1f6feb !important; color: #fff !important; }}

  /* DataFrame */
  .stDataFrame {{ border: 1px solid #30363d; border-radius: 8px; }}

  /* Buttons */
  .stButton button {{
    background: linear-gradient(135deg, #1f6feb, #388bfd) !important;
    color: white !important; border: none !important;
    border-radius: 8px !important; font-weight: 600 !important;
    padding: 0.6rem 1.5rem !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
  }}
  .stButton button:hover {{ opacity: 0.85 !important; }}

  /* Info / warning boxes */
  .info-box {{
    background: #161b22; border-left: 3px solid #58a6ff;
    border-radius: 0 8px 8px 0; padding: 0.8rem 1rem;
    font-size: 0.85rem; color: #8b949e; margin: 0.5rem 0;
  }}

  /* Progress bar */
  .risk-bar-wrap {{ background: #21262d; border-radius: 100px; height: 10px; width: 100%; }}
  .risk-bar {{ height: 10px; border-radius: 100px; transition: width 0.6s ease; }}

  /* Table styling */
  .coef-table {{ width: 100%; border-collapse: collapse; font-size: 0.83rem; }}
  .coef-table th {{
    background: #21262d; color: #8b949e; padding: 8px 12px;
    text-align: left; font-weight: 600; text-transform: uppercase;
    font-size: 0.72rem; letter-spacing: 0.06em;
  }}
  .coef-table td {{ padding: 7px 12px; border-bottom: 1px solid #21262d; color: #e6edf3; }}
  .coef-table tr:hover td {{ background: #1c2128; }}
  .badge-pos {{ background:#1a3a1a; color:#3fb950; padding:2px 8px; border-radius:100px; font-size:0.75rem; }}
  .badge-neg {{ background:#1a1a3a; color:#58a6ff; padding:2px 8px; border-radius:100px; font-size:0.75rem; }}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# FONCTIONS UTILITAIRES
# ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data(file):
    """Charge le CSV et retourne le DataFrame."""
    return pd.read_csv(file)


def feature_engineering(df, ref_date=None):
    """Applique le feature engineering métier."""
    df = df.copy()
    if ref_date is None:
        ref_date = pd.Timestamp('2023-12-31')

    for col in ['Date_octroi', 'Date_maturite']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    df['duree_contrat_mois']   = ((df['Date_maturite'] - df['Date_octroi']).dt.days / 30).round(1)
    df['age_contrat_mois']     = ((ref_date - df['Date_octroi']).dt.days / 30).round(1)
    df['duree_restante_mois']  = ((df['Date_maturite'] - ref_date).dt.days / 30).clip(lower=0).round(1)
    df['pct_vie_ecoulee']      = (df['age_contrat_mois'] / df['duree_contrat_mois']).clip(0, 1).round(4)
    df['ratio_crd_nominal']    = (df['CRD'] / df['Nominal']).round(4)
    df['diff_taux']            = (df['Taux_credit'] - df['Taux_marche']).round(4)
    df['economie_potentielle'] = (df['diff_taux'] * df['CRD']).round(2)
    df['penalite_relative']    = (df['Penalite'] / df['Taux_credit']).round(4)
    df['ratio_revenu_crd']     = (df['Revenu'] / df['CRD']).round(4)
    return df


FEATURE_COLS = [
    'Nominal', 'CRD', 'Taux_credit', 'Penalite', 'Revenu', 'Anciennete', 'Taux_marche',
    'duree_contrat_mois', 'age_contrat_mois', 'duree_restante_mois', 'pct_vie_ecoulee',
    'ratio_crd_nominal', 'diff_taux', 'economie_potentielle', 'penalite_relative', 'ratio_revenu_crd',
    'Type_taux', 'Type_credit', 'Type_client'
]
CAT_COLS = ['Type_taux', 'Type_credit', 'Type_client']


@st.cache_resource
def train_model(df_raw):
    """Entraîne le modèle et retourne tout ce dont on a besoin."""
    df = feature_engineering(df_raw)
    df_model = df[FEATURE_COLS + ['Y']].copy()
    df_enc = pd.get_dummies(df_model, columns=CAT_COLS, drop_first=True)

    X = df_enc.drop('Y', axis=1)
    y = df_enc['Y']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    model = LogisticRegression(
        C=1.0, penalty='l2', solver='lbfgs',
        max_iter=1000, class_weight='balanced', random_state=42
    )
    model.fit(X_train_sc, y_train)

    y_pred  = model.predict(X_test_sc)
    y_proba = model.predict_proba(X_test_sc)[:, 1]

    pipe_cv = Pipeline([('sc', StandardScaler()), ('lr', LogisticRegression(
        C=1.0, penalty='l2', solver='lbfgs', max_iter=1000,
        class_weight='balanced', random_state=42))])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_auc = cross_val_score(pipe_cv, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)

    metrics = {
        'accuracy':  accuracy_score(y_test, y_pred),
        'auc':       roc_auc_score(y_test, y_proba),
        'f1':        f1_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall':    recall_score(y_test, y_pred),
        'brier':     brier_score_loss(y_test, y_proba),
        'logloss':   log_loss(y_test, y_proba),
        'cv_auc_mean': cv_auc.mean(),
        'cv_auc_std':  cv_auc.std(),
    }

    return {
        'model':        model,
        'scaler':       scaler,
        'X_train':      X_train,
        'X_test':       X_test,
        'y_train':      y_train,
        'y_test':       y_test,
        'y_pred':       y_pred,
        'y_proba':      y_proba,
        'feature_names': list(X.columns),
        'X_columns':    list(X.columns),
        'metrics':      metrics,
    }


def predict_new_client(client_dict, model_bundle, ref_date=None):
    """Prédit la probabilité de rachat pour un nouveau client."""
    if ref_date is None:
        ref_date = pd.Timestamp('2024-06-30')

    df_nc = pd.DataFrame([client_dict])
    df_nc = feature_engineering(df_nc, ref_date=ref_date)

    nc_features = df_nc[FEATURE_COLS].copy()
    nc_enc = pd.get_dummies(nc_features, columns=CAT_COLS, drop_first=True)
    nc_aligned = nc_enc.reindex(columns=model_bundle['X_columns'], fill_value=0)
    nc_scaled  = model_bundle['scaler'].transform(nc_aligned)

    proba = model_bundle['model'].predict_proba(nc_scaled)[0, 1]
    pred  = int(proba >= 0.5)
    return proba, pred, df_nc.iloc[0]


def risk_color(p):
    if p > 0.7:  return '#f85149', '🔴 RISQUE ÉLEVÉ'
    if p > 0.5:  return '#f0883e', '🟠 RISQUE MOYEN'
    if p > 0.3:  return '#e3b341', '🟡 RISQUE FAIBLE'
    return '#3fb950', '🟢 STABLE'


# ─────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style='text-align:center; padding: 1rem 0 1.5rem;'>
      <img src="data:image/jpeg;base64,{CIH_LOGO_B64}"
           style="width:140px; margin-bottom:0.8rem; border-radius:8px;
                  background:#fff; padding:8px;" />
      <div style='font-family: IBM Plex Mono, monospace; font-size:1.1rem;
                  font-weight:700; color:#58a6ff;'>RachatML</div>
      <div style='color:#8b949e; font-size:0.78rem; margin-top:0.3rem;'>
        Modélisation des Rachats Anticipés
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title"> Données</div>', unsafe_allow_html=True)
    data_source = st.radio("Source des données", ["Utiliser le dataset fourni", "Uploader mon CSV"], label_visibility="collapsed")

    df_raw = None
    if data_source == "Uploader mon CSV":
        uploaded = st.file_uploader("Votre fichier CSV", type=['csv'])
        if uploaded:
            df_raw = load_data(uploaded)
            st.success(f"✓ {len(df_raw)} lignes chargées")
    else:
        try:
            df_raw = load_data("dataset_rachat_anticipe_1000.csv")
            st.success(f"✓ Dataset chargé — {len(df_raw)} clients")
        except FileNotFoundError:
            st.error("dataset_rachat_anticipe_1000.csv introuvable dans le dossier courant.")

    if df_raw is not None:
        st.markdown('<div class="section-title" style="margin-top:1.5rem">⚙️ Paramètres Modèle</div>', unsafe_allow_html=True)
        C_param    = st.select_slider("Régularisation C", options=[0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0], value=1.0)
        test_size  = st.slider("Taille jeu de test (%)", 10, 40, 20, step=5)
        threshold  = st.slider("Seuil de décision", 0.3, 0.8, 0.5, step=0.05,
                                help="Probabilité au-dessus de laquelle on prédit un rachat")

        st.markdown("---")
        st.markdown("""
        <div style='font-size:0.75rem; color:#484f58; text-align:center;'>
          Régression Logistique · sklearn · v1.0<br>
          <span style='color:#30363d'>─────────────────</span>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero-header">
  <div style="display:flex;align-items:center;gap:1.2rem;">
    <img src="data:image/jpeg;base64,{CIH_LOGO_B64}"
         style="height:52px;border-radius:6px;background:#fff;padding:5px;" />
    <div>
      <div class="hero-title">Rachat Anticipé — Scoring Dashboard</div>
      <div class="hero-sub">CIH Bank · Régression Logistique · Feature Engineering · Prévision en temps réel</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

if df_raw is None:
    st.markdown("""
    <div class="info-box">
       Chargez votre dataset dans la barre latérale pour commencer.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ─── Entraîner le modèle ─────────────────────────────────────────
with st.spinner("🔄 Entraînement du modèle en cours..."):
    @st.cache_resource
    def get_model(df_hash, c, ts):
        df = feature_engineering(df_raw)
        df_model = df[FEATURE_COLS + ['Y']].copy()
        df_enc = pd.get_dummies(df_model, columns=CAT_COLS, drop_first=True)
        X = df_enc.drop('Y', axis=1)
        y = df_enc['Y']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ts/100, random_state=42, stratify=y)
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc  = scaler.transform(X_test)
        model = LogisticRegression(C=c, penalty='l2', solver='lbfgs', max_iter=1000, class_weight='balanced', random_state=42)
        model.fit(X_train_sc, y_train)
        y_pred  = model.predict(X_test_sc)
        y_proba = model.predict_proba(X_test_sc)[:, 1]
        pipe_cv = Pipeline([('sc', StandardScaler()), ('lr', LogisticRegression(C=c, penalty='l2', solver='lbfgs', max_iter=1000, class_weight='balanced', random_state=42))])
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_auc = cross_val_score(pipe_cv, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred), 'auc': roc_auc_score(y_test, y_proba),
            'f1': f1_score(y_test, y_pred), 'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred), 'brier': brier_score_loss(y_test, y_proba),
            'logloss': log_loss(y_test, y_proba), 'cv_auc_mean': cv_auc.mean(), 'cv_auc_std': cv_auc.std(),
        }
        return {'model': model, 'scaler': scaler, 'X_train': X_train, 'X_test': X_test,
                'y_train': y_train, 'y_test': y_test, 'y_pred': y_pred, 'y_proba': y_proba,
                'feature_names': list(X.columns), 'X_columns': list(X.columns), 'metrics': metrics}

    bundle = get_model(len(df_raw), C_param, test_size)

m = bundle['metrics']

# ─────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Vue d'ensemble",
    "📈 Performance",
    "🔍 Interprétation",
    "🔮 Prédiction Client",
    "📋 Données"
])


# ══════════════════════════════════════════════════════════════════
# TAB 1 — VUE D'ENSEMBLE
# ══════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-title">📊 Métriques Clés du Modèle</div>', unsafe_allow_html=True)

    col1, col2, col3, col4, col5 = st.columns(5)
    cards = [
        (col1, f"{m['auc']:.4f}",      "AUC-ROC",       "metric-blue"),
        (col2, f"{m['accuracy']:.4f}", "Accuracy",       "metric-green"),
        (col3, f"{m['f1']:.4f}",       "F1-Score",       "metric-orange"),
        (col4, f"{m['precision']:.4f}","Précision",      "metric-blue"),
        (col5, f"{m['recall']:.4f}",   "Rappel (Recall)","metric-green"),
    ]
    for col, val, label, cls in cards:
        with col:
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-value {cls}">{val}</div>
              <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown('<div class="section-title">📦 Dataset</div>', unsafe_allow_html=True)
        n_rachat = int(df_raw['Y'].sum())
        n_total  = len(df_raw)
        st.markdown(f"""
        <div class="metric-card" style="text-align:left">
          <div style="margin-bottom:0.7rem">
            <span style="color:#8b949e;font-size:0.8rem">Total clients</span><br>
            <span style="font-size:1.4rem;font-weight:700;font-family:'IBM Plex Mono',monospace;color:#e6edf3">{n_total:,}</span>
          </div>
          <div style="margin-bottom:0.7rem">
            <span style="color:#8b949e;font-size:0.8rem">Rachats (Y=1)</span><br>
            <span style="font-size:1.4rem;font-weight:700;font-family:'IBM Plex Mono',monospace;color:#f85149">{n_rachat:,}</span>
            <span style="color:#8b949e;font-size:0.8rem"> ({n_rachat/n_total:.1%})</span>
          </div>
          <div>
            <span style="color:#8b949e;font-size:0.8rem">Stables (Y=0)</span><br>
            <span style="font-size:1.4rem;font-weight:700;font-family:'IBM Plex Mono',monospace;color:#3fb950">{n_total-n_rachat:,}</span>
            <span style="color:#8b949e;font-size:0.8rem"> ({(n_total-n_rachat)/n_total:.1%})</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="section-title">🔄 Cross-Validation (5-Fold)</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="metric-card" style="text-align:left">
          <div style="margin-bottom:0.7rem">
            <span style="color:#8b949e;font-size:0.8rem">AUC moyen</span><br>
            <span style="font-size:1.8rem;font-weight:700;font-family:'IBM Plex Mono',monospace;color:#58a6ff">{m['cv_auc_mean']:.4f}</span>
            <span style="color:#8b949e;font-size:0.85rem"> ± {m['cv_auc_std']:.4f}</span>
          </div>
          <div>
            <span style="color:#8b949e;font-size:0.8rem">Stabilité</span><br>
            <span style="font-size:1rem;font-weight:600;color:{'#3fb950' if m['cv_auc_std'] < 0.04 else '#f0883e'}">
              {'✓ Modèle stable' if m['cv_auc_std'] < 0.04 else '⚠ Variabilité élevée'}
            </span>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with col_c:
        st.markdown('<div class="section-title">📐 Calibration</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="metric-card" style="text-align:left">
          <div style="margin-bottom:0.7rem">
            <span style="color:#8b949e;font-size:0.8rem">Brier Score</span><br>
            <span style="font-size:1.4rem;font-weight:700;font-family:'IBM Plex Mono',monospace;color:#e3b341">{m['brier']:.4f}</span>
            <span style="color:#8b949e;font-size:0.8rem"> (0=parfait, 0.25=aléatoire)</span>
          </div>
          <div>
            <span style="color:#8b949e;font-size:0.8rem">Log-Loss</span><br>
            <span style="font-size:1.4rem;font-weight:700;font-family:'IBM Plex Mono',monospace;color:#e3b341">{m['logloss']:.4f}</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">📉 Distribution des Probabilités Prédites</div>', unsafe_allow_html=True)

    fig_dist, ax_dist = plt.subplots(figsize=(12, 3.5))
    fig_dist.patch.set_facecolor('#161b22')
    ax_dist.set_facecolor('#161b22')
    p_r = bundle['y_proba'][bundle['y_test'] == 1]
    p_s = bundle['y_proba'][bundle['y_test'] == 0]
    ax_dist.hist(p_s, bins=30, alpha=0.7, color='#1f6feb', label='Réel : Stable (0)', edgecolor='none')
    ax_dist.hist(p_r, bins=30, alpha=0.7, color='#f85149', label='Réel : Rachat (1)', edgecolor='none')
    ax_dist.axvline(threshold, color='#e3b341', linestyle='--', lw=1.5, label=f'Seuil = {threshold}')
    ax_dist.set_xlabel('P(rachat)', color='#8b949e'); ax_dist.set_ylabel('Effectif', color='#8b949e')
    ax_dist.tick_params(colors='#8b949e'); ax_dist.spines[:].set_color('#30363d')
    ax_dist.legend(fontsize=9, facecolor='#21262d', labelcolor='#e6edf3', edgecolor='#30363d')
    fig_dist.tight_layout()
    st.pyplot(fig_dist)
    plt.close(fig_dist)


# ══════════════════════════════════════════════════════════════════
# TAB 2 — PERFORMANCE
# ══════════════════════════════════════════════════════════════════
with tab2:
    col_roc, col_cm = st.columns([1.3, 1])

    with col_roc:
        st.markdown('<div class="section-title">📈 Courbe ROC</div>', unsafe_allow_html=True)
        fig_roc, ax_roc = plt.subplots(figsize=(6, 4.5))
        fig_roc.patch.set_facecolor('#161b22'); ax_roc.set_facecolor('#161b22')
        fpr, tpr, _ = roc_curve(bundle['y_test'], bundle['y_proba'])
        ax_roc.fill_between(fpr, tpr, alpha=0.12, color='#1f6feb')
        ax_roc.plot(fpr, tpr, color='#58a6ff', lw=2.5, label=f'AUC = {m["auc"]:.4f}')
        ax_roc.plot([0,1],[0,1],'--',color='#484f58',lw=1)
        ax_roc.set_xlabel('Faux Positifs', color='#8b949e')
        ax_roc.set_ylabel('Vrais Positifs', color='#8b949e')
        ax_roc.set_title('Courbe ROC', color='#e6edf3', fontsize=11)
        ax_roc.tick_params(colors='#8b949e'); ax_roc.spines[:].set_color('#30363d')
        ax_roc.legend(fontsize=10, facecolor='#21262d', labelcolor='#e6edf3', edgecolor='#30363d')
        fig_roc.tight_layout(); st.pyplot(fig_roc); plt.close(fig_roc)

    with col_cm:
        st.markdown('<div class="section-title">🔢 Matrice de Confusion</div>', unsafe_allow_html=True)
        y_pred_thresh = (bundle['y_proba'] >= threshold).astype(int)
        cm = confusion_matrix(bundle['y_test'], y_pred_thresh)
        fig_cm, ax_cm = plt.subplots(figsize=(5, 4.5))
        fig_cm.patch.set_facecolor('#161b22'); ax_cm.set_facecolor('#161b22')
        im = ax_cm.imshow(cm, cmap='Blues', vmin=0)
        labels_cm = [['VN','FP'],['FN','VP']]
        for (i,j), val in np.ndenumerate(cm):
            clr = 'white' if val > cm.max()*0.5 else '#e6edf3'
            ax_cm.text(j, i, f'{val}\n({labels_cm[i][j]})', ha='center', va='center',
                       fontsize=12, fontweight='bold', color=clr)
        ax_cm.set_xticks([0,1]); ax_cm.set_xticklabels(['Prédit : 0','Prédit : 1'], color='#8b949e')
        ax_cm.set_yticks([0,1]); ax_cm.set_yticklabels(['Réel : 0','Réel : 1'], color='#8b949e')
        ax_cm.set_title(f'Seuil = {threshold}', color='#e6edf3', fontsize=10)
        ax_cm.spines[:].set_color('#30363d')
        fig_cm.tight_layout(); st.pyplot(fig_cm); plt.close(fig_cm)

    col_pr, col_cal = st.columns(2)

    with col_pr:
        st.markdown('<div class="section-title">🎯 Courbe Précision-Rappel</div>', unsafe_allow_html=True)
        fig_pr, ax_pr = plt.subplots(figsize=(5.5, 3.5))
        fig_pr.patch.set_facecolor('#161b22'); ax_pr.set_facecolor('#161b22')
        prec_c, rec_c, _ = precision_recall_curve(bundle['y_test'], bundle['y_proba'])
        ap = average_precision_score(bundle['y_test'], bundle['y_proba'])
        ax_pr.fill_between(rec_c, prec_c, alpha=0.12, color='#f85149')
        ax_pr.plot(rec_c, prec_c, color='#f85149', lw=2, label=f'AP = {ap:.4f}')
        ax_pr.axhline(bundle['y_test'].mean(), color='#484f58', linestyle='--', lw=1)
        ax_pr.set_xlabel('Rappel', color='#8b949e'); ax_pr.set_ylabel('Précision', color='#8b949e')
        ax_pr.set_title('Précision-Rappel', color='#e6edf3')
        ax_pr.tick_params(colors='#8b949e'); ax_pr.spines[:].set_color('#30363d')
        ax_pr.legend(fontsize=9, facecolor='#21262d', labelcolor='#e6edf3', edgecolor='#30363d')
        fig_pr.tight_layout(); st.pyplot(fig_pr); plt.close(fig_pr)

    with col_cal:
        st.markdown('<div class="section-title">📐 Courbe de Calibration</div>', unsafe_allow_html=True)
        fig_cal, ax_cal = plt.subplots(figsize=(5.5, 3.5))
        fig_cal.patch.set_facecolor('#161b22'); ax_cal.set_facecolor('#161b22')
        frac_pos, mean_pred = calibration_curve(bundle['y_test'], bundle['y_proba'], n_bins=10)
        ax_cal.plot([0,1],[0,1],'--',color='#484f58',lw=1, label='Parfait')
        ax_cal.fill_between(mean_pred, frac_pos, mean_pred, alpha=0.15, color='#f0883e')
        ax_cal.plot(mean_pred, frac_pos, 's-', color='#f0883e', lw=2, markersize=5, label='Modèle')
        ax_cal.set_xlabel('Probabilité prédite', color='#8b949e')
        ax_cal.set_ylabel('Fraction positifs réels', color='#8b949e')
        ax_cal.set_title('Calibration des probabilités', color='#e6edf3')
        ax_cal.tick_params(colors='#8b949e'); ax_cal.spines[:].set_color('#30363d')
        ax_cal.legend(fontsize=9, facecolor='#21262d', labelcolor='#e6edf3', edgecolor='#30363d')
        fig_cal.tight_layout(); st.pyplot(fig_cal); plt.close(fig_cal)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">📄 Rapport de Classification</div>', unsafe_allow_html=True)
    report_str = classification_report(
        bundle['y_test'], y_pred_thresh,
        target_names=['Pas de rachat (0)', 'Rachat anticipé (1)']
    )
    st.code(report_str, language='text')


# ══════════════════════════════════════════════════════════════════
# TAB 3 — INTERPRÉTATION
# ══════════════════════════════════════════════════════════════════
with tab3:
    coef_s = pd.Series(bundle['model'].coef_[0], index=bundle['feature_names'])
    odds_r = np.exp(coef_s)

    col_coef, col_or = st.columns(2)

    with col_coef:
        st.markdown('<div class="section-title">📊 Coefficients β (standardisés)</div>', unsafe_allow_html=True)
        fig_coef, ax_coef = plt.subplots(figsize=(6, 7))
        fig_coef.patch.set_facecolor('#161b22'); ax_coef.set_facecolor('#161b22')
        coef_sorted = coef_s.sort_values()
        colors_coef = ['#f85149' if c > 0 else '#1f6feb' for c in coef_sorted]
        ax_coef.barh(range(len(coef_sorted)), coef_sorted.values, color=colors_coef, edgecolor='none', height=0.7)
        ax_coef.set_yticks(range(len(coef_sorted)))
        ax_coef.set_yticklabels(coef_sorted.index, fontsize=8, color='#8b949e')
        ax_coef.axvline(0, color='#484f58', lw=0.8)
        ax_coef.set_xlabel('β', color='#8b949e'); ax_coef.set_title('Coefficients', color='#e6edf3', fontsize=10)
        ax_coef.tick_params(colors='#8b949e', axis='x'); ax_coef.spines[:].set_color('#30363d')
        fig_coef.tight_layout(); st.pyplot(fig_coef); plt.close(fig_coef)

    with col_or:
        st.markdown('<div class="section-title">🎲 Odds Ratios (e^β)</div>', unsafe_allow_html=True)
        top_or = odds_r.sort_values(ascending=False).head(15).sort_values()
        fig_or, ax_or = plt.subplots(figsize=(6, 7))
        fig_or.patch.set_facecolor('#161b22'); ax_or.set_facecolor('#161b22')
        col_or_bar = ['#f85149' if v > 1 else '#1f6feb' for v in top_or]
        ax_or.barh(range(len(top_or)), top_or.values, color=col_or_bar, edgecolor='none', height=0.7)
        ax_or.set_yticks(range(len(top_or)))
        ax_or.set_yticklabels(top_or.index, fontsize=8, color='#8b949e')
        ax_or.axvline(1, color='#484f58', lw=0.8)
        ax_or.set_xlabel('Odds Ratio', color='#8b949e'); ax_or.set_title('Top 15 Odds Ratios', color='#e6edf3', fontsize=10)
        ax_or.tick_params(colors='#8b949e', axis='x'); ax_or.spines[:].set_color('#30363d')
        fig_or.tight_layout(); st.pyplot(fig_or); plt.close(fig_or)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">📋 Tableau Complet des Coefficients</div>', unsafe_allow_html=True)
    coef_df = pd.DataFrame({
        'Variable':        coef_s.index,
        'Coefficient β':   coef_s.values.round(4),
        'Odds Ratio':      odds_r.values.round(4),
        'Impact':          ['Favorise rachat ↑' if c > 0 else 'Réduit rachat ↓' for c in coef_s.values],
    }).sort_values('Coefficient β', ascending=False)
    st.dataframe(coef_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════
# TAB 4 — PRÉDICTION NOUVEAU CLIENT
# ══════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-title">🔮 Évaluer un Nouveau Client</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
      Renseignez les caractéristiques du client. Le modèle estimera immédiatement
      la probabilité de rachat anticipé.
    </div>
    """, unsafe_allow_html=True)

    with st.form("client_form"):
        st.markdown("**Informations du crédit**")
        c1, c2, c3 = st.columns(3)
        with c1:
            nominal      = st.number_input("Montant nominal (MAD)", 50000, 2000000, 300000, step=10000)
            taux_credit  = st.number_input("Taux du crédit (%)", 1.0, 15.0, 5.5, step=0.1, format="%.2f") / 100
            type_credit  = st.selectbox("Type de crédit", ['Immobilier', 'Conso', 'Professionnel'])
        with c2:
            crd          = st.number_input("Capital Restant Dû (MAD)", 10000, 1900000, 250000, step=10000)
            penalite     = st.number_input("Pénalité de rachat (%)", 0.5, 5.0, 2.0, step=0.1, format="%.2f") / 100
            type_taux    = st.selectbox("Type de taux", ['Fixe', 'Variable'])
        with c3:
            taux_marche  = st.number_input("Taux marché actuel (%)", 1.0, 10.0, 4.0, step=0.1, format="%.2f") / 100
            revenu       = st.number_input("Revenu mensuel (MAD)", 3000, 100000, 25000, step=1000)
            type_client  = st.selectbox("Type de client", ['Physique', 'Morale'])

        st.markdown("**Durée & Ancienneté**")
        d1, d2, d3 = st.columns(3)
        with d1:
            date_octroi   = st.date_input("Date d'octroi", value=pd.Timestamp('2019-01-01'))
        with d2:
            date_maturite = st.date_input("Date de maturité", value=pd.Timestamp('2034-01-01'))
        with d3:
            anciennete    = st.number_input("Ancienneté client (années)", 0, 30, 5)

        submitted = st.form_submit_button("🔮 Estimer la probabilité de rachat", use_container_width=True)

    if submitted:
        client = {
            'Nominal':       nominal,
            'CRD':           crd,
            'Taux_credit':   taux_credit,
            'Penalite':      penalite,
            'Revenu':        revenu,
            'Anciennete':    anciennete,
            'Taux_marche':   taux_marche,
            'Type_taux':     type_taux,
            'Type_credit':   type_credit,
            'Type_client':   type_client,
            'Date_octroi':   pd.Timestamp(date_octroi),
            'Date_maturite': pd.Timestamp(date_maturite),
        }

        proba, pred, derived = predict_new_client(client, bundle)
        color, risk_label = risk_color(proba)

        st.markdown("<br>", unsafe_allow_html=True)

        card_class = "pred-card-rachat" if pred == 1 else "pred-card-stable"
        icon        = "🔴" if pred == 1 else "🟢"
        verdict     = "RACHAT ANTICIPÉ PROBABLE" if pred == 1 else "CLIENT STABLE"
        verdict_col = "#f85149" if pred == 1 else "#3fb950"

        col_res, col_detail = st.columns([1, 1.5])
        with col_res:
            st.markdown(f"""
            <div class="{card_class}">
              <div class="pred-title" style="color:{verdict_col}">{icon} {verdict}</div>
              <div class="pred-prob" style="color:{color}">{proba:.1%}</div>
              <div class="pred-sub">Probabilité de rachat anticipé</div>
              <br>
              <div style="font-size:0.85rem;color:#8b949e">
                Seuil appliqué : <strong style="color:#e3b341">{threshold:.2f}</strong>
              </div>
              <div style="margin-top:0.8rem">
                <div class="risk-bar-wrap">
                  <div class="risk-bar" style="width:{proba*100:.1f}%;background:{color}"></div>
                </div>
              </div>
              <div style="font-size:0.9rem;margin-top:0.6rem;color:{color};font-weight:600">{risk_label}</div>
            </div>
            """, unsafe_allow_html=True)

        with col_detail:
            st.markdown('<div class="section-title">📊 Variables Dérivées</div>', unsafe_allow_html=True)
            diff = taux_credit - taux_marche
            eco  = diff * crd
            items = [
                ("Différentiel de taux", f"{diff:+.2%}", "#f85149" if diff > 0 else "#3fb950"),
                ("Économie potentielle", f"{eco:,.0f} MAD", "#f85149" if eco > 0 else "#3fb950"),
                ("% Vie écoulée", f"{float(derived.get('pct_vie_ecoulee', 0)):.1%}", "#e3b341"),
                ("Durée restante", f"{float(derived.get('duree_restante_mois', 0)):.0f} mois", "#58a6ff"),
                ("Ratio CRD/Nominal", f"{crd/nominal:.2%}", "#8b949e"),
                ("Pénalité relative", f"{penalite/taux_credit:.2%}", "#8b949e"),
            ]
            for label, val, clr in items:
                st.markdown(f"""
                <div style="display:flex;justify-content:space-between;align-items:center;
                     padding:6px 0;border-bottom:1px solid #21262d;">
                  <span style="color:#8b949e;font-size:0.85rem">{label}</span>
                  <span style="color:{clr};font-weight:600;font-family:'IBM Plex Mono',monospace;font-size:0.9rem">{val}</span>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">💡 Analyse Métier</div>', unsafe_allow_html=True)
        if diff > 0:
            st.success(f"✅ Le taux contractuel ({taux_credit:.2%}) est **supérieur** au taux du marché ({taux_marche:.2%}). Le client économiserait **{eco:,.0f} MAD** en rachetant son crédit → **forte incitation financière**.")
        else:
            st.info(f"ℹ️ Le taux contractuel ({taux_credit:.2%}) est **inférieur ou égal** au taux du marché ({taux_marche:.2%}). Pas d'incitation financière directe au rachat.")
        if proba > 0.5:
            st.warning(f"⚠️ Probabilité de rachat : **{proba:.1%}** → Action recommandée : contacter le client pour une proposition de renégociation.")


# ══════════════════════════════════════════════════════════════════
# TAB 5 — DONNÉES
# ══════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-title">📋 Aperçu du Dataset</div>', unsafe_allow_html=True)

    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        filter_y = st.multiselect("Filtrer par Y (cible)", [0, 1], default=[0, 1])
    with col_f2:
        filter_type = st.multiselect("Type de crédit", df_raw['Type_credit'].unique().tolist(), default=df_raw['Type_credit'].unique().tolist())
    with col_f3:
        filter_taux = st.multiselect("Type de taux", df_raw['Type_taux'].unique().tolist(), default=df_raw['Type_taux'].unique().tolist())

    df_filtered = df_raw[
        df_raw['Y'].isin(filter_y) &
        df_raw['Type_credit'].isin(filter_type) &
        df_raw['Type_taux'].isin(filter_taux)
    ]
    st.markdown(f"<div class='info-box'>🔎 {len(df_filtered):,} clients affichés sur {len(df_raw):,}</div>", unsafe_allow_html=True)
    st.dataframe(df_filtered, use_container_width=True, height=400)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">🎯 Scores du Jeu de Test</div>', unsafe_allow_html=True)

    scores_df = bundle['X_test'].copy()
    scores_df['Y_réel']   = bundle['y_test'].values
    scores_df['P_rachat'] = bundle['y_proba'].round(4)
    scores_df['Prédiction'] = (bundle['y_proba'] >= threshold).astype(int)
    scores_df['Segment'] = pd.cut(
        scores_df['P_rachat'],
        bins=[0, 0.3, 0.5, 0.7, 1.0],
        labels=['Faible', 'Moyen', 'Élevé', 'Très élevé']
    )
    st.dataframe(scores_df[['Y_réel','P_rachat','Prédiction','Segment']].sort_values('P_rachat', ascending=False),
                 use_container_width=True, height=350)

    csv_buf = io.StringIO()
    scores_df.to_csv(csv_buf, index=False)
    st.download_button(
        "⬇️ Télécharger les scores (CSV)",
        data=csv_buf.getvalue(),
        file_name="scores_rachat_anticipe.csv",
        mime="text/csv"
    )
