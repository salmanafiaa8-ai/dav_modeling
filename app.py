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
# CSS CUSTOM
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

  html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

  .stApp { background: #0d1117; color: #e6edf3; }

  /* Header principal */
  .hero-header {
    background: linear-gradient(135deg, #161b22 0%, #1c2128 50%, #161b22 100%);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
  }
  .hero-header::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #1f6feb, #58a6ff, #3fb950, #58a6ff, #1f6feb);
    background-size: 200% 100%;
    animation: shimmer 3s linear infinite;
  }
  @keyframes shimmer { 0%{background-position:200% 0} 100%{background-position:-200% 0} }
  .hero-title {
    font-size: 2rem; font-weight: 700; color: #58a6ff;
    font-family: 'IBM Plex Mono', monospace; margin: 0;
    letter-spacing: -0.5px;
  }
  .hero-sub { color: #8b949e; font-size: 0.95rem; margin-top: 0.3rem; }

  /* Metric cards */
  .metric-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    transition: border-color 0.2s;
  }
  .metric-card:hover { border-color: #58a6ff; }
  .metric-value { font-size: 2rem; font-weight: 700; font-family: 'IBM Plex Mono', monospace; }
  .metric-label { font-size: 0.78rem; color: #8b949e; margin-top: 0.2rem; text-transform: uppercase; letter-spacing: 0.05em; }
  .metric-blue  { color: #58a6ff; }
  .metric-green { color: #3fb950; }
  .metric-orange{ color: #f0883e; }
  .metric-red   { color: #f85149; }

  /* Prediction card */
  .pred-card-rachat {
    background: linear-gradient(135deg, #1f1009, #2d1a0e);
    border: 2px solid #f85149;
    border-radius: 12px; padding: 1.8rem; text-align: center;
  }
  .pred-card-stable {
    background: linear-gradient(135deg, #0d1f0d, #0f2a0f);
    border: 2px solid #3fb950;
    border-radius: 12px; padding: 1.8rem; text-align: center;
  }
  .pred-title { font-size: 1.5rem; font-weight: 700; margin-bottom: 0.3rem; }
  .pred-prob  { font-family: 'IBM Plex Mono', monospace; font-size: 3rem; font-weight: 700; }
  .pred-sub   { color: #8b949e; font-size: 0.85rem; }

  /* Section titles */
  .section-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem; font-weight: 600;
    color: #58a6ff; text-transform: uppercase; letter-spacing: 0.1em;
    border-bottom: 1px solid #30363d;
    padding-bottom: 0.5rem; margin-bottom: 1rem;
  }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: #161b22 !important;
    border-right: 1px solid #30363d;
  }

  /* Input labels */
  label { color: #c9d1d9 !important; font-size: 0.85rem !important; }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] { background: #161b22; border-radius: 8px; padding: 4px; }
  .stTabs [data-baseweb="tab"] { color: #8b949e !important; border-radius: 6px; }
  .stTabs [aria-selected="true"] { background: #1f6feb !important; color: #fff !important; }

  /* DataFrame */
  .stDataFrame { border: 1px solid #30363d; border-radius: 8px; }

  /* Buttons */
  .stButton button {
    background: linear-gradient(135deg, #1f6feb, #388bfd) !important;
    color: white !important; border: none !important;
    border-radius: 8px !important; font-weight: 600 !important;
    padding: 0.6rem 1.5rem !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
  }
  .stButton button:hover { opacity: 0.85 !important; }

  /* Info / warning boxes */
  .info-box {
    background: #161b22; border-left: 3px solid #58a6ff;
    border-radius: 0 8px 8px 0; padding: 0.8rem 1rem;
    font-size: 0.85rem; color: #8b949e; margin: 0.5rem 0;
  }

  /* Progress bar */
  .risk-bar-wrap { background: #21262d; border-radius: 100px; height: 10px; width: 100%; }
  .risk-bar { height: 10px; border-radius: 100px; transition: width 0.6s ease; }

  /* Table styling */
  .coef-table { width: 100%; border-collapse: collapse; font-size: 0.83rem; }
  .coef-table th {
    background: #21262d; color: #8b949e; padding: 8px 12px;
    text-align: left; font-weight: 600; text-transform: uppercase;
    font-size: 0.72rem; letter-spacing: 0.06em;
  }
  .coef-table td { padding: 7px 12px; border-bottom: 1px solid #21262d; color: #e6edf3; }
  .coef-table tr:hover td { background: #1c2128; }
  .badge-pos { background:#1a3a1a; color:#3fb950; padding:2px 8px; border-radius:100px; font-size:0.75rem; }
  .badge-neg { background:#1a1a3a; color:#58a6ff; padding:2px 8px; border-radius:100px; font-size:0.75rem; }
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

    # Cross-validation
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
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 1.5rem;'>
      <div style='font-family: IBM Plex Mono, monospace; font-size:1.3rem;
                  font-weight:700; color:#58a6ff;'>🏦 RachatML</div>
      <div style='color:#8b949e; font-size:0.78rem; margin-top:0.3rem;'>
        Modélisation des Rachats Anticipés
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">📂 Données</div>', unsafe_allow_html=True)
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
st.markdown("""
<div class="hero-header">
  <div class="hero-title">🏦 Rachat Anticipé — Scoring Dashboard</div>
  <div class="hero-sub">Régression Logistique · Feature Engineering · Prévision en temps réel</div>
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
with st.spinner(" Entraînement du modèle en cours..."):
    # On cache en fonction des paramètres
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
    " Vue d'ensemble",
    " Performance",
    " Interprétation",
    " Prédiction Client",
    " Données"
])


# ══════════════════════════════════════════════════════════════════
# TAB 1 — VUE D'ENSEMBLE
# ══════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-title"> Métriques Clés du Modèle</div>', unsafe_allow_html=True)

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
        st.markdown('<div class="section-title"> Dataset</div>', unsafe_allow_html=True)
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
        st.markdown('<div class="section-title"> Cross-Validation (5-Fold)</div>', unsafe_allow_html=True)
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
        st.markdown('<div class="section-title"> Calibration</div>', unsafe_allow_html=True)
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

    # Distribution des probabilités
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title"> Distribution des Probabilités Prédites</div>', unsafe_allow_html=True)

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

    # ROC curve
    with col_roc:
        st.markdown('<div class="section-title"> Courbe ROC</div>', unsafe_allow_html=True)
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

    # Confusion matrix
    with col_cm:
        st.markdown('<div class="section-title"> Matrice de Confusion</div>', unsafe_allow_html=True)
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

    # Précision-Rappel + Calibration
    col_pr, col_cal = st.columns(2)

    with col_pr:
        st.markdown('<div class="section-title"> Courbe Précision-Rappel</div>', unsafe_allow_html=True)
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
        st.markdown('<div class="section-title"> Courbe de Calibration</div>', unsafe_allow_html=True)
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

    # Rapport textuel
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title"> Rapport de Classification</div>', unsafe_allow_html=True)
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
        st.markdown('<div class="section-title"> Coefficients β (standardisés)</div>', unsafe_allow_html=True)
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
        st.markdown('<div class="section-title"> Odds Ratios (e^β)</div>', unsafe_allow_html=True)
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

    # Tableau détaillé
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title"> Tableau Complet des Coefficients</div>', unsafe_allow_html=True)
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
    st.markdown('<div class="section-title"> Évaluer un Nouveau Client</div>', unsafe_allow_html=True)
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

        submitted = st.form_submit_button(" Estimer la probabilité de rachat", use_container_width=True)

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

        # Résultat principal
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
            st.markdown('<div class="section-title"> Variables Dérivées</div>', unsafe_allow_html=True)
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

        # Explication métier
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

    # Scores du portefeuille
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title"> Scores du Jeu de Test</div>', unsafe_allow_html=True)

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

    # Export
    csv_buf = io.StringIO()
    scores_df.to_csv(csv_buf, index=False)
    st.download_button(
        "⬇️ Télécharger les scores (CSV)",
        data=csv_buf.getvalue(),
        file_name="scores_rachat_anticipe.csv",
        mime="text/csv"
    )
