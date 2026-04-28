import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import warnings
import io

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score, roc_curve, brier_score_loss, 
    classification_report, confusion_matrix
)

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────
# CONFIGURATION & STYLE
# ─────────────────────────────────────────────────────────────────
st.set_page_config(page_title="CIH Bank - Asset Liability Management", layout="wide")

CIH_LOGO_B64 = "..." # Gardez votre variable logo ici

st.markdown(f"""
<style>
    .stApp {{ background-color: #0d1117; color: #e6edf3; }}
    .main-header {{
        background: linear-gradient(90deg, #161b22, #1f6feb);
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #58a6ff;
        margin-bottom: 25px;
    }}
    .metric-container {{
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
    }}
</style>
<div class="main-header">
    <h1 style='margin:0; color:white;'>🏦 ALM Engine : Rachat Anticipé</h1>
    <p style='margin:0; color:#c9d1d9;'>Modélisation du risque de remboursement prématuré des crédits</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# LOGIQUE MÉTIER & PIPELINE
# ─────────────────────────────────────────────────────────────────

def feature_engineering(df):
    """Prétraitement enrichi pour le contexte marocain / bancaire"""
    df = df.copy()
    # Calcul des spreads de taux
    if 'Taux_credit' in df.columns and 'Taux_marche' in df.columns:
        df['Spread_Taux'] = df['Taux_credit'] - df['Taux_marche']
    
    # Calcul de la maturité résiduelle
    if 'CRD' in df.columns and 'Nominal' in df.columns:
        df['Ratio_Amortissement'] = df['CRD'] / df['Nominal']
        
    return df

@st.cache_resource
def train_professional_model(df):
    df = feature_engineering(df)
    
    # Identification automatique des colonnes
    target = 'Y'
    num_features = ['Nominal', 'CRD', 'Taux_credit', 'Anciennete', 'Taux_marche', 'Spread_Taux', 'Ratio_Amortissement']
    cat_features = ['Type_taux', 'Type_credit', 'Type_client']
    
    # Filtrage des colonnes présentes
    num_features = [f for f in num_features if f in df.columns]
    cat_features = [f for f in cat_features if f in df.columns]
    
    X = df[num_features + cat_features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # Pipeline de transformation
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_features),
            ('cat', categorical_transformer, cat_features)
        ])

    # Modèle avec équilibrage des classes (crucial en banque)
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(class_weight='balanced', C=0.5))
    ])
    
    model.fit(X_train, y_train)
    return model, X_test, y_test, num_features, cat_features

# ─────────────────────────────────────────────────────────────────
# INTERFACE UTILISATEUR
# ─────────────────────────────────────────────────────────────────

uploaded_file = st.sidebar.file_uploader("📁 Charger les données historiques (CSV)", type="csv")

if uploaded_file:
    raw_data = pd.read_csv(uploaded_file)
    model, X_test, y_test, num_f, cat_f = train_professional_model(raw_data)
    
    # Prédictions
    y_probs = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_probs)
    brier = brier_score_loss(y_test, y_probs)

    # Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Vue d'ensemble", "🔍 Analyse de Portefeuille", "⚙️ Performance Machine Learning"])

    with tab1:
        st.subheader("Indicateurs Stratégiques")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("AUC-ROC", f"{auc:.2%}")
        c2.metric("Score de Brier", f"{brier:.3f}", help="Plus proche de 0 est meilleur")
        c3.metric("Population Test", len(y_test))
        c4.metric("Taux de Rachat moyen", f"{y_test.mean():.1%}")

        # Graphique de répartition du risque
        fig_hist = px.histogram(y_probs, nbins=50, title="Distribution des Probabilités de Rachat",
                               labels={'value': 'Probabilité de Rachat'}, color_discrete_sequence=['#58a6ff'])
        st.plotly_chart(fig_hist, use_container_width=True)

    with tab2:
        st.subheader("Exploration du Portefeuille à Risque")
        results_df = X_test.copy()
        results_df['Probabilité_Rachat'] = y_probs
        results_df['Realité'] = y_test.values
        
        # Filtre interactif
        threshold = st.slider("Seuil de vigilance (%)", 0, 100, 70)
        high_risk = results_df[results_df['Probabilité_Rachat'] > (threshold/100)].sort_values(by='Probabilité_Rachat', ascending=False)
        
        st.write(f"Nombre de dossiers identifiés au dessus de {threshold}% : **{len(high_risk)}**")
        st.dataframe(high_risk.style.background_gradient(subset=['Probabilité_Rachat'], cmap='Reds'))
        
        # Export
        csv = high_risk.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Exporter la liste des clients à cibler", csv, "targeting_list.csv", "text/csv")

    with tab3:
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.markdown("**Interprétabilité (Coefficients)**")
            # Extraction des noms de variables après OneHotEncoding
            ohe_names = model.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(cat_f)
            feature_names = num_f + list(ohe_names)
            coefs = model.named_steps['classifier'].coef_[0]
            
            coef_df = pd.DataFrame({'Variable': feature_names, 'Impact': coefs}).sort_values(by='Impact')
            fig_coef = px.bar(coef_df, x='Impact', y='Variable', orientation='h', 
                             title="Impact des variables sur le rachat",
                             color='Impact', color_continuous_scale='RdBu_r')
            st.plotly_chart(fig_coef, use_container_width=True)

        with col_b:
            st.markdown("**Courbe ROC**")
            fpr, tpr, _ = roc_curve(y_test, y_probs)
            fig_roc = px.area(x=fpr, y=tpr, title=f"Courbe ROC (AUC={auc:.2f})",
                             labels={'x': 'Taux de Faux Positifs', 'y': 'Taux de Vrais Positifs'})
            fig_roc.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
            st.plotly_chart(fig_roc, use_container_width=True)

else:
    st.info("👋 Veuillez charger un fichier CSV dans la barre latérale pour activer l'analyse ALM.")
    st.write("Le fichier doit contenir au minimum : `Nominal`, `CRD`, `Taux_credit`, `Taux_marche` et la cible `Y` (1 pour rachat, 0 pour stable).")
