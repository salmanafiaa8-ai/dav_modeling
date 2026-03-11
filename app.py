import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.graph_objects as go
from st_aggrid import AgGrid, GridOptionsBuilder

# ==============================
# Page config
# ==============================
st.set_page_config(page_title="Terminal ALM DAV", layout="wide")

primary_color = "#0055A4"
secondary_color = "#C8102E"
accent_color = "#FFD700"

# ==============================
# Header
# ==============================
st.markdown(
    f"<h1 style='color:{primary_color}; text-align:center'>📊 Terminal ALM - DAV</h1>",
    unsafe_allow_html=True
)
st.markdown(
    f"<h4 style='color:{secondary_color}; text-align:center'>CIH Bank - Gestion Actif-Passif</h4>",
    unsafe_allow_html=True
)
st.markdown("---")

# ==============================
# Sidebar - Navigation
# ==============================
st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Choisir la page",
    ["Import & Préparation", "Estimation & Simulation", "Dashboard interactif"]
)

# ==============================
# Nettoyage des données
# ==============================
def clean_regression_data(X, y):
    data = pd.concat([y, X], axis=1)
    data = data.apply(pd.to_numeric, errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna()
    y_clean = data.iloc[:, 0]
    X_clean = data.iloc[:, 1:]
    return X_clean, y_clean

# ==============================
# Page 1: Import & Préparation
# ==============================
if page == "Import & Préparation":
    st.header("1️⃣ Import et préparation des données DAV")
    
    dav_type = st.radio(
        "Type de dépôt",
        ["Compte courant / Chèque", "Compte épargne"]
    )
    st.session_state['dav_type'] = dav_type
    
    file = st.file_uploader("Uploader CSV ou Excel", type=["csv","xlsx"])
    
    if file is not None:
        if file.name.endswith(".csv"):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
        
        df.columns = df.columns.str.strip().str.lower()
        required_cols = ["date","dk","rk"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            st.error(f"Colonnes manquantes : {missing}")
            st.stop()
        
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        df["dk"] = pd.to_numeric(df["dk"].astype(str).str.replace(" ","").str.replace(",", "."), errors="coerce")
        df["rk"] = pd.to_numeric(df["rk"], errors="coerce")
        if "ik" in df.columns:
            df["ik"] = pd.to_numeric(df["ik"], errors="coerce")
        df = df.dropna(subset=["dk","rk"])
        
        # Préparation
        df = df.sort_values("date").reset_index(drop=True)
        df = df[df["dk"]>0]
        df["logDk"] = np.log(df["dk"])
        df["logDk_lag"] = df["logDk"].shift(1)
        df["Rk"] = df["rk"]
        df["Rk_lag"] = df["Rk"].shift(1)
        df["dRk"] = df["Rk"] - df["Rk_lag"]
        df["trend"] = np.arange(len(df))
        if dav_type == "Compte épargne" and "ik" in df.columns:
            df["spread"] = df["Rk"] - df["ik"]
        df = df.replace([np.inf,-np.inf],np.nan)
        df = df.dropna()
        
        st.session_state['df'] = df
        st.success("Données prêtes ✅")
        st.dataframe(df.head())

# ==============================
# Page 2: Estimation & Simulation
# ==============================
elif page == "Estimation & Simulation":
    st.header("2️⃣ Estimation des modèles et simulations")
    
    if 'df' not in st.session_state:
        st.warning("Importez d'abord les données à la page précédente.")
        st.stop()
    
    df = st.session_state['df']
    dav_type = st.session_state['dav_type']
    
    # Sélection des modèles
    models = st.multiselect(
        "Choisir les modèles à estimer",
        ["Selvaggio", "Dupre", "Jarrow-Van Deventer", "OBrien", "OTS"],
        default=["Selvaggio","Dupre"]
    )
    
    if st.button("Estimer modèles"):
        results = {}
        
        # Selvaggio
        if "Selvaggio" in models:
            X = sm.add_constant(df[["logDk_lag","Rk","trend"]])
            y = df["logDk"]
            X, y = clean_regression_data(X,y)
            if len(X)>5:
                results["Selvaggio"] = sm.OLS(y,X).fit()
        
        # Dupre
        if "Dupre" in models:
            df["delta_logD"] = df["logDk"] - df["logDk_lag"]
            X = sm.add_constant(df[["Rk"]])
            y = df["delta_logD"]
            X, y = clean_regression_data(X,y)
            if len(X)>5:
                results["Dupre"] = sm.OLS(y,X).fit()
        
        st.session_state['results'] = results
        
        if len(results)==0:
            st.warning("Aucun modèle n'a pu être estimé.")
        else:
            st.success("Estimation terminée ✅")
            for name, model in results.items():
                st.subheader(name)
                st.text(model.summary())

# ==============================
# Page 3: Dashboard interactif
# ==============================
elif page == "Dashboard interactif":
    st.header("3️⃣ Dashboard interactif")
    
    if 'df' not in st.session_state:
        st.warning("Importez d'abord les données à la page 1.")
        st.stop()
    
    df = st.session_state['df']
    dav_type = st.session_state['dav_type']
    results = st.session_state.get('results', {})
    
    # ------------------------
    # KPIs
    # ------------------------
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Encours total DAV", f"{df['dk'].sum():,.0f} MAD", delta=f"{df['dk'].pct_change().iloc[-1]*100:.2f}%")
    col2.metric("Encours moyen", f"{df['dk'].mean():,.0f} MAD")
    col3.metric("Taux moyen", f"{df['Rk'].mean()*100:.2f} %")
    if "spread" in df.columns:
        col4.metric("Spread moyen", f"{df['spread'].mean():.2f} %")
    
    st.markdown("---")
    
    # ------------------------
    # Filtre interactif par période
    # ------------------------
    start_date, end_date = st.select_slider(
        "Sélectionnez la période",
        options=df['date'].sort_values(),
        value=(df['date'].min(), df['date'].max())
    )
    df_filtered = df[(df['date']>=start_date) & (df['date']<=end_date)]
    
    # ------------------------
    # Graphiques interactifs
    # ------------------------
    st.subheader("Évolution des dépôts et taux")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_filtered['date'], y=df_filtered['dk'], mode='lines+markers', name='Encours'))
    fig.add_trace(go.Scatter(x=df_filtered['date'], y=df_filtered['Rk'], mode='lines+markers', name='Taux'))
    if "spread" in df_filtered.columns:
        fig.add_trace(go.Scatter(x=df_filtered['date'], y=df_filtered['spread'], mode='lines+markers', name='Spread'))
    fig.update_layout(title="DAV et Taux", xaxis_title="Date", yaxis_title="Montant / %")
    st.plotly_chart(fig, use_container_width=True)
    
    # ------------------------
    # Table interactive type bourse
    # ------------------------
    st.subheader("Tableau type Bourse")
    df_table = df_filtered.copy()
    df_table['delta_month'] = df_table['dk'].pct_change()*100
    df_table['delta_month'] = df_table['delta_month'].round(2)
    
    gb = GridOptionsBuilder.from_dataframe(df_table)
    gb.configure_default_column(filterable=True, sortable=True)
    gb.configure_columns(['delta_month'], cellStyle={'color': 'black', 'backgroundColor': '#FFD700'})
    AgGrid(df_table, gridOptions=gb.build(), height=300)
    
    # ------------------------
    # Comparaison des modèles (si estimés)
    # ------------------------
    if results:
        st.subheader("Comparaison des modèles")
        comparison = pd.DataFrame({
            "Modèle": list(results.keys()),
            "R2": [m.rsquared for m in results.values()],
            "AIC": [m.aic for m in results.values()]
        })
        st.dataframe(comparison)
        fig2 = go.Figure()
        fig2.add_bar(x=comparison["Modèle"], y=comparison["R2"], name="R2", marker_color=primary_color)
        fig2.add_bar(x=comparison["Modèle"], y=comparison["AIC"], name="AIC", marker_color=secondary_color)
        fig2.update_layout(title="Comparaison modèles", barmode="group")
        st.plotly_chart(fig2)
