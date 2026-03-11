import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.graph_objects as go

# ==============================
# Page config
# ==============================
st.set_page_config(page_title="Plateforme ALM DAV", layout="wide")

# ==============================
# Branding couleurs CIH
# ==============================
primary_color = "#0055A4"
secondary_color = "#C8102E"

# ==============================
# Header
# ==============================
st.image("cih_logo.jpg", width=200)

st.markdown(
f"<h1 style='color:{primary_color}; text-align:center'>Plateforme - Modélisation DAV</h1>",
unsafe_allow_html=True
)

st.markdown(
f"<h4 style='color:{secondary_color}; text-align:center'>CIH Bank - ALM</h4>",
unsafe_allow_html=True
)

# ==============================
# Sidebar
# ==============================
st.sidebar.image("cih_logo.jpg", width=150)

page = st.sidebar.radio(
"Navigation",
[
"1️⃣ Import données",
"2️⃣ Préparation",
"3️⃣ Estimation",
"4️⃣ Comparaison",
"5️⃣ Visualisation"
]
)

# ==============================
# 1️⃣ Import données
# ==============================
if page == "1️⃣ Import données":

    st.header("Import des données")

    with st.expander("Uploader les données DAV"):

        dav_type = st.radio(
        "Type de dépôt",
        ["Compte courant / Chèque","Compte épargne"]
        )

        st.session_state.dav_type = dav_type

        file = st.file_uploader(
        "Uploader fichier CSV ou Excel",
        type=["csv","xlsx"]
        )

        if file is not None:

            if file.name.endswith(".csv"):
                df = pd.read_csv(file)
            else:
                df = pd.read_excel(file)

            df.columns = df.columns.str.strip().str.lower()

            # Vérification colonnes
            required_cols = ["date","dk","rk"]

            for col in required_cols:
                if col not in df.columns:
                    st.error(f"❌ La colonne {col} est obligatoire")

            df["date"] = pd.to_datetime(df["date"], dayfirst=True)

            df["dk"] = (
            df["dk"]
            .astype(str)
            .str.replace(" ","")
            .str.replace(",",".")
            .astype(float)
            )

            df["rk"] = pd.to_numeric(df["rk"], errors="coerce")

            if "ik" in df.columns:
                df["ik"] = pd.to_numeric(df["ik"], errors="coerce")

            st.success("Données importées avec succès")

            st.dataframe(df.head())

            st.session_state.df_raw = df

# ==============================
# 2️⃣ Préparation
# ==============================
if page == "2️⃣ Préparation":

    if "df_raw" in st.session_state:

        df = st.session_state.df_raw.copy()
        dav_type = st.session_state.dav_type

        st.header("Préparation des variables")

        df = df.sort_values("date").reset_index(drop=True)

        df["logDk"] = np.log(df["dk"])

        df["logDk_lag"] = df["logDk"].shift(1)

        df["Rk"] = df["rk"]

        df["Rk_lag"] = df["Rk"].shift(1)

        df["dRk"] = df["Rk"] - df["Rk_lag"]

        df["trend"] = np.arange(len(df))

        if dav_type == "Compte épargne" and "ik" in df.columns:
            df["spread"] = df["Rk"] - df["ik"]

        df = df.dropna()

        st.success("Variables préparées")

        st.dataframe(df.head())

        st.session_state.df = df

# ==============================
# 3️⃣ Estimation
# ==============================
if page == "3️⃣ Estimation":

    if "df" in st.session_state:

        df = st.session_state.df
        dav_type = st.session_state.dav_type

        st.header("Estimation des modèles")

        models = st.multiselect(
        "Choisir les modèles",
        ["Selvaggio","Dupre","Jarrow-Van Deventer","OBrien","OTS"],
        default=["Selvaggio","Dupre"]
        )

        if st.button("Lancer estimation"):

            results = {}

            if "Selvaggio" in models:

                X = sm.add_constant(df[["logDk_lag","Rk","trend"]])
                y = df["logDk"]

                results["Selvaggio"] = sm.OLS(y,X).fit()

            if "Dupre" in models:

                df["delta_logD"] = df["logDk_lag"] - df["logDk"]

                X = sm.add_constant(df["Rk"])
                y = df["delta_logD"]

                results["Dupre"] = sm.OLS(y,X).fit()

            if "Jarrow-Van Deventer" in models:

                X = sm.add_constant(df[["logDk_lag","Rk","dRk","trend"]])
                y = df["logDk"]

                results["JVD"] = sm.OLS(y,X).fit()

            if "OBrien" in models and dav_type == "Compte épargne":

                X = sm.add_constant(df[["logDk_lag","spread","trend"]])
                y = df["logDk"]

                results["OBrien"] = sm.OLS(y,X).fit()

            if "OTS" in models:

                X = sm.add_constant(df["dk"].shift(1))
                y = df["dk"]

                results["OTS"] = sm.OLS(y.dropna(),X.dropna()).fit()

            st.session_state.results = results

            st.success("Estimation terminée")

# ==============================
# 4️⃣ Comparaison
# ==============================
if page == "4️⃣ Comparaison":

    if "results" in st.session_state:

        results = st.session_state.results

        comparison = pd.DataFrame({

        "Model":[k for k in results],

        "R2":[m.rsquared for m in results.values()],

        "AIC":[m.aic for m in results.values()]

        })

        st.header("Comparaison des modèles")

        st.dataframe(comparison)

        fig = go.Figure()

        fig.add_bar(
        x=comparison["Model"],
        y=comparison["R2"],
        name="R2",
        marker_color=primary_color
        )

        fig.add_bar(
        x=comparison["Model"],
        y=comparison["AIC"],
        name="AIC",
        marker_color=secondary_color
        )

        fig.update_layout(
        title="Comparaison modèles",
        barmode="group"
        )

        st.plotly_chart(fig)

# ==============================
# 5️⃣ Visualisation
# ==============================
if page == "5️⃣ Visualisation":

    if "df" in st.session_state:

        df = st.session_state.df
        dav_type = st.session_state.dav_type

        st.header("Visualisation des séries")

        options = df.columns.tolist()

        default_vars = ["dk","Rk"] if dav_type=="Compte courant / Chèque" else ["dk","Rk","ik"]

        selected = st.multiselect(
        "Variables à afficher",
        options,
        default=default_vars
        )

        if selected:

            fig = go.Figure()

            for col in selected:

                fig.add_trace(
                go.Scatter(
                x=df["date"],
                y=df[col],
                mode="lines",
                name=col
                )
                )

            st.plotly_chart(fig)

        csv = df.to_csv(index=False).encode("utf-8")

        st.download_button(
        "Télécharger les données préparées",
        data=csv,
        file_name="DAV_model_data.csv",
        mime="text/csv"
        )
