import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import statsmodels.api as sm
import plotly.express as px
import plotly.graph_objects as go

# ==============================
# Page config
# ==============================
st.set_page_config(page_title="Plateforme ALM DAV", layout="wide")

# ==============================
# Branding couleurs CIH
# ==============================
primary_color = "#0055A4"  # Bleu CIH
secondary_color = "#C8102E"  # Rouge CIH

# ==============================
# Header avec logo
# ==============================
st.image("cih_logo.jpg", width=200)  # Logo en haut
st.markdown(f"<h1 style='color:{primary_color}; text-align:center'>📊 Plateforme ALM - Modélisation DAV</h1>", unsafe_allow_html=True)
st.markdown(f"<h4 style='color:{secondary_color}; text-align:center'>CIH Bank - Département ALM</h4>", unsafe_allow_html=True)

# ==============================
# Sidebar avec logo et navigation
# ==============================
st.sidebar.image("cih_logo.jpg", width=150)
st.sidebar.title("Navigation")
page = st.sidebar.radio("Choisir la section :", [
    "1️⃣ TMP BAM",
    "2️⃣ Import DAV",
    "3️⃣ Préparation",
    "4️⃣ Estimation",
    "5️⃣ Comparaison",
    "6️⃣ Visualisation"
])



# ==============================
# Fonction pour récupérer TMP BAM
# ==============================
def get_tmp_bam(start, end):
    url = "https://www.bkam.ma/fr/Marches/Principaux-indicateurs/Marche-monetaire/Marche-monetaire-interbancaire"
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "lxml")
        table = soup.find("table")
        df = pd.read_html(str(table))[0]

        # Nettoyage des données
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
        df["Taux Moyen Pondéré"] = df["Taux Moyen Pondéré"].str.replace("%","").str.replace(",",".").astype(float)
        df = df[(df["Date"] >= pd.to_datetime(start)) & (df["Date"] <= pd.to_datetime(end))]

        # TMP agrégé par mois
        df_monthly = df.groupby(df["Date"].dt.to_period("M"))["Taux Moyen Pondéré"].mean().reset_index()
        df_monthly["Date"] = df_monthly["Date"].dt.to_timestamp()

        return df, df_monthly
    except Exception as e:
        st.error("Impossible de récupérer le TMP BAM.")
        st.error(str(e))
        return None, None

# ==============================
# 1️⃣ TMP BAM
# ==============================
st.header("1️⃣ TMP BAM")
start_date = st.date_input("Date de début TMP", value=pd.to_datetime("2020-01-01"))
end_date = st.date_input("Date de fin TMP", value=pd.to_datetime("2023-12-31"))

if st.button("Télécharger TMP BAM"):
    tmp_df, tmp_monthly_df = get_tmp_bam(start_date, end_date)
    if tmp_df is not None and tmp_monthly_df is not None:
        st.success("✅ TMP BAM téléchargé !")

        st.subheader("TMP journalier")
        st.dataframe(tmp_df)  # Affiche le TMP jour par jour

        st.subheader("TMP agrégé par mois")
        st.dataframe(tmp_monthly_df)  # Affiche le TMP par mois

        # Sauvegarder dans session_state pour réutilisation
        st.session_state.tmp_df = tmp_df
        st.session_state.tmp_monthly_df = tmp_monthly_df

# ==============================
# 2️⃣ Import DAV
# ==============================
if page == "2️⃣ Import DAV":
    with st.expander("📂 Import des données DAV"):
        dav_type = st.radio("Type de dépôt :", ["Compte courant / Chèque", "Compte épargne"])
        st.session_state.dav_type = dav_type

        file = st.file_uploader("Uploader fichier Excel ou CSV", type=["csv","xlsx"])
        if file is not None:
            if file.name.endswith(".csv"):
                df_dav = pd.read_csv(file)
            else:
                df_dav = pd.read_excel(file)
            df_dav.columns = df_dav.columns.str.strip().str.lower()
            if 'date' not in df_dav.columns:
                st.error("❌ Votre fichier doit contenir une colonne 'Date'")
            else:
                df_dav['date'] = pd.to_datetime(df_dav['date'], dayfirst=True, errors='coerce')
                df_dav = df_dav.dropna(subset=['date']).reset_index(drop=True)
                if 'dk' in df_dav.columns:
                    df_dav['dk'] = df_dav['dk'].astype(str).str.replace(" ", "").str.replace(",", ".").astype(float)
                st.success("✅ Fichier chargé et nettoyé !")
                st.dataframe(df_dav.head())
                st.session_state.df_dav = df_dav

# ==============================
# 3️⃣ Préparation
# ==============================
if page == "3️⃣ Préparation":
    if "df_dav" in st.session_state and "tmp_df" in st.session_state:
        df_dav = st.session_state.df_dav
        tmp_df = st.session_state.tmp_df
        dav_type = st.session_state.dav_type

        with st.expander("🔧 Préparation des variables"):
            df = pd.merge(df_dav, tmp_df, on="Date", how="left").sort_values("Date").reset_index(drop=True)
            df["logDk"] = np.log(df["dk"])
            df["logDk_lag"] = df["logDk"].shift(1)
            df["Rk"] = df["Taux Moyen Pondéré"]
            df["Rk_lag"] = df["Rk"].shift(1)
            df["dRk"] = df["Rk"] - df["Rk_lag"]
            df["trend"] = np.arange(len(df))
            if dav_type=="Compte épargne" and "ik" in df.columns:
                df["spread"] = df["Rk"] - df["ik"]
            df = df.dropna()
            st.success("✅ Variables préparées !")
            st.dataframe(df.head())
            st.session_state.df = df

# ==============================
# 4️⃣ Estimation
# ==============================
if page == "4️⃣ Estimation":
    if "df" in st.session_state:
        df = st.session_state.df
        dav_type = st.session_state.dav_type

        with st.expander("⚙️ Sélection des modèles à estimer"):
            models_to_run = st.multiselect("Choisir les modèles :", 
                                           ["Selvaggio","Dupre","Jarrow-Van Deventer","OBrien","OTS"],
                                           default=["Selvaggio","Dupre"])
        if st.button("Lancer estimation"):
            results = {}
            if "Selvaggio" in models_to_run:
                X = sm.add_constant(df[["logDk_lag","Rk","trend"]])
                y = df["logDk"]
                results["Selvaggio"] = sm.OLS(y, X).fit()
            if "Dupre" in models_to_run:
                df["delta_logD"] = df["logDk_lag"] - df["logDk"]
                X_dupre = sm.add_constant(df["Rk"])
                y_dupre = df["delta_logD"]
                results["Dupre"] = sm.OLS(y_dupre, X_dupre).fit()
            if "Jarrow-Van Deventer" in models_to_run:
                X_jvd = sm.add_constant(df[["logDk_lag","Rk","dRk","trend"]])
                y_jvd = df["logDk"]
                results["Jarrow-Van Deventer"] = sm.OLS(y_jvd, X_jvd).fit()
            if "OBrien" in models_to_run and dav_type=="Compte épargne":
                X_obrien = sm.add_constant(df[["logDk_lag","spread","trend"]])
                y_obrien = df["logDk"]
                results["OBrien"] = sm.OLS(y_obrien, X_obrien).fit()
            if "OTS" in models_to_run:
                X_ots = sm.add_constant(df["dk"].shift(1))
                y_ots = df["dk"]
                results["OTS"] = sm.OLS(y_ots.dropna(), X_ots.dropna()).fit()

            st.success("✅ Estimation terminée !")
            st.session_state.results = results

# ==============================
# 5️⃣ Comparaison
# ==============================
if page == "5️⃣ Comparaison":
    if "results" in st.session_state:
        results = st.session_state.results
        comparison = pd.DataFrame({
            "Model": [name for name in results],
            "R2": [model.rsquared for model in results.values()],
            "AIC": [model.aic for model in results.values()]
        })
        st.header("📊 Comparaison des modèles")
        st.dataframe(comparison)
        # Graphique Plotly avec logo en fond
        fig = go.Figure()
        fig.add_trace(go.Bar(x=comparison["Model"], y=comparison["R2"], name="R2", marker_color=primary_color))
        fig.add_trace(go.Bar(x=comparison["Model"], y=comparison["AIC"], name="AIC", marker_color=secondary_color))
        fig.update_layout(
            title="R2 et AIC des modèles",
            barmode="group",
            images=[dict(
                source="cih_logo.jpg",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                sizex=0.3, sizey=0.3,
                xanchor="center", yanchor="middle",
                opacity=0.15,
                layer="below"
            )]
        )
        st.plotly_chart(fig)

# ==============================
# 6️⃣ Visualisation
# ==============================
if page == "6️⃣ Visualisation":
    if "df" in st.session_state:
        df = st.session_state.df
        dav_type = st.session_state.dav_type
        st.header("📈 Visualisation des séries")
        options = df.columns.tolist()
        selected_vars = st.multiselect("Variables à afficher", options, default=["dk","Rk"] if dav_type=="Compte courant / Chèque" else ["dk","Rk","ik"])
        if selected_vars:
            fig = go.Figure()
            for col in selected_vars:
                fig.add_trace(go.Scatter(x=df["Date"], y=df[col], mode="lines", name=col))
            # Ajouter logo en fond
            fig.update_layout(images=[dict(
                source="cih_logo.jpg",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                sizex=0.4, sizey=0.4,
                xanchor="center", yanchor="middle",
                opacity=0.15,
                layer="below"
            )])
            st.plotly_chart(fig)

        # Télécharger les données préparées
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("💾 Télécharger les données préparées", data=csv, file_name='DAV_model_data.csv', mime='text/csv')
