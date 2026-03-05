import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import statsmodels.api as sm
import matplotlib.pyplot as plt

st.set_page_config(page_title="Modélisation DAV", layout="wide")
st.title("Plateforme de Modélisation des Dépôts à Vue (DAV)")

# ==============================
# 1️⃣ Télécharger TMP BAM avec choix de la période
# ==============================
st.header("1️⃣ Récupération du TMP BAM")

start_date = st.date_input("Date de début pour TMP", value=pd.to_datetime("2020-01-01"))
end_date = st.date_input("Date de fin pour TMP", value=pd.to_datetime("2023-12-31"))

def get_tmp_bam(start=start_date, end=end_date):
    url = "https://www.bkam.ma/fr/Marches/Principaux-indicateurs/Marche-monetaire/Marche-monetaire-interbancaire"
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "lxml")
        table = soup.find("table")
        df = pd.read_html(str(table))[0]
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
        df["Taux Moyen Pondéré"] = df["Taux Moyen Pondéré"].str.replace("%","").str.replace(",",".").astype(float)
        # Filtrer par dates
        df = df[(df["Date"] >= pd.to_datetime(start)) & (df["Date"] <= pd.to_datetime(end))]
        # Calculer la moyenne mensuelle
        df_monthly = df.groupby(df["Date"].dt.to_period("M"))["Taux Moyen Pondéré"].mean().reset_index()
        df_monthly["Date"] = df_monthly["Date"].dt.to_timestamp()
        return df_monthly
    except Exception as e:
        st.error("Impossible de récupérer le TMP BAM. Vérifiez votre connexion ou le site de BAM.")
        st.error(str(e))
        return None

tmp_df = get_tmp_bam()
if tmp_df is not None:
    st.success("TMP BAM téléchargé et agrégé par mois avec succès !")
    st.dataframe(tmp_df.head())

# ==============================
# 2️⃣ Choisir le type de DAV
# ==============================
st.header("2️⃣ Choix du type de DAV")
dav_type = st.radio("Sélectionnez le type de dépôt :", ["Compte courant / Chèque", "Compte épargne"])

st.write(f"Vous avez choisi : {dav_type}")

# ==============================
# 3️⃣ Importer données DAV
# ==============================
st.header("3️⃣ Importer vos données DAV")
st.write("Le fichier doit contenir : Date | Dk" + (" | ik (taux rémunération)" if dav_type=="Compte épargne" else ""))

file = st.file_uploader("Uploader fichier Excel ou CSV", type=["csv","xlsx"])
df_dav = None

if file is not None:
    if file.name.endswith(".csv"):
        df_dav = pd.read_csv(file)
    else:
        df_dav = pd.read_excel(file)
  # Convertir en datetime et remplacer les erreurs par NaT
df_dav["Date"] = pd.to_datetime(df_dav["Date"], errors="coerce")

# Supprimer les lignes où la date est invalide
df_dav = df_dav.dropna(subset=["Date"]).reset_index(drop=True)
    st.dataframe(df_dav.head())

# ==============================
# 4️⃣ Préparation des variables
# ==============================
if df_dav is not None and tmp_df is not None:
    st.header("4️⃣ Préparation des variables économétriques")
    # Fusion avec TMP
    df = pd.merge(df_dav, tmp_df, on="Date", how="left")
    df = df.sort_values("Date").reset_index(drop=True)

    # Variables
    df["logDk"] = np.log(df["Dk"])
    df["logDk_lag"] = df["logDk"].shift(1)
    df["Rk"] = df["Taux Moyen Pondéré"]
    df["Rk_lag"] = df["Rk"].shift(1)
    df["dRk"] = df["Rk"] - df["Rk_lag"]
    df["trend"] = np.arange(len(df))

    if dav_type=="Compte épargne":
        df["spread"] = df["Rk"] - df["ik"]

    df = df.dropna()
    st.success("Variables préparées !")
    st.dataframe(df.head())

# ==============================
# 5️⃣ Estimation des modèles
# ==============================
if df_dav is not None and tmp_df is not None:
    st.header("5️⃣ Estimation des modèles économétriques")
    results = {}

    # 5.1 Modèle Selvaggio
    X = df[["logDk_lag","Rk","trend"]]
    X = sm.add_constant(X)
    y = df["logDk"]
    model_selvaggio = sm.OLS(y, X).fit()
    results["Selvaggio"] = model_selvaggio

    # 5.2 Modèle Dupré
    df["delta_logD"] = df["logDk_lag"] - df["logDk"]
    X_dupre = sm.add_constant(df["Rk"])
    y_dupre = df["delta_logD"]
    model_dupre = sm.OLS(y_dupre, X_dupre).fit()
    results["Dupre"] = model_dupre

    # 5.3 Modèle Jarrow & Van Deventer
    X_jvd = df[["logDk_lag","Rk","dRk","trend"]]
    X_jvd = sm.add_constant(X_jvd)
    y_jvd = df["logDk"]
    model_jvd = sm.OLS(y_jvd, X_jvd).fit()
    results["Jarrow-Van Deventer"] = model_jvd

    # 5.4 Modèle O'Brien (seulement si compte épargne)
    if dav_type=="Compte épargne":
        X_obrien = df[["logDk_lag","spread","trend"]]
        X_obrien = sm.add_constant(X_obrien)
        y_obrien = df["logDk"]
        model_obrien = sm.OLS(y_obrien, X_obrien).fit()
        results["OBrien"] = model_obrien

    # 5.5 Modèle OTS
    X_ots = sm.add_constant(df["Dk"].shift(1))
    y_ots = df["Dk"]
    model_ots = sm.OLS(y_ots.dropna(), X_ots.dropna()).fit()
    results["OTS"] = model_ots

    st.success("Estimation terminée !")

# ==============================
# 6️⃣ Comparaison des modèles
# ==============================
if results:
    st.header("6️⃣ Comparaison des modèles")
    comparison = pd.DataFrame({
        "Model": [],
        "R2": [],
        "AIC": []
    })

    for name, model in results.items():
        comparison = pd.concat([comparison, pd.DataFrame({
            "Model":[name],
            "R2":[model.rsquared],
            "AIC":[model.aic]
        })])

    st.dataframe(comparison)

    best_model = comparison.loc[comparison["AIC"].idxmin()]["Model"]
    st.success(f"Meilleur modèle selon AIC : {best_model}")

# ==============================
# 7️⃣ Graphiques et téléchargement
# ==============================
if df_dav is not None and tmp_df is not None:
    st.header("7️⃣ Visualisation")
    if dav_type=="Compte épargne":
        st.line_chart(df.set_index("Date")[["Dk","Rk","ik"]])
    else:
        st.line_chart(df.set_index("Date")[["Dk","Rk"]])

    # Option de téléchargement
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Télécharger les données préparées et les résultats",
        data=csv,
        file_name='DAV_model_data.csv',
        mime='text/csv',
    )

st.write("✅ Application complète pour modéliser et analyser les dépôts à vue.")
