import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ==============================
# Page config
# ==============================
st.set_page_config(page_title="Dashboard ALM - DAV", layout="wide")

# ==============================
# Couleurs CIH
# ==============================
primary_color = "#0055A4"  # Bleu CIH
secondary_color = "#C8102E"  # Rouge CIH

# ==============================
# Header
# ==============================
st.markdown(f"""
<div style="display:flex; align-items:center; justify-content:center; margin-bottom:20px;">
    <img src="cih_logo.jpg" width="100"/>
    <h1 style="color:{primary_color}; margin-left:20px;">Dashboard ALM - Modélisation DAV</h1>
</div>
""", unsafe_allow_html=True)

# ==============================
# Sidebar
# ==============================
st.sidebar.image("cih_logo.jpg", width=120)
st.sidebar.title("Navigation")
page = st.sidebar.radio("Choisir la section :", [
    "KPI",
    "TMP BAM",
    "Import DAV",
    "Préparation",
    "Estimation Modèles",
    "Visualisation"
])

# ==============================
# Exemple de données fictives
# ==============================
# Pour démo, on crée des données aléatoires
dates = pd.date_range(start="2023-01-01", periods=24, freq="M")
df = pd.DataFrame({
    "Date": dates,
    "Dk": np.random.randint(1000, 5000, size=24),
    "Rk": np.random.uniform(1.5, 3.0, size=24),
    "ik": np.random.uniform(1.0, 2.5, size=24)
})

# ==============================
# 1️⃣ KPI en haut
# ==============================
if page == "KPI":
    st.header("📊 KPI Clés")
    total_dav = df["Dk"].sum()
    avg_tmp = df["Rk"].mean()
    avg_spread = (df["Rk"] - df["ik"]).mean()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div style="background-color:{primary_color};padding:20px;border-radius:10px;text-align:center;color:white;">
            <h3>Total DAV</h3>
            <h2>{total_dav:,.0f} MAD</h2>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div style="background-color:{secondary_color};padding:20px;border-radius:10px;text-align:center;color:white;">
            <h3>TMP Moyen</h3>
            <h2>{avg_tmp:.2f} %</h2>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div style="background-color:{primary_color};padding:20px;border-radius:10px;text-align:center;color:white;">
            <h3>Spread Moyen</h3>
            <h2>{avg_spread:.2f} %</h2>
        </div>
        """, unsafe_allow_html=True)

# ==============================
# 2️⃣ TMP BAM
# ==============================
if page == "TMP BAM":
    st.header("📥 TMP BAM")
    st.dataframe(df[["Date","Rk"]])
    fig = px.line(df, x="Date", y="Rk", title="Taux Moyen Pondéré TMP BAM", markers=True)
    fig.update_layout(plot_bgcolor='white', paper_bgcolor='white')
    # Ajouter logo en watermark
    fig.add_layout_image(
        dict(source="cih_logo.jpg", xref="paper", yref="paper",
             x=0.5, y=0.5, sizex=0.4, sizey=0.4,
             xanchor="center", yanchor="middle",
             opacity=0.1, layer="below")
    )
    st.plotly_chart(fig, use_container_width=True)

# ==============================
# 3️⃣ Import DAV
# ==============================
if page == "Import DAV":
    st.header("📂 Import des données DAV")
    file = st.file_uploader("Uploader fichier Excel ou CSV", type=["csv","xlsx"])
    if file:
        if file.name.endswith(".csv"):
            df_dav = pd.read_csv(file)
        else:
            df_dav = pd.read_excel(file)
        st.success("✅ Fichier chargé !")
        st.dataframe(df_dav.head())

# ==============================
# 4️⃣ Préparation
# ==============================
if page == "Préparation":
    st.header("🔧 Préparation des variables")
    df["logDk"] = np.log(df["Dk"])
    df["spread"] = df["Rk"] - df["ik"]
    st.dataframe(df.head())

# ==============================
# 5️⃣ Estimation Modèles
# ==============================
if page == "Estimation Modèles":
    st.header("⚙️ Estimation des modèles")
    X = sm.add_constant(df[["logDk","Rk"]])
    y = df["logDk"]
    model = sm.OLS(y, X).fit()
    st.text(model.summary())

# ==============================
# 6️⃣ Visualisation
# ==============================
if page == "Visualisation":
    st.header("📈 Visualisation interactive")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Dk"], mode="lines+markers", name="Dk", line=dict(color=primary_color)))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Rk"], mode="lines+markers", name="Rk", line=dict(color=secondary_color)))
    # Logo CIH en fond
    fig.add_layout_image(
        dict(source="cih_logo.jpg", xref="paper", yref="paper",
             x=0.5, y=0.5, sizex=0.4, sizey=0.4,
             xanchor="center", yanchor="middle",
             opacity=0.1, layer="below")
    )
    fig.update_layout(title="DAV et TMP", plot_bgcolor='white', paper_bgcolor='white')
    st.plotly_chart(fig, use_container_width=True)
