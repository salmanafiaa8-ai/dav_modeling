import streamlit as st
import pandas as pd
import numpy as np
import time
from st_aggrid import AgGrid, GridOptionsBuilder
import plotly.graph_objects as go
from datetime import datetime

# ==============================
# Page config
# ==============================
st.set_page_config(page_title="Plateforme ALM DAV - Live", layout="wide", page_icon="💰")

primary_color = "#0055A4"
secondary_color = "#C8102E"
bg_color = "#1e1e1e"
text_color = "#ffffff"

# ==============================
# Header
# ==============================
st.markdown(
    f"<h1 style='color:{primary_color}; text-align:center;'>Plateforme ALM DAV - LIVE</h1>",
    unsafe_allow_html=True
)
st.markdown(
    f"<h4 style='color:{secondary_color}; text-align:center;'>CIH Bank - ALM Dashboard</h4>",
    unsafe_allow_html=True
)

# ==============================
# Sidebar : filtrage
# ==============================
st.sidebar.header("Filtres")
dav_type = st.sidebar.radio("Type de dépôt", ["Compte courant / Chèque", "Compte épargne"])
update_interval = st.sidebar.slider("Intervalle de mise à jour (s)", 1, 10, 5)

# ==============================
# Génération / chargement de données simulées
# ==============================
@st.cache_data
def generate_initial_data(rows=30):
    dates = pd.date_range(end=datetime.today(), periods=rows, freq='D')
    df = pd.DataFrame({
        "Date": dates,
        "DK": np.random.randint(1000, 2000, size=rows),
        "Rk": np.random.uniform(1.0, 3.0, size=rows)
    })
    if dav_type == "Compte épargne":
        df["ik"] = np.random.uniform(0.5, 2.5, size=rows)
        df["Spread"] = df["Rk"] - df["ik"]
    return df

df = generate_initial_data()

# ==============================
# Section KPIs
# ==============================
st.subheader("📊 KPIs")
kpi1, kpi2, kpi3 = st.columns(3)

kpi1.metric("Total DK", f"{df['DK'].sum():,.0f}")
kpi2.metric("Taux moyen Rk", f"{df['Rk'].mean():.2f}%")
if dav_type == "Compte épargne":
    kpi3.metric("Spread moyen", f"{df['Spread'].mean():.2f}%")
else:
    kpi3.metric("Spread moyen", "-")

# ==============================
# Tableau interactif AgGrid
# ==============================
st.subheader("📈 Tableau interactif")
gb = GridOptionsBuilder.from_dataframe(df)
gb.configure_pagination(paginationAutoPageSize=True)
gb.configure_side_bar()
gb.configure_default_column(editable=False, groupable=True)
gridOptions = gb.build()
AgGrid(df, gridOptions=gridOptions, theme="dark", height=300, fit_columns_on_grid_load=True)

# ==============================
# Graphiques dynamiques Plotly
# ==============================
st.subheader("📉 Graphiques dynamiques")

fig = go.Figure()
fig.add_trace(go.Scatter(x=df["Date"], y=df["DK"], mode="lines+markers", name="DK", line=dict(color="#00ffcc")))
fig.add_trace(go.Scatter(x=df["Date"], y=df["Rk"], mode="lines+markers", name="Rk", line=dict(color="#ff9900")))
if dav_type == "Compte épargne":
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Spread"], mode="lines+markers", name="Spread", line=dict(color="#ff3366")))

fig.update_layout(
    template="plotly_dark",
    xaxis_title="Date",
    yaxis_title="Valeur",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    height=400
)
st.plotly_chart(fig, use_container_width=True)

# ==============================
# Live update simulation
# ==============================
st.subheader("🔄 Live update simulation")
placeholder = st.empty()

for _ in range(5):  # tu peux mettre while True pour réel live
    # Nouvelle ligne simulée
    new_row = {
        "Date": datetime.now(),
        "DK": np.random.randint(1000, 2000),
        "Rk": np.random.uniform(1.0, 3.0)
    }
    if dav_type == "Compte épargne":
        new_row["ik"] = np.random.uniform(0.5, 2.5)
        new_row["Spread"] = new_row["Rk"] - new_row["ik"]
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    
    # Mettre à jour KPIs
    kpi1.metric("Total DK", f"{df['DK'].sum():,.0f}")
    kpi2.metric("Taux moyen Rk", f"{df['Rk'].mean():.2f}%")
    if dav_type == "Compte épargne":
        kpi3.metric("Spread moyen", f"{df['Spread'].mean():.2f}%")
    
    # Mettre à jour tableau
    AgGrid(df.tail(10), gridOptions=gridOptions, theme="dark", height=300, fit_columns_on_grid_load=True)
    
    # Mettre à jour graphique
    fig.data[0].y = df["DK"]
    fig.data[0].x = df["Date"]
    fig.data[1].y = df["Rk"]
    fig.data[1].x = df["Date"]
    if dav_type == "Compte épargne":
        fig.data[2].y = df["Spread"]
        fig.data[2].x = df["Date"]
    placeholder.plotly_chart(fig, use_container_width=True)
    
    time.sleep(update_interval)
