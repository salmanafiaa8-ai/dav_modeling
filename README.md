# 🏦 Rachat Anticipé — ML Dashboard

Application Streamlit interactive pour la modélisation et la prédiction des rachats anticipés de crédits par régression logistique.

## 🚀 Lancer l'application

### En local
```bash
pip install -r requirements.txt
streamlit run app.py
```

### Sur Streamlit Cloud (GitHub)
1. Pushez ce dépôt sur GitHub
2. Rendez-vous sur [share.streamlit.io](https://share.streamlit.io)
3. Connectez votre repo → sélectionnez `app.py`
4. Déployez !

## 📦 Structure du projet
```
rachat_app/
├── app.py                           # Application principale
├── dataset_rachat_anticipe_1000.csv # Dataset d'entraînement
├── requirements.txt                 # Dépendances Python
└── README.md
```

## 🎯 Fonctionnalités

| Onglet | Description |
|--------|-------------|
| 📊 Vue d'ensemble | Métriques clés, distribution des probabilités |
| 📈 Performance | Courbes ROC, Précision-Rappel, Calibration, Matrice de confusion |
| 🔍 Interprétation | Coefficients β, Odds Ratios, tableau complet |
| 🔮 Prédiction Client | Formulaire temps réel → probabilité + analyse métier |
| 📋 Données | Exploration filtrée + export des scores CSV |

## ⚙️ Paramètres ajustables (barre latérale)
- **Régularisation C** : contrôle l'intensité de la pénalisation L2
- **Taille jeu de test** : proportion train/test
- **Seuil de décision** : probabilité à partir de laquelle on prédit un rachat

## 📐 Variables du modèle

### Variables originales
`Nominal`, `CRD`, `Taux_credit`, `Penalite`, `Revenu`, `Anciennete`, `Taux_marche`, `Type_taux`, `Type_credit`, `Type_client`

### Variables dérivées (Feature Engineering)
| Variable | Description |
|----------|-------------|
| `diff_taux` | Taux crédit − Taux marché (principal levier) |
| `economie_potentielle` | diff_taux × CRD |
| `pct_vie_ecoulee` | % de la durée du contrat écoulée |
| `duree_restante_mois` | Mois restants avant maturité |
| `ratio_crd_nominal` | CRD / Nominal |
| `penalite_relative` | Pénalité / Taux crédit |
| `ratio_revenu_crd` | Revenu / CRD |

## 📊 Performances typiques

| Métrique | Valeur |
|----------|--------|
| AUC-ROC | ~0.693 |
| F1-Score | ~0.654 |
| CV-AUC (5-fold) | ~0.700 ± 0.032 |

## 🛠️ Technologies
- **Python** 3.9+
- **Streamlit** — interface web
- **scikit-learn** — modélisation ML
- **pandas / numpy** — traitement des données
- **matplotlib** — visualisations
