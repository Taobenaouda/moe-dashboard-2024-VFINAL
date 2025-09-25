# MOE Dashboard 2024

Dashboard analytique interactif pour Marseille Outdoor Experiences 2024.

## Description

Ce dashboard Streamlit présente les analyses des inscriptions et participations à l'événement MOE 2024. Il inclut :

- **Analyses démographiques** : répartition par âge, sexe, département
- **Analyses temporelles** : évolution des inscriptions, lead time
- **Analyses géographiques** : carte de France avec répartition des participants
- **Analyses comportementales** : taux de présence, utilisation des promos
- **Personas** : segmentation des participants avec recommandations marketing

## Installation et utilisation locale

```bash
# Cloner le repository
git clone [URL_DU_REPO]
cd SEB-ANALYSE

# Installer les dépendances
pip install -r requirements.txt

# Lancer le dashboard
streamlit run dashboard_app.py
```

Le dashboard sera accessible sur http://localhost:8501

## Structure du projet

- `dashboard_app.py` : Application Streamlit principale
- `run_analysis.py` : Script d'analyse des données
- `out/` : Fichiers de données générés par l'analyse
- `assets/` : Ressources (logo, fichiers geojson)
- `IMAGES/` : Images du projet

## Déploiement

Ce dashboard peut être déployé sur :
- **Streamlit Community Cloud** (recommandé)
- **Heroku**
- **Railway**
- **DigitalOcean**

## Données

Les analyses sont basées sur l'export des inscriptions MOE 2024. Les données sont prétraitées et anonymisées pour respecter la confidentialité.

## Contact

Pour toute question sur ce dashboard, contactez l'équipe MOE.

---

© MOE 2024 – Programme 2025: https://www.marseilleoutdoorexperiences.fr/programme-2025
