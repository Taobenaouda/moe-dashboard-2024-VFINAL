# Guide d'Upload Manuel vers GitHub

## Fichiers ESSENTIELS à uploader (dans cet ordre) :

### 1. Fichiers de configuration
- `requirements.txt`
- `.streamlit/config.toml`
- `.gitignore`
- `README.md`

### 2. Application principale
- `dashboard_app.py`

### 3. Données (dossier out/)
- `out/kpis.json`
- `out/registrations_slim.csv`
- `out/registrations_with_segments.csv`
- Tous les autres fichiers du dossier `out/`

### 4. Assets
- `assets/departements.geojson`
- `IMAGES/MOE.png`

### 5. Scripts utiles
- `run_analysis.py`
- `export_pdf.py`
- `check_data.py`

## ❌ Fichiers à NE PAS uploader :
- `export_marseille_outdoor_experiences_archive_2024.xlsx` (trop gros)
- `SEB ANALYSE COURSE - sheet1.csv` (données sources)
- Dossier `.git/`
- Dossier `.venv/`

## URL GitHub : https://github.com/taobenaouda199/MOE-Dashboard-2024

## Après upload → Déployer sur Streamlit Cloud !
