import json
import os
import base64
import mimetypes

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import urllib.request

st.set_page_config(
    page_title="MOE Dashboard 2024",
    page_icon="🏃‍♂️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.marseilleoutdoorexperiences.fr',
        'Report a bug': None,
        'About': "# MOE Dashboard 2024\nAnalyses des inscriptions et participations MOE 2024"
    }
)

# Fonction de login
def check_login():
    """Vérifie l'authentification de l'utilisateur"""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    if not st.session_state.authenticated:
        st.title("🔐 Accès sécurisé - MOE Dashboard 2024")
        st.markdown("---")
        
        with st.form("login_form"):
            col1, col2, col3 = st.columns([1,2,1])
            with col2:
                st.markdown("### Connexion requise")
                username = st.text_input("Nom d'utilisateur:")
                password = st.text_input("Mot de passe:", type="password")
                submitted = st.form_submit_button("Se connecter", use_container_width=True)
                
                if submitted:
                    if username == "admin" and password == "AdminMOE13":
                        st.session_state.authenticated = True
                        st.success("✅ Connexion réussie ! Redirection...")
                        st.rerun()
                    else:
                        st.error("❌ Identifiants incorrects")
                        st.info("💡 Utilisez les identifiants fournis")
        
        st.stop()

# Vérifier l'authentification au début
check_login()

# ---- THEME (dark with pink accent) ----
PRIMARY_PINK = "#f6b3d8"
ACCENT_VIOLET = "#b08ae6"  # violet pâle pour tags/puces
DARK_BG = "#0b0b0b"
SIDEBAR_BG = "#101010"

# Plotly defaults
px.defaults.template = "plotly_dark"
px.defaults.color_discrete_sequence = [
    PRIMARY_PINK,
    "#8e6bae",
    "#3a86ff",
    "#2ec4b6",
    "#ffbe0b",
]

# Global UI CSS
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@600;700&display=swap');
    .stApp {{ background-color: {DARK_BG}; }}
    section[data-testid="stSidebar"] {{ background-color: {SIDEBAR_BG}; }}
    html, body, [class^="css"], .stMarkdown, .stText, .stMetric, .stCaption {{ color: #f0f0f0; }}
    a {{ color: {PRIMARY_PINK}; }}
    .stButton>button, .stDownloadButton>button {{
        background-color: {PRIMARY_PINK};
        color: #111;
        border: none;
        border-radius: 6px;
    }}
    /* Inputs */
    div[role="combobox"], div[data-baseweb], .stSelectbox div, .stMultiSelect div {{ background-color: #171717 !important; color: #f0f0f0 !important; }}
    /* Tags/puces multiselect (BaseWeb Tag) */
    div[data-baseweb="tag"] {{
        background-color: {ACCENT_VIOLET} !important;
        border-color: {ACCENT_VIOLET} !important;
        color: #1a1a1a !important;
    }}
    div[data-baseweb="tag"] span, div[data-baseweb="tag"] div {{ color: #1a1a1a !important; }}
    div[data-baseweb="tag"] svg {{ fill: #1a1a1a !important; stroke: #1a1a1a !important; }}
    /* Boutons close sur tags */
    div[data-baseweb="tag"] button {{ filter: none !important; }}

    /* Header title */
    .moe-title {{
        color: {PRIMARY_PINK};
        font-family: 'Montserrat', 'Poppins', 'Inter', 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
        font-weight: 700;
        font-size: 34px;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        text-align: center;
        margin-top: 0;
        line-height: 1.25;
    }}
    .moe-header {{
        display: flex;
        align-items: center;
        gap: 18px;
        justify-content: center;
        margin: 6px 0 10px 0;
    }}
    .moe-logo {{ width: 260px; height: auto; display:block; }}
    </style>
    """,
    unsafe_allow_html=True,
)

def _find_logo_path() -> str | None:
    """Find a logo file in common locations, case-insensitive, with multiple formats."""
    search_dirs = ["images", "assets", "."]
    preferred_names = ["moe_logo", "logo"]
    exts = [".png", ".jpg", ".jpeg", ".webp", ".svg"]
    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        try:
            files = os.listdir(d)
        except Exception:
            files = []
        # 1) exact preferred names
        for name in preferred_names:
            for ext in exts:
                for f in files:
                    if f.lower() == f"{name}{ext}":
                        return os.path.join(d, f)
        # 2) any file containing moe or logo
        for f in files:
            lf = f.lower()
            if any(ext in lf for ext in exts) and ("moe" in lf or "logo" in lf):
                return os.path.join(d, f)
        # 3) any image file
        for f in files:
            lf = f.lower()
            if any(lf.endswith(ext) for ext in exts):
                return os.path.join(d, f)
    return None

def render_logo_header() -> None:
    """Render header with logo (left) and title (right) on the same line using columns."""
    logo_path = _find_logo_path()
    col_left, col_mid, col_right = st.columns([1, 4, 1])
    with col_left:
        if logo_path and logo_path.lower().endswith(".svg"):
            try:
                with open(logo_path, "rb") as f:
                    data = base64.b64encode(f.read()).decode("utf-8")
                st.markdown(f"<img class='moe-logo' src='data:image/svg+xml;base64,{data}' />", unsafe_allow_html=True)
            except Exception:
                st.markdown("<div style='font-size:28px;color:#f6b3d8;'>MOE</div>", unsafe_allow_html=True)
        elif logo_path:
            st.image(logo_path, width=260)
        else:
            st.markdown("<div style='font-size:28px;color:#f6b3d8;'>MOE</div>", unsafe_allow_html=True)
    with col_mid:
        st.markdown("<div class='moe-title'>Analytics Dashboard Of Last Year's Dataset</div>", unsafe_allow_html=True)

@st.cache_data
def load_data():
    kpis_path = os.path.join("out", "kpis.json")
    slim_path = os.path.join("out", "registrations_slim.csv")
    seg_path = os.path.join("out", "registrations_with_segments.csv")
    logit_parcours_path = os.path.join("out", "logit_parcours21_coefs.csv")
    logit_presence_path = os.path.join("out", "logit_presence_coefs.csv")

    try:
        with open(kpis_path, "r", encoding="utf-8") as f:
            kpis = json.load(f)
    except FileNotFoundError:
        st.error(f"Fichier KPIs non trouvé : {kpis_path}")
        st.stop()
    except Exception as e:
        st.error(f"Erreur lors du chargement des KPIs : {e}")
        st.stop()

    try:
        df = pd.read_csv(slim_path, dtype=str)
    except FileNotFoundError:
        st.error(f"Fichier de données non trouvé : {slim_path}")
        st.stop()
    except Exception as e:
        st.error(f"Erreur lors du chargement des données : {e}")
        st.stop()
    
    # Parse dates and numeric fields
    for c in ["date_inscription"]:
        if c in df:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    for c in ["age", "lead_days"]:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["present"] = df["present"].astype(str).str.upper().eq("TRUE") | df["present"].astype(str).str.upper().eq("OUI")
    df["paid_ok"] = df["paid_ok"].astype(str).str.upper().eq("TRUE") | df["paid_ok"].astype(str).str.upper().eq("OUI")
    df["promo_used"] = df["promo_used"].astype(str).str.upper().isin(["TRUE", "OUI"]) | (df.get("CODE PROMO", pd.Series([""]*len(df))).fillna("").astype(str).str.len() > 0)

    # Try to merge segments
    if os.path.exists(seg_path):
        df_seg = pd.read_csv(seg_path, dtype=str)
        df = df.merge(df_seg[["REF", "segment"]], on="REF", how="left")
    else:
        df["segment"] = np.nan

    # Load models
    logit_parcours = pd.read_csv(logit_parcours_path) if os.path.exists(logit_parcours_path) else pd.DataFrame()
    logit_presence = pd.read_csv(logit_presence_path) if os.path.exists(logit_presence_path) else pd.DataFrame()

    return df, kpis, logit_parcours, logit_presence

df, kpis, logit_parcours, logit_presence = load_data()

# Sidebar filters
st.sidebar.markdown("### Filtres")
parcours_opts = sorted([p for p in df["parcours_norm"].dropna().unique()])
sexe_opts = ["H", "F"]
tranche_opts = [t for t in df["tranche_age"].dropna().unique()]
dep_opts = sorted([d for d in df["dep_code"].dropna().unique()])
lead_opts = [t for t in df.get("lead_bucket", pd.Series(dtype=str)).dropna().unique()]
seg_opts = sorted([s for s in df["segment"].dropna().unique()])

f_parcours = st.sidebar.multiselect("Parcours", parcours_opts, default=parcours_opts)
f_sexe = st.sidebar.multiselect("Sexe", sexe_opts, default=sexe_opts)
f_tranches = st.sidebar.multiselect("Tranches d'âge", tranche_opts, default=tranche_opts)
f_deps = st.sidebar.multiselect("Départements", dep_opts, default=dep_opts)
f_leads = st.sidebar.multiselect("Lead time", lead_opts, default=lead_opts)
f_seg = st.sidebar.multiselect("Segments", seg_opts, default=seg_opts)

# Apply filters
filt = (
    df["parcours_norm"].isin(f_parcours)
    & df["sexe"].isin(f_sexe)
    & df["tranche_age"].isin(f_tranches)
)
if f_deps:
    filt &= df["dep_code"].isin(f_deps)
if "lead_bucket" in df.columns and f_leads:
    filt &= df["lead_bucket"].isin(f_leads)
if f_seg:
    filt &= df["segment"].isin(f_seg)

view = df.loc[filt].copy()

# Header
render_logo_header()
st.caption("Programme 2025: marseilleoutdoorexperiences.fr/programme-2025")

# Download report (PDF)
try:
    pdf_path = os.path.join("out", "report.pdf")
    if os.path.exists(pdf_path):
        with open(pdf_path, "rb") as _pf:
            st.download_button(
                label="Download report (PDF)",
                data=_pf.read(),
                file_name="MOE_report_2024.pdf",
                mime="application/pdf",
                key="dl_report_pdf_top",
            )
except Exception:
    pass

# KPI row
c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1:
    st.metric("Inscrits (filtré)", f"{len(view):,}".replace(",", " "))
with c2:
    st.metric("Présence %", f"{view['present'].mean()*100:.1f}%")
with c3:
    st.metric("Local 13 %", f"{(view['dep_code']=='13').mean()*100:.1f}%")
with c4:
    st.metric("Promo %", f"{view['promo_used'].mean()*100:.1f}%")
with c5:
    st.metric("Âge médian", f"{view['age'].median():.0f}" if view['age'].notna().any() else "–")
with c6:
    if view["date_inscription"].notna().any():
        lead_med = (pd.Timestamp("2024-11-09") - view["date_inscription"]).dt.days.median()
        st.metric("Lead médian (j)", f"{lead_med:.0f}")
    else:
        st.metric("Lead médian (j)", "–")

st.markdown("---")

# 🎯 NAVIGATION AVEC MENU
st.markdown("### 📊 Navigation")
selected_page = st.selectbox(
    "Choisissez une section à analyser :",
    ["🏠 Vue d'ensemble", "🗺️ Géographie", "⏰ Temporalité", "💰 Commercial", "👥 Personas"],
    index=0
)

# 🏠 SECTION 1: VUE D'ENSEMBLE
if selected_page == "🏠 Vue d'ensemble":
    st.header("Vue d'ensemble des inscriptions")
    
    # Mix parcours, sexe, âge
    cc1, cc2, cc3 = st.columns([1,1,1])
    with cc1:
        st.subheader("Mix parcours")
        vc = view["parcours_norm"].value_counts().reset_index()
        vc.columns = ["Parcours", "Inscriptions"]
        fig = px.pie(vc, names="Parcours", values="Inscriptions", hole=0.3)
        st.plotly_chart(fig, use_container_width=True)
    with cc2:
        st.subheader("Répartition par sexe")
        vc = view["sexe"].value_counts().reset_index()
        vc.columns = ["Sexe", "Inscriptions"]
        fig = px.bar(vc, x="Sexe", y="Inscriptions", text="Inscriptions")
        st.plotly_chart(fig, use_container_width=True)
    with cc3:
        st.subheader("Tranches d'âge")
        vc = view["tranche_age"].value_counts().sort_index().reset_index()
        vc.columns = ["Tranche", "Inscriptions"]
        fig = px.bar(vc, x="Tranche", y="Inscriptions", text="Inscriptions")
        st.plotly_chart(fig, use_container_width=True)
    
    # Présence par tranche d'âge et parcours (heatmap)
    st.subheader("Présence par tranche d'âge et parcours")
    try:
        pres_pivot = (
            view.pivot_table(index="tranche_age", columns="parcours_norm", values="present", aggfunc="mean")
                .round(3)
                .sort_index()
        )
        if pres_pivot.notna().any().any():
            fig_hm_pres = px.imshow(
                pres_pivot,
                color_continuous_scale="Viridis",
                aspect="auto",
                labels=dict(color="Présence")
            )
            fig_hm_pres.update_layout(margin=dict(l=0, r=0, t=30, b=0), coloraxis_colorbar=dict(tickformat=".0%"))
            st.plotly_chart(fig_hm_pres, use_container_width=True)
        else:
            st.info("Pas assez de données pour la heatmap de présence.")
    except Exception:
        st.info("Heatmap de présence indisponible sur ce sous-ensemble.")

# 🗺️ SECTION 2: GÉOGRAPHIE
elif selected_page == "🗺️ Géographie":
    st.header("Analyses géographiques")
    
    # Top départements
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Top départements")
        dep_counts = view["dep_code"].value_counts().head(15).reset_index()
        dep_counts.columns = ["Département", "Inscriptions"]
        fig_dep = px.bar(dep_counts.sort_values("Inscriptions"), x="Inscriptions", y="Département", orientation="h")
        st.plotly_chart(fig_dep, use_container_width=True)
    
    with col2:
        st.subheader("Taux de présence par département")
        try:
            top_dep_index = view["dep_code"].dropna().value_counts().head(15).index
            dep_rate = (
                view[view["dep_code"].isin(top_dep_index)]
                .groupby("dep_code")["present"].mean().mul(100).round(1)
                .sort_values(ascending=False).reset_index()
            )
            dep_rate.columns = ["Département", "Présence %"]
            fig_dep_rate = px.bar(dep_rate, x="Présence %", y="Département", orientation="h", text="Présence %")
            st.plotly_chart(fig_dep_rate, use_container_width=True)
        except Exception:
            st.info("Taux de présence par département indisponible.")
    
    # Focus Local (13) vs Hors‑13 vs Paris/Lyon
    st.subheader("Analyse géographique détaillée")
    try:
        def _segment_geo(row):
            dep = str(row.get("dep_code", ""))
            if dep == "13":
                return "Local 13"
            if dep in {"75", "69"}:
                return "Paris+Lyon"
            return "Hors 13"

        geo = view.copy()
        geo["geo_grp"] = geo.apply(_segment_geo, axis=1)
        # KPIs par groupe
        kpi_geo = geo.groupby("geo_grp").agg(
            Taille=("REF", "count"),
            Lead_méd_j=("lead_days", lambda s: float(pd.Series(s).median()) if pd.Series(s).notna().any() else None),
            Promo_pct=("promo_used", "mean"),
            Presence_pct=("present", "mean"),
        ).reset_index()
        if not kpi_geo.empty:
            kpi_geo["Promo_pct"] = (kpi_geo["Promo_pct"] * 100).round(1)
            kpi_geo["Presence_pct"] = (kpi_geo["Presence_pct"] * 100).round(1)
            st.markdown("**KPIs par groupe géographique**")
            st.dataframe(kpi_geo, use_container_width=True)
        
        # Tendance hebdo
        if geo["date_inscription"].notna().any():
            trend = (
                geo.dropna(subset=["date_inscription"]) 
                   .assign(week=lambda x: x["date_inscription"].dt.to_period("W").dt.start_time)
                   .groupby(["week", "geo_grp"]).size().rename("Inscriptions").reset_index()
            )
            if not trend.empty:
                fig_trend = px.line(trend, x="week", y="Inscriptions", color="geo_grp", markers=True)
                st.markdown("**Tendance hebdomadaire par groupe géographique**")
                st.plotly_chart(fig_trend, use_container_width=True)
    except Exception:
        st.info("Section géographique indisponible.")
    
    # Carte France – répartition des inscrits (cercles)
    st.subheader("Carte France – répartition des inscrits")

    def _load_fr_departements_geojson(path: str = os.path.join("assets", "departements.geojson")):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Essayer d'abord le fichier compressé
        if os.path.exists(path + ".gz"):
            try:
                import gzip
                with gzip.open(path + ".gz", "rt", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        try:
            url = "https://france-geojson.gregoiredavid.fr/repo/departements.geojson"
            urllib.request.urlretrieve(url, path)
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def _centroid_of_polygon(coords):
        if not coords:
            return None
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        return sum(xs) / len(xs), sum(ys) / len(ys)

    def _compute_dept_centroids(geojson_obj):
        centroids = {}
        if not geojson_obj:
            return centroids
        for feat in geojson_obj.get("features", []):
            props = feat.get("properties", {})
            code = str(props.get("code", "")).zfill(2)
            geom = feat.get("geometry", {})
            gtype = geom.get("type")
            coords = geom.get("coordinates")
            lon, lat = None, None
            try:
                if gtype == "Polygon":
                    ring = coords[0] if coords else []
                    c = _centroid_of_polygon(ring)
                    if c:
                        lon, lat = c
                elif gtype == "MultiPolygon":
                    ring = coords[0][0] if coords else []
                    c = _centroid_of_polygon(ring)
                    if c:
                        lon, lat = c
            except Exception:
                lon, lat = None, None
            if lon is not None and lat is not None:
                centroids[code] = {"lon": float(lon), "lat": float(lat)}
        return centroids

    geojson_deps = _load_fr_departements_geojson()
    centroids = _compute_dept_centroids(geojson_deps)

    dep_counts_map = (
        view["dep_code"].dropna().astype(str).value_counts().reset_index()
        if "dep_code" in view.columns else pd.DataFrame(columns=["index", "dep_code"])
    )
    if not dep_counts_map.empty:
        dep_counts_map.columns = ["dep_code", "count"]
        dep_counts_map["dep_code"] = dep_counts_map["dep_code"].str.upper()
        dep_counts_map["lon"] = dep_counts_map["dep_code"].map(lambda c: centroids.get(c, {}).get("lon"))
        dep_counts_map["lat"] = dep_counts_map["dep_code"].map(lambda c: centroids.get(c, {}).get("lat"))
        missing = dep_counts_map[dep_counts_map["lon"].isna()].index
        for i in missing:
            c = dep_counts_map.at[i, "dep_code"]
            c2 = c.lstrip("0")
            if c2 in centroids:
                dep_counts_map.at[i, "lon"] = centroids[c2]["lon"]
                dep_counts_map.at[i, "lat"] = centroids[c2]["lat"]
            elif len(c) == 1 and ("0" + c) in centroids:
                dep_counts_map.at[i, "lon"] = centroids["0" + c]["lon"]
                dep_counts_map.at[i, "lat"] = centroids["0" + c]["lat"]

        data_map = dep_counts_map.dropna(subset=["lon", "lat"]).copy()
        if not data_map.empty:
            fig_map = px.scatter_mapbox(
                data_map,
                lat="lat",
                lon="lon",
                size="count",
                hover_name="dep_code",
                hover_data={"count": True, "lat": False, "lon": False},
                color_discrete_sequence=[PRIMARY_PINK],
                zoom=4.2,
                size_max=70,
                height=520,
                title="Répartition géographique des inscriptions"
            )
            fig_map.update_layout(mapbox_style="carto-darkmatter", margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.info("Centroides indisponibles pour les départements présents.")
    else:
        st.info("Aucun département disponible pour la carte.")

# ⏰ SECTION 3: TEMPORALITÉ
elif selected_page == "⏰ Temporalité":
    st.header("Analyses temporelles")
    
    # Temporalité des inscriptions
    st.subheader("Évolution des inscriptions")
    if view["date_inscription"].notna().any():
        wv = view.dropna(subset=["date_inscription"]).assign(week=lambda x: x["date_inscription"].dt.to_period("W").dt.start_time)
        line = (
            wv.groupby("week").size().rename("Inscriptions").reset_index()
        )
        fig_line = px.line(line, x="week", y="Inscriptions", markers=True)
        st.plotly_chart(fig_line, use_container_width=True)

        st.subheader("Inscriptions par parcours dans le temps")
        stacked = wv.pivot_table(index="week", columns="parcours_norm", values="REF", aggfunc="count").fillna(0).reset_index()
        fig_stack = px.area(stacked, x="week", y=[c for c in stacked.columns if c != "week"], groupnorm=None)
        st.plotly_chart(fig_stack, use_container_width=True)
    else:
        st.info("Pas de dates d'inscription disponibles après filtrage.")
    
    # Lead time analyses
    st.subheader("Analyses de lead time")
    col1, col2 = st.columns(2)
    with col1:
        if view["lead_days"].notna().any():
            fig_lead = px.histogram(view, x="lead_days", nbins=30, title="Distribution des lead times")
            st.plotly_chart(fig_lead, use_container_width=True)
        else:
            st.info("Lead time non disponible.")
    with col2:
        if "lead_bucket" in view.columns and view["lead_bucket"].notna().any():
            pres = view.groupby(["lead_bucket", "parcours_norm"])['present'].mean().reset_index()
            fig_pres = px.line(pres, x="lead_bucket", y="present", color="parcours_norm", markers=True, title="Présence par lead time")
            fig_pres.update_layout(yaxis_tickformat=".0%")
            st.plotly_chart(fig_pres, use_container_width=True)
        else:
            st.info("Pas de bucket de lead disponible.")
    
    # Heatmap jour × heure
    st.subheader("Inscriptions par jour et heure")
    try:
        if view["date_inscription"].notna().any():
            tmp = view.dropna(subset=["date_inscription"]).copy()
            tmp["jour_idx"] = tmp["date_inscription"].dt.dayofweek
            jour_map = {0: "Lundi", 1: "Mardi", 2: "Mercredi", 3: "Jeudi", 4: "Vendredi", 5: "Samedi", 6: "Dimanche"}
            tmp["jour"] = tmp["jour_idx"].map(jour_map)
            tmp["heure"] = tmp["date_inscription"].dt.hour
            # Pivot counts
            pv = tmp.pivot_table(index="jour_idx", columns="heure", values="REF", aggfunc="count").fillna(0)
            # Remap index to day labels and order
            pv.index = [jour_map.get(i, str(i)) for i in pv.index]
            pv = pv.reindex(["Lundi","Mardi","Mercredi","Jeudi","Vendredi","Samedi","Dimanche"]) 
            fig_do_h = px.imshow(pv, color_continuous_scale="Oranges", aspect="auto", labels=dict(color="Inscriptions"))
            fig_do_h.update_layout(margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_do_h, use_container_width=True)
            
            # Top créneaux
            long = pv.reset_index().melt(id_vars="index", var_name="heure", value_name="inscriptions").rename(columns={"index":"jour"})
            top_slots = long.sort_values("inscriptions", ascending=False).head(5)
            if not top_slots.empty:
                st.subheader("Créneaux recommandés (top 5)")
                st.dataframe(top_slots.reset_index(drop=True), use_container_width=True)
        else:
            st.info("Pas de dates pour construire la heatmap jour × heure.")
    except Exception:
        st.info("Heatmap jour × heure indisponible.")

# 💰 SECTION 4: COMMERCIAL
elif selected_page == "💰 Commercial":
    st.header("Analyses commerciales")
    
    # Promos & Paiement
    st.subheader("Promotions et paiements")
    colp1, colp2 = st.columns(2)
    with colp1:
        st.markdown("**Utilisation des promos par parcours**")
        promo_rate = view.groupby("parcours_norm")["promo_used"].mean().reset_index()
        promo_rate["promo_%"] = (promo_rate["promo_used"] * 100).round(1)
        fig_promo = px.bar(promo_rate, x="parcours_norm", y="promo_%", text="promo_%")
        fig_promo.update_layout(yaxis_title="Promo %")
        st.plotly_chart(fig_promo, use_container_width=True)
    with colp2:
        if "TYPE PAIEMENT" in view.columns:
            st.markdown("**Types de paiement**")
            tp = view["TYPE PAIEMENT"].fillna("").value_counts().head(10).reset_index()
            tp.columns = ["Type paiement", "Inscriptions"]
            st.plotly_chart(px.bar(tp, x="Type paiement", y="Inscriptions", text="Inscriptions"), use_container_width=True)
        else:
            st.info("Types de paiement non disponibles dans le sous-ensemble.")
    
    # Heatmap promos par lead time et parcours
    if "lead_bucket" in view.columns and view["lead_bucket"].notna().any():
        st.subheader("Utilisation des promos par lead time et parcours")
        try:
            promo_pivot = (
                view.pivot_table(index="lead_bucket", columns="parcours_norm", values="promo_used", aggfunc="mean")
                    .round(3)
            )
            if promo_pivot.notna().any().any():
                fig_hm_promo = px.imshow(
                    promo_pivot,
                    color_continuous_scale="Blues",
                    aspect="auto",
                    labels=dict(color="Promo %")
                )
                fig_hm_promo.update_layout(margin=dict(l=0, r=0, t=30, b=0), coloraxis_colorbar=dict(tickformat=".0%"))
                st.plotly_chart(fig_hm_promo, use_container_width=True)
            else:
                st.info("Pas assez de données pour la heatmap promos.")
        except Exception:
            st.info("Heatmap promos indisponible sur ce sous-ensemble.")
    
    # Clubs
    st.subheader("Analyses clubs")
    club_counts = view.get("CLUB")
    if club_counts is not None:
        cc = club_counts.fillna("").value_counts().drop(labels="", errors="ignore").head(20).reset_index()
        cc.columns = ["Club", "Inscriptions"]
        fig_clubs = px.bar(cc, x="Club", y="Inscriptions", title="Top 20 clubs")
        fig_clubs.update_layout(xaxis_tickangle=45)
        st.plotly_chart(fig_clubs, use_container_width=True)
    else:
        st.info("Données clubs non disponibles.")

# 👥 SECTION 5: PERSONAS
elif selected_page == "👥 Personas":
    st.header("Cibles & Personas")
    tab_simple, tab_data = st.tabs(["Personas (simple)", "Détails data (segments)"])
    
    # Base enrichie pour personas
    def _enrich_base_for_personas(df_all: pd.DataFrame) -> pd.DataFrame:
        b = df_all.copy()
        b["is_local"] = b["dep_code"].eq("13").astype(int)
        b["is_female"] = b["sexe"].eq("F").astype(int)
        b["has_club_int"] = b.get("CLUB", pd.Series([np.nan]*len(b))).fillna("").ne("").astype(int)
        b["promo_int"] = b["promo_used"].astype(int)
        b["p21"] = b["parcours_norm"].eq("21km").astype(int)
        b["p12"] = b["parcours_norm"].eq("12km").astype(int)
        b["late_decider"] = (b["lead_days"] <= 14).fillna(False).astype(int)
        b["non_local"] = (b["is_local"] == 0).astype(int)
        b["paris_lyon"] = b["dep_code"].isin(["75","69"]).astype(int)
        b["age_ok_25_44"] = b["age"].between(25, 44, inclusive="both").fillna(False).astype(int)
        b["age_ok_25_34"] = b["age"].between(25, 34, inclusive="both").fillna(False).astype(int)
        return b

    base_all = _enrich_base_for_personas(df)

    with tab_simple:
        st.write("Cartes synthétiques des 6 personas clés, avec messages et offres.")

        def card(title: str, size: int, percent: float, bullets: list[str]):
            c = st.container(border=True)
            c.markdown(f"### {title}")
            cols = c.columns(2)
            cols[0].metric("Taille", f"{size:,}".replace(","," "))
            cols[1].metric("Part", f"{percent:.1f}%")
            for b in bullets:
                c.markdown(f"- {b}")

        total = len(base_all)

        # 1. Locaux 25–44 – 12km urbain fun
        m1 = (base_all["is_local"].eq(1) & base_all["age_ok_25_44"].eq(1) & base_all["p12"].eq(1))
        n1 = int(m1.sum()); p1 = (n1/total*100) if total else 0
        card("Locaux 25–44 – 12 km urbain fun", n1, p1, [
            "Message: Vivre Marseille entre amis; finish party",
            "Offres: TEAM5 (‑10% dès 5), CE locaux",
            "Canaux: IG/TikTok local, affichage micro‑local",
        ])

        # 2. Défi 21 km – hommes 25–44 (clubs)
        m2 = (base_all["p21"].eq(1) & base_all["age_ok_25_44"].eq(1) & base_all["is_female"].eq(0))
        n2 = int(m2.sum()); p2 = (n2/total*100) if total else 0
        card("Défi 21 km – hommes 25–44 (clubs)", n2, p2, [
            "Message: Conquérir la ville; segments mythiques; sas chrono",
            "Offres: CLUB10, dotations capitaines",
            "Canaux: Strava, clubs FFA, retailers running",
        ])

        # 3. Femmes 25–34 – 12 km crew
        m3 = (base_all["p12"].eq(1) & base_all["age_ok_25_34"].eq(1) & base_all["is_female"].eq(1))
        n3 = int(m3.sum()); p3 = (n3/total*100) if total else 0
        card("Femmes 25–34 – 12 km crew", n3, p3, [
            "Message: Accessible, fun, instagrammable",
            "Offres: Girls Crew (‑10% à 3)",
            "Canaux: Créateurs locaux, UGC",
        ])

        # 4. Non‑locaux Paris/Lyon (week‑end destination)
        m4 = base_all["paris_lyon"].eq(1)
        n4 = int(m4.sum()); p4 = (n4/total*100) if total else 0
        card("Non‑locaux Paris/Lyon – week‑end destination", n4, p4, [
            "Message: City‑trail + mer + culture – 48h à Marseille",
            "Offres: WEEKENDMOE (train/hôtel limité)",
            "Canaux: Meta ciblage 75/69, OT/hôtels",
        ])

        # 5. Décideurs tardifs (≤14 j)
        m5 = base_all["late_decider"].eq(1)
        n5 = int(m5.sum()); p5 = (n5/total*100) if total else 0
        card("Décideurs tardifs (≤14 j)", n5, p5, [
            "Message: Derniers dossards; logistique simple; météo",
            "Tactiques: relances J‑14/J‑7/J‑3, stock restreint",
        ])

        # 6. Entreprises/Clubs locaux
        m6 = (base_all["has_club_int"].eq(1) & base_all["is_local"].eq(1))
        n6 = int(m6.sum()); p6 = (n6/total*100) if total else 0
        card("Entreprises/Clubs locaux", n6, p6, [
            "Message: Team‑building urbain",
            "Offres: conventions clubs/CE, quotas, classement inter‑clubs",
        ])

        st.markdown("—")
        st.markdown("Télécharger la synthèse des personas (CSV)")
        import io
        persona_csv = io.StringIO()
        pd.DataFrame([
            {"Persona":"Locaux 25–44 – 12 km","Taille":n1,"Part_%":round(p1,1)},
            {"Persona":"Défi 21 km – H 25–44","Taille":n2,"Part_%":round(p2,1)},
            {"Persona":"Femmes 25–34 – 12 km","Taille":n3,"Part_%":round(p3,1)},
            {"Persona":"Paris/Lyon – week‑end","Taille":n4,"Part_%":round(p4,1)},
            {"Persona":"Décideurs tardifs","Taille":n5,"Part_%":round(p5,1)},
            {"Persona":"Entreprises/Clubs locaux","Taille":n6,"Part_%":round(p6,1)},
        ]).to_csv(persona_csv, index=False)
        st.download_button("Exporter CSV", persona_csv.getvalue(), file_name="personas_moe.csv", mime="text/csv")

    with tab_data:
        # Reprise de la vue data détaillée (segments + top targets)
        base = _enrich_base_for_personas(df)
        if base["segment"].notna().any():
            grp = base.groupby("segment")
            persona = pd.DataFrame({
                "Taille": grp.size(),
                "Âge_moy": grp["age"].mean().round(1),
                "Lead_méd_j": grp["lead_days"].median(),
                "Local13_%": (grp["is_local"].mean()*100).round(1),
                "F_%": (grp["is_female"].mean()*100).round(1),
                "Club_%": (grp["has_club_int"].mean()*100).round(1),
                "Promo_%": (grp["promo_int"].mean()*100).round(1),
                "21km_%": (grp["p21"].mean()*100).round(1),
                "Présence_%": (grp["present"].mean()*100).round(1),
                "NonLocal_%": (grp["non_local"].mean()*100).round(1),
                "ParisLyon_%": (grp["paris_lyon"].mean()*100).round(1),
                "Late_%": (grp["late_decider"].mean()*100).round(1),
            }).reset_index()
            st.dataframe(persona.sort_values(["Taille"], ascending=False), use_container_width=True)
        else:
            st.info("Segments indisponibles pour construire les personas.")

# Analyses visuelles utiles (heatmaps)
st.markdown("---")
st.subheader("Analyses visuelles utiles")

# Heatmap: présence par tranche d'âge et parcours
try:
    pres_pivot = (
        view.pivot_table(index="tranche_age", columns="parcours_norm", values="present", aggfunc="mean")
            .round(3)
            .sort_index()
    )
    if pres_pivot.notna().any().any():
        fig_hm_pres = px.imshow(
            pres_pivot,
            color_continuous_scale="Viridis",
            aspect="auto",
            labels=dict(color="Présence"),
            title="Taux de présence par tranche d'âge et parcours"
        )
        fig_hm_pres.update_layout(margin=dict(l=0, r=0, t=30, b=0), coloraxis_colorbar=dict(tickformat=".0%"))
        st.plotly_chart(fig_hm_pres, use_container_width=True)
    else:
        st.info("Heatmap de présence indisponible sur ce sous-ensemble.")
except Exception:
    st.info("Heatmap de présence indisponible sur ce sous-ensemble.")

# Heatmap: inscriptions par tranche d'âge et parcours  
try:
    count_pivot = (
        view.pivot_table(index="tranche_age", columns="parcours_norm", values="REF", aggfunc="count")
            .fillna(0)
            .sort_index()
    )
    if count_pivot.notna().any().any():
        fig_hm_count = px.imshow(
            count_pivot,
            color_continuous_scale="Blues",
            aspect="auto",
            labels=dict(color="Inscriptions"),
            title="Nombre d'inscriptions par tranche d'âge et parcours"
        )
        fig_hm_count.update_layout(margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_hm_count, use_container_width=True)
    else:
        st.info("Heatmap d'inscriptions indisponible sur ce sous-ensemble.")
except Exception:
    st.info("Heatmap d'inscriptions indisponible sur ce sous-ensemble.")

st.markdown("---")
st.caption("© MOE 2024 – Dashboard analytique. Programme 2025: https://www.marseilleoutdoorexperiences.fr/programme-2025")
