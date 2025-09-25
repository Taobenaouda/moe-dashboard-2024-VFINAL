import os
import json
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Nouvelles importations pour ML/plots avancés
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


def ensure_out_dirs() -> None:
    os.makedirs("out/figures", exist_ok=True)


def read_registrations(csv_path: str) -> pd.DataFrame:
    encodings_to_try = ["utf-8-sig", "cp1252", "latin1"]
    last_error = None
    for enc in encodings_to_try:
        try:
            df = pd.read_csv(csv_path, dtype=str, encoding=enc)
            return df
        except Exception as e:  # noqa: BLE001
            last_error = e
    raise RuntimeError(f"Failed to read CSV with encodings {encodings_to_try}: {last_error}")


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    def parse_col(col: str) -> pd.Series:
        s = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True, dayfirst=True)
        # Retry with dayfirst False for US-like strings
        retry_mask = s.isna() & df[col].notna()
        if retry_mask.any():
            s2 = pd.to_datetime(df.loc[retry_mask, col], errors="coerce", infer_datetime_format=True, dayfirst=False)
            s.loc[retry_mask] = s2
        return s

    if "DATE INSCRIPTION" in df.columns:
        df["date_inscription"] = parse_col("DATE INSCRIPTION")
    if "DATE DE NAISSANCE" in df.columns:
        df["date_naissance"] = parse_col("DATE DE NAISSANCE")
    if "DATE EMARGEMENT" in df.columns:
        df["date_emargement"] = parse_col("DATE EMARGEMENT")
    return df


def derive_features(df: pd.DataFrame, race_day: pd.Timestamp) -> pd.DataFrame:
    # Clean whitespace
    df = df.apply(lambda col: col.str.strip() if col.dtype == "object" else col)

    # Age and buckets
    if "date_naissance" in df:
        age_years = ((race_day - df["date_naissance"]).dt.days / 365.25)
        df["age"] = age_years.round().astype("Int64")
    else:
        df["age"] = pd.Series([pd.NA] * len(df), dtype="Int64")

    bins = [0, 17, 24, 34, 44, 54, 64, 200]
    labels = ["<18", "18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
    df["tranche_age"] = pd.cut(df["age"].astype("float"), bins=bins, labels=labels, right=True, include_lowest=True)

    # Lead time
    if "date_inscription" in df:
        df["lead_days"] = (race_day - df["date_inscription"]).dt.days
    else:
        df["lead_days"] = pd.NA

    # Normalize parcours to one of {21km,12km,6km/5km,other}
    def normalize_parcours(val: Any) -> str:
        text = (str(val) if pd.notna(val) else "").lower()
        if "21" in text:
            return "21km"
        if "12" in text:
            return "12km"
        if "6" in text or "5" in text:
            return "6km"
        return "autre"

    df["parcours_norm"] = df.get("PARCOURS", pd.Series([pd.NA] * len(df))).apply(normalize_parcours)

    # Geography
    dep_code = df.get("DEPARTEMENT (CODE)")
    if dep_code is not None:
        df["dep_code"] = dep_code.str.extract(r"(\d{2,3})", expand=False)
        df["is_local_13"] = df["dep_code"].eq("13")
    else:
        df["dep_code"] = pd.NA
        df["is_local_13"] = False

    # Status flags
    df["present"] = df.get("EMARGEMENT", pd.Series([""] * len(df))).fillna("").str.upper().eq("OUI")
    # Payment status: PAIEMENT column contains 'OUI' when paid
    df["paid_ok"] = df.get("PAIEMENT", pd.Series([""] * len(df))).fillna("").str.upper().eq("OUI")

    # Early ticket flag
    df["is_early_ticket"] = df.get("PARCOURS", pd.Series([""] * len(df))).fillna("").str.upper().str.contains("EARLY")

    # Promo flag
    df["promo_used"] = df.get("CODE PROMO", pd.Series([pd.NA] * len(df))).fillna("").astype(str).str.len() > 0

    # Gender normalize
    df["sexe"] = df.get("CIVILITE", pd.Series([pd.NA] * len(df))).str.upper().map({"HOMME": "H", "FEMME": "F"})

    # Country
    df["pays"] = df.get("PAYS", pd.Series([pd.NA] * len(df))).fillna("")
    df["is_fr"] = df["pays"].str.contains("france", case=False, na=False)

    # Club present
    df["has_club"] = df.get("CLUB", pd.Series([pd.NA] * len(df))).fillna("").str.len() > 0

    return df


def compute_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    kpis: Dict[str, Any] = {}

    kpis["inscrits_total"] = int(len(df))
    kpis["parcours_mix"] = df["parcours_norm"].value_counts(dropna=False).to_dict()
    kpis["sexe_mix"] = df["sexe"].value_counts(dropna=False).to_dict()
    kpis["tranches_age"] = (
        df["tranche_age"].value_counts(dropna=False).sort_index().astype(int).to_dict()
    )
    kpis["local_13_pct"] = round(float(df["is_local_13"].mean() * 100), 1)
    kpis["presence_pct"] = round(float(df["present"].mean() * 100), 1)
    kpis["paid_ok_pct"] = round(float(df["paid_ok"].mean() * 100), 1)
    kpis["promo_pct"] = round(float(df["promo_used"].mean() * 100), 1)
    kpis["early_ticket_pct"] = round(float(df["is_early_ticket"].mean() * 100), 1)
    kpis["france_pct"] = round(float(df["is_fr"].mean() * 100), 1)

    # Licence / certificat and payment types
    if "LICENCE/CERTIFICAT" in df.columns:
        kpis["licence_ok_pct"] = round(float(df["LICENCE/CERTIFICAT"].fillna("").str.upper().eq("VALIDÉ").mean() * 100), 1)
    if "TYPE PAIEMENT" in df.columns:
        kpis["type_paiement_mix"] = df["TYPE PAIEMENT"].fillna("").value_counts().head(10).to_dict()

    # Top geos
    kpis["top_departements"] = df["dep_code"].value_counts().head(15).to_dict()
    kpis["top_villes"] = df.get("VILLE", pd.Series([pd.NA] * len(df))).value_counts().head(15).to_dict()

    # Timing
    if "date_inscription" in df:
        ts = (
            df.dropna(subset=["date_inscription"]).assign(week=lambda x: x["date_inscription"].dt.to_period("W").dt.start_time)
        )
        kpis["inscriptions_weeks_min"] = str(ts["week"].min()) if not ts.empty else None
        kpis["inscriptions_weeks_max"] = str(ts["week"].max()) if not ts.empty else None
        kpis["lead_days_median"] = float(df["lead_days"].median()) if df["lead_days"].notna().any() else None

    # Segment presence
    seg_presence = (
        df.groupby(["parcours_norm", "tranche_age"]) ["present"].mean().unstack(0)
    )
    kpis["presence_by_parcours_tranche"] = (
        seg_presence.round(3).fillna(0.0).to_dict()
    )

    # Promo by parcours
    promo_by_parcours = df.groupby("parcours_norm")["promo_used"].mean().round(3).to_dict()
    kpis["promo_by_parcours"] = {k: float(v) for k, v in promo_by_parcours.items()}

    return kpis


def save_tables(df: pd.DataFrame) -> None:
    # Aggregates to CSV
    (
        df.groupby("parcours_norm").size().rename("count").to_frame()
        .to_csv("out/parcours_counts.csv")
    )
    (
        df.groupby(["parcours_norm", "sexe"]).size().rename("count").to_frame()
        .to_csv("out/parcours_by_sexe.csv")
    )
    (
        df.groupby("tranche_age").size().rename("count").to_frame()
        .to_csv("out/tranches_age_counts.csv")
    )
    (
        df.groupby("dep_code").size().rename("count").to_frame().sort_values("count", ascending=False)
        .to_csv("out/dep_counts.csv")
    )
    (
        df.assign(week=lambda x: x["date_inscription"].dt.to_period("W").dt.start_time)
        .groupby("week").size().rename("count").to_frame()
        .to_csv("out/inscriptions_by_week.csv")
    )


def plot_figures(df: pd.DataFrame) -> List[str]:
    fig_paths: List[str] = []
    # Weekly curve
    weekly = (
        df.dropna(subset=["date_inscription"]) \
          .assign(week=lambda x: x["date_inscription"].dt.to_period("W").dt.start_time) \
          .groupby("week").size().rename("Inscriptions")
    )
    if not weekly.empty:
        plt.figure(figsize=(10, 5))
        weekly.plot(marker="o")
        plt.title("Inscriptions par semaine")
        plt.xlabel("Semaine")
        plt.ylabel("Inscriptions")
        plt.grid(True, alpha=0.3)
        path = "out/figures/inscriptions_by_week.png"
        plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
        fig_paths.append(path)

    # Age distribution
    ages = df["age"].dropna().astype(int)
    if not ages.empty:
        plt.figure(figsize=(8, 5))
        plt.hist(ages, bins=range(int(ages.min()), int(ages.max()) + 2, 2), color="#2a9d8f")
        plt.title("Distribution des âges")
        plt.xlabel("Âge")
        plt.ylabel("Participants")
        plt.grid(True, alpha=0.3)
        path = "out/figures/age_distribution.png"
        plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
        fig_paths.append(path)

    # Parcours by age bucket
    cross = df.pivot_table(index="tranche_age", columns="parcours_norm", values="REF", aggfunc="count").fillna(0)
    if not cross.empty:
        plt.figure(figsize=(9, 5))
        cross.plot(kind="bar", stacked=False)
        plt.title("Volume par parcours et tranche d'âge")
        plt.xlabel("Tranche d'âge")
        plt.ylabel("Inscriptions")
        plt.grid(True, axis="y", alpha=0.3)
        plt.legend(title="Parcours")
        path = "out/figures/parcours_by_tranche.png"
        plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
        fig_paths.append(path)

    # Top départements
    dep_counts = df["dep_code"].value_counts().head(15)
    if not dep_counts.empty:
        plt.figure(figsize=(8, 6))
        dep_counts.iloc[::-1].plot(kind="barh", color="#264653")
        plt.title("Top 15 départements")
        plt.xlabel("Inscriptions")
        plt.ylabel("Département")
        plt.grid(True, axis="x", alpha=0.3)
        path = "out/figures/top_departements.png"
        plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
        fig_paths.append(path)

    return fig_paths


def bucketize_lead_days(df: pd.DataFrame) -> pd.DataFrame:
    bins = [-1, 14, 30, 60, 90, 180, 10000]
    labels = ["≤14j", "15-30j", "31-60j", "61-90j", "91-180j", ">180j"]
    if "lead_days" in df.columns:
        df["lead_bucket"] = pd.cut(df["lead_days"], bins=bins, labels=labels)
    else:
        df["lead_bucket"] = pd.NA
    return df


def plot_more_figures(df: pd.DataFrame) -> List[str]:
    paths: List[str] = []
    # Weekly by parcours (stacked)
    wk = (
        df.dropna(subset=["date_inscription"]) \
          .assign(week=lambda x: x["date_inscription"].dt.to_period("W").dt.start_time)
    )
    if not wk.empty:
        pivot = wk.pivot_table(index="week", columns="parcours_norm", values="REF", aggfunc="count").fillna(0)
        if not pivot.empty:
            pivot.sort_index(inplace=True)
            pivot.plot(kind="bar", stacked=True, figsize=(12, 6))
            plt.title("Inscriptions par semaine et par parcours")
            plt.xlabel("Semaine")
            plt.ylabel("Inscriptions")
            plt.tight_layout()
            p = "out/figures/inscriptions_by_week_parcours.png"
            plt.savefig(p, dpi=150); plt.close(); paths.append(p)

    # Lead days histogram
    if df["lead_days"].notna().any():
        plt.figure(figsize=(9, 5))
        plt.hist(df["lead_days"].dropna(), bins=30, color="#6c5ce7")
        plt.title("Distribution du lead time (jours entre inscription et course)")
        plt.xlabel("Jours avant course")
        plt.ylabel("Participants")
        plt.grid(True, alpha=0.3)
        p = "out/figures/lead_days_hist.png"
        plt.tight_layout(); plt.savefig(p, dpi=150); plt.close(); paths.append(p)

    # Presence by lead bucket and parcours
    if "lead_bucket" in df.columns:
        pres = df.groupby(["lead_bucket", "parcours_norm"])['present'].mean().unstack(1).fillna(0)
        if not pres.empty:
            pres.plot(marker='o', figsize=(9, 5))
            plt.title("Présence (émargement) par lead time et parcours")
            plt.xlabel("Lead time (bucket)")
            plt.ylabel("Taux de présence")
            plt.ylim(0.7, 1.0)
            plt.grid(True, alpha=0.3)
            p = "out/figures/presence_by_lead_parcours.png"
            plt.tight_layout(); plt.savefig(p, dpi=150); plt.close(); paths.append(p)
    return paths


def run_kmeans_segmentation(df: pd.DataFrame, n_clusters: int = 5) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    work = df.copy()
    # Features numériques
    work["is_local"] = work["is_local_13"].astype(int)
    work["is_female"] = work["sexe"].eq("F").astype(int)
    work["has_club_int"] = work["has_club"].astype(int)
    work["promo_int"] = work["promo_used"].astype(int)
    work["p21"] = work["parcours_norm"].eq("21km").astype(int)
    features = ["age", "lead_days", "is_local", "is_female", "has_club_int", "promo_int", "p21"]
    X = work[features].astype(float).fillna(0.0)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    km = KMeans(n_clusters=n_clusters, n_init=20, random_state=42)
    labels = km.fit_predict(Xs)
    work["segment"] = labels

    prof = work.groupby("segment")[features].mean()
    sizes = work["segment"].value_counts().sort_index()

    # Sauvegarde des attributions
    cols_to_save = ["REF", "parcours_norm", "sexe", "age", "tranche_age", "dep_code", "is_local_13", "lead_days", "has_club", "promo_used", "present", "segment"]
    work[cols_to_save].to_csv("out/registrations_with_segments.csv", index=False)

    return prof, sizes, work


def run_logistic_models(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    results: Dict[str, pd.DataFrame] = {}
    base = df.copy()
    base["is_local"] = base["is_local_13"].astype(int)
    base["is_female"] = base["sexe"].eq("F").astype(int)
    base["has_club_int"] = base["has_club"].astype(int)
    base["promo_int"] = base["promo_used"].astype(int)

    # Features communs
    feat = ["age", "lead_days", "is_local", "is_female", "has_club_int", "promo_int"]
    X = base[feat].astype(float).fillna(0.0)

    # Modèle 1: choix 21km (vs 12km)
    y_parcours = base["parcours_norm"].map({"21km": 1, "12km": 0})
    mask = y_parcours.notna()
    Xp, yp = X[mask], y_parcours[mask].astype(int)
    pipe_p = Pipeline(steps=[("scaler", StandardScaler()), ("logit", LogisticRegression(max_iter=200, solver="lbfgs"))])
    pipe_p.fit(Xp, yp)
    coefs_p = pipe_p.named_steps["logit"].coef_[0]
    out_p = pd.DataFrame({"feature": feat, "coef": coefs_p}).sort_values("coef", ascending=False)
    out_p.to_csv("out/logit_parcours21_coefs.csv", index=False)
    results["parcours21_coefs"] = out_p

    # Modèle 2: présence (émargement)
    y_pres = base["present"].astype(int)
    Xm, ym = X, y_pres
    pipe_m = Pipeline(steps=[("scaler", StandardScaler()), ("logit", LogisticRegression(max_iter=200, solver="lbfgs"))])
    pipe_m.fit(Xm, ym)
    coefs_m = pipe_m.named_steps["logit"].coef_[0]
    out_m = pd.DataFrame({"feature": feat, "coef": coefs_m}).sort_values("coef", ascending=False)
    out_m.to_csv("out/logit_presence_coefs.csv", index=False)
    results["presence_coefs"] = out_m

    return results


def analyze_promos_and_clubs(df: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    # Promos: top codes, usage par parcours
    codes = df.get("CODE PROMO", pd.Series([pd.NA] * len(df))).fillna("")
    top_codes = codes.value_counts().drop(index="", errors="ignore").head(15)
    out["top_codes"] = top_codes.to_dict()
    promo_by_week = (
        df.assign(week=lambda x: x["date_inscription"].dt.to_period("W").dt.start_time)
          .groupby("week")["promo_used"].mean().round(3)
    )
    promo_by_week.to_csv("out/promo_rate_by_week.csv")

    # Clubs
    club = df.get("CLUB", pd.Series([pd.NA] * len(df))).fillna("")
    club_counts = club.value_counts().drop(index="", errors="ignore")
    top_clubs = club_counts.head(20)
    out["top_clubs"] = top_clubs.to_dict()

    # Présence par club (top)
    if not top_clubs.empty:
        pres_by_club = (
            df[df["CLUB"].isin(top_clubs.index)]
            .groupby("CLUB")["present"].mean().sort_values(ascending=False)
        )
        pres_by_club.to_csv("out/presence_rate_by_top_clubs.csv")

    # Sauvegardes
    top_codes.to_csv("out/top_promo_codes.csv")
    top_clubs.to_csv("out/top_clubs.csv")

    return out


def write_insights_md(kpis: Dict[str, Any], fig_paths: List[str]) -> None:
    lines: List[str] = []
    lines.append("# Insights MOE 2024 (inscriptions)\n")

    lines.append("## KPIs clés\n")
    lines.append(f"- Inscriptions totales: **{kpis.get('inscrits_total')}**\n")
    pmix = kpis.get("parcours_mix", {})
    if pmix:
        mix_str = ", ".join([f"{k}: {v}" for k, v in pmix.items()])
        lines.append(f"- Mix parcours: {mix_str}\n")
    lines.append(f"- Présence (émargement): **{kpis.get('presence_pct')}%**\n")
    lines.append(f"- Paiement validé: **{kpis.get('paid_ok_pct')}%**\n")
    lines.append(f"- Local (dép.13): **{kpis.get('local_13_pct')}%**\n")
    lines.append(f"- France: **{kpis.get('france_pct')}%**\n")
    lines.append(f"- Early ticket: **{kpis.get('early_ticket_pct')}%**\n")
    lines.append(f"- Utilisation promo: **{kpis.get('promo_pct')}%**\n")

    # Geography
    lines.append("\n## Géographie\n")
    top_dep = kpis.get("top_departements", {})
    if top_dep:
        top_dep_list = list(top_dep.items())[:10]
        for dep, cnt in top_dep_list:
            lines.append(f"- Dép. {dep}: {cnt}\n")

    # Timing
    lines.append("\n## Temporalité\n")
    if kpis.get("inscriptions_weeks_min") and kpis.get("inscriptions_weeks_max"):
        lines.append(
            f"- Fenêtre d'inscriptions: {kpis['inscriptions_weeks_min']} → {kpis['inscriptions_weeks_max']}\n"
        )
    if kpis.get("lead_days_median") is not None:
        lines.append(f"- Délai médian avant course: {int(kpis['lead_days_median'])} jours\n")

    # Segments
    lines.append("\n## Segments \n")
    pres = kpis.get("presence_by_parcours_tranche", {})
    if pres:
        lines.append("- Présence par parcours et tranche d'âge (taux): voir détails JSON `out/kpis.json`\n")

    # Figures
    if fig_paths:
        lines.append("\n## Figures générées\n")
        for p in fig_paths:
            lines.append(f"- {p}\n")

    with open("out/insights_summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def compute_non_local_insights(df: pd.DataFrame) -> Dict[str, Any]:
    """KPIs et agrégats pour la cible hors département 13 (Paris, Lyon, etc.)."""
    ext = df[df["is_local_13"] == False].copy()  # noqa: E712
    out: Dict[str, Any] = {}
    out["inscrits_non_locaux"] = int(len(ext))
    out["share_non_locaux_pct"] = round(100 * len(ext) / max(len(df), 1), 1)
    out["parcours_mix"] = ext["parcours_norm"].value_counts().to_dict()
    out["sexe_mix"] = ext["sexe"].value_counts().to_dict()
    # Top départements hors 13
    out["top_departements_ext"] = (
        ext["dep_code"].value_counts().head(20).to_dict()
    )
    # Focus Paris (75) et Lyon (69)
    for dep in ["75", "69"]:
        sub = ext[ext["dep_code"] == dep]
        out[f"dep_{dep}_count"] = int(len(sub))
        out[f"dep_{dep}_lead_median"] = float(sub["lead_days"].median()) if sub["lead_days"].notna().any() else None
        out[f"dep_{dep}_promo_pct"] = round(float(sub["promo_used"].mean() * 100), 1) if not sub.empty else 0.0
        out[f"dep_{dep}_presence_pct"] = round(float(sub["present"].mean() * 100), 1) if not sub.empty else 0.0
    # Global non-locaux
    out["lead_median"] = float(ext["lead_days"].median()) if ext["lead_days"].notna().any() else None
    out["promo_pct"] = round(float(ext["promo_used"].mean() * 100), 1) if not ext.empty else 0.0
    out["presence_pct"] = round(float(ext["present"].mean() * 100), 1) if not ext.empty else 0.0

    # Sauvegarde JSON
    with open("out/non_local_kpis.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    return out


def plot_non_local_figures(df: pd.DataFrame) -> List[str]:
    paths: List[str] = []
    ext = df[df["is_local_13"] == False].copy()  # noqa: E712
    if not ext.empty:
        # Top départements hors 13
        dep_counts = ext["dep_code"].value_counts().head(15)
        if not dep_counts.empty:
            plt.figure(figsize=(8, 6))
            dep_counts.iloc[::-1].plot(kind="barh", color="#1d3557")
            plt.title("Top 15 départements (hors 13)")
            plt.xlabel("Inscriptions")
            plt.ylabel("Département")
            plt.grid(True, axis="x", alpha=0.3)
            p = "out/figures/top_departements_non_locaux.png"
            plt.tight_layout(); plt.savefig(p, dpi=150); plt.close(); paths.append(p)
        # Distribution des âges non-locaux
        ages = ext["age"].dropna().astype(int)
        if not ages.empty:
            plt.figure(figsize=(8, 5))
            plt.hist(ages, bins=range(int(ages.min()), int(ages.max()) + 2, 2), color="#457b9d")
            plt.title("Âges – participants non-locaux")
            plt.xlabel("Âge")
            plt.ylabel("Participants")
            plt.grid(True, alpha=0.3)
            p = "out/figures/ages_non_locaux.png"
            plt.tight_layout(); plt.savefig(p, dpi=150); plt.close(); paths.append(p)
    return paths


def write_full_report(kpis: Dict[str, Any], seg_profiles: pd.DataFrame, seg_sizes: pd.Series, logit: Dict[str, pd.DataFrame], aux: Dict[str, Any], fig_paths: List[str], non_local: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.append("# Rapport complet MOE 2024 – Inscriptions\n")

    # Lien programme 2025
    lines.append("Source programme 2025: [`marseilleoutdoorexperiences.fr/programme-2025`](https://www.marseilleoutdoorexperiences.fr/programme-2025)\n")

    lines.append("\n## 1. KPIs clés\n")
    lines.append(f"- Total inscrits: **{kpis.get('inscrits_total')}**\n")
    pmix = kpis.get("parcours_mix", {})
    if pmix:
        mix_str = ", ".join([f"{k}: {v}" for k, v in pmix.items()])
        lines.append(f"- Mix parcours: {mix_str}\n")
    lines.append(f"- Sexe: {kpis.get('sexe_mix')}\n")
    lines.append(f"- Tranches d'âge (principales): {kpis.get('tranches_age')}\n")
    lines.append(f"- Local (13): **{kpis.get('local_13_pct')}%** | France: **{kpis.get('france_pct')}%**\n")
    lines.append(f"- Présence (émargement): **{kpis.get('presence_pct')}%** | Paiement ok: **{kpis.get('paid_ok_pct')}%**\n")
    if "licence_ok_pct" in kpis:
        lines.append(f"- Licences/Certificats validés: **{kpis.get('licence_ok_pct')}%**\n")
    lines.append(f"- Promo utilisée: **{kpis.get('promo_pct')}%** | Early ticket: **{kpis.get('early_ticket_pct')}%**\n")
    lines.append(f"- Fenêtre inscriptions: {kpis.get('inscriptions_weeks_min')} → {kpis.get('inscriptions_weeks_max')} | Lead médian: {kpis.get('lead_days_median')} j\n")

    lines.append("\n## 2. Géographie\n")
    lines.append(f"Top départements: {kpis.get('top_departements')}\n")
    lines.append(f"Top villes: {kpis.get('top_villes')}\n")

    lines.append("\n## 3. Cible hors région (non-locaux)\n")
    lines.append(f"- Part non-locaux: **{non_local.get('share_non_locaux_pct')}%** ({non_local.get('inscrits_non_locaux')} pers.)\n")
    lines.append(f"- Top départements non-locaux: {list(non_local.get('top_departements_ext', {}).items())[:10]}\n")
    lines.append(f"- Focus Paris (75): {non_local.get('dep_75_count')} | lead médian: {non_local.get('dep_75_lead_median')} j | promo: {non_local.get('dep_75_promo_pct')}% | présence: {non_local.get('dep_75_presence_pct')}%\n")
    lines.append(f"- Focus Lyon (69): {non_local.get('dep_69_count')} | lead médian: {non_local.get('dep_69_lead_median')} j | promo: {non_local.get('dep_69_promo_pct')}% | présence: {non_local.get('dep_69_presence_pct')}%\n")
    lines.append(f"- Non-locaux – lead médian: {non_local.get('lead_median')} j | promo: {non_local.get('promo_pct')}% | présence: {non_local.get('presence_pct')}%\n")
    lines.append("Recommandations: offres week‑end (train/hôtel), contenus destination, partenariats (SNCF/hôtels), ciblage paid sur 75/69 et lookalikes inscrits non‑locaux.\n")

    lines.append("\n## 4. Temporalité\n")
    lines.append("Courbes hebdomadaires et répartition par parcours (voir figures).\n")

    lines.append("\n## 5. Promos & Clubs\n")
    if "type_paiement_mix" in kpis:
        lines.append(f"- Types de paiement: {kpis['type_paiement_mix']}\n")
    if aux.get("top_codes"):
        lines.append(f"- Top codes promo: {list(aux['top_codes'].items())[:10]}\n")
    if aux.get("top_clubs"):
        lines.append(f"- Top clubs: {list(aux['top_clubs'].items())[:10]}\n")

    lines.append("\n## 6. Segmentation (KMeans)\n")
    lines.append("Segments sur variables: âge, lead, local, sexe, club, promo, parcours (indic).\n")
    lines.append("Tailles de segments:\n")
    for s, n in seg_sizes.sort_index().items():
        lines.append(f"- Segment {s}: {n}\n")
    lines.append("\nProfils moyens par segment:\n")
    lines.append(seg_profiles.round(2).to_string())
    lines.append("\n")

    lines.append("\n## 7. Modèles explicatifs (régressions logistiques)\n")
    if "parcours21_coefs" in logit:
        lines.append("- Choix 21km (vs 12km) – coefficients (features standardisées):\n")
        lines.append(logit["parcours21_coefs"].round(3).to_string(index=False))
        lines.append("\n")
    if "presence_coefs" in logit:
        lines.append("- Présence (émargement) – coefficients (features standardisées):\n")
        lines.append(logit["presence_coefs"].round(3).to_string(index=False))
        lines.append("\n")

    lines.append("\n## 8. Recommandations marketing 2025\n")
    lines.append("- Renforcer le ciblage local (13) et voisins (83/84), créas locales et partenariats clubs.\n")
    lines.append("- Acquisition non-locaux (75/69/IDF): bundles week‑end, contenus destination, ciblage paid dédié, offres limitées.\n")
    lines.append("- Calendrier d’activation: pics à exploiter et relances J‑90/J‑60/J‑30; pousser early‑bird.\n")
    lines.append("- 21km: messages performance/défi, codes ciblés; 12km: fun/entre amis, volumes.\n")
    lines.append("- Optimiser la conversion tardive via rappels logistiques S‑2/S‑1 pour maintenir la présence élevée.\n")

    if fig_paths:
        lines.append("\n## 9. Figures\n")
        for p in fig_paths:
            lines.append(f"- {p}\n")

    with open("out/report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    ensure_out_dirs()

    csv_path = "SEB ANALYSE COURSE - sheet1.csv"
    df = read_registrations(csv_path)
    df = parse_dates(df)

    # Race day inferred from dataset (2024 edition)
    race_day = pd.Timestamp("2024-11-09")
    df = derive_features(df, race_day)
    df = bucketize_lead_days(df)

    # Save a cleaned, slimmed dataset
    keep_cols = [
        "REF", "PARCOURS", "parcours_norm", "CIVILITE", "sexe", "date_naissance", "age", "tranche_age",
        "PAYS", "pays", "is_fr", "VILLE", "CODE POSTAL", "DEPARTEMENT (CODE)", "dep_code", "is_local_13",
        "CLUB", "has_club", "PAIEMENT", "paid_ok", "DATE INSCRIPTION", "date_inscription", "lead_days",
        "CODE PROMO", "promo_used", "EMARGEMENT", "present", "is_early_ticket", "lead_bucket"
    ]
    slim_df = df[[c for c in keep_cols if c in df.columns]].copy()
    slim_df.to_csv("out/registrations_slim.csv", index=False)

    # KPIs
    kpis = compute_kpis(df)
    with open("out/kpis.json", "w", encoding="utf-8") as f:
        json.dump(kpis, f, ensure_ascii=False, indent=2)

    # Aggregated tables & plots
    save_tables(df)
    fig_paths = plot_figures(df)
    fig_paths += plot_more_figures(df)
    # Non-locaux
    non_local = compute_non_local_insights(df)
    fig_paths += plot_non_local_figures(df)

    # Segmentation, modèles, promos/clubs
    seg_profiles, seg_sizes, df_with_segments = run_kmeans_segmentation(df, n_clusters=5)
    logit = run_logistic_models(df)
    aux = analyze_promos_and_clubs(df)

    # Insights MD (court) + Rapport complet
    write_insights_md(kpis, fig_paths)
    write_full_report(kpis, seg_profiles, seg_sizes, logit, aux, fig_paths, non_local)

    # Also print a short console summary
    print(json.dumps({
        "inscrits_total": kpis["inscrits_total"],
        "parcours_mix": kpis["parcours_mix"],
        "presence_pct": kpis["presence_pct"],
        "local_13_pct": kpis["local_13_pct"],
        "promo_pct": kpis["promo_pct"],
        "segments": {int(k): int(v) for k, v in seg_sizes.to_dict().items()},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main() 