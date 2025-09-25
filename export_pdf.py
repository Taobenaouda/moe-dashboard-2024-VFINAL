from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
import json
import os
import datetime as dt
import csv

OUT_DIR = "out"
FIG_DIR = os.path.join(OUT_DIR, "figures")


def safe_img(path: str, width: float = 16*cm) -> Image | None:
    if os.path.exists(path):
        img = Image(path)
        # Resize keeping aspect ratio
        iw, ih = img.wrap(0, 0)
        ratio = width / iw
        img._restrictSize(width, ih * ratio)
        return img
    return None


def make_table(data, col_widths=None):
    t = Table(data, colWidths=col_widths)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F0F0F0")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#333333")),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
        ("TOPPADDING", (0, 0), (-1, 0), 6),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#DDDDDD")),
    ]))
    return t


def sanitize_text(text: str) -> str:
    if not isinstance(text, str):
        return text
    replacements = {
        "•": "-",
        "–": "-",
        "—": "-",
        "‑": "-",
        "’": "'",
        "“": '"',
        "”": '"',
        "…": "...",
        "€": " EUR",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text


def P(text: str, style) -> Paragraph:
    return Paragraph(sanitize_text(text), style)


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_pdf():
    kpis = load_json(os.path.join(OUT_DIR, "kpis.json"))
    non_local = {}
    non_local_path = os.path.join(OUT_DIR, "non_local_kpis.json")
    if os.path.exists(non_local_path):
        non_local = load_json(non_local_path)

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TitleBig", fontSize=20, leading=24, spaceAfter=12))
    styles.add(ParagraphStyle(name="H2", fontSize=14, leading=18, spaceBefore=10, spaceAfter=6))
    styles.add(ParagraphStyle(name="Body", fontSize=10.5, leading=14))
    styles.add(ParagraphStyle(name="Small", fontSize=9, leading=12, textColor=colors.HexColor("#666666")))

    doc = SimpleDocTemplate(os.path.join(OUT_DIR, "report.pdf"), pagesize=A4,
                            leftMargin=2*cm, rightMargin=2*cm, topMargin=1.5*cm, bottomMargin=1.5*cm)

    story = []

    # Cover
    story.append(P("Marseille Outdoor Experiences - Rapport Analytique 2024", styles["TitleBig"]))
    story.append(P(f"Date: {dt.date.today().isoformat()}", styles["Small"]))
    story.append(Spacer(1, 12))
    story.append(P("Objectif: fournir une synthese data-centree de l'edition 2024 et des recommandations operables pour 2025.", styles["Body"]))
    story.append(P("Programme 2025: marseilleoutdoorexperiences.fr/programme-2025", styles["Small"]))
    story.append(Spacer(1, 14))

    # KPIs clés
    story.append(P("1. Inscriptions", styles["H2"]))
    bullets = [
        f"Total inscrits: <b>{kpis.get('inscrits_total')}</b>",
        f"Mix parcours: {kpis.get('parcours_mix')}",
        f"Sexe: {kpis.get('sexe_mix')}",
        f"Tranches d'âge: {kpis.get('tranches_age')}",
        f"Local (13): <b>{kpis.get('local_13_pct')}%</b> | France: {kpis.get('france_pct')}%",
        f"Présence (émargement): <b>{kpis.get('presence_pct')}%</b> | Paiement ok: {kpis.get('paid_ok_pct')}%",
        f"Promo utilisée: <b>{kpis.get('promo_pct')}%</b> | Early ticket: {kpis.get('early_ticket_pct')}%",
        f"Fenêtre d'inscriptions: {kpis.get('inscriptions_weeks_min')} → {kpis.get('inscriptions_weeks_max')} | Lead médian: {kpis.get('lead_days_median')} j",
    ]
    story.extend([P(f"- {b}", styles["Body"]) for b in bullets])

    # Figures principales
    for fig in [
        "inscriptions_by_week.png",
        "parcours_by_tranche.png",
        "age_distribution.png",
        "top_departements.png",
        "lead_days_hist.png",
        "presence_by_lead_parcours.png",
    ]:
        img = safe_img(os.path.join(FIG_DIR, fig))
        if img:
            story.append(Spacer(1, 8))
            story.append(img)

    # Non-locaux
    story.append(PageBreak())
    story.append(P("3. Analyse geographique et temporelle - Public hors region", styles["H2"]))
    if non_local:
        bullets2 = [
            f"Part non-locaux: <b>{non_local.get('share_non_locaux_pct')}%</b> ({non_local.get('inscrits_non_locaux')} pers.)",
            f"Top départements non-locaux: {list(non_local.get('top_departements_ext', {}).items())[:10]}",
            f"Paris (75): {non_local.get('dep_75_count')} | lead médian: {non_local.get('dep_75_lead_median')} j | promo: {non_local.get('dep_75_promo_pct')}% | présence: {non_local.get('dep_75_presence_pct')}%",
            f"Lyon (69): {non_local.get('dep_69_count')} | lead médian: {non_local.get('dep_69_lead_median')} j | promo: {non_local.get('dep_69_promo_pct')}% | présence: {non_local.get('dep_69_presence_pct')}%",
            f"Non-locaux global: lead médian: {non_local.get('lead_median')} j | promo: {non_local.get('promo_pct')}% | présence: {non_local.get('presence_pct')}%",
        ]
        story.extend([P(f"- {b}", styles["Body"]) for b in bullets2])

    for fig in [
        "top_departements_non_locaux.png",
        "ages_non_locaux.png",
        "inscriptions_by_week_parcours.png",
    ]:
        img = safe_img(os.path.join(FIG_DIR, fig))
        if img:
            story.append(Spacer(1, 8))
            story.append(img)

    # Data-driven insights (moins marketing, plus analytique)
    story.append(PageBreak())
    story.append(P("3.2 Temporalite & lead time", styles["H2"]))
    story.append(P("Evolutions hebdomadaires des inscriptions et distribution du lead time.", styles["Body"]))

    # 3.1 Géographie – Top départements (table) + figure
    story.append(Spacer(1, 6))
    story.append(P("3.1 Geographie (top departements)", styles["H2"]))
    dep_counts_csv = os.path.join(OUT_DIR, "dep_counts.csv")
    if os.path.exists(dep_counts_csv):
        rows = [["Département", "Inscriptions"]]
        try:
            with open(dep_counts_csv, newline="", encoding="utf-8") as f:
                r = csv.reader(f)
                next(r, None)  # header
                for i, row in enumerate(r):
                    if i >= 15:
                        break
                    rows.append([row[0], row[1]])
            story.append(make_table([[sanitize_text(c) for c in row] for row in rows], [3*cm, 3*cm]))
        except Exception:
            pass
    img = safe_img(os.path.join(FIG_DIR, "top_departements.png"))
    if img:
        story.append(Spacer(1, 6))
        story.append(img)

    # 3.2 Temporalité & lead time – figures
    story.append(Spacer(1, 10))
    story.append(P("3.3 Temporalite & lead time (figures)", styles["H2"]))
    for fig in [
        "inscriptions_by_week.png",
        "lead_days_hist.png",
        "presence_by_lead_parcours.png",
    ]:
        img = safe_img(os.path.join(FIG_DIR, fig))
        if img:
            story.append(Spacer(1, 6))
            story.append(img)

    # 3.3 Promotions & clubs – tableaux
    story.append(Spacer(1, 10))
    story.append(P("3.4 Promotions & clubs (synthese)", styles["H2"]))
    promo_csv = os.path.join(OUT_DIR, "promo_rate_by_week.csv")
    if os.path.exists(promo_csv):
        try:
            rows = [["Semaine", "Promo %"]]
            with open(promo_csv, newline="", encoding="utf-8") as f:
                r = csv.reader(f)
                next(r, None)
                for i, row in enumerate(r):
                    if i >= 10:
                        break
                    rows.append([row[0], row[1]])
            story.append(make_table([[sanitize_text(c) for c in row] for row in rows], [4*cm, 3*cm]))
        except Exception:
            pass
    clubs_csv = os.path.join(OUT_DIR, "presence_rate_by_top_clubs.csv")
    if os.path.exists(clubs_csv):
        try:
            rows = [["Club", "Présence %"]]
            with open(clubs_csv, newline="", encoding="utf-8") as f:
                r = csv.reader(f)
                for i, row in enumerate(r):
                    if i >= 15:
                        break
                    rows.append([row[0], row[1]])
            story.append(Spacer(1, 6))
            story.append(make_table([[sanitize_text(c) for c in row] for row in rows], [9*cm, 3*cm]))
        except Exception:
            pass

    # 4. Recommandations data‑centrées (professionnelles)
    story.append(PageBreak())
    story.append(P("3. Recommandations", styles["H2"]))
    recos_data = [
        "Standardiser le schéma de données d’inscription (types, encodage, dictionnaire).",
        "Suivre systématiquement le lead time et l’attrition entre inscription/paiement/présence par parcours.",
        "Mettre en place des cohortes hebdomadaires avec objectifs cibles (inscrits, présence).",
        "Construire des segments opérationnels simples (local 13 / hors‑13 / 75‑69 / clubs) et suivre leur évolution.",
        "Mesurer l’effet des codes promos sur la présence (A/B simple par parcours).",
        "Prioriser les départements à fort volume mais présence moyenne pour actions ciblées (voir top départements).",
        "Exporter mensuellement des listes ‘clubs à fort potentiel’ (volume élevé, présence < moyenne).",
    ]
    story.extend([P(f"- {r}", styles["Body"]) for r in recos_data])

    # 5. Limites & pistes d’approfondissement
    # (Optional) Annexes / limites – omis pour version courte centrée data

    # Footer
    story.append(Spacer(1, 16))
    story.append(P("Contact & suivis: pixels actifs, UTM, cohortes lead-time; tableaux et figures details dans le dossier out/.", styles["Small"]))

    doc.build(story)


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    build_pdf() 