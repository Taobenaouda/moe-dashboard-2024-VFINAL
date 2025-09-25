# Rapport complet MOE 2024 – Inscriptions

Source programme 2025: [`marseilleoutdoorexperiences.fr/programme-2025`](https://www.marseilleoutdoorexperiences.fr/programme-2025)


## 1. KPIs clés

- Total inscrits: **2393**

- Mix parcours: 12km: 1463, 21km: 930

- Sexe: {'H': 1470, 'F': 923}

- Tranches d'âge (principales): {'<18': 14, '18-24': 313, '25-34': 946, '35-44': 548, '45-54': 366, '55-64': 173, '65+': 33}

- Local (13): **76.3%** | France: **98.8%**

- Présence (émargement): **90.8%** | Paiement ok: **100.0%**

- Licences/Certificats validés: **97.2%**

- Promo utilisée: **20.3%** | Early ticket: **100.0%**

- Fenêtre inscriptions: 2024-01-01 00:00:00 → 2024-12-09 00:00:00 | Lead médian: 63.0 j


## 2. Géographie

Top départements: {'13': 1826, '83': 66, '75': 49, '69': 40, '34': 36, '84': 34, '30': 21, '38': 18, '04': 17, '42': 17, '05': 17, '26': 16, '06': 15, '92': 12, '31': 12}

Top villes: {'Marseille': 1194, 'MARSEILLE': 129, 'Aix-en-Provence': 49, 'Paris': 43, 'marseille': 33, 'Lyon': 23, 'Allauch': 19, 'Aubagne': 19, 'La Ciotat': 18, 'Aix en Provence': 16, 'Toulon': 16, 'Vitrolles': 13, 'Montpellier': 12, 'Velaux': 9, 'Grenoble': 9}


## 3. Cible hors région (non-locaux)

- Part non-locaux: **23.7%** (567 pers.)

- Top départements non-locaux: [('83', 66), ('75', 49), ('69', 40), ('34', 36), ('84', 34), ('30', 21), ('38', 18), ('04', 17), ('42', 17), ('05', 17)]

- Focus Paris (75): 49 | lead médian: 58.0 j | promo: 20.4% | présence: 81.6%

- Focus Lyon (69): 40 | lead médian: 79.0 j | promo: 22.5% | présence: 97.5%

- Non-locaux – lead médian: 73.0 j | promo: 15.0% | présence: 89.1%

Recommandations: offres week‑end (train/hôtel), contenus destination, partenariats (SNCF/hôtels), ciblage paid sur 75/69 et lookalikes inscrits non‑locaux.


## 4. Temporalité

Courbes hebdomadaires et répartition par parcours (voir figures).


## 5. Promos & Clubs

- Types de paiement: {'Carte bancaire': 2390, 'Inscription manuelle': 3}

- Top codes promo: [('RCCMARSEILLEBB', 103), ('MOEMAISONMERE', 44), ('MOE2024CHALLENGE', 40), ('MOEMESINFOS', 39), ('GATEMOE2024', 31), ('MOEPORT', 30), ('MOEMC', 25), ('RECOMOE2024', 11), ('MOEJAM', 11), ('RTL2MOE', 10)]

- Top clubs: [('RCC', 15), ('Running Club Catalans', 14), ('SCO STE-MARGUERITE MARSEILLE*', 12), ('GRC', 9), ('LE CLUB CAMPUS', 6), ('WAKE UP CAFE', 5), ('MMRC', 4), ('OLYMPIQUE CABRIES CALAS', 4), ('Running club catalans', 4), ('Courir à Meyreuil', 4)]


## 6. Segmentation (KMeans)

Segments sur variables: âge, lead, local, sexe, club, promo, parcours (indic).

Tailles de segments:

- Segment 0: 375

- Segment 1: 407

- Segment 2: 853

- Segment 3: 346

- Segment 4: 412


Profils moyens par segment:

           age  lead_days  is_local  is_female  has_club_int  promo_int   p21
segment                                                                      
0        40.37      94.79      0.71       0.38           1.0       0.37  0.56
1        35.86     101.24      0.00       0.37           0.0       0.00  0.46
2        36.06      91.25      1.00       0.46           0.0       0.00  0.00
3        34.46      92.50      0.85       0.42           0.0       1.00  0.35
4        34.96      97.86      1.00       0.21           0.0       0.00  1.00



## 7. Modèles explicatifs (régressions logistiques)

- Choix 21km (vs 12km) – coefficients (features standardisées):

     feature   coef
has_club_int  0.321
   lead_days  0.055
   promo_int  0.013
         age -0.079
    is_local -0.254
   is_female -0.523


- Présence (émargement) – coefficients (features standardisées):

     feature   coef
    is_local  0.124
   is_female  0.074
has_club_int  0.045
   lead_days  0.012
         age -0.014
   promo_int -0.151



## 8. Recommandations marketing 2025

- Renforcer le ciblage local (13) et voisins (83/84), créas locales et partenariats clubs.

- Acquisition non-locaux (75/69/IDF): bundles week‑end, contenus destination, ciblage paid dédié, offres limitées.

- Calendrier d’activation: pics à exploiter et relances J‑90/J‑60/J‑30; pousser early‑bird.

- 21km: messages performance/défi, codes ciblés; 12km: fun/entre amis, volumes.

- Optimiser la conversion tardive via rappels logistiques S‑2/S‑1 pour maintenir la présence élevée.


## 9. Figures

- out/figures/inscriptions_by_week.png

- out/figures/age_distribution.png

- out/figures/parcours_by_tranche.png

- out/figures/top_departements.png

- out/figures/inscriptions_by_week_parcours.png

- out/figures/lead_days_hist.png

- out/figures/presence_by_lead_parcours.png

- out/figures/top_departements_non_locaux.png

- out/figures/ages_non_locaux.png
