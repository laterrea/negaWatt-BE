# Workshop review — international mobility · progress tracker

Review comments by Sylvain Quoilin, worked through sequentially.
Status: ⬜ todo · 🟡 in progress · ✅ done · ⛔ blocked / dropped (with reason)

Files expected to be touched: `website/workshop/content/international-mobility.yaml`,
`website/workshop/content/ui.yaml`, `workshop_levers/international_mobility.py`,
`scripts/build_workshop_content.py`, `nW_BE_demand_model_transports.ipynb`.

Rebuild after each batch:
```bash
python scripts/build_workshop_content.py --check
```
Levers-module changes additionally need the transport notebook re-run.

---

## Task list (from the raw notes, reorganised)

### Q1 — `long-haul-flights` (Voyages long-courriers)

| # | Task | Status |
|---|---|---|
| 1.1 | Fiche "Tendance passée": remplacer le texte par un graphique | ⬜ |
| 1.2 | Enlever la fiche "Contexte belge" | ⬜ |
| 1.3 | Titre de la fiche benchmark → "Ailleurs" (les chiffres sont mondiaux) | ⬜ |
| 1.4 | Enlever la fiche "Concrètement" | ⬜ |
| 1.5 | Ajouter une fiche: part de la population mondiale n'ayant jamais pris l'avion | ⬜ |

### Q2 — `short-haul-flights` (Voyages court-courriers)

| # | Task | Status |
|---|---|---|
| 2.1 | Fiche "Tendance passée": graphique au lieu du texte (comme 1.1) | ⬜ |
| 2.2 | Fiche "Contexte belge": préciser que la comparaison voiture vaut pour un seul occupant | ⬜ |
| 2.3 | Ajouter une fiche sur le potentiel de report modal avion → train | ⬜ |
| 2.4 | Courbe de Lorenz de la répartition des vols, si les données le permettent | ⬜ |

### Q3 — `long-haul-load` (Passagers par vol long-courrier)

| # | Task | Status |
|---|---|---|
| 3.1 | Titre → "Remplissage des vols long-courriers en 2050" | ⬜ |
| 3.2 | Enlever la fiche "Contexte belge" | ⬜ |
| 3.3 | Clarifier que la consommation moyenne par avion est constante dans la feuille de calcul; quantifier l'effet "avions plus gros = plus de conso" et justifier qu'il est faible — sinon revoir la formulation du levier | ⬜ |
| 3.4 | Données sur le taux de remplissage moyen actuel + expliquer le passage taux de remplissage ↔ passagers/vol | ⬜ |

### Q4 — `long-haul-fuel` (Consommation par km-avion, long-courrier)

| # | Task | Status |
|---|---|---|
| 4.1 | Trouver des données historiques; les tracer; si la tendance est visible sur le graphique, retirer la fiche "Tendance passée" | ⬜ |
| 4.2 | Enlever la fiche "Contexte belge" | ⬜ |
| 4.3 | Renommer "Comparaison internationale" → "Décarbonation de l'aviation"; insister sur les 3 % qui correspondent au levier | ⬜ |
| 4.4 | Enlever la fiche "Concrètement" | ⬜ |

### Q5 — `short-haul-load` (Passagers par vol court-courrier)

| # | Task | Status |
|---|---|---|
| 5.1 | Déplacer la question juste avant/après Q3 (levers similaires) | ⬜ |
| 5.2 | Mêmes remarques que Q3 (3.2, 3.3, 3.4) | ⬜ |
| 5.3 | Enlever la fiche "À manier avec prudence" | ⬜ |

### Q6 — `short-haul-fuel` (Consommation par km-avion, court-courrier)

| # | Task | Status |
|---|---|---|
| 6.1 | Fusionner avec Q4 (questions quasi équivalentes) | ⬜ |
| 6.2 | Reprendre les remarques 4.1 à 4.4 sur la question fusionnée | ⬜ |

### Q7 — `hydrogen-flights`

Rien à faire. ✅

### Q8 — Question supplémentaire: report modal du fret aérien

| # | Task | Status |
|---|---|---|
| 8.1 | Vérifier si le report modal du fret aérien est une hypothèse de la feuille de calcul | ⬜ |
| 8.2 | Si oui (ou si documentable): nouveau levier + fiches (quelles marchandises volent, tendances historiques, hub Alibaba à Liège) | ⬜ |

### Infrastructure

| # | Task | Status |
|---|---|---|
| I.1 | Vérifier le support des images dans les fiches (le tracker inland ne mentionne que les graphiques) | ⬜ |
| I.2 | Étiquette de fiche personnalisée (`label:`) — nécessaire pour 1.3 et 4.3 | ⬜ |

---

## Log

- Start: read the topic YAML, the lever module and the inland-mobility review notes
  (`notes_workshop.md`) for the conventions established there.
