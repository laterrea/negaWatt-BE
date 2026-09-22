# Workshop review — tertiary heat

Review by Sylvain, 2026-09-21, on the seven question screens. This file is the **working
tracker**: the original comments are kept verbatim under each item, followed by what was
decided and what was done.

Legend: `[ ]` open · `[x]` done · `[~]` done with a deviation, explained.

---

## G — Global

### G1 — "Bâtiments de services" → "Bâtiments tertiaires" (FR only)
> in French "Bâtiments de services" should be changed by "Bâtiments tertiaires" everywhere

- [x] **Done.** 20 French strings in `tertiary-heat.yaml`: the topic title, the seven
  questions, the subtitles, the `tangible` read-backs, the fact prose and the
  `historyNote`. The elliptical forms went with it — `short:` "Thermostat / Climatisation /
  Eau chaude **des services**" → "**du tertiaire**", and the unit label "m² de
  services/pers." → "m² tertiaires/pers.". Dutch and English are untouched (the comment is
  explicitly FR-only, and *dienstengebouwen* / *service buildings* are the right words in
  those languages). Only English prose and source citations still contain "services".

---

## Q1/7 — `ter-floor-area` (surface tertiaire par personne)

### Q1a — remove the "Concrètement" card
> enlèver la ficher "concrètement", qui n'apporte pas grand chose

- [x] **Done.** The `kind: tangible` fact ("{refValue} m² par personne, c'est une pièce de
  4 mètres sur 5…") is gone. The card said nothing the *benchmark* card next to it did not
  say better, and it spent a quarter of the paper card on an arithmetic restatement.
  The slider's own read-back (the top-level `tangible:` field, "{value} m² de bâtiments
  tertiaires pour chaque habitant") is a different thing and stays.

### Q1b — new card on the teleworking potential
> voir si on peut faire une nouvelle fiche avec le potentiel du télé-travail (vérifier
> également les hypothèses negawatt sur ce sujet)

- [x] **Done — one new card, `kind: lever`, headed "Le télétravail", placed last before the
  reveal card.** It carries a plot.

  **First, the check you asked for. négaWatt-BE says almost nothing about telework.**
  §1.2.2 of the buildings notebook motivates the −10 % in one sentence — "reduce the
  material footprint and the footprint related to heating and cooling" — and carries your
  own margin note, *"Should further motivate this!"*. The only place the word appears is
  the label of the public-site hypothesis, `reference="Telework, space rationalisation"`
  (cell 99). So telework is *named* as the mechanism and never quantified anywhere in the
  model. The workshop already said as much on the reveal side, and still does: the
  justification lists telework and desk sharing as the "leviers implicites", and the debate
  says telework "libère des bureaux sans les faire disparaître".

  **What the new card says** — the one thing worth knowing before setting a m²/person
  target is that *the take-up already happened and the floor area did not follow*:

  | figure | source |
  |---|---|
  | Belgians working from home, usually or sometimes: 24,6 % (2019) → 39,9 % (2021 peak) → **37,1 % (2025)** — plotted as the card's line chart | Eurostat `lfsa_ehomp`, 15-64, BE |
  | Brussels office stock **12 489 055 m²** end-2022, −212 918 m² against 2020, the lowest in fifteen years | Observatoire des bureaux n°40, perspective.brussels, mars 2024 |
  | marketed vacancy **8,7 %** in Nov 2023, **> 1 million m²** empty (plus an unmeasured "vacance cachée") | *idem* |
  | conversions since 1997: **1 968 421 m²** — about 16 % of today's stock in 26 years — **70 % of it to housing** | *idem* |

  The card's own line is *"vider un bureau n'est pas le supprimer"*, which is the honest
  reading and sets up the debate card without spoiling the target. Two sources, so the plot
  carries its own `source`/`url` pair (Eurostat) while the fact carries the Observatoire —
  the mechanism the README documents for exactly this case.

  Q1 keeps four pre-answer cards: trend, benchmark, structure, télétravail.

---

## Q2/7 — `ter-insulation`

> (no comment)

---

## Q3/7 — `ter-thermostat`

### Q3a — retitle the question (FR)
> changer le titre de la question "De combien de degrés la sobriété peut-elle abaisser le
> thermostat des bâtiments de services d'ici 2050 ?" en "De combien peut-on raisonablement
> abaisser le thermostat des bâtiments de bureau d'ici 2050?"

- [~] **Done, with one word changed — flagging it.** The question now reads *"De combien
  peut-on raisonnablement abaisser le thermostat des bâtiments **tertiaires** d'ici
  2050 ?"* (and NL/EN were rephrased to match). Two deliberate departures from the note:
  *raisonablement* → *raisonnablement*, and **bureau → tertiaire**.
  The second is the one worth a look: the lever is the average setpoint of the *whole*
  tertiary stock — its own subtitle says "sur tout le parc tertiaire", its read-back says
  "dans les bureaux, les classes et les magasins", and négaWatt's 1 °C is a stock average
  that includes hospitals and shops. A question headed "bâtiments de bureau" would collect
  an office answer and feed it to a hospital, and it would also contradict G1 two lines
  further down. If you want the office wording anyway, it is one line:
  `fr:` in `tertiary-heat.yaml:464`.

### Q3b — invert the y axis on the charts
> Inverser les ordonnées sur les graphes ? (diminution va vers le haut actuellement,
> contre-intuitif -> aussi adapter caption du plot)

- [x] **Done, as an opt-in flag rather than a hard-coded special case (D60).**
  `chartInvertY: true` on a lever makes its chart plot the *signed change* instead of the
  lever's own "so much less" value. The axis now runs **+1 °C at the top to −5 °C at the
  bottom**, the 2019 anchor sits at 0, and an answer of "3 °C en moins" lands at **−3,
  below the anchor** — verified in the browser, not assumed. Nothing else moves: the
  slider, the readout ("3 °C en moins"), the stored answer, the summary chart and every
  number in the prose keep the lever's own sign. Three files: `spark.js` (a `flipped()`
  copy of the options — it must not mutate, `responsive()` redraws with the same object),
  `play.js` (one line), `build_workshop_content.py` (accepts the flag, refuses a non-boolean).
- [x] **Caption adapted**, in all three languages. The `historyNote` now opens with *"Le
  graphe porte l'écart de consigne par rapport à 2019 — abaisser le thermostat fait
  descendre la courbe"* before the existing explanation of why there is no observed series.
- **Not done, on purpose — your call:** `residential-heat`'s `thermostat` lever is the
  identical case (same unit, same `historyAbsent`, same upside-down line) and was signed
  off in its own review round. One line in `residential-heat.yaml` turns it on the same
  way, plus the same sentence in front of its `historyNote`. Say the word.

---

## Q4/7 — `ter-cooling`

### Q4a — new card on Belgian law on maximum office temperature
> ajouter une fiche sur la législation belge en terme de température maximale dans le
> bureaux (+ si possible trouver des statistiques?)

- [x] **Done — one new card, `kind: structure` headed "La loi belge", with a plot.**

  **What the law actually says** (Code du bien-être au travail, Livre V, Titre 1, art.
  V.1-3 and V.1-4, consolidated 2024 — read, not summarised from a secondary page):
  - There is **no maximum air temperature** for an office anywhere in Belgian law. The
    ceiling is an *action value* on the **WBGT index — 29 for light or very light work**
    (26 medium, 22 heavy, 18 very heavy). WBGT is a heat-stress index, not a thermostat
    reading, and the SPF Emploi's own page says the index "may be lower than the
    temperature shown by an ordinary thermometer" — so the threshold bites well above
    29 °C on the wall.
  - Above it, art. V.1-4 requires a *programme* of technical and organisational measures:
    ventilation, workload, schedules, rest periods, clothing, free cool drinks. **Air
    conditioning is one option among seven, never an obligation.**
  - The one place the law does speak in plain degrees is the **floor**: a minimum of 18 °C
    for very light work and 16 °C for light work (art. V.1-3 §1). Office desk work sits on
    the boundary of those two classes (117 W), so the card says "16 à 18 °C" rather than
    picking one.

  That is the card's point, and it is the right one for this question: nothing in Belgian
  law forces a building to be cooled, so the 2050 level is a choice and not a given. It
  also plays against the Spanish decree quoted on Q3, which *did* legislate a setpoint.

  **The statistics you asked for.** No Belgian series measures office temperatures, and the
  cooling-degree-day card next to it already carries the climate trend. What the card adds
  instead is the *frequency* of the days when the question arises, plotted as a two-bar
  chart with its own source: at Uccle, **20 summer days (max ≥ 25 °C) a year in 1961-1990,
  30 in 1991-2020**; heat days (≥ 30 °C) went from **2 to 5** (caption). Source: IRM/KMI
  climate normals.

  Q4 keeps four pre-answer cards: trend, structure, benchmark, la loi belge.

### Q4b — remove the "Concrètement" card
> Enlever la fiche "concrètement"

- [x] **Done.** The `kind: tangible` fact (the 5 000 m² office block needing 86 000 kWh of
  cooling) is gone; the slider read-back stays.

---

## Q5/7 — `ter-hot-water`

### Q5a — remove the "À manier avec prudence" card
> enlever la fiche "a manier avec prudence"

- [x] **Done.** The `kind: caution` fact ("attention à l'année de référence… 2019 est un
  creux local") is gone. Its substance is not lost: the *trend* card immediately above it
  already says 2019-2020 is a COVID trough and that 2023 is the highest point of the whole
  series, and the reveal's justification still quotes how far above the target 2023 sits.

### Q5b — new card on the sufficiency potential
> ajouter une fiche sur le potentiel de sobriété dans ce domaine, via des mesures
> concrètes, des retours d'expérience, etc etc.

- [x] **Done — one new card, `kind: lever`, headed "Où la sobriété mord".** It is built on
  the one piece of real field evidence that exists for tertiary hot water: COSTIC's
  monitoring for ADEME and GRDF — **3 500 meter readings on about 400 sites**, 45
  instrumented follow-ups, published September 2020. French, and the card says so; there is
  no Belgian equivalent.

  What it puts on the table, all from that guide's recap table (p. 150) and §5.3:

  | | need at 40 °C |
  |---|---|
  | an office worker (basins + cleaning) | **5 to 10 L per working day** |
  | one hotel night, 3★ / 4★ | **78 ± 21 / 108 ± 31 L** |
  | a pupil, school with no canteen | 2 to 4 L per school day |
  | …with a canteen, dishwashers on **hot** water | **7 to 12 L** |
  | …the same canteen, dishwashers on **cold** water | **4 to 7 L** |
  | shower fittings | 6 l/min today against 8-10 l/min on old ones |

  The card leads with *"et ce ne sont pas les robinets des bureaux"*, which is the concrete
  measure hiding in the numbers: the volume is in showers, hotel rooms, care and kitchens,
  so that is where a target has to bite. The school dishwasher line is the one measured
  before/after in the source — a single plumbing choice, roughly half the need.

### Q5c — say what the hot water is actually used for
> Fiche Concrètement: expliquer pour quoi est utilisée cette eau chaude sanitaire (au moins
> de façon qualitative)

- [x] **Done.** The "Concrètement" card kept its arithmetic ({litres2019} L of 40 °C water
  a day per person, against 64 at home) and gained the answer to the question, in the
  order of the stock: *lavabos et nettoyage des bureaux, écoles et commerces ; douches et
  vestiaires des salles de sport, piscines et ateliers ; toilette des patients et des
  résidents et blanchisserie des hôpitaux et maisons de repos ; salles de bains des
  hôtels ; plonge des cuisines de collectivité.* The old card named four of those in a
  trailing fragment; this one is a full sentence and covers care and hotels, which are
  where the volume actually sits (see Q5b).

---

## Q6/7 — `ter-catering` · Q7/7 — `ter-district-heat`

> (no comment)

---

## Print budget (checked, because both new cards carry a plot)

Measured with the print stylesheet applied at 190 mm, the method the README prescribes.
The first draft ran Q1 to 147,9 mm (fr) / 147,6 (nl) and pushed the **Dutch** print run
from four A4 sheets to five. Fixed in prose — the two new texts tightened, and the two
chart `source:` strings shortened, because a long source line wraps the figcaption onto a
second line and that is 3,4 mm. Final card heights:

| | Q1 | Q2 | Q3 | Q4 | Q5 | Q6 | Q7 | sheets |
|---|---|---|---|---|---|---|---|---|
| fr | 136,4 | 128 | 128 | 135,1 | 129,5 | 128 | 138,6 | 4 |
| nl | 140,3 | 128 | 128 | 139,3 | 133,5 | 128 | 138,6 | 4 |
| en | 136,4 | 128 | 128 | 135,1 | 128 | 128 | 128 | 4 |

128 mm is the `min-height`; a page holds 277 mm and each card carries a 6 mm bottom margin.
Q7 at 138,6 mm predates this round.

---

## Checks and deployment

- [x] `python scripts/build_workshop_content.py` — 30 levers, 160 facts, 32 with a plot
- [x] `python scripts/build_workshop_content.py --check`
- [x] `python scripts/verify_workshop_export.py` — 327 checks, all passed
- [x] `python scripts/test_workshop_helpers.py` — 8 passed
- [x] `python scripts/test_workshop_api.py` — 59 passed (local shim)
- [x] all four new/changed screens read in **fr, nl and en** in the browser, play + cards +
  reveal; the inverted axis verified by reading the rendered SVG, not by eye
- [x] the four new source URLs fetched and each returns 200
- [ ] `bash scripts/deploy_website.sh` (no commit — per instruction)

---

## Still open for you

1. **Q3a** — the question says "bâtiments **tertiaires**", not "bâtiments de bureau". One
   line if you want it the other way: `tertiary-heat.yaml:464`.
2. **Q3b** — `residential-heat`'s `thermostat` lever has the identical upside-down chart and
   was not touched. `chartInvertY: true` plus a sentence in front of its `historyNote` does
   it.
