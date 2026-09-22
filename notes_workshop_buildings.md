# Workshop review — residential heat (`residential-heat`)

Review by Sylvain, 2026-09-21, on the seven question screens of the *Chaleur des
logements* topic. This file is the **working tracker**: the original comments are kept
verbatim under each item, followed by what was decided and what was done.

Files touched by this round:

- `website/workshop/content/residential-heat.yaml` — the wording, the facts, the charts
- `website/workshop/content/ui.yaml` — shared strings (chart axis units)
- `workshop_levers/residential_heat.py` — which notebook quantities are levers, and the
  `facts` values the YAML interpolates
- `nW_BE_demand_model_buildings.ipynb` / `nW_BE_demand_data_aux.ipynb` — only where the
  review found a genuine modelling problem
- `nW_BE_demand_model_sub_functions.py`, `website/assets/js/workshop/impact.js` — the new
  `renovation` response kind (Q2)
- `docs/workshop_module.md` — the "why", as a review-round section (§16) and decisions
  D49-D55

Status legend: ☐ todo · ◐ in progress · ☑ done · ⊘ dropped (with reason)

---

## Second pass — 2026-09-21, review of the round itself

Sylvain's instruction: *"some items have not been properly addressed (or too fast) — review
the last changes and correct what is not correct or too quick and dirty."* This section is
that pass. It re-derives the figures from the live sources rather than trusting the round,
lists the six things that were wrong or cut short, and says plainly what is still open.

### What was re-checked against the source, and held

Every load-bearing number of the round was fetched again, not re-read:

| claim | checked against | verdict |
|---|---|---|
| Eurostat hot water flat 2016-2024 (the nine TJ figures hard-coded in the aux notebook) | `nrg_d_hhq` API, `FC_OTH_HH_E_WH`/`TOTAL`/TJ | **exact**, all nine |
| PV concentration, 51,5 % in May-August | `nrg_cb_pem` API, `RA420`, BE, 2025 | **exact**: 5 360,3 of 10 404,1 GWh |
| District-heat bars BE/FR/DE/EU27/DK/SE 2024 | `nrg_d_hhq`, derived heat ÷ (space heating + water heating) | **exact to two decimals**: 0,18 / 4,41 / 7,91 / 10,97 / 49,63 / 52,17 |
| The curated 2016-2024 curve, "Eurostat scaled to the JRC 2019 point" | recomputed from the API at factor 0,25010/0,20638 | **all nine points reproduce**, ±0,005 pt |
| Pipe losses 7,5 % of Belgian derived heat, 2019 | `nrg_bal_c`, `H8000`: DL 1 539,7 TJ over AFC+DL 20 569,6 | **7,49 %**, and the denominator is well chosen |
| Flanders 908 delivered / 955 injected GWh, 2024 | VEKA *Warmtenetrapportering 2024*, PDF fetched | **exact** (loss 4,9 %) |
| HRE4 Belgium 37 %, from 2 % in 2015, range 20-54 %, "slightly lower than the overall HRE average, which lies around 45 %" | HRE4 country roadmap, PDF fetched | **exact, and the 45 %-is-the-average reading is the report's own sentence** |
| Q2's arithmetic (−0,458 kWh/m²/yr from 79,5; JRC rate 2,2610 %/yr sd 0,0084; implied depth 25,49 %; 0,60 × 2,24 = 1,344) | recomputed from `history_buildings.js` | **exact**, and the `renovation` impact response reproduces the summary bars (−2,97 and −4,93 TWh) to the digit |
| the 21 source URLs of the topic | HTTP | all resolve; only tandfonline 403s to a bot (opens in a browser, but see below) |
| "nothing downstream moved" | re-ran both notebooks | `buildings.js` and `data/energy_totals_overrides.csv` reproduce **byte for byte** |

That is the good news, and it is most of the round: the bibliography, the arithmetic and the
Q7 harmonisation are sound. What follows is what was not.

### Fixed in this pass

- ☑ **R1 — A negative floor on charts of a quantity that cannot be negative.**
      The district-heat screen printed **−0,66** (and −2,22 for a larger answer) as the
      y-axis minimum. `domain()` in `spark.js` pads the range by 12 % and then tries to snap
      to a zero baseline, but the guard tested the *padded* minimum: once padding had pushed
      it below zero the snap could never fire — exactly the case that needs it. Now tested on
      the unpadded minimum. Also affected `cooling`, `ter-cooling` and `air-freight`; the
      round exposed it on district-heat by giving the lever an observed curve, where it
      previously had none. Verified in the browser: the floor now reads 0,00.
- ☑ **R2 — `decimals: 2` padded every answer on the screen, not just the reveal.**
      The round set two decimals so the 0,25 % reference would stop rounding to 0,3, and
      recorded the "15,00 %" on the reveal as *"ugly, but it is what stops 0,25 becoming
      0,3"*. It was worse than recorded: the play readout and the `tangible` sentence showed
      **"7,50 %"** for every answer on a 0,5-point slider, and cooling showed "2,70". Fixed
      at the root, in the three places a number is rendered: `i18n.js num()` (the page),
      `spark.js fmt()` (chart and dot-plot labels) and `format_number` in
      `build_workshop_content.py` (the `{placeholder}`s baked into the prose). `decimals` is
      a **ceiling** in all three now. The third is the one that mattered: *"négaWatt retient
      15,00 %"* is written into the reveal **at build time**, so the two JS fixes alone left
      it standing — worth knowing that a formatting change here needs a rebuild, not just a
      reload. An explicit `{x:d2}` still forces two decimals, which is what it is for
      (checked: `{airKwhPerTkm2019:d2}` still prints 0,90). Checked across all four topics:
      0,25 → 15 %, 0 → 2 °C, 1,22 → 2 personnes/voiture, 7,5 – 27 min-max. No figure loses a
      digit. *Three lines; revert them if the padded style was wanted.*
- ☑ **R3 — The Flemish 113 GWh was read as an end-use when it is a carrier.**
      VEKA's Figure 4 splits delivered energy by *fluid* — `Warm water`, `Stoom`, `Koude` —
      crossed with sector. "Warm water aan residentiële sector 113 GWh" is **all heat piped
      to homes** (steam goes to industry; residential steam was 0 in 2024), not domestic hot
      water. The card said *"113 GWh d'eau chaude aux logements"*, which a Flemish reader
      will hear as sanitary hot water. Now "113 GWh de chaleur aux logements (des réseaux à
      eau chaude ; la vapeur va à l'industrie)", in the three languages, and the same
      correction in `docs/workshop_module.md`. The argument it carries is unchanged, and in
      fact stronger: 113 GWh is *all* of Flemish household network heat against Eurostat's
      129 GWh for Belgium.
- ☑ **R4 — The cooking trend was two thirds data break, and the card explained it with
      behaviour.** `ref_RS_cook_trend = −1,6 kWh/household/year` is a 2000-2023 fit of a JRC
      series that steps from 102,0 ktoe (2021) to 77,1 (2022) — a quarter of the total in one
      year. Eurostat cooking is flat across that step (5 507 TJ in 2021, 5 517 in 2022) and
      falls only in 2023-24, where a gas-to-induction switch belongs, in *delivered* energy.
      Over the clean window the slope is **−0,94 kWh/household/year (2000-2019)**, barely
      −4 % in twenty years. The round flagged this and left the number on the card, under the
      sentence *"l'explication avancée est le changement des habitudes culinaires"* — a data
      artefact wearing a behavioural story, on a card a participant is invited to check.
      Now: notebook §2.1.2 restates the trend on 2000-2019 with a note on why it stops there,
      the tertiary §-cell's companion figure likewise (−0,523 → **−0,237 kWh/person/year**),
      `ref_RS_cook_trend` follows, and the card and its `historyNote` say the window and the
      reason. `ref_RS_tes_cok` (the 2019 anchor) and the +15 % assumption read nothing after
      2019, so **the scenario does not move**: `buildings.js` and
      `data/energy_totals_overrides.csv` reproduce byte for byte.
- ☑ **R5 — The Q4 `debate` hand-typed the model's shower recipe.** It read
      *"la recette (5 min × 7 l/min) fait 35 litres"* while the `justification` three lines
      above interpolates `{showerMinutes}` and `{showerFlow}` for the same two numbers —
      rule 3, and the build cannot see it. Now `{showerMinutes} min × {showerFlow} l/min` and
      a new model fact `showerLitresMixed` (= 35, spoiler-flagged) for the product.
- ☑ **R6 — The new PV chart cost five sixths of a printed card.** Measured at print width,
      the twelve monthly bars made one fact **105,7 mm** tall on a 128 mm A5 card — the
      module's stated budget is ~25 mm a plot. A horizontal bar costs ~6 mm, so the cost is
      the bar *count*, which the README did not say. Recast as **six two-month bars**
      (688 / 2 303 / **2 743** / **2 617** / 1 477 / 575 GWh): the seasonal hump survives,
      and the two amber bars now sum to exactly the 5 360 GWh the sentence quotes. The card
      drops from 224,5 to 186,4 mm. README's plot-cost rule corrected.
- ☑ **R7 — Q7's two 2024 figures, reconciled on the card that shows them.**
      The benchmark bars put Belgium at **0,18 %** (raw Eurostat) while the main curve ends
      at **0,22 %** (the same series rescaled to the JRC anchor) — the same country, the same
      year, two numbers on one screen, which is the complaint that opened Q7. Both are right
      and the trend card explains the splice, but the reader meets the bars first. One clause
      added to the benchmark card in the three languages.

- ☑ **R8 — "2,3 % du parc rénové chaque année" is right, and reads as wrong.** Sylvain's
      objection: the renovation rate is *notoriously around 1 %*, so a slider opening at 2,3
      looks like an error. Checked, and the number is arithmetically exactly what JRC-IDEES
      books — but it is not a renovation rate in any sense a practitioner would recognise.

      **The check.** `new_ren_households_useful_surface_area` reproduces to within 0,13 Mm²
      (0,02 % of the stock) over twenty-three years as `ΔStock + 0,02261 × Stock`. The
      "renovation" component is therefore not a residual of anything observed: JRC-IDEES
      *books it as a fixed parameter*, 2,261 %/year, sd 0,0084 points — **one pass over the
      stock every 44,2 years**. The tracker already called it "a convention of the dataset";
      the reconstruction proves it, and gives the convention a physical meaning.

      **Why 2,3 and 1 are both right.** They count different things. *A Renovation Wave for
      Europe* (COM(2020) 662) states all three of the numbers in circulation, verbatim:
      "only **11%** of the EU existing building stock undergoes some level of renovation each
      year"; "The **weighted annual energy renovation rate** is low at **some 1%**"; "deep
      renovations that reduce energy consumption by **at least 60%** are carried out only in
      **0.2%** of the building stock per year". The familiar 1 % is the *energy-weighted*
      rate; JRC's 2,26 % is a headcount of floor area at any depth. Nothing was wrong — the
      card simply never said which of the five or six possible rates it meant.

      **The number the split cannot move.** Any (rate, depth) pair on the observed trajectory
      gives the same product, so the honest invariant is `rate × depth` — the pace expressed
      as *share of the stock taken to zero heating need each year*. Observed: **0,58 %/year**.
      That is the like-for-like companion to the Commission's weighted 1 %, and it is a new
      model fact (`deepEquivObs`), not a hand-computed one.

      **What this does to the reveal, which is the real gain.** The Commission's threshold for
      a *deep* renovation is ≥ 60 % — numerically the depth this scenario assumes. So
      négaWatt's pair reads directly as **2,24 % of the stock deep-renovated every year
      against the 0,2 %/year observed EU-wide**, about ten times more, and **1,34 %/year in
      full-renovation equivalent against 0,58 % observed, 2,33×**. That is a far stronger and
      far less complacent punchline than "the same number of renovations, only deeper", which
      is what the screen said before and which invites exactly Sylvain's reaction in reverse —
      *"so nothing much has to change"*.

      **Changed:** the subtitle now warns, at the point of contact, that this is not the 1 %
      figure and says why; the card *Ce que compte un « taux de rénovation »* carries the
      three Commission figures with the primary source, states the 44-year cycle and the
      0,58 % invariant, and its plot is now **seven rates on one scale** — 11 / 2,26 / 1 /
      0,88 / 0,58 / 0,2 / 0,09 — with the project's 2,26 in amber, so a participant sees
      exactly where the anchor sits among the numbers they have heard. The reveal and the
      `historyNote` follow. Three new facts: `deepEquivObs`, `deepEquivTarget`,
      `deepEquivRatio` (the last two spoiler-flagged).

      **Paid for by dropping a chart.** The Walloon four-bar plot went: its 0,88 and 0,09 are
      now on the seven-rate chart, and its 3 % / 5,22 % remain in its own sentence. The card
      goes from three plots at 249 mm to two at 263 mm — still the tallest in the topic, and
      the one to trim first if the printed set has to shrink.

      **Not changed: any model input.** `ref_RS_cook_trend`-style corrections were not needed
      here — the rate/depth pair still reproduces the observed −0,458 kWh/m²/year exactly, and
      `buildings.js` and `data/energy_totals_overrides.csv` reproduce byte for byte.

### Print density, finally measured

The tracker twice deferred this ("Sylvain's call", "do that before a session"). Measured at
print width with the print stylesheet applied, card height in mm (A5 target 128 mm; an A4
page holds 277 mm of cards):

| card | facts | plots | before | after |
|---|---|---|---|---|
| `floor-area` | 6 | 1 | 192 | 192 |
| `renovation-rate` | 4 | 3 → **2** | 249 | **263** |
| `renovation-depth` | 4 | 1 | 191 | 191 |
| `thermostat` | 4 | 0 | **128** | 128 |
| `hot-water` | 4 | 1 | 172 | 172 |
| `cooling` | 5 | 1 | 225 | **186** |
| `cooking` | 4 | 0 | **128** | 134 |
| `district-heat` | 5 | 1 | 239 | 243 |

Nothing is clipped — the print CSS is `min-height: 128mm; height: auto; break-inside: avoid`,
so a long card grows rather than losing a fact. But "two A5 cards to an A4 page" is fiction
for this topic: it needs about **six pages, not four**. Two readings the tracker did not have:

- **Plots, not fact counts, drive the height.** `renovation-rate` is the tallest card in the
  topic with only four facts, because of its plots. `floor-area`, the card the tracker
  worried about at six facts, is 71 mm shorter. (R8 later traded one of its three plots for
  a longer and much more useful one, so it now stands at 263 mm with two.)
- **Prose length is the other half.** The single longest fact in the topic is
  district-heat's *Trois Régions* at 60 mm — four regional figures in one paragraph.

Trimming is editorial and is left alone here. The cheapest cuts, if wanted: drop
`renovation-rate`'s heating-need curve (−35 mm; its four figures are all in the sentence
beside it), and split or shorten *Trois Régions*.

### Still open, deliberately

- **History series labels are English on the FR and NL pages.** Every main chart prints
  *"Source: Residential floor area per person — JRC-IDEES-2023…"* in all three languages,
  because `make_history_series` exports a single English `label`. Systemic and pre-existing
  — all four topics, every generated series. The curated district-heat block is the only one
  with a trilingual label, which is what makes the gap visible. Fix route: a trilingual
  `label` in `make_history_series`, or a `history.<key>` string family in `ui.yaml`. Not done
  here because it touches the transport topics, which this review does not cover.
- **The prebound source is paywalled.** Sunikka-Blank & Galvin (2012) is on tandfonline
  behind a paywall, against rule 2's *"a participant who doubts a figure must be one click
  from checking it"*. It is the canonical source and the numbers are right; an open mirror or
  a second, open citation would close it.
- **Q6's slider still has no observed curve.** With the trend now honest, exporting
  `res_cooking_per_household` truncated at 2019 would be defensible. Not done: the same
  Eurostat reconstruction that fixed hot water is *not* valid here, because a gas-to-induction
  switch moves delivered energy while leaving useful heat alone, so scaling useful by
  delivered would import the electrification into a series that should not see it. That is
  the difference between Q4 and Q6, and it is why one was reconstructed and the other stops.
- **Tertiary.** `ter-district-heat` still quotes Eurostat on an all-energy denominator (the
  same fault as the old 0,2 %), and `ter-insulation` still asks kWh/m² where residential now
  asks rate × depth. Both were already recorded as the next topic's work.
- **The 45 % slider ceiling** is still a literal in `_add("district-heat", …)` rather than
  `ref_dhn_potential_pct`, so the comment explaining the 45/37 distinction sits next to a
  constant the slider does not read. Harmless today; worth one line when the lever is next
  touched.
- **One aux-notebook cell has no `id`.** Cell 31 (the aviation detail) predates this round —
  it is already id-less at HEAD — but `nbformat` now warns that this "will become a hard
  error in future versions". A one-key fix, left alone here only because re-serialising that
  notebook would churn a file this review did not otherwise touch.

---

## State of play — 2026-09-21, Q7 closed

**Done:** Q1, Q2, Q3, Q4, Q5, Q6, **Q7**, Q8, and the four `tangible` removals.
The residential-heat tracker is empty.

Nothing is committed. The build is green:

```bash
python scripts/build_workshop_content.py --check
```

30 levers · 160 facts · 31 plots. The rest of the suite is green too:

```bash
python scripts/verify_workshop_export.py    # 327 checks
python scripts/test_workshop_helpers.py     # 8
python scripts/test_workshop_api.py --base http://127.0.0.1:8787   # 59
```

To look at the result:

```bash
python scripts/dev_static.py --port 8080
```

then `http://127.0.0.1:8080/workshop/play.html?topic=residential-heat&lang=fr`.

**Working method that proved right, worth keeping.** Every figure on a card was fetched and
read before being written; where a number could not be verified at a live URL it was left
off the card and recorded in this file instead (see 1.2 and 1.3). The build enforces that a
card *has* a source and a link — it cannot enforce that the link opens the number, and that
is exactly where a workshop card fails in front of a participant who checks.

---

## Q1 — `floor-area` · Surface du logement

> - ajouter une fiche sur les résidences secondaires
> - ajouter une fiche sur les logements inoccupés
> - ajouter des données sur la quantité de logements jugés divisibles (ex kangourou) ?
> - Enlever la fiche "concrètement" qui est moins intéressante

- ☑ **1.1** Fact card: second homes — *done*. `structure`, retitled *Résidences secondaires*.
      The coast alone holds >106 000 of them, lived in 87 nights a year (74 by the owners).
      The point for this question: those m² sit in the denominator all year, their heating
      does not — but it does not fall to zero either (frost protection).
      Source Westtoer, *Onderzoek tweede verblijfstoerisme aan de Kust* (2024), URL checked.
- ☑ **1.2** Fact card: vacant dwellings — *done*. `caution`, retitled *Logements inoccupés*.
      Uses the only rigorous Belgian count I could verify: Brussels crossed the cadastre,
      the Registre national and Vivaqua's low-water records and found **4 500** presumed
      vacant, <2 % of the regional stock — *below* what was expected. The card says plainly
      that re-letting them does not move m²/person (stock and population both unchanged);
      what it buys is housing people without building. Source Bruxelles Logement / BSI
      (BRIO-VUB, IGEAT-ULB), 2024-11-22, URL checked.
      **Not done:** no national vacancy figure. The ones in circulation (30–50 k Wallonia,
      17–26 k Brussels) come from advocacy pages and press, not from a register; the
      Brussels study is precisely the demonstration that those were too high. Left out
      rather than sourced to a blog.
- ☑ **1.3** Divisible dwellings — *done*, and now with real counts. `lever` card, retitled
      *Le gisement, chiffré*: **46,6 %** of occupied Belgian dwellings (2,3 million) had
      three rooms or more per occupant at the 2021 census, and IWEPS singles out
      **233 617** Walloon houses owned and occupied by someone 65+, of which 99,6 % hold
      one or two people and 89 % have five rooms or more — IWEPS itself calls this
      *"un potentiel de logements sous-occupés"*. The mechanisms (kangourou, colocation,
      division) and the obstacle (municipal permit, *statut de cohabitant*) close the card.
      Source IWEPS *Décryptage n°10* (2026) + Statbel Census 2021 T04_DRM, URL checked.
      Replaces the Eurostat under-occupation figures I first used there — those are
      people-based and were already half-quoted on the comparison card.

      **Verified but not used, kept here for a future pass** (all fetched and checked):

      | figure | value | source |
      |---|---|---|
      | Kangaroo housing actually built, Flanders | 7 794 *zorgwonen* dossiers since 2009 (654 in 2024) | Dept. Omgeving, hearing deck 23-09-2025 |
      | …and how often the rules stop it | 33 % of notifications ruled invalid, "often because a local authority tightened the Flemish rules"; 8 of 28 sandbox pilots inhabited after six years | Vlaams Parlement, 18-04-2023 and 08-11-2023 |
      | The cohabitant penalty, priced | GRAPA 1 644,28 € → 1 096,18 €/month; RIS 1 367,34 € → 911,56 €. Both exactly two thirds | SFP and SPP Intégration Sociale, rates at 01-09-2026 |
      | Under-occupation, Flanders | 39 % of dwellings in 2013 (35 % in 2001); 211 000 *structurally* under-occupied — the authors' own "margin for subdivision" | Steunpunt Wonen, GWO 2013, Ad hoc 14 |
      | Under-occupation, Wallonia | 65 % of households, 28,7 % with two spare bedrooms or more | CEHD, EQH 2012-13 |
      | Splitting potential, Flanders | +30 000 dwellings from a cautious 10 % densification of well-located subdivision neighbourhoods; +250 000 at 25 dw/ha | Vlaams Bouwmeester, 2025 |

      Two of these would make strong cards on their own — the priced cohabitant penalty, and
      "8 of 28 pilots inhabited after six years" as the regulation-is-the-bottleneck number.
      Both are held back only by Q1's card count (see the print note below).

      **Still not found, do not invent:** any count of dwellings *physically assessed* as
      splittable (the VLAIO *SplitKit* project is working on exactly this and has published
      nothing yet), any habitat-kangourou count for Wallonia or Brussels, and any live,
      checkable Samenhuizen vzw project count — the "133 groups / 1 282 dwellings" figure
      that circulates is a dead link today (samenhuizen.be 404s, the legacy site 503s).
- ☑ **1.5** *(added)* Chart on the international-comparison card: rooms per person, EU27 /
      DE / FR / BE, Belgium highlighted. The numbers were already in the sentence, so the
      plot adds a shape, not a claim. Needed a new `unit.rooms/person` entry in `ui.yaml` —
      the `unit.<raw>` lookup in `i18n.js` existed but had never been used by any card.
- ⚠ **Print density.** Q1 now carries **six** pre-answer cards plus a plot, against the four
      the module aims for. On screen it reads fine; on the A5 printed card it will not fit.
      Options, Sylvain's call: merge 1.1+1.2 into one "logements que personne n'habite"
      card, or drop the trend card (its numbers are already on the main curve).
- ☑ **1.4** Remove the `tangible` ("Concrètement") card — *done*

## Q2 — `insulation` → `renovation-rate` + `renovation-depth`

> Il faut revoir cette question.
> les kwh/m² ne parlent pas aux participants des workshop. Il faut mieux un levier basé
> sur les taux de rénovation (historique, planifié).
> Les stats de consommation sont intéressantes malgré tout et peuvent faire l'objet d'une
> fiche.
> il faut également parler de l'effet rebond (la diminution de conso n'est pas directement
> proportionnelle au niveau d'isolation cfr
> https://publications.ibpsa.org/proceedings/bs/2021/papers/bs2021_30245.pdf)
> enlever la fiche "concrètement"
> Cette question est à revoir en profondeur. Il est possible qu'il faille modifier la
> feuille de calcul, ou qu'il faille diviser la question en deux.

- ☑ **2.1** Re-base the lever on a **renovation rate** (%/year) — *done, as half of a pair*
- ☑ **2.2** Keep the kWh/m² consumption statistics as a fact card (with its curve) — *done*,
      it is now the `trend` card of `renovation-rate`, plotting `res_heat_per_m2` 2000-2023
- ☑ **2.3** Fact card on the **rebound effect** — *done*, two cards (see the correction below)
- ☑ **2.4** Remove the `tangible` card — *done*
- ☑ **2.5** Split the question in two (rate × depth) — *done*

### What was done, and why it is admissible

The blocker recorded at the end of the last session was real but had a clean way through.
Restated: the model has one degree of freedom here (`acc_RS_tes_sht_ren`), so a rate and a
depth cannot both be free — and only their *product* is observed, so splitting them costs
one assumption. The resolution:

**The arithmetic.** Renovate a constant share `r` of the stock each year, each renovation
cutting a fraction `d` off that dwelling's heating need, and the stock average falls
**linearly**: `I(t) = I(0)·[1 − d·r·t]` for `r·t ≤ 1`. That is *exactly* the shape §2.1.1
already assumes, which is what makes the two-lever reading exact rather than approximate.
(The geometric variant in last session's note — `(1−r)^31` — would have been a different
trajectory from the notebook's; the linear one is the notebook's own.) Hence
`d × r = −acc·cur/ref = 1.344 %/year`, and fixing either fixes the other.

**The one assumption, and where it is written.** The notebook (cell 24, §2.1.1) now sets the
depth at **60 %** — the threshold the European Commission uses to call a renovation *deep*,
Recommendation (EU) 2019/786, Annex 2.3.1.3 — and *derives* the rate, **2.24 %/year**, which
renovates 69.4 % of the stock by 2050. The assumption is in the notebook, not in
`workshop_levers/`, so that module still adds none of its own.

**The historical anchor, which is what made the choice defensible.** The same reading run
backwards: at the JRC renovation rate of 2.261 %/year, the observed −0.458 kWh/m²/year on a
trend starting at 79.5 kWh/m² implies an average depth of **25.5 %**. Independently, the
Walloon draft renovation plan (Nov 2025) measures **20 % of final energy** across its 15 743
grant-backed renovations of 2023 — the only depth figure published anywhere in Belgium. The
two agree in order of magnitude, which is the corroboration the desk analysis was missing.
So the scenario's "doubling of the improvement rate" reads as **the same number of
renovations, each about 2.4× deeper** — not as more renovations. That is the reveal's
punchline and it is a genuinely surprising one.

**Nothing downstream moved.** `trg_RS_tes_sht` is identical to the last digit;
`website/data/buildings.js` differs from its committed version only in the generation date;
`data/energy_totals_overrides.csv` is untouched, so the CI check against the PyPSA-Eur fork's
`nW_BE.py` is unaffected. Verified, not assumed.

### Files changed for Q2

| file | change |
|---|---|
| `nW_BE_demand_model_buildings.ipynb` | §2.1.1 prose gains the renovation reading; cell 24 gains `dep_/shr_/rat_RS_tes_sht_ren` and the `obs_RS_*` historical decomposition, with two asserts; reference [8] added |
| `nW_BE_demand_data_aux.ipynb` | cell 34: `res_renovation_rate` was mislabelled — it held renewal (renovation + new build). Split into `res_renewal_rate`, `res_renovation_rate`, `res_new_build_rate`; same for tertiary |
| `workshop_levers/residential_heat.py` | `insulation` replaced by `renovation-rate` and `renovation-depth`; the three renewal constants now read from the notebook; four new asserts |
| `nW_BE_demand_model_sub_functions.py` | `"renovation"` added to `LEVER_IMPACT_KINDS` |
| `website/assets/js/workshop/impact.js` | the `renovation` response, with its `axis`/`other` parameters and the saturation cap |
| `website/workshop/content/residential-heat.yaml` | the two blocks, eight new cards, four plots; "sept" → "huit" hypotheses; two new unit strings |
| `website/workshop/content/ui.yaml` | `unit.kWh/m²/year`, `unit.% of the stock per year` — chart axes were printing English on the FR/NL pages |

### A correction about the IBPSA paper

`bs2021_30245` is **not** a prebound/rebound study. It is Van Hove et al. (Ghent University +
VEKA), *Data-driven statistical modelling of real energy use…*, a calculated-versus-metered
study of 47 082 Flemish houses built or thoroughly renovated since 2006; the words "rebound"
and "prebound" appear nowhere in its body. Its finding is stronger for our purpose anyway —
the EPB calculation **overestimates real gas use by 103 % on average** (14-214 % depending on
E-level) and explains only 25 % of the variance — so it is kept, as the "calculation vs
meter" card. The actual prebound and rebound numbers were sourced separately:

- **prebound** — Sunikka-Blank & Galvin (2012), *Building Research & Information* 40(3):
  30 % less than calculated across 3 400 German dwellings; the gap vanishes at 50 kWh/m²·year
  and reverses below it (~65 % *above* calculation under 75 kWh/m²·year); comprehensive
  retrofits really save 25-35 %, not the 70-80 % claimed. Contains a Belgian datapoint
  (Hens et al., 964 dwellings).
- **rebound after renovation** — Aydin, Kok & Brounen (2017), *RAND Journal of Economics*
  48(3): 26.7 % for owners, 41.3 % for tenants across 563 000 Dutch households, and **~56 %**
  on the dwellings actually treated by a retrofit subsidy programme. Used in the `debate`.

### Verified but not used — kept for a future pass

| figure | value | source |
|---|---|---|
| EU renovation rates by depth, 2012-2016 | EU28 12.3 %/year total, 0.2 % deep; **Belgium 15.6 %** total, 6.5 light, 1.0 medium, **0.2 % deep** | EC / Ipsos-Navigant 2019, Table 2 p. 15 |
| …and what each depth class actually saves | EU28 light 12.7 / medium 41.1 / deep 66.0 %; **BE 12.4 / 40.8 / 66.4**, all-renovations average **9.0 %** | same, Table 4 p. 21 |
| EPBD recast binding trajectory | residential stock primary energy −16 % by 2030, −20-22 % by 2035; ≥55 % of it from the **43 % worst-performing** | Directive (EU) 2024/1275, Art. 9(2) |
| Walloon investment need | **110 bn €** residential to 2050 (175 bn € with non-residential) | Plan wallon (projet, nov. 2025), pp. 212-213 |
| Walloon required rates by depth | deep 1.82 %/year (2024-30) → 2.98 → **3.59 %/year** (2041-50) | same, Tableau 8 p. 66 |
| Flemish stock averages | single-family **397**, apartments **238** kWh/m²·year primary (early 2026); 9 % of dwellings at label A (early 2024) | Statistiek Vlaanderen / VEKA |
| Brussels stock trend | certified dwellings **317 → 254 kWh/m²·year primary, 2011 → 2024**; 28.5 % still class G | Bruxelles Environnement, *Certification PEB… données 2024* |

**Explicitly not available, do not invent:** Flanders publishes no average label jump or
kWh/m² improvement per renovation — a minister confirmed in writing (WQ 102, 3/12/2024) that
"deze analyse is momenteel niet beschikbaar", because no new EPC is required after works.
Brussels' 2019 strategy states no required renovation rate at all. Wallonia's 20 % depth is
*modelled* from the works done, not measured before/after; the plan proposes creating an
observatory precisely because the data does not exist.

**Left open on purpose: the tertiary twin.** `ter-insulation` still asks for kWh/m², on the
same shape of assumption (`acc_TS_tes_sht_ren = 5` on an observed -0.154 kWh/m²/year). The
same split would work there and would make the two heat topics read alike for a facilitator
running both — but the depth anchor would have to be argued again for offices, shops,
schools and hospitals, where both the European "deep renovation" threshold and the Belgian
evidence are thinner. Out of scope for this tracker, which is the residential review; worth
a decision of its own.

**Print density.** Both new screens carry four pre-answer cards and between one and three
plots. `cards.html` renders eight cards for the topic with no overflow in the DOM, but the A5
print has not been eyeballed — do that before a session, together with Q1's six-card problem.

## Q3 — `thermostat` · Thermostat

> Renommer la question en quelque chose comme: "Niveau moyen raisonnable du thermostat
> sans perte de confort?"

- ☑ **3.1** Reword the question — *done*. Kept as a **delta in °C**, not an absolute
      setpoint: no one measures the absolute setpoint of Belgian homes (the model's own
      caution card says so), so an absolute question would have no reference year and no
      observed curve. The subtitle now spells that out.

## Q4 — `hot-water` · Eau chaude sanitaire

> - enlever "concrètement"
> - vérifier les données. La diminution depuis 2019 ne semble pas justifiée par des
>   éléments concrets. Cross-checker avec des sources de données différentes. Modifier les
>   feuilles de calcul si des erreurs manifestes sont détectées.
> - Trouver des sources avec des moyens concrets de diminuer la conso d'eau chaude
>   sanitaire et quantifier dans une fiche dédiée (pommeau de douche économique, douche vs
>   bain, ...)

- ☑ **4.1** Remove the `tangible` card — *done*
- ☑ **4.2** Cross-check the 2019→2023 fall (675 → 435 kWh/person) — *done, and JRC is
      wrong.* Eurostat `nrg_d_hhq`, Belgium, water heating, TOTAL, TJ (updated 2026-06-09):
      43 175 (2019), 43 131 (2020), 43 409 (2021), 43 645 (2022), 43 510 (2023), 43 268
      (2024). Flat. JRC useful DHW 663,4 → 437,7 ktoe would imply conversion efficiency
      collapsing from 64 % to 42 %. Cooking from the same Eurostat table is also flat
      through 2022 (+0,7 %), so the 2020 JRC DHW drop is not "people were home".
      **Choices:** (1) keep `ref_RS_tes_shw` at the 2019 JRC value — Eurostat shows 2019 is
      typical, not a peak; rebasing on 2023 JRC would lock the artefact in. (2) reconstruct
      2020-2023 from Eurostat delivered energy at the 2019 useful/final ratio, rather than
      truncate — truncating hid four years of a series that is, on the delivered-energy
      evidence, essentially flat. (3) leave the 2050 target alone — it is a shower recipe,
      not a reading of this series. Documented in the buildings notebook §2.1.2, the aux
      notebook next to the DHW array, D54, and `res_hot_water_per_person`'s `note`. The JRC
      arrays themselves stay as transcribed.
- ☑ **4.3** Fact card quantifying concrete ways to cut hot water — *done*, `lever` card
      *Ce qui coupe vraiment les kWh*. Flanders 2023: shower is 26 % of 80 l/person/day of
      tap water ≈ 21 litres (VMM, URL checked). Drain heat exchanger: up to 40 % of DHW
      energy on a shower, nothing on a bath, because it needs simultaneous flow; European
      showers last 4,5–8,5 minutes (Sevela et al., REHVA Journal, URL fetched). Low-flow
      and shorter showers are stated as proportional cuts, without an unsourced "12 l/min
      standard" figure.
      **Not found, do not invent:** a Belgian litre-per-bath volume. The bath point on the
      card is the one REHVA actually measures (WWHR cannot recover a tub fill).
      **One URL for two sources**, same pattern as Q1's IWEPS+Statbel: the card opens the
      VMM indicator (the Belgian number a participant will want to check); REHVA is named
      in `source:` and recorded here.

      The reveal `debate` now carries the friction this card creates with the scenario:
      5 min × 7 l/min = 35 litres of mixed water, against 21 litres already used for
      showers in Flanders. The "sufficient" shower is more generous than today's Flemish
      average.

## Q5 — `cooling` · Climatisation

> - Ajouter une fiche sur la récente canicule (2026) belge, le débat qui en a suivi et
>   l'augmentation de la vente de climatiseurs
> - Ajouter une fiche sur le fait que les clims sont en général assez corrélées de façon
>   locale avec la production photovoltaïque, ce qui n'est que partiellement pris en compte
>   dans le modèle negawatt BE

- ☑ **5.1** Fact card: the 2026 Belgian heatwave, the debate, AC sales — *done*.
      Replaces the `tangible` 100 m² arithmetic (the 44 kWh was already implied by the
      trend number; print space was needed). IRM summer bulletin (PDF, 2026-09-01):
      20,3 °C at Uccle, first time above 20 °C, three heatwaves, 40,4 °C at Houyet on
      27 June. Frixis via Belga/De Tijd (2026-08-07): 134 178 fixed ACs in H1, +47 %,
      740/day, *before* the July and August waves. URL opens the IRM PDF; Frixis is
      named in `source:` (same dual-citation pattern as 4.3).
- ☑ **5.2** Fact card: AC vs local PV, and how the model treats it — *done*, `caution`.
      Eurostat `nrg_cb_pem` 2025: 51,5 % of Belgian PV electricity in May–August
      (5 360 / 10 404 GWh), with the monthly bars. The demand model only carries an
      annual kWh/m², so the hourly coincidence is not calculated here; downstream the
      annual total is shaped into a load curve approximately. The scenario's rejection
      of "AC to soak up PV" stays on the reveal (justification), not on this card.

## Q6 — `cooking` · Cuisine domestique

> - enlever "concrètement"
> - ajouter une fiche sur l'électrification de la cuisine (eg gas vs induction), son
>   influence, et comment c'est pris en compte dans le modèle

- ☑ **6.1** Remove the `tangible` card — *done*
- ☑ **6.2** Fact card: cooking electrification (gas vs induction), efficiency, and how the
      model handles it — *done*. `lever` card *Gaz ou induction*, before the reveal.
      ENERGY STAR (Frontier Energy, July 2019, URL checked): ~85 % of induction energy
      reaches the vessel, vs a third for gas. Eurostat `nrg_d_hhq` cooking, Belgium,
      electricity / TOTAL (updated 2026-06-09): 67,1 % in 2019 (3 673 / 5 476 TJ),
      73,8 % in 2024 (3 388 / 4 590 TJ) — written 67 % and 74 %. The slider is useful
      heat per household; the gas share is a separate carrier assumption
      (`ref_RS_tes_cok_gas` = 26,8 % of *useful* heat in 2019, not the same as
      Eurostat's 28,8 % of *delivered* energy). The 2 % 2050 target stays on the
      reveal (`gasShareTarget` is already a spoiler). URL opens ENERGY STAR; Eurostat
      is named in `source:` (same dual-citation as 4.3 / 5.1).
      **Not fixed here:** JRC cooking useful energy still breaks in 2022 (102,0 →
      77,1 ktoe) while Eurostat cooking is flat through 2022 then drops in 2023–24.
      The play page already declares `historyAbsent`; the trend card's
      −1,6 kWh/household/year still uses the contaminated 2000–2023 polyfit. Same
      stain as item A, recorded rather than spliced.

## Q7 — `district-heat` · Réseaux de chaleur

> - il y a une inconsistance: 0.3% dans la fiche "contexte belge" et 0.2% dans
>   "comparaison internationale". Il faut que le dénominateur soit la chaleur. Il faut
>   également revoir ces chiffres, qui ne me paraissent pas exacts. De mémoire c'est 0.5%
>   pour la Wallonie et beaucoup plus pour la Flandre
> - Typiquement cette question doit faire l'objet d'une recherche biblio plus poussée. Il
>   faut harmoniser les chiffres et trouver les tendances historiques (à afficher sur le
>   graphique principal). Modifier les feuilles de calcul si nécessaire
> - Rajouter dans les fiches ce qui est considéré comme un potentiel raisonnable par des
>   analyses GIS (eg heat roadmap europe). Comparer aux ambitions de la Flandre, la
>   Wallonie et Bruxelles (il s'agit d'une compétence régionale).
> - Rajouter une fiche sur les avantages et inconvénients de réseaux de chaleur:
>   possibilité de générateurs centralisés, possibilité de stockage possiblement
>   saisonnier, accessibilité de la source froide, potentiel de récupération de chaleur
>   etc). Préciser qu'en termes de conso uniquement, ce n'est pas forcément un avantage car
>   les pertes sont non négligeables!
> Question à revoir en profondeur!

- ☑ **7.1** Fix the 0,3 % / 0,2 % inconsistency; put every figure on a *heat* denominator
      — *done* (D55). The 0,3 % was JRC useful energy 0,25 % rounded by a 1-decimal
      slider; the 0,2 % was Eurostat derived heat over *all* household energy. On heating
      + hot water, Eurostat is 0,21 % in 2019 and 0,18 % in 2024, next to JRC 0,25 %.
      `decimals: 2` so `{refValue}` prints 0,25. International bars are now that same
      heat basis (BE 0,18 / FR 4,4 / DE 7,9 / EU27 11,0 / DK 49,6 / SE 52,2 in 2024).
- ☑ **7.2** Bibliographic pass: Belgian share, regional figures, historical trend —
      *done*. Eurostat `nrg_d_hhq` 2016-2024 is the main-chart series (D15), scaled to
      the JRC 2019 point so slider and curve meet. Raw delivered shares stay between
      0,17 and 0,22 %. Regional volumes, all opened:
      Flanders VEKA 2024: 908 GWh all customers, 113 GWh hot water to homes, 955 GWh
      injected; Wallonia SPW Art. 25: 237 GWh (2016) → 304 GWh (2021), all sectors;
      Brussels Environnement 2024: 99 GWh on six networks in 2021, campuses/hospitals.
      **The recollection of "0,5 % Wallonia and much more Flanders" is this GWh gap,
      not a household-heat share.** Flanders' 113 GWh is most of Eurostat's 129 GWh
      (465 TJ) of Belgian household district heat; a Flemish *share* of Flemish
      household heat is not on the card because no regional household-heat
      denominator opened. The 0,4 % of Walloon *all-sector* heat in 2016 (237 GWh /
      ~63 TWh) is the slide that memory comes from; the official 2024 synthesis quotes
      GWh, not that percentage, so the card quotes GWh.
- ☑ **7.3** GIS potential vs regional ambitions — *done*. Heat Roadmap Belgium (HRE4)
      retains **37 %** of built-environment heat excluding industry by 2050 (from ~2 %
      in 2015), economic range 20–54 %. The notebook's 45 % is the Heat Roadmap
      *average* (Lund et al.); §2.1.3 now says so. PATHS2050 stays at the notebook's
      13 % for all buildings. Flanders VEKP: 2 400 GWh delivered by 2030 (all
      customers, ~2,6× 2024). Wallonia: no share target of its own. Brussels: 70 % of
      2050 demand in a very-high-density zone, and for ~40 % of that demand a
      low-temperature network would be the best techno-economic option (évaluation
      chaleur-froid 2024). Slider max stays 45 %; 15 % target untouched.
- ☑ **7.4** Fact card: what a network buys you, and the losses — *done*. `caution`,
      retitled *Ce que le réseau achète*: central generation, seasonal storage, cold
      source (district cooling excluded). Flanders pipe losses ~5 % (955 injected /
      908 delivered, 2024); Eurostat national distribution losses 7,5 % of derived
      heat in 2019 (1 540 / 20 570 TJ). Demand-model blindness kept: the lever does
      not move useful energy at the dwelling.

      ⚠ **Print density.** Q7 now carries **five** pre-answer cards plus a bar chart,
      against the four the module aims for. GIS is the densest; on A5 it will wrap.
      A side effect of `decimals: 2` is the reveal printing **15,00 %**; ugly, but it
      is what stops 0,25 becoming 0,3 again.

      **Not done / next topic:** `ter-district-heat` still quotes Eurostat on an
      all-energy denominator (2,2 % of service-sector energy). Same stain as the old
      0,2 %, left for the tertiary pass. Cooking's JRC 2022 break (Q6) is still
      flagged, not spliced.

## Q8 — Summary screen ("C'est envoyé, merci")

> le graphique qui résume l'effet des mesures sur la demande (cfr inland transport)
> n'apparaît pas sur cette page. A corriger. Vérifier qu'il apparaît bien pour toutes les
> autres sous-sections

- ☑ **8.1** Enable the ± summary chart on `residential-heat` — *done*
- ☑ **8.2** Same for `tertiary-heat`; the two mobility topics are untouched and still
      carry theirs — *done*

---

## Why the summary chart was missing (item 8)

Not a bug in the page: the build **refused** to turn the chart on. Both heat topics carry a
lever whose `impact.kind` is `negligible` — `district-heat` / `ter-district-heat` are
carrier splits, and by construction they move no end-use demand at all (`share_heat_dhn` and
`share_heat_ihs` add to 1 in cell 57 of the buildings notebook). The rule written for the
mobility topics was "every lever must have a usable response, or no chart", which cost the
other six levers their picture because of one that is *provably* flat.

Fixed by making that case explicit rather than fatal:

- `scripts/build_workshop_content.py` now refuses only a lever with **no impact record at
  all** (a response nobody worked out). `negligible` is an answer, and it is accepted.
- `website/assets/js/workshop/play.js` marks such a lever `neutral` instead of passing a
  null value that the chart would have drawn as *"not answered"*.
- `website/assets/js/workshop/spark.js` prints the new `play.effects.neutral` string on that
  row — "sans effet sur cette demande" — so the flat lever teaches something instead of
  looking like a missing answer.

## Model / calculation issues found on the way

*(inconsistencies in the notebook itself — things the workshop cannot fix by wording)*

**A. The hot-water series had a step the world does not.** (Q4, item 4.2 — fixed, D54.)
JRC-IDEES residential hot water runs **663,4 ktoe (2019) → 570,6 (2020) → 456,8 (2022) →
437,7 (2023)**. Eurostat `nrg_d_hhq` delivered energy for the same end-use is flat at
~43 000 TJ every year 2016-2024. The workshop curve keeps JRC through 2019 and
reconstructs 2020-2023 from Eurostat at the 2019 useful/final ratio; `ref_RS_tes_shw`
stays the 2019 JRC value; the 2050 target is still the shower recipe. The JRC arrays in
the aux notebook are unchanged (they are the transcription). Cooking on the same JRC
sheet has a 2022 break that Eurostat also does not show — later item, Q6.

**B. The two district-heating shares were on two different denominators.** (Q7, item 7.1 —
fixed, D55.) `ref_RS_tes_dhn = 10.823/(3664.049+663.435)` = 0,25 %, formerly rendered as
"0,3 %", is a share of **heat** (space heating + sanitary hot water). The 0,2 % on the
old international card was Eurostat derived heat over **all household energy**. On a heat
denominator Eurostat is 0,21 % in 2019. Regional GWh (Flanders 113 GWh to homes, Wallonia
304 GWh all sectors, Brussels 99 GWh) are volumes, not household-heat shares.

**C. A floor-area unit trap, already handled but worth knowing.** `nW_BE_demand_data_aux.ipynb`
cell 34 documents it: cell 15's `floor_area` scales the JRC tertiary figures by `1e6` where
cell 24 uses `1e3`; `1e3` is the correct one. The history export uses the cell-24 variables.
If anything else in the notebooks reads cell 15's `floor_area`, it is six times too large.

## Log

- 2026-09-21 — **R8: the renovation rate that looked wrong.** 2,26 %/year is exactly what
  JRC-IDEES books — proved by reconstructing the series as `ΔStock + 0,02261 × Stock` to
  0,02 % of the stock, i.e. one pass every 44 years — but it is a headcount at any depth, not
  the energy-weighted ~1 % everyone quotes. Card now carries the Commission's own three
  figures (11 % / 1 % / 0,2 % at ≥ 60 %) and a seven-rate plot with the anchor in amber, plus
  the split-invariant 0,58 %/year full-renovation equivalent. The reveal gains its real
  punchline: 2,24 %/year *at the EU's own "deep" threshold*, against 0,2 % observed. D59.
- 2026-09-21 — **Review pass over the round** (R1-R7, see "Second pass" above). Six defects
  fixed: the negative chart floor in `spark.js domain()`; `decimals` now a ceiling rather
  than padding, killing "7,50 %" and "15,00 %"; VEKA's 113 GWh re-read as a carrier, not an
  end-use; the cooking trend rebased on 2000-2019 (−0,94, not −1,6 — the rest was a JRC
  step); the Q4 debate's shower recipe read from the model instead of typed; the twelve-bar
  PV plot compacted to six two-month bars after measuring it at 106 mm on a 128 mm card.
  Every load-bearing figure of the round was re-fetched from Eurostat, VEKA and HRE4 and
  held. Print density measured at last: the topic needs six A4 pages, not four. Scenario
  numerically unchanged — `buildings.js` and `energy_totals_overrides.csv` reproduce byte
  for byte. D56-D58.
- 2026-09-21 — **Q7 closed.** All district-heat shares sit on heating + hot water.
  Slider prints 0,25 % (was 0,3). Main chart is Eurostat 2016-2024 scaled to JRC 2019.
  Regional GWh (Flanders 113 / Wallonia 304 / Brussels 99), HRE4 37 % vs Lund 45 %,
  losses 5 % Flanders / 7,5 % Eurostat. Reveal now says 4,2 TWh connected, not 27,9.
  D55. Tertiary sister lever still on an all-energy denominator.
- 2026-09-21 — **Q6 closed.** New `lever` card on induction vs gas (ENERGY STAR 85 % /
  one third; Eurostat electricity share of delivered cooking energy 67 % → 74 %). The
  slider is useful heat; the carrier split is a separate assumption, not spoiled.
  JRC cooking's 2022 break is flagged, not spliced (`historyAbsent` already).
- 2026-09-21 — **Q5 closed.** Summer-2026 card (IRM + Frixis via Belga) replaces the
  cooling `tangible`; AC/PV caution with Eurostat `nrg_cb_pem` 2025 monthly bars
  (51,5 % of Belgian PV in May–August). Hourly coincidence is not in the demand model.
- 2026-09-21 — **Q4 curve reconstructed, not truncated.** 2020-2023 useful DHW is now
  Eurostat `nrg_d_hhq` scaled at the 2019 useful/final ratio (flat ~665–675 kWh/person,
  against JRC's crash to 435). 2019 JRC reference and 2050 shower recipe unchanged.
- 2026-09-21 — **Q4 closed.** Eurostat `nrg_d_hhq` is flat 2016-2024 for Belgian water
  heating; JRC useful-energy DHW after 2019 is an artefact (implied efficiency 64 % → 42 %).
  Workshop curve reconstructed 2020-2023 from Eurostat at the 2019 useful/final ratio
  (later correction: not truncated). 2019 reference kept; 2050 recipe untouched. New `lever`
  card on flow / duration / WWHR (VMM + REHVA). Debate now flags that the recipe's 35 l
  mixed shower exceeds Flanders' 21 l average. D54.
- 2026-09-21 — **Q2 closed.** `insulation` split into `renovation-rate` ×
  `renovation-depth`; notebook §2.1.1 gained the renovation reading (one new assumption, the
  60 % depth; the rate derived); `res_renovation_rate` un-mislabelled in the aux notebook;
  new `renovation` impact kind; eight new cards, four plots. Scenario numerically unchanged.
- 2026-09-21 — session ended on token budget. Q2 analysed but not implemented;
  the design, the arithmetic and the blocker are written up under Q2 above.
- 2026-09-21 — Q1 closed: items 1.1, 1.2, 1.3, 1.5 done (1.4 earlier).
- 2026-09-21 — items 1.4, 3.1, 4.1, 6.1, 8.1, 8.2 done.
- 2026-09-21 — tracker created from Sylvain's notes; residential-heat inventory: 7 levers,
  33 facts, 0 charts, no summary chart.
