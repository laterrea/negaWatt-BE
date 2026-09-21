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
  D49-D53

Status legend: ☐ todo · ◐ in progress · ☑ done · ⊘ dropped (with reason)

---

## State of play — 2026-09-21, end of session

**Done:** Q1 (all of it), **Q2 (all of it — the deep one)**, Q3, Q8, and the four
`tangible` removals that were asked for.
**Next up:** Q4, then Q5, Q6, Q7 in that order. Q4 carries a real modelling problem
(item A below), Q7 carries two (items B and B-regional).

Nothing is committed. `git status` should show the eight modified files listed above plus
this tracker. The build is green:

```bash
python scripts/build_workshop_content.py --check
```

30 levers · 156 facts · 28 plots (the residential topic now has eight questions, not
seven). The rest of the suite is green too:

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
- ☐ **4.2** Cross-check the 2019→2023 fall (675 → 435 kWh/person) against independent
      sources; fix the notebook if the series is wrong
- ☐ **4.3** Fact card quantifying concrete ways to cut hot water (low-flow head, shower
      vs bath, shower length, waste-water heat recovery)

## Q5 — `cooling` · Climatisation

> - Ajouter une fiche sur la récente canicule (2026) belge, le débat qui en a suivi et
>   l'augmentation de la vente de climatiseurs
> - Ajouter une fiche sur le fait que les clims sont en général assez corrélées de façon
>   locale avec la production photovoltaïque, ce qui n'est que partiellement pris en compte
>   dans le modèle negawatt BE

- ☐ **5.1** Fact card: the 2026 Belgian heatwave, the debate, AC sales
- ☐ **5.2** Fact card: AC demand vs local PV production, and how the model treats it

## Q6 — `cooking` · Cuisine domestique

> - enlever "concrètement"
> - ajouter une fiche sur l'électrification de la cuisine (eg gas vs induction), son
>   influence, et comment c'est pris en compte dans le modèle

- ☑ **6.1** Remove the `tangible` card — *done*
- ☐ **6.2** Fact card: cooking electrification (gas vs induction), efficiency, and how the
      model handles it

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

- ☐ **7.1** Fix the 0.3 % / 0.2 % inconsistency; put every figure on a *heat* denominator
- ☐ **7.2** Bibliographic pass: harmonise the Belgian share, find the regional figures
      (Flanders ≫ Wallonia) and a historical trend for the main chart
- ☐ **7.3** GIS potential (Heat Roadmap Europe & co) vs the regional ambitions of
      Flanders, Wallonia and Brussels
- ☐ **7.4** Fact card: what a network buys you (central generation, seasonal storage, cold
      source, waste-heat recovery) — and that in *consumption* terms it is not an
      advantage, because network losses are real

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

**A. The hot-water series has a step the world does not.** (Q4, item 4.2 — confirmed, not yet
fixed.) JRC-IDEES residential hot water runs **663,4 ktoe (2019) → 570,6 (2020) → 456,8
(2022)**. A −14 % step in 2020 is the wrong direction: people were at home more that year, and
the cooking series *from the same dataset* moves **+7 %** over the same step. Something is
wrong upstream of the model, and the workshop inherits it — the curve the participant extends
in Q4 falls from 675 to 435 kWh/person for no reason anyone can name at the table. Cross-check
against Eurostat `nrg_d_hhq` before touching the notebook; if JRC is wrong, the reference year
choice for `ref_RS_tes_shw` needs revisiting too.

**B. The two district-heating shares are on two different denominators.** (Q7, item 7.1 —
diagnosed, not yet fixed.) `ref_RS_tes_dhn = 10.823/(3664.049+663.435)` = 0,25 %, rendered as
"0,3 %", is a share of **heat** (space heating + sanitary hot water). The 0,2 % on the
international-comparison card is Eurostat derived heat over **all household energy**,
appliances included. Neither is regional, which is the third problem: your recollection of
~0,5 % for Wallonia and much more for Flanders is a regional split the model does not carry at
all. Fixing 7.1 means picking one denominator (heat) and putting every figure on it.

**C. A floor-area unit trap, already handled but worth knowing.** `nW_BE_demand_data_aux.ipynb`
cell 34 documents it: cell 15's `floor_area` scales the JRC tertiary figures by `1e6` where
cell 24 uses `1e3`; `1e3` is the correct one. The history export uses the cell-24 variables.
If anything else in the notebooks reads cell 15's `floor_area`, it is six times too large.

## Log

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
