# Workshop review — inland mobility · progress tracker

Review comments by Sylvain Quoilin, worked through sequentially.
Status: ⬜ todo · 🟡 in progress · ✅ done · ⛔ blocked / dropped (with reason)

Files touched: `website/workshop/content/inland-mobility.yaml`,
`website/workshop/content/ui.yaml`, `workshop_levers/inland_mobility.py`,
`scripts/build_workshop_content.py`, `website/assets/css/workshop.css`.

Rebuild after each batch:
```bash
python scripts/build_workshop_content.py --check
```
Levers module changes additionally need the transport notebook re-run.

---

## Q1 — `ground-km-day` (Distance parcourue par jour)

| # | Task | Status |
|---|---|---|
| 1.1 | Préciser les moyens de transport que "mobilité motorisée" englobe | ✅ subtitle now lists car / bus & coach / train / tram & metro / motorised two-wheelers |
| 1.2 | Enlever la fiche "tendance passée" (déjà visible sur le graphique) | ✅ (pre-existing edit) |
| 1.3 | Enlever la fiche "concrètement" | ✅ (pre-existing edit) |
| 1.4 | "Contexte belge": détailler les distances par mode | ✅ fact rewritten with the 5-mode split (+ bike, walk, air) **and a bar chart**; new `facts` keys exported from `workshop_levers/inland_mobility.py` |
| 1.5a | Benchmark en km/pers/jour (Peeters et al. converti) | ✅ |
| 1.5b | Label de la fiche: "Ailleurs en Europe" → "Comparaison internationale" | ✅ (pre-existing edit in `ui.yaml`) |
| 1.5c | Ajouter des valeurs hors Europe | ✅ new `benchmark` card + chart: Japan 29.4, Belgium 32.7, USA 62.5 km/p/day (same-scope), world 21.7 and Sub-Saharan Africa ~5.7 (all-modes, caveated in the text and caption) |

Rejected as not comparable: China's official 6.6 km/day (commercial transport only,
private cars excluded), India (rail only), Australia (scope unverified).
Note: the US figure is 62.5 on FHWA's 2023 occupancy basis; the same table gives
70.7 for 2019 and ORNL gives 60.8 — the honest range is 60–71 km/day.

## Q2 — `car-share` (Part de la voiture)

| # | Task | Status |
|---|---|---|
| 2.1 | Enlever la fiche "contexte belge" (structure) | ✅ (pre-existing edit) |
| 2.2 | Ajouter une fiche: possession de voiture(s) par habitant selon le revenu | ✅ Brussels ladder 26 % (<2 000 €/month) → 91 % (>10 000 €) + Eurostat: 22.7 % below the poverty line cannot afford a car vs 3.6 % above |
| 2.3 | Ajouter une fiche: usage de la voiture en milieu rural + nécessité de TC ruraux | ✅ EFM 2025: rural Wallonia 25.8 car-km/p/day vs Brussels 7.8 (**chart**), 86.7 % vs 52.3 % of distance; BFP: 5 % carless where transit is poor vs 37 % where it is good |
| 2.4 | Ajouter une fiche: les tramways vicinaux disparus | ✅ ~4 800 km of line in 1945 (more than today's 3 619 km railway) → 978 km in 1960 → last lines closed 1978; only the 67 km Coast Tram left |

Deliberately avoided: the often-quoted "5 200 km in 1925" (no primary source found —
the openable sources put 1925 at ~3 940 km non-electrified) and "world's longest tram
line" (lost to the LA A Line in 2023).
**No Belgian national source publishes cars-per-household by income quintile** — checked
Statbel and the Eurostat HBS family; the Brussels series is the closest real thing, and
its caveat (Brussels is also the most transit-rich place) is stated on the card.

### ⚠ Side effect to decide on: printed-card length

`cards.html` lays out two A5 cards per A4 page. Fact counts after all the additions:

| lever | facts (pre-answer) | charts | height* |
|---|---|---|---|
| ground-km-day | 4 (3) | 2 | 212 mm |
| **car-share** | **8 (7)** | 1 | 231 mm |
| car-occupancy | 5 (3) | 1 | 172 mm |
| car-energy | 6 (5) | 0 | 193 mm |
| **bike-km-day** | **6 (6)** | **3** | 267 mm |
| freight-tkm | 4 (4) | 0 | 128 mm |
| truck-share | 4 (3) | 2 | 180 mm |
| truck-load | 5 (4) | 0 | 128 mm |

\* measured in the browser at the print column width (190 mm) but with screen font
sizes, so these over-estimate; the print stylesheet drops the question to 11 pt and
captions to 6.5 pt. The A5 target is 128 mm.

Five of the eight no longer pair two-to-a-page. Nothing gets clipped — `break-inside:
avoid` lets a long card grow — so the deck simply goes from 4 pages to roughly 7.
**Needs your call**: accept the longer deck, or add a `print: false` flag so a fact can
be shown on screen but left off the paper card. The screen itself is fine; it scrolls.

## Q3 — `car-occupancy` (Occupation des voitures)

| # | Task | Status |
|---|---|---|
| 3.1 | Enlever la fiche "contexte belge" | ✅ (pre-existing edit) |
| 3.2 | Enlever la fiche "concrètement" | ✅ (pre-existing edit) |
| 3.3 | Chercher des pays à taux plus élevé + explication; fiche seulement si bon exemple | ✅ good example found → card added |

**The card**: Romania 1.87, Latvia 1.74 vs Belgium 1.28 and Italy 1.17 (Eurostat
*Passenger mobility statistics* fig. 3, urban mobility, pkm/vkm, 2015–2019 surveys) **with a
chart**. The explanation is the point: it is **car scarcity, not carpooling** — Latvia has
381 cars/1 000 inhabitants against Belgium's 510 at an identical household size, and Italy,
the most car-equipped country, comes last.
Basis caveat stated on the card and in the caption: that table is urban trips only, so
Belgium reads 1.28 there against 1.22 all-travel in the model.

The existing `trend` card was **reworded**: it claimed 1.2–1.5 "across the EU", which the
Romanian and Latvian figures now contradict on the very same screen. It now says western
Europe, and carries the real flatness evidence instead — England 1.6 every year 2002–2019,
France "quasi stable 2008–2019" with only 3 % of passengers having carpooled.

## Q4 — `car-energy` (Consommation par km)

| # | Task | Status |
|---|---|---|
| 4.1 | Préciser que l'indicateur porte sur l'énergie mécanique / hors motorisation | ✅ subtitle rewritten — see the caveat below |
| 4.2 | Enlever la fiche "contexte belge" (conso essence) | ✅ (pre-existing edit) |
| 4.3 | Enlever la fiche "concrètement" | ✅ (pre-existing edit) |
| 4.4 | Enlever la fiche "ailleurs en Europe" (traînée aéro) | ✅ (pre-existing edit) |
| 4.5 | Ajouter une fiche chiffrée et sourcée **par levier** | ✅ three `lever` cards, see below |
| 4.6 | Ajouter une fiche: tendance d'augmentation de la taille moyenne des voitures | ✅ `trend` card |

The three `lever` cards (new fact kind, already wired into the build script, CSS and ui.yaml):

| lever | figure | source |
|---|---|---|
| Speed | 90 km/h limit → −4 to −10 % energy/km for cars and vans (80 km/h → −6 to −14 %), against a real motorway average of 102–105 km/h | KiM 2023, §3.3 |
| Eco-driving | style swings consumption 15–25 %; training decays 4.6 % → 2.5 % in ten months, undetectable on motorway after 30 weeks; permanent in-car assistance 5–10 % | Xu et al., *Sensors*, 2021 |
| Size/weight | −10 % mass → −6 to −7 % fuel with engine downsizing, −4 to −5 % without (≈3–5 % per 100 kg on a 1 400 kg car); an SUV is 200–300 kg heavier and ~20 % more CO₂ | ICCT 2017 ; IEA 2024 |

Trend card: EU average new-car mass ~1 275 kg (2001) → **1 518 kg (2022)**, +19 %; width
170.5 → 180.2 cm (2001→2020), one centimetre every two years; Belgian SUV share 42 % (2020)
→ 47.2 % (H1 2021).

**Rejected as unverifiable**: the widely-quoted ADEME "110 instead of 130 km/h = −20 %"
(ADEME sites unreachable) and the "IEA: 10 km/h slower = −5 to −10 % per driver" attribution
(not in the IEA 10-Point Plan — the real IEA figure is a national total, 290 kb/d). A GIZ
eco-driving decay figure was dropped because it measures *truck* drivers.

### ⚠ Q4.1 — a mismatch between your comment and the notebook

You wrote that the indicator covers the mechanical energy to move the vehicle and that the
powertrain is *not* included. Notebook §2.3.5 lists four things behind the −25 %, and the
fourth is *"more efficient power trains (including regenerative braking for electric cars,
non-plug-in hybrid for thermal)"*. And `redu_fuel_PM_car` is applied to the kWh/km of **each
powertrain separately**, so what it really means is "energy per km at an unchanged powertrain
mix" — the thermal→electric switch is what is counted elsewhere, not powertrain efficiency
as such.

What I wrote, which is true of the model as it stands: the subtitle says this is the energy
needed to *move the vehicle* — speed, mass, aerodynamics, driving style — and that the switch
to electric is counted separately; the first fact card names all four ingredients honestly,
including the powertrain-efficiency one, and says the split between them is not quantified.
**If you would rather the model matched your description, the notebook text is what needs to
change** — tell me and I will drop the fourth ingredient from §2.3.5 and from the card.

## Q5 — `bike-km-day` (Kilomètres à vélo)

| # | Task | Status |
|---|---|---|
| 5.1 | Chercher des valeurs historiques du vélo en Belgique | ✅ **found** — Flemish OVG series, 11 waves 2007→2020, in km/person/day, **plotted as a line chart** |
| 5.2 | Fiche benchmark: km/pers/jour uniquement + Danemark | ✅ rewritten, DK added, **chart** DK 1.5 / BE 1.68 / NL 2.9 |
| 5.3 | Ajouter une fiche: pistes cyclables NL / DK / Flandre / Wallonie | ⚠ **partly — the four-way comparison is not defensible**, see below |
| 5.4 | Fiche "tendance passée": chiffre 2017 obsolète | ✅ replaced by EFM 2025 (11 % of trip loops) |
| 5.5 | Enlever la fiche "À manier avec prudence" | ✅ removed |
| 5.6 | *(added)* La moyenne belge cache deux pays | ✅ Flanders 2.5 vs Brussels 0.4 vs Wallonia 0.2 km/p/day, **chart** |

**5.1 — the Flemish series (OVG 3.0 → 5.5)**, from the analysis report's tabl. 8
(km/person/day) × tabl. 49 (bike share). 1.77 (2007-08) · 1.68 · 1.65 · 1.46 · 1.32
(2011-12) · 1.50 · 1.42 · 1.64 · 1.66 · 1.86 (2018-19) · 1.48 (2019-20). The products are
arithmetic, but the report prints the absolute km for the last two waves (1.85 and 1.47)
and they match to within 1 %, which validates the method. **The finding is that there is no
trend**: thirteen years of Flemish cycling policy did not move km-per-person; what rises is
the bike's *share*, because total distance travelled is falling. That is now the card.

**5.2 — the Danish surprise.** Denmark is at **1.5 km/person/day (2024, DTU)** — *below*
Belgium's 1.68. Denmark's cycling reputation is built on trip share (14.7 % of journeys),
not distance. Only the Netherlands (2.9) is a genuine outlier. The card says so, since a
group that assumes "be like Denmark" would be aiming *below* today's Belgian level.
Basis caveat: DK and NL are age 6+, the Belgian 1.68 is whole-population.

**5.3 — why the four-way comparison was not built.** The available figures count
incompatible objects, and the definitional spread is far larger than the spread between
countries:

| | figure | what it actually counts |
|---|---|---|
| Netherlands | 153 000 km (CBS, 2022) | every road where cyclists are allowed; only ~40 % is car-free |
| Flanders | 7 712 km (AWV, 2023) | cycle paths along **regional roads only** — municipal and provincial excluded |
| Wallonia | 1 557 km (SPW) | the RAVeL — former railways and towpaths, largely recreational |
| Denmark | — | no primary source openable; ~870 km is state roads only, most Danish paths are municipal |

Charting 153 000 / 7 712 / 1 557 would tell a room that the Netherlands has a hundred times
Flanders' cycle infrastructure, which is false. **What was built instead** is a card that
names the Flemish and Walloon networks side by side while saying explicitly that they are
different things — and adds the quality figure that actually carries the point: **40.8 % of
the cycle paths along Flemish regional roads score inadequate on design**.

## Q6 — `freight-tkm`

Rien à faire. ✅

## Q7 — `truck-share` (Part de la route, fret)

| # | Task | Status |
|---|---|---|
| 7.1 | Enlever la fiche "à manier avec prudence" sur les sources discordantes | ✅ removed (the peer-review objection about the same gap stays in `debate`, which is reveal-only) |
| 7.2 | Clarifier le report route → autres modes; fret aérien non désirable (↑ énergie) | ✅ subtitle names the air-freight residual; the intensity `caution` now ranks all four modes **with a bar chart** and says a shift to air raises demand; a new `reveal` fact spells out the destinations (rail {toRailPct} pts, water {toWaterPct} pts, **nothing to air**, LCVs untouched) |

New exported fact keys: `airKwhPerTkm` (0.893 kWh/tkm in 2050 — 8× a truck, 30× rail)
and `airShareTkm2019` (5.0 %), added to `workshop_levers/inland_mobility.py`.
Verified against the notebook: §3.1.1 sends heavy-truck tkm 15 pts to rail and
10 pts to inland waterways, and to nothing else.

## Q8 — `truck-load` (Charge des camions)

| # | Task | Status |
|---|---|---|
| 8.1 | Ajouter une fiche: charge moyenne si les camions étaient chargés à 100% | ✅ card + chart, plus two supporting cards |

**The ceiling card**: a Belgian 5-axle articulated lorry is capped at **44 t** gross
(Directive 96/53/EC; art. 32bis of the Belgian technical regulation), of which ~15 t is the
vehicle → **29 t of payload**. With one vehicle-km in five run empty — the EU figure already
on the neighbouring card — the all-trips average could not exceed about **23 t** even if
every laden truck left full. Today's 12.65 t is barely half of that. Charted as
today / laden EU average 14.4 t / 23 t / 29 t.

**A caution card was added with it**, because 29 t is an upper bound on an upper bound: it is
the figure for a 44 t artic, while the fleet also contains rigid trucks with far smaller
payloads and heavy trailer types (tankers, refrigerated, car transporters) whose tare exceeds
15 t. No source consulted quantifies the fleet-weighted ceiling, and the card says so.

**The volume card was rescued.** It previously asserted "trucks fill by volume before weight"
with `source: pratique logistique courante` and no URL — the one unsourced claim in the
lever. It now carries the d-fine/UIC figure: a 100 m³ trailer of goods at 72 kg/m³ weighs
**7.2 t** packed to the roof, a quarter of its weight capacity, and freight density is
*falling* (McKinnon for ACEA).

**Bonus: the model's 2019 value is independently confirmed.** Eurostat `road_go_ta_tott` for
Belgium 2019, on exactly the lever's definition (tkm ÷ all vehicle-km), gives **12.41 t**
against JRC-IDEES' **12.65 t** — within 2 %.

Deliberately not used: the circulating "trucks run at ~50 % of capacity" claim (no primary
source findable; the EEA load-factor indicator is discontinued, covers only DK/NL/UK and
defines load factor differently), and McKinnon's UK volume-constrained share (only visible
in search snippets). Provenance flagged: the 15 t tare comes from a study commissioned by
the rail lobby — its numbers cut against its own interest here, which is why they are usable.

## Q9 — Page finale "C'est envoyé, merci !"

| # | Task | Status |
|---|---|---|
| 9.1 | Graphique à barres +/− des effets sur la conso finale du secteur vs 2019 | ⬜ |
| 9.2 | Valeurs alignées sur la feuille de calcul principale (notebook) | ✅ verified, see below |
| 9.3 | Activable/désactivable par topic dans le YAML | ⬜ |

### Design decision, and why

Two shapes were tested against the notebook before writing any code.

**A true cumulative waterfall (2019 → group's 2050 total) does not work.** The levers
compound multiplicatively, so single-lever effects do not add up over a span that large:
summing the eight "back to 2019" effects gives 33.9 TWh where the notebook gives 39.3 —
a 5.4 TWh (14 %) residual bar. Around négaWatt's own point the interaction is only ~3 %
(measured: all seven levers pushed hard together → −0.18 TWh of interaction on a
+5.92 TWh joint change), but a chart anchored there would print négaWatt's 2050 total on
the participant's screen *before* the reveal, which D17 forbids.

**What is being built instead**: a diverging ± bar chart, one bar per lever, each showing
that lever's own effect on the sector's 2050 final demand against *keeping today's level
of the same indicator* — exactly `NW_IMPACT.contribution()`, the number already shown on
each question screen. No total to close, nothing revealed, and every bar exact.

### Q9.2 — verification against the notebook

`scripts/` scratch harness re-ran the transport notebook with each assumption overridden
and compared the resulting 2050 inland total against `impact.js`'s arithmetic:

| lever | value tested | notebook | impact.js | error |
|---|---|---|---|---|
| ground-km-day | 23.157 km/day | 20.220 | 20.250 | +0.030 |
| car-share | 68.44 % | 23.782 | 23.653 | −0.129 |
| car-share | 45.02 % | 22.546 | 22.610 | +0.064 |
| car-occupancy | 1.500 | 25.392 | 25.391 | −0.001 |
| car-energy | 95 % | 24.914 | 24.928 | +0.014 |
| freight-tkm | 7 715 tkm | 25.564 | 25.564 | +0.000 |
| truck-share | 63.50 % | 23.200 | 23.200 | +0.000 |
| truck-share | 40.61 % | 22.982 | 22.983 | +0.001 |
| truck-load | 16.45 t | 22.242 | 22.242 | −0.000 |

Worst error 0.13 TWh on a 23.08 TWh total (0.6 %). Note the two modal-shift levers can
only be moved with their destination splits rescaled to match — otherwise the notebook
itself prints *"There is an error in the modal shift"* and the comparison is meaningless.

---

## Found on the way (not in your list — no action taken)

**A removable division by zero in the transport notebook, cell 27.**

```python
dlt_PM_spe     = pro_PM_spe * ref_PM_spe
rem_dlt_PM_rel = 1 - dlt_PM_spe_avi_lng / dlt_PM_spe     # <- 0/0 when pro_PM_spe == 0
rem_dlt_PM_abs = rem_dlt_PM_rel * dlt_PM_spe
```

`rem_dlt_PM_abs` is algebraically just `dlt_PM_spe - dlt_PM_spe_avi_lng`, so the two
forms agree everywhere except at `pro_PM_spe == 0`, where the written form gives
`(1 - a/0) * 0` → NaN and the model silently returns nonsense (11.2 TWh instead of
~24.5). The limit is perfectly well behaved: `pro_PM_spe = -1e-6` gives 24.503 TWh.

So **the model cannot currently be run with no overall mobility reduction** — a natural
thing for a reviewer or a sensitivity sweep to try. The fix is one line and is
behaviour-neutral. Found while checking the waterfall arithmetic (Q9); not needed for
it in the end, because the 2019 anchor sits at `pro_PM_spe` ≈ −0.04, away from the
singularity. **Not changed — your call.**

---

## Log

- Start: repo had uncommitted work-in-progress (fact deletions for Q1–Q4, a new
  `lever` fact kind, extra `facts` keys in `workshop_levers/inland_mobility.py`
  not yet regenerated into `website/data/levers_transport.js`).
- All nine items addressed. Nothing committed.

### Final state

8 levers, 44 facts, 10 charts (was 39 facts, 1 chart).

Verified after the last rebuild:
- `jupyter nbconvert --execute nW_BE_demand_model_transports.ipynb` — exit 0
- `python scripts/build_workshop_content.py` — clean, no warnings (every lever now
  carries at least three pre-answer facts)
- `python scripts/test_workshop_helpers.py` — 8/8
- `python scripts/verify_workshop_export.py` — 295/295
- `python scripts/test_workshop_api.py --base http://127.0.0.1:8787` — 59/59
- a scripted sweep over the built bundle: **no lever's target value appears in any
  question, subtitle, tangible, non-reveal fact or chart caption in any of the three
  languages**, and every fact quoting a figure has either a URL or an nW-BE section
- FR and NL walked in the browser: play screens, printed cards, all 10 charts render
- the summary chart driven end to end: full answers, partial answers, phone width,
  and a topic with the chart switched off

### Engineering changes made along the way

| file | what |
|---|---|
| `scripts/build_workshop_content.py` | chart `y:` accepts `"{key}"` placeholders, so plotted model numbers resolve from the notebook instead of being typed by hand; new `lever` fact kind; `summaryChart:` block validated per topic, and refused when any lever lacks a usable impact |
| `website/assets/js/workshop/spark.js` | new `NW_SPARK.effects()` — the diverging ± chart, with a stacked layout below 430 px |
| `website/assets/js/workshop/play.js` | renders the effects chart on the done screen |
| `website/workshop/play.html` | the `#effects` figure |
| `website/assets/css/workshop.css` | `lever` fact-kind colour; `.ws-effects` styles |
| `website/workshop/content/ui.yaml` | benchmark label → "Comparaison internationale"; `lever` kind label; four `play.effects.*` strings |
| `workshop_levers/inland_mobility.py` | new exported facts: per-mode km/day for 2019, Peeters bounds in km/day, `airKwhPerTkm`, `airShareTkm2019` |

### The five open points — all resolved 2026-09-20

1. **Printed deck length** — accepted. ~7 pages instead of 4, nothing clipped.
2. **Q4.1** — **notebook updated.** §2.3.5 no longer lists powertrain efficiency among the
   ingredients of the −25 %; it now states that `redu_fuel_PM_car` multiplies the kWh/km of
   each propulsion type separately and therefore carries the energy needed to *move the
   vehicle*, with the propulsion mix counted in §2.2.5. The section also gained the
   literature orders of magnitude for the three remaining ingredients. The −25 % value is
   unchanged. The `car-energy` fact card and justification were rewritten to match.
3. **Cell 27 division by zero** — **fixed.** `rem_dlt_PM_abs` is now written as
   `dlt_PM_spe - dlt_PM_spe_avi_lng`; `rem_dlt_PM_rel` is kept for reporting, guarded.
   Verified behaviour-neutral: the négaWatt point is 23.076 TWh before and after, and
   `pro_PM_spe = 0` now returns 24.503 TWh instead of a NaN-poisoned 11.198.
4. **5.3** — left as built (no four-way comparison).
5. **Docs** — `docs/workshop_module.md` gained **D40** (the ± chart and why it is not a
   waterfall), **D41** (chart values as placeholders), **D42** (the `lever` fact kind) and a
   new §12 review-round section. `README.md` gained the `summaryChart:` contract, the chart
   placeholder rule and the `lever` kind.

### Found and fixed while verifying

The `car-energy` subtitle shipped with literal `*asterisks*` on screen: `NW_I18N.rich()`
renders emphasis only for a fact's text, the justification and the debate — never for a
question, short, subtitle, tangible, historyNote or chart caption. The subtitle was reworded,
and **the build now refuses `*emphasis*` in any field that does not render it** (negative
test: adding stars to a `short:` fails the build).

### Deployed 2026-09-20

`bash scripts/deploy_website.sh` — rsync clean, the five spot-checked files match the server.

- `test_workshop_api.py --base https://negawatt.squoilin.eu/api` — **59/59**
- workshop index, play, reveal, cards, `workshop_content.js`, `spark.js` — all **200**
- `workshop/content/ui.yaml` — still **403**, as intended
- live site driven in the browser: 8 cards, 10 charts, no unresolved placeholder, no
  literal asterisk; the eight-lever run reaches the summary screen and the ± chart draws
