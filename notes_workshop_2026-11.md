# Workshop of 2 November 2026 — adaptation tracker

Request (2026-10-30, Sylvain): adapt the workshop questions and cards to
`scenario_fiches.docx` (repository root, untracked — text extraction with its tracked
changes and comments in `docs/workshop_2026-11/request_scenario_fiches.md`).
The covering e-mail mentioned in the request was **not included in the message**; the
.docx is the only written source used.

Constraints from the request:

- never delete a question: modify it, or retire it and keep it as legacy;
- a config to switch questions on and off easily (fewer questions next time);
- commit locally, **never push**; deploying to the live site for testing is allowed if
  the previous state can be restored (`git tag pre-workshop-2026-11` + server backup);
- answers already in the database stay readable: modified questions keep their id,
  replaced questions keep their id and their content, and the link is recorded;
- one Sonnet agent at a time, sequential work, documented here so it can be resumed;
- broad public, concrete and clear, but scientifically correct and sourced.

Status: ⬜ todo · 🟡 in progress · ✅ done · ⛔ dropped (with reason)

---

## Resume here

> **All work packages done (2026-10-30).** The live site plays the 2 November line-up.
> What is left is for the organisers — see "Before the workshop" at the end.

Environment: no `data_processing` env on this machine; the `pypsa-eur` env has every
dependency (numpy, pandas, scipy, matplotlib, nbformat, nbconvert, yaml, jupyter):

```bash
PY=/sylvain/mount/pypsa-wal-data/miniforge3/envs/pypsa-eur/bin
$PY/python scripts/build_workshop_content.py --check
$PY/jupyter nbconvert --to notebook --execute --output /tmp/o.ipynb nW_BE_demand_model_transports.ipynb
```

Restore point: `git checkout pre-workshop-2026-11` (tag on 7e0d3be, the state before any
of this). The live site was identical to that tag on 2026-10-30 (checked file by file).

**Putting the live site back as it was** (either way works; answers in the database are
never touched by a deploy):

```bash
# a) from the server-side snapshot taken before the first deploy (whole site but intervec/,
#    RFNBO_final_results/ and api/config.php, which were left out of it and are never touched)
ssh negawatt@negawatt.squoilin.eu 'tar -xzf ~/backups/public_html_2026-10-30_pre-workshop-2026-11.tgz -C ~'
# b) from git: publish the tagged website/ (rsync without --delete, like the deploy script)
git worktree add /tmp/nw-restore pre-workshop-2026-11
rsync -avz --no-perms --omit-dir-times --exclude '*.md' /tmp/nw-restore/website/ negawatt@negawatt.squoilin.eu:public_html/
git worktree remove /tmp/nw-restore
```

A deploy of this work changes exactly 18 files on the server (dry run with `--checksum`):
`assets/css/workshop.css`, `assets/js/workshop/{cards,impact,join,play,reveal}.js`,
`data/{history_transport,levers_transport,workshop_content}.js`,
`workshop/{index,play}.html`, and the seven YAML files under `workshop/content/` (two of them
new; YAML is never served, `.htaccess` answers 403).

---

## What the .docx asks for

1. **Fewer questions, 9 cards in 3 themes**, "focus on the collective, sufficiency and
   clarity (a few efficiency ones are fine)":

   | theme (docx) | card (old letter) | lever id | change |
   |---|---|---|---|
   | Mobilité intérieure | B car share | `car-share` | reworded |
   | | C people per car | `car-occupancy` | reworded |
   | | D energy per km (3 sliders) | `car-energy` | reworded |
   | Mobilité internationale | A long-haul trips **+ B short-haul trips** | new `plane-trips` | **merged**: one card, two sliders |
   | | C long-haul load **+ D short-haul load** | new `plane-load` | **merged**: one figure for all flights |
   | | G air freight | `air-freight` | reworded |
   | Chaleur des bâtiments | residential A floor area | `floor-area` | reworded |
   | (residential + tertiary | residential D thermostat | `thermostat` | reworded |
   | in one theme) | tertiary C thermostat | `ter-thermostat` | reworded |

2. **New structure for every card** (proposal by Bieze): Question → Introduction to the
   question → Reference points (plain language) → Answer (chart) → « Pour que cela
   arrive, il faudrait que… » → « Ce que votre réponse change » / conclusion.
3. Clearer questions and reference points: the tracked changes give the new French
   wording card by card; 14 reviewer comments (C0–C13) and 2 footnotes to address.

---

## Design (decisions D69–D72 in `docs/workshop_module.md`)

- **D69 — Line-ups.** `website/workshop/content/programme.yaml` names the active line-up
  and lists, per theme, the questions played, in order. A question not listed is not
  deleted: it is still built and validated, kept out of the play screen and the printed
  cards, and shown on the reveal when the date window holds answers to it ("archived").
  Switching a question on or off = editing one list + rebuilding the content (no
  notebook run).
- **D70 — Themes made of other themes' questions.** `buildings-heat.yaml` ("Chaleur des
  bâtiments") carries only its title, lead, intro and summary chart, and plays
  `floor-area`, `thermostat` (residential) and `ter-thermostat` (tertiary) from their own
  files. New groups are stored under topic `buildings-heat`; past groups stay under
  `residential-heat` / `tertiary-heat`, whose reveal still shows them.
- **D71 — The card follows the docx structure.** Play screen and printed card reordered:
  question, introduction, reference points, answer (chart + slider), the condition,
  what the answer changes.
- **D72 — Two merged aviation questions, new ids.** `plane-trips` (two sliders: long-haul
  and short-haul return trips per lifetime, adding up to air km per person per year, the
  measured curve) and `plane-load` (passengers per flight, all flights). The four old
  levers keep their ids, their content and their answers, as archived questions;
  `plane-trips__long-haul` / `__short-haul` mean exactly what `long-haul-flights` /
  `short-haul-flights` meant (same unit), which the lever records as `supersedes`.

---

## Work packages

| WP | What | Who | Status |
|---|---|---|---|
| WP0 | Restore point (tag), docx extraction, this tracker | orchestrator | ✅ |
| WP1 | Line-ups: `programme.yaml`, build, join/play/cards/reveal, per-lever sector | orchestrator | ✅ |
| WP2 | New card structure: play screen + printed card | orchestrator | ✅ |
| WP3 | Model: `plane-trips`, `plane-load`, weighted parts, per-part impacts, new history series, notebook re-runs | orchestrator | ✅ |
| WP4 | Content — inland mobility (3 cards) | agent 1 (Sonnet) + review | ✅ |
| WP5 | Content — buildings heat (theme + 3 cards) | agent 2 (Sonnet) + review | ✅ |
| WP6 | Content — international mobility (3 cards) | agent 3 (Sonnet) + review | ✅ |
| WP7 | Tests: build, helpers, export, API, browser FR/NL/EN, print | orchestrator | ✅ |
| WP8 | Server backup, deploy for testing, live check | orchestrator | ✅ |
| WP9 | README + `docs/workshop_module.md` (D69–D75, §18), final notes | orchestrator | ✅ |

---

## Log

- 2026-10-30 — WP0 done: tag `pre-workshop-2026-11`, extraction saved, tracker written.
- 2026-10-30 — WP1 done. `programme.yaml` (active: `full` until the new questions exist);
  `build_workshop_content.py` reads it (`--lineup` to override), compiles each question for
  the theme that plays it (`compile_lever`), keeps the others as `archive` with their old
  letter, resolves `{questionCount}`/`{otherCount}` in theme-level text, accepts
  `subtitleSource`/`subtitleUrl`, and refuses a played card that points at an archived one.
  Without a line-up file the bundle is identical to before (checked against the committed
  one). JS: entry page offers only line-up themes; play refuses archived themes; cards
  print the line-up (`?all=1` adds archived cards); reveal walks played questions plus
  archived ones that have answers in the window (dashed grey letter, "question archivée").
  Each question carries its `sector`, so a theme can mix files and sectors.
- 2026-10-30 — WP2 done. Play screen: question → introduction (subtitle, with an optional
  source line) → "Quelques repères" → "Votre réponse pour 2050" (chart, slider, number) →
  "Pour que cela arrive, il faudrait que…" (3-line field) → "Ce que votre réponse change"
  (the tangible reading, then the TWh of this lever alone). Printed card: question +
  introduction → facts → 2019 value and range → two lines for the condition. Checked in
  the browser (FR, inland mobility, slider moved: box fills, no console error).
- 2026-10-30 — WP3 done. Helpers: a "sum" part may carry a `weight`; impact kind `parts`
  (each part carries its own impact record, the lever's effect is their sum —
  `NW_IMPACT.leverContribution`, used by the readout and the ± summary chart). Module:
  `plane-trips` (km/person/year by air; parts `long-haul` 225 km/yr per trip and
  `short-haul` 27.1 km/yr per trip, each exactly the old lever: same unit, values,
  targets and impact record — checked) and `plane-load` (passengers per flight, kerosene
  pkm / kerosene aircraft-km; inverse on the kerosene TWh; a uniform +10 % gives
  −0.6518 TWh, the same as the two old levers summed). Both carry
  `model.supersedes`. data_aux: series `aviation_km_year` (3 176 km in 2019) and
  `avia_occupancy_all` (152.4). Notebooks re-run: only those additions changed;
  `energy_totals_overrides.csv` byte-identical, no existing lever changed. Skeleton
  wording for the two new questions appended at the end of
  `international-mobility.yaml` (so archived cards keep letters A–G). `full` line-up now
  lists the seven old aviation questions explicitly. Tests: helpers 17/17, export
  438/438 (verifier taught `parts` impacts and weights), API 61/61 on dev_api, browser:
  two sliders → 2 264 km/pers./an and −3 TWh for 4 long-haul trips.
- 2026-10-30 — Side work while agent 1 ran: server tarball
  `~negawatt/backups/public_html_2026-10-30_pre-workshop-2026-11.tgz` (live == tag, checked
  on five files); the workshop is now served at https://workshop.negawatt.be/ (the old
  URLs redirect; same `public_html`), so `deploy_website.sh` was made path-independent and
  verifies there; D69–D72 written; README updated; `scripts/question_registry.py` →
  `docs/workshop_question_ids.md` (every id, incl. `retired:` ids in programme.yaml —
  `truck-load`, `car-energy__speed`, `car-energy__size` hold live answers). Live DB at that
  time: 106 groups, 213 answers. The API contract test writes into the real topic: never
  run it against the live server. Reveal checked on dev_api: 3 played cards, then the
  archived ones that have answers in the window ("question archivée").
- 2026-10-30 — WP4 done (agent 1 + review). The three inland cards follow the docx;
  C0 (national average hides town/country) in the intro; C1 (80.4 vs 78.5: Eurostat counts
  car/bus/train only) stated in the comparison; C2/C3 (speed) rewritten: the slider is the
  average speed over all car-km, a limit change only acts on its road type and is diluted
  by its share of km. Docx corrections, all verified by the agent: "Vision Rail **2040**"
  (adopted 2022; baseline 8 % is **2019**, Eurostat), "76,3 %" is Flanders *outside the
  cities* (all Flanders 71.9 %), federal survey 1.21/1.03/1.40 = adults, fieldwork
  2024-25 (report Nov 2025, updated May 2026), with `subtitleSource`/`subtitleUrl`.
  Vision Rail url is an archive.org copy (the official PDF is offline). Reviewer fix:
  "Quelle place à la voiture" → "Quelle place **pour** la voiture". Topic intro/notes use
  `{questionCount}`/`{otherCount}`. Archived inland blocks byte-identical. Printed cards
  of this theme are each longer than half a page (as before): one card per A4 page.
- 2026-10-30 — While agent 2 ran: per-haul curves for C4 (`avia_lng_trips_life`,
  `avia_srt_trips_life`, in return trips per lifetime; 2019 = the sliders' start, checked by
  the verifier, 450/450) and the fact keys the merged aviation cards need (people/years per
  long-haul trip, months per short-haul trip, plane vs car per pkm, 2000-2023 occupancy
  gains). Notebooks run in a throw-away git worktree and the two generated files swapped
  in atomically, so the agent's builds never read a half-written file. Additive only.
  §18 of `docs/workshop_module.md` drafted.
- 2026-10-30 — WP5 done (agent 2 + review). New `buildings-heat.yaml` (title "Chaleur des
  bâtiments", `include: [residential-heat, tertiary-heat]`, summary chart on the heat demand
  of homes and tertiary buildings). `floor-area`: the reviewers' personal introduction plus
  one clause saying the answer is the national average (empty and second homes included);
  the docx's "59,4 % des logements sous-occupés" is in fact **59,2 % of people** (Eurostat
  `ilc_lvho50a` measures people; 59,4 % was the women-only row), EU 33,4 %; Brussels
  overcrowding 30 % (Baromètre social 2025, EU-SILC 2024; IBSA gives 31 % for 2023) — two
  cards so that each figure has its link. `thermostat` and `ter-thermostat`: the reviewers'
  introductions (the tertiary one said "logements" by mistake); the new "En complément de
  l'isolation" card is sourced to Aydin, Kok & Brounen 2017 (RAND J. Econ., 563 000 Dutch
  households, rebound 27 % owners / 41 % tenants) and says "en partie", because the
  performance gap has several causes and the study measures consumption, not setpoints
  (Fowlie et al. 2018 found no higher indoor temperatures); "dans les logements" on the
  tertiary card since the study covers homes only. The 12 °C humidity sentence is checked
  in the SlowHeat book (p. 118). The residential thermostat chart is now inverted like the
  tertiary one (D60), so both read the same way in one theme — also in the legacy theme.
  `short` labels "Thermostat du logement" / "Thermostat du tertiaire" tell them apart.
  Home files: `{questionCount}`/`{otherCount}`. Archived blocks byte-identical.
  Build guard added: an id listed under `retired:` cannot come back (tested).
  Follow-up noted, not done: the archived `renovation-depth` debate quotes Aydin et al. as
  "about 56 %" — check against the paper; the archived `cooking` block carries two `url:`
  keys (YAML keeps the last). Printed card A is 244 mm (8 reference points).
- 2026-10-30 — While agent 3 ran: buildings-heat checked in the browser (entry page, the three
  cards with readouts and TWh effects, summary chart with three bars, reveal with négaWatt
  values and the summary table). Entry page: the facilitator panel now links the reveal of
  every theme left out of the line-up (from 2026-01-01 to today). Print: `cards.html` to PDF
  with headless Chromium — inland 3 pages, buildings 3 pages (FR), one card per A4 page, the
  new order visible (question+intro, facts, answer box, two lines for the condition).
- 2026-10-30 — Agent 3 stopped on the session limit right after starting (nothing written;
  checked: only the bundle differed, by the new `start.archived` string, rebuilt and
  committed). Relaunched with the same brief once the limit reset.
- 2026-10-30 — WP6 done (agent 3 + review). `plane-trips`: introduction with return-trip
  distances, an 80-year life and the km/person/year sum; trend fact rewritten with Statbel
  (C5/C6: airports passed 2019 in 2025 — 36 M passengers against 35.5 M — flights still
  below; airport passengers ≠ km flown by residents; the km series stops in 2023 at −12 %);
  two curves per flight type in trips per lifetime (C4); "Qui vole et ne vole pas" with ICCT
  moved to the source (fn2); the western-Europe survey; the 80 % caution dropped (C7); "Et
  les émissions de CO₂" from primary sources instead of the Canopea dossier (fn3): EU
  aviation ≈ 4 % of EU GHG in 2022 (European Commission), Belgian international aviation
  +68 % 1990→2022 (Eurostat env_air_gge, a memo item outside the national total);
  technologies (Airbus ZEROe postponed 2025; hydrogen mostly fossil, IEA; SAF 0.3 % of jet
  fuel and 11 % of renewable-fuel *capacity*, IATA Dec 2024; SAF are mostly biofuels, not
  "carburants de synthèse" as the docx said); "Avion ou voiture" keeps both points (C8);
  the train facts kept. "Répartition des vols" dropped (C9). `plane-load`: all-flights trend
  on the chart's basis (110 → 152 → 164) then per flight type; C10 explained (an
  aircraft-km costs about the same, 67.3 vs 69.1 kWh; 188 vs 122 passengers make the
  0.358 vs 0.566 kWh/pkm); C11/C12 rewritten (passengers/flight +22 %, fuel per aircraft-km
  −38 %, 2000→2019: progress more than offset size); Ryanair is **94 %** on the year (FY26),
  96 % in summer only (the docx's 96 % annual held in FY19). `air-freight`: tracked changes,
  C13 ("fret général" = everything but express, which is ~18 %, Boeing WACF 2024–2043),
  parcels "131 milliards en 2020, doublement attendu d'ici 2026" (Pitney Bowes 2021), no more
  `{question:…}` to archived cards. Review fix: the trend fact no longer repeats the answer
  chart (third chart on the card). Line-up `workshop-2026-11` **activated**; all checks green
  (build both line-ups, registry, helpers 17, export 450, API 61 on dev_api).
- 2026-10-30 — WP7 done. Scripted walk of every card (dev servers): inland FR, aviation NL
  and FR, buildings EN — question, introduction, reference points (4–10), charts, sliders,
  readout, "Ce que votre réponse change" (tangible + TWh) filled on every card, no raw
  placeholder, no console error; end screen with the summary table and the ± chart (3 bars);
  e.g. 4 long-haul + 30 short-haul trips → 1 713 km/pers./an and −5,7 TWh, which matches
  the hand computation (−2,96 − 2,78). Reveal: the three themes walk A–C with négaWatt's
  values; the aviation one shows the combined km per group. Print (`cards.html` → PDF,
  headless Chromium): inland 3 pages, aviation 5 (the two-slider card runs over two pages),
  buildings 3 — the same in FR, NL and EN.
- 2026-10-30 — WP8 done. Final dry run: the same 18 files. `scripts/deploy_website.sh` →
  all 12 verified files identical at https://workshop.negawatt.be/. Read-only checks on the
  live site: entry page offers the three themes and links the two archived ones; reveal of
  *Mobilité intérieure* from 2026-01-01: 151 answers / 70 groups (3 played cards with 20, 20,
  19 groups, then the 5 archived ones with their old letters); *Chaleur des logements*: 12
  answers / 15 groups on its 8 archived cards; play page opened with `api=http://127.0.0.1:9`
  (offline, nothing written; storage cleared). Live DB counts unchanged:
  106 groups / 213 answers / 1 459 log rows.
- 2026-10-30 — WP9 done: D73 (both thermostat charts inverted), D74 (archived themes on
  the entry page), D75 (ids never reused; registry); §18 completed.

## Before the workshop (for the organisers)

1. **Read the nine cards** in the language(s) you will use:
   https://workshop.negawatt.be/cards.html?topic=inland-mobility (then
   `international-mobility`, `buildings-heat`; `&lang=nl` / `&lang=en`). The Dutch was not
   written by a native speaker.
2. **Print** them from the same page: 3 + 5 + 3 A4 pages, one card per page (the two-slider
   aviation card runs over two pages).
3. **On the day**: share https://workshop.negawatt.be/ — the entry page offers the three
   themes; open the reveal from its facilitator panel. Groups playing different themes are
   kept apart; a group plays one theme, so a table doing all nine questions plays the three
   themes one after the other (three groups on the reveal).
4. **To change the questions**: edit `active:` or the lists in
   `website/workshop/content/programme.yaml`, run `python scripts/build_workshop_content.py`,
   then `bash scripts/deploy_website.sh`. `full` gives back the thirty questions of
   September–October.
5. **To undo everything** on the live site: see "Putting the live site back" at the top.
