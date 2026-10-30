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

> **Next step:** WP5 — brief and launch agent 2 (buildings heat: new `buildings-heat.yaml` + `floor-area`, `thermostat`, `ter-thermostat`), then review.

Environment: no `data_processing` env on this machine; the `pypsa-eur` env has every
dependency (numpy, pandas, scipy, matplotlib, nbformat, nbconvert, yaml, jupyter):

```bash
PY=/sylvain/mount/pypsa-wal-data/miniforge3/envs/pypsa-eur/bin
$PY/python scripts/build_workshop_content.py --check
$PY/jupyter nbconvert --to notebook --execute --output /tmp/o.ipynb nW_BE_demand_model_transports.ipynb
```

Restore point: `git checkout pre-workshop-2026-11` (tag on 7e0d3be, the state before any
of this). Server backup of the deployed files: see WP8.

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
| WP5 | Content — buildings heat (theme + 3 cards) | agent 2 (Sonnet) + review | ⬜ |
| WP6 | Content — international mobility (3 cards) | agent 3 (Sonnet) + review | ⬜ |
| WP7 | Tests: build, helpers, export, API, browser FR/NL/EN, print | orchestrator | ⬜ |
| WP8 | Server backup, deploy for testing, live check | orchestrator | ⬜ |
| WP9 | README + `docs/workshop_module.md` (D69–D72, review round), final notes | orchestrator | ⬜ |

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
