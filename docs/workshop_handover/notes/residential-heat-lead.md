# Recovered work — residential-heat (lead)

Agent `a7e3eccc354bc98be`, stopped by the session limit before it could report.
Everything below is recovered verbatim from its transcript, in order. **Nothing here has been checked** — treat every figure as unverified until re-sourced.

15 entries, of which 0 web results.

---

## reasoning

I'm puzzled — export_topics should raise a RuntimeError if no module declares SECTOR = "buildings", yet levers_buildings.js exists with an empty levers object. That means something must have generated it despite there being no matching modules, so I need to figure out how that empty file got created in the first place.

I want to double check the actual git status and directory contents to see if workshop_levers is tracked or hidden by gitignore, and whether a sibling agent's file is lurking somewhere I haven't checked yet. I'll also verify the notebook and review document to see if they reference files that don't actually exist yet.

## reasoning

I need unit strings for my topic YAML, but I'm worried about collisions with the sibling agent's tertiary YAML — both might declare `unit.m²/person` or `unit.kWh/m²`, and the build treats duplicate keys as a hard error rather than silently overriding them.

I'm weighing options: prefixing keys with the topic id would break the UI's lookup convention, making the raw unit strings deliberately distinct would look ugly in the interface, and just omitting the strings entirely would fall back to the raw English string across all languages — acceptable but not ideal for French translations like "m²/pers.". I want to check how i18n.js resolves unit keys before deciding.

Since I don't know the sibling agent's id to coordinate directly, I'm mapping out which units both residential and tertiary topics will likely need — floor area per person, thermostat degrees, renovation rate percentages, heat per m², hot water per person, district heating share — and there's a real risk of key collisions since topic files load in alphabetical order with residential-heat processed before tertiary-heat.

Given the guidance to report shared-file conflicts rather than editing them myself, I'm thinking through whether to rely on fallback behavior in the i18n handling for shared units and only declare truly topic-specific strings, checking whether that fallback degrades gracefully.

I'm also reconsidering the temperature unit — rather than a delta value, expressing it as an absolute indoor temperature in °C might read better to participants, but I need to check how `d_temp` and `d_cons_temp` are separately assigned across the residential and tertiary cells to make sure they don't collide as shared global names.

I should check cells 61-76 to see the tertiary values and confirm which variables get overwritten.

## reasoning

I'm realizing `d_temp` and `d_cons_temp` get overwritten by cell 65, so at export time I can't read the residential value directly. Instead I need to back it out from `suf_RS_tes_sht`, calculating d_temp_RS as (1 - 0.86)/0.07 = 2.0.

Rather than hard-coding this fragile derivation, I want to follow the inland_mobility.py pattern of naming documented prose constants and asserting them against the model — so I'll define `ref_RS_setpoint_drop_c = 2.0` and `ref_RS_pct_per_degc = 7.0`, then assert `suf_RS_tes_sht` matches. I also need to flag that `share_heat_dhn` and `share_heat_ihs` get overwritten too, so I should rely on the RS-prefixed variables like `ref_RS_tes_dhn` and `trg_RS_tes_dhn` instead, and report this variable-reuse issue as a finding.

The reuse of `share_heat_dhn`, `share_heat_ihs`, `d_temp`, and `d_cons_temp` across residential and tertiary sections is a genuine fragility — cell 75 silently clobbers residential values, and everything only works because of the current execution order in "Run All." If someone re-ran cell 57 in isolation after a full pass, residential results would silently corrupt. This ordering dependency is worth flagging as a notebook-level bug alongside the same issue for `d_temp`/`d_cons_temp`, plus I want to check the cell 22 comment about space heating being reported per average household surface area.

Now I'm verifying the residential cooling growth figures — checking that the 2000 to 2023 increase from 0.0448 to 1.0096 kWh/m² gives the stated 22.5x factor, and that the 2019-2023 jump of 0.5725 kWh/m² matches what's described in the prose. Both check out against the source numbers.

Maybe the discrepancy is a different denominator — the history series may use floor areas from the EU Building Observatory while my notebook uses a different reference surface value. Let me try converting the ktoe figure through kWh using the standard ktoe-to-GWh factor and dividing by the surface area to see if I land closer to 0.035 or 0.0448.

Working it out: 272.78 GWh converts to 272.78e6 kWh, divided by 625.236e6 m² gives about 0.4363 kWh/m², which lines up well with the historical 2019 figure of 0.4371 — the small gap is likely due to differing population or floor-area reference years.

Now checking the cooling numbers: the notebook's growth rate of 0.035 kWh/m²/year versus the prose claim of rising "from 0,035 kWh/m² to 1,010 kWh/m²" looks like an error, since 0.035 is actually the rate, not the year-2000 starting level, which was closer to 0.045. I need to verify whether the stated "+0,035 kWh/m²/year" growth assumption actually matches the historical trend.

Calculating the actual historical average rate from 2000 to 2023 gives roughly 0.042 kWh/m²/yr, using both the 1.0096 and slightly adjusted 1.010 endpoint values — so the assumed 0.035 rate is somewhat below the historical average rather than matching it as claimed. I'm trying alternate ways they might have derived 0.035, like projecting a 2050 target from the 2019 baseline, but none of my calculations land exactly on that figure yet.

I should double-check this by actually running the notebook and computing the numbers directly rather than guessing.

For the renovation rate, the notebook states the average annual renewal rate of floor area was around 3% between 2000 and 2023, with an associated reduction in end-use demand of about -0.458 kWh/m²/year, and the efficiency assumption doubles this to -0.916 kWh/m²/year — so I'm trying to pin down exactly what lever this "renovation rate" assumption controls.

The actual degree of freedom in the model is `acc_RS_tes_sht_ren`, a dimensionless multiplier currently set to 2, and the tangible unit for the audience should be "% of stock renovated per year" — with 2019 observed at roughly 3.07% and the 2050 target being double that, around 6.1%. I need to be precise here since the model actually doubles the improvement rate in kWh/m²/year, not the renovation rate directly, so I want to define both the model's dimensionless multiplier and the historical ~3% renewal rate carefully to get the reveal exactly right.

I'm now trying to nail down whether to base the mapping on the 2019 observed value (3.0673%) or the mean of the full 2001-2023 series, since the lever's ref value should probably be the 2019 endpoint per the design principle, while the target scales off the historical mean — though I need to check these two bases don't create an inconsistency.

Tying `cur` to that ~3%/yr renewal rate means the acceleration factor `acc` equals rate divided by 3.0, so a target rate of 6%/yr gives acc=2.0. There's a slight inconsistency between the prose's round "3.0" baseline and the actual observed 2019 figure of 3.0673, but that's acceptable given the year-to-year wobble in the historical curve (2.55 to 3.88).

I'm considering instead anchoring to the mean of the full 2001–2023 series rather than the round prose figure, since that's what the "-0.458 kWh/m²/year average" claim actually pairs with — but that series is computed in a separate data_aux notebook, not accessible from the buildings notebook where residential_heat.py lives, so I likely have to fall back on the prose's declared value of 3.0 as a cross-check constant.

I'm verifying the 2019 observed heat intensity by working through the unit conversion from ktoe to kWh/m², confirming the 68.19 figure closely matches the 68.15 value in the history series.

The reference values in the notebook line up well with the history data, confirming the slider represents the 2050 endpoint of an actual trend curve. I'm now going through the residential model's adjustable parameters, starting with floor area per person, which drops from about 54.69 to 49.22 m²/person under a -10% change.

Next I'm looking at renovation rate, which goes from a 2019 baseline of 3.07% up to a target of 6.0% of stock per year, directly scaling heating intensity. Then I'm considering the thermostat lever, weighing whether it should represent an absolute indoor temperature or a temperature reduction, since there isn't an observed historical series for indoor temperature to anchor it to.

Since I can't invent a 2019 baseline setpoint, maybe I should frame the lever as degrees below today's thermostat setting, using historyAbsent since no measured Belgian indoor temperature series exists — this seems consistent with how several other inland levers already lack curves. The core question becomes how many degrees sufficiency could lower the average heating setpoint by 2050.

I'm considering whether the EU Building Stock Observatory or Odyssee-Mure has a citable Belgian average indoor temperature figure (around 19-20°C), but without a verified source and URL I shouldn't state that number given the constraint against unsourced figures.

Since I can't add a new unverified setpoint value, I'm working out the lever design differently: a −2°C change from a zero reference, with the slider range checked so the target isn't dead-centre in the min-max span—adjusting the bounds from -5/+1 to -6/+1 to avoid that edge case.

With decimals computed, I settle on one decimal place giving "−2,0 °C". I'm also weighing whether to frame this as degrees of setpoint reduction instead, with positive numbers and "more is better" framing, and thinking through how that phrasing would translate across French, English, and other languages.

For the French version, I land on "Combien de degrés en moins sur le thermostat la sobriété peut-elle permettre en 2050 ?" with a value of 2,0, giving "2,0 °C de moins" — this resolves the sign confusion nicely. I'm now working through the leverage calculation, computing the difference between the answer and the 2019 baseline of zero, with more reduction counting as more ambitious, and figuring out that the impact scales linearly with a slope tied to the space heating baseline. I need to check how the impact calculation logic handles this kind of linear-shift relationship, then move on to look at the hot water lever, where the target shifts from 675 to 533 kWh per person.

I'm considering whether to express hot water usage in litres per day per person instead of kWh, since the notebook already computes that tangible equivalent (64 down to 50 litres/day). But the underlying history series is in kWh/person/year, and since the chart pulls from that series directly, switching units would create a mismatch between the slider and the chart — so I need to keep kWh/person/year as the actual unit and just surface the litres/day figure in the tangible description and supporting facts instead.

I'm also noticing the series has a suspicious jump from 675 down to 435 between 2019 and 2023, likely a data break, so I should still show the real series but flag it with a caution note rather than building the lever's narrative around that discontinuity — and I'm weighing whether the 2023 value already dipping below the 2050 target creates an interesting but tricky framing issue.

Given the brief's emphasis on sliders being real curve endpoints, I think showing the full 2000→2019 rise followed by the suspicious post-2019 drop with a prominent caution fact is more informative than hiding it. I'll go with that approach and note the reasoning in the report before moving to the next parameter, trg_RS_tes_cok.

Cooking energy shows a genuinely counter-intuitive 15% increase under sufficiency assumptions, but it lacks a history series, uses household rather than per-person units, and is small in absolute magnitude (1.12 TWh in 2019) — so its workshop impact would likely be limited. I'm now working through candidates for the 6-8 levers I want, starting with floor-area per person which does have history data.

Other candidates: renovation rate (with history), thermostat setting (no history), hot-water use per person (with history), cooling per square meter (with history), and district-heating share (only two data points, no real history). I'm also considering whether a heat-carrier shift belongs on the list — the brief mentions this, but in the residential model the only true carrier shifts are the district-heating share and the cooking gas share, since the buildings notebook otherwise treats heat as a single undifferentiated category rather than splitting fossil vs. heat-pump.

I'm settling on seven variables to display: floor area per person, renovation rate, thermostat setting, hot-water use, cooling demand, district-heating share, and cooking energy per household, each with their historical-to-target value ranges.

Cooking-gas electrification (roughly 26.8% down to 2%) feels more like a fixed policy shift than a sufficiency lever, so I'll list it as a spare, unshown variable alongside things like bus and train occupancy, since spare levers don't need full YAML content and aren't cross-checked during the build. Now I need to double-check the units for collisions—m²/person and % of stock per year both risk overlapping with other tertiary-sector variables, so I need to make sure each unit is distinct.

Several other units also risk colliding with existing tertiary-sector definitions, and one, kWh/household/year, might actually be safe since tertiary tends to be measured per person rather than per household. I'm considering whether to make the unit strings inherently distinct, but both residential and tertiary would naturally want the same phrasing, so that's not a clean fix. I'm now checking how T.unit() resolves and whether a fallback is acceptable, and also whether the sibling variable would actually be the one to hit the build error since residential-heat sorts before tertiary-heat alphabetically.

Either way, the combined build must end up error-free, so I'm weighing options: declare the unit strings I need and flag the collision risk in my final report, recommending these shared unit keys move to ui.yaml or that the collision rule exempt identical unit. translations — though that risks leaving the build broken. The safer alternative is to skip declaring unit strings entirely and let the fallback display the raw unit text, which avoids collisions altogether.

I'm checking whether raw units like "m²/person", "kWh/m²/year", "°C", and "% of heat" read naturally in French and Dutch — floor area and energy units seem fine, but phrases like "% of heat" or "% of the stock per year" read as awkward English prose in FR/NL. So I'm leaning toward picking raw unit strings that are language-neutral symbols needing no translation, weighing options like "m²/pers." against the D3 principle that units should be the tangible thing shown to participants.

Then I reconsider the collision rule — a key defined elsewhere is an error unless prefixed with the topic id, but unit keys must exactly match `unit.<raw unit string>` to be found by the page, so making residential and tertiary use different raw unit phrasing feels ugly. I'm now exploring making the raw unit strings genuinely topic-specific in a way that also improves the UX, starting with the residential floor area unit.

Actually, I want to check how the unit chip is displayed before deciding on phrasing, so I'm going to look through i18n.js, impact.js, play.js, and spark.js.

## reasoning

I'm working out the contribution formula for each lever type: for linear-shift and proportional levers, energy responds linearly to the lever value, so contribution reduces to a clean difference from the reference times a slope factor, while inverse levers give a hyperbolic relationship instead. Now I'm applying this to specific levers, starting with floor-area, since heating and cooling energy both scale proportionally with total floor area.

Continuing through the other levers, I'm confirming that renovation-rate and thermostat both fit the linear-shift model with derivable slopes, while hot-water and cooling fit the proportional model with scaled equal to their 2050 TWh values. Now I'm examining district-heat, where the network share splits energy between two heat sources without changing the total end-use demand.

Since the DHN share leaves total residential demand exactly unchanged, this lever reads as "negligible" -- an honest and interesting reveal, similar in spirit to the truck-share surprise. The real benefit of district heating happens upstream on the supply side, which this demand model doesn't capture, so I'm weighing whether a zero-leverage lever still offers workshop value as an insight in itself.

For cooking, the demand simply scales proportionally with households and per-household consumption, which is straightforward. I'm now working out how "total" should be defined for residential heat in the same calibrated way as inland mobility -- deciding whether the topic's scope should be residential heating and cooling combined.

I need to check whether the total actually matters for the contribution formula, since for proportional and linear-shift kinds the total may cancel out of the contribution calculation entirely, which would simplify the scoping decision.

I check the floor-area lever and renovation-rate lever too, confirming both satisfy the invariant that f(vTarget) equals the total. Now I want to look at play.js to verify how the chart consumes `history`, how the placeholders like `{valuePerDay}`/`{valuePerYear}` get filled in, and how the leverage box is rendered.

## reasoning

A "negligible" lever hides the leverage box entirely, which makes sense for cases like district heating where the readout would be empty — but I want to check how the reveal explains this absence, so I'm looking at how justification coverage handles a negligible verdict.

I'm considering pairing that with a caution fact noting the lever doesn't reduce heat demand, only its source — a structural claim about the model rather than négaWatt's choice, so it should stay pre-answer safe. I need to stay careful with banned words like "scenario" or naming négaWatt directly, using neutral phrasing like "this model" instead.

For the lever count, I'm settling on including district-heating share as the 7th shown lever alongside floor-area, renovation-rate, thermostat, hot-water, cooling, and cooking, with cooking-gas kept as a spare. I'm also weighing whether the cooking lever's kWh/household/year unit is graspable enough, thinking about converting it into something tangible like oven-hours to make the 15% rise more intuitive.

Cooking stays in since it keeps the total coherent across thermal services and structural drivers. I'm now laying out the seven levers to show: floor area per person, renovation rate, heating setpoint, domestic hot water, home cooling, and continuing to define the rest with their units, historical baselines, and target values.

For renovation rate specifically, I need to think through whether to represent it as a percent of the housing stock renovated per year versus the resulting improvement in heating intensity per square meter, since the brief frames it in terms of a roughly doubled historical rate expressed as a change in space-heating kWh/m²/year, targeting about 6 percent.

Now I'm calculating the average annual renovation rate from the observed 2001-2023 data series to establish the historical baseline the lever should double from.

Finishing the sum across all 23 years, I get a mean of about 3.007 %/yr, which matches the prose reference of "around 3%." So using 3.0 as the base rate is reasonable, giving a target of acc × 3.0 = 6.0 %/yr, though I'm noting some inconsistency between the 2019 observed reference value and the expected value.

That said, I want to double-check whether the "renovation rate" in the history series has the same basis as the notebook's stated rate. The series is defined as newly renovated floor area over total floor area, while the notebook prose describes it as renovation plus new construction combined — but since both average to roughly 3%, they seem to align well enough.

I'm going to check the data_aux notebook's history export cell to confirm exactly what it computes, and also review `verify_workshop_export.py` since it was modified and may have a residential check list. I also want to pull in `ktoe_to_GWh`, `rho_h2o`, `cp_h2o`, `population_dict`, and `households_dict` from the macro notebook.

## reasoning

I notice the naming discrepancy with `dhw` vs `shw` in the data_aux notebook variable. I'm also puzzling over where the tertiary neighbour comparison series came from — whether the sibling notebook `nW_BE_demand_data_aux.ipynb` added them since it's shown as modified in git status, given that this shared export apparently serves both topics.

Since editing a shared file would violate §12's restrictions, either the sibling did it against the rules or the base author included them originally — but either way, I lack residential EU floor-area comparisons in my data, so I'll need to source residential m²/person figures for other countries from Odyssee or Eurostat with proper citations. Next I want to check verify_workshop_export.py.

## reasoning

I notice the verification script only checks transport files, so it'll pass regardless, but I should still respect its constraints since buildings likely mirror the same pattern: edge margin ≥0.12, reference value within slider range, and the 2019 history point matching the reference within a tight tolerance of max(0.05, 0.3%).

Going through each lever: renovation-rate needs refValue 3.0673 to match its 2019 history point; floor-area's reference (~54.7 from population data) should fall within its looser 0.164 tolerance; cooling's computed reference of 0.4363 is close enough to history's 0.4371; and I still need to verify hot-water's reference against its 2.02 tolerance window.

I'm also checking that verify_workshop_export.py doesn't actually validate buildings at all, so it'll pass trivially — but I still want my levers to follow the same invariants as a matter of good practice, since the script's checks (like matching "inlandTwh" keys or requiring positive scaled values for non-linear-shift kinds) simply won't apply to buildings but could be extended later.

If I use a "negligible" impact kind for the DHN lever, I want to be careful since that could break the positive-scaled check if extended, so I'll set scaled to the residential heat total covering space heating and hot water, which is what DHN actually splits.

I'll verify that district heating truly has zero impact on end-use demand by checking the calculation — heat_ihs and heat_dhn split from the same total (sht + shw), so shifting the DHN share only changes the carrier split, not the total demand. That's a striking, designed surprise worth flagging as a caution fact before participants answer.

Now I need to actually run the notebook to get real numbers. Executing the whole thing with nbconvert seems like the most reliable path, so I can inspect the resulting globals rather than trying to run cells piecemeal through IPython magics.

Running that right now would fail at the export cell since no buildings topic module exists yet, so I should instead build a minimal version of residential_heat.py that just prints diagnostics first, run the notebook to see the output, then refine from there. Actually, extracting the needed cells into a plain script and running under IPython might be cleaner — let me check the macro notebook for what's available.

## reasoning

I'm setting up a plain Python script to replicate the residential computation, working through population and household figures for 2019 and 2050 to calculate reference surface per capita and apply a -10% target adjustment.

Now I'm deriving the reference and target values for space heating, cooling, hot water, and cooking energy demand per unit area or population, computing specific energy intensities from the raw energy totals and applying growth/decline rates over 31 years. I'm calculating shower and other hot water energy needs using water heat capacity and temperature deltas, then combining these into district heating network targets, and finally computing the 2050 total energy demand in TWh for each end-use category by multiplying target intensities by projected 2050 area, population, or household counts.

## reasoning

I'm cross-checking the numbers against the notebook prose: per-person area drops 10% to 49.2 m², household area falls 14.9% to 107.5 m², reference heat demand comes out to 68.15 kWh/m² matching the stated 68.2, target efficiency-only drops to 39.76 and with SlowHeat to 34.19, both matching prose values, cooling demand rises from 0.436 to 1.521 kWh/m², and hot water demand falls 21.2% from 674.96 to 531.88 — everything lines up.

Continuing to verify: cooking energy goes from 226.55 to 260.54 kWh/household, though I notice cell 32 has a stray comment saying "kWh/person" when it should read "kWh/household" — a small notebook labeling error worth flagging. Gas cooking share drops from 26.77% to 2%, and district heating penetration jumps from a mere 0.25% to a target of 15%. I want to double check that 0.25% figure since it seems tiny compared to the tertiary sector's 2.4% share mentioned elsewhere in the notebook — but the residential prose does say district heating is currently "anecdotal," covering less than 1% of demand, so the 0.25% baseline is consistent.

For the DHN slider, I'm checking the edge position: with range 0–45 and target 15, that lands at about a third of the way across, comfortably away from dead-center. The max of 45% is well justified since the notebook prose cites studies showing district heating's techno-economic potential could reach 45% of residential heating demand by 2050 — that's a solid, sourced slider maximum.

Then for the thermostat setting, I'm checking a reference of 0 and target of 2°C lower, with slider bounds -1 to 6. The edge lands around 43% of the way across, which is far enough from center that no special note is needed, and the reference value of 0 sits properly within the slider range.

Landing on min=−1, max=6, step 0.5, decimals=1 — the −1 end represents heating slightly more than today, tying into the rebound-effect argument that better-insulated homes often get heated warmer. Now I'm figuring out the unit label for the readout, deciding between just "°C" or something like "°C on the thermostat".

I realize several unit keys I'm about to declare — like "m²/person", "% of the stock per year", and "°C less" — risk colliding with similar units used elsewhere, particularly in the tertiary sector levers which likely reuse the same setpoint concept. I need to think through how to handle these collisions pragmatically.

Since I can't coordinate declarations across the combined build, my safest bet is to skip declaring unit strings entirely and instead just use raw unit strings that read fine in all three languages, since the system falls back to printing the raw string directly.

I could make the unit string more descriptive per topic, like "m² of home per person" versus "m² of tertiary floor space per person" — distinct and more informative, though it risks making the unit chip too long wherever it gets rendered in the readout and scale-max labels.

I'm weighing the tradeoffs: declaring a shared key like `unit.m²/person` in both places causes a build collision, while declaring it nowhere just falls back to the raw string untranslated but without errors. The cleanest path seems to be declaring only residential-specific unit keys that won't collide, using raw unit strings that read fine untranslated in all three languages for the shared cases, and flagging in my report that `unit.*` keys are shared vocabulary belonging in `ui.yaml` with the exact FR/NL/EN translations needed so the maintainer can add them centrally.

That's honest, but re-reading the spec's intent — that a new topic shouldn't need to touch the shared file, and that a key collision should be a hard error rather than a silent override — I realize the collision is really just an unforeseen side effect of two sibling topics in the same sector needing the same units, and since I'm meant to report rather than edit shared files myself, I need to think through how the sibling topic would actually be affected if I claim that key.

Racing to grab the unit key first would just break the sibling's build instead of mine, which feels like the wrong way to resolve this. The better path is to pick distinct, residential-specific unit strings so there's no collision at all and both topics build green with proper translations — I'm weighing slightly longer but still concise unit labels across English, French, and Dutch, checking how they'd render in the layout by looking at the readout-unit and scale-max styling in play.html and workshop.css to make sure a short, distinctly-prefixed unit string still fits cleanly.

Now I'm reconsidering whether sharing the raw units without declaring them is actually fine, judging how bad an untranslated fallback would look per unit — "m²/person" reads acceptably close to French and Dutch, but something like "% of the stock per year" is clearly English and would need a shorter, more neutral form like "%/year" to avoid looking foreign in French and Dutch text.

Since the fallback isn't good enough across several units, I'm deciding to declare proper unit keys with residential-specific raw unit strings, starting with something like "m² of home" for floor area.

For floor area I'm setting the trilingual translations to "m² de logement/pers.", "m² woning/pers.", and "m² of home/person". For renovation rate, I initially consider "% of homes/year" but realize the underlying data is floor area not dwelling count, so I settle on "% of the housing stock/year" with matching French and Dutch versions, keeping it distinct from the tertiary sector's equivalent unit.

For thermostat, since °C needs no translation and declaring it risks a collision with the tertiary sector's identical unit, I decide to skip declaring it explicitly and let the fallback handle it correctly across all languages. For hot water, I'm weighing options since "kWh/person/year" also risks colliding with another unit, but appending clarifying text feels clunky, so I'm leaning toward keeping it simple since domestic hot water context should be clear enough.

I'm second-guessing this whole approach now, wondering if I'm overcomplicating something that has a simpler fix. Rather than relying on the shared `strings:` unit mechanism which keeps causing collision risks between sibling topics, I'm considering embedding the unit text directly into the tangible sentence and subtitle instead, sidestepping the shared-key problem entirely.

Then I start thinking about which units could actually stay language-neutral symbols rather than needing translation — `%` works cleanly for renovation rate and district heating since the surrounding sentence already conveys the "per year" or "of stock" context, and `kWh/pers.` seems understandable enough across French, Dutch, and English without extra wording.

Rather than keep guessing at what the rendered chip actually shows, I decide to check the real templates — play.html, workshop.css, spark.js — to see how the unit and source labels get printed before settling on a final decision.

## reasoning

That original scale-max label is too long for the 375px width, so I'm checking existing short unit examples like "km/pers./jour" and "% des km motorisés" to find a workable length target of around 18 characters. I'm settling on concise French unit labels for each metric, starting with floor area, that stay short while remaining distinct from other keys.

For renovation-rate, I'm weighing "% of homes/year" against the actual floor-area basis of the chart data — since the denominator needs to match the question, I'm shifting toward "% of the stock/year" as a safer, more accurate framing, and testing its French and Dutch translations for length.

I should check whether `unit` is also used in reveal.js and cards.js, not just wherever I've been assuming, so I'll grep for all display paths that go through T.unit(). For the thermostat unit I'll use a unique raw key like `°C less at home` with fr: `°C en moins`, nl: `°C minder`, en: `°C less`, avoiding collision with the tertiary variant.

For hot-water, embedding "(hot water)" into the raw key feels hacky, so I'm weighing alternatives like `kWh/person/year at home`, but honestly these workarounds all feel ugly. The cleanest path forward is to keep the raw unit strings short and matching what a tertiary topic would naturally use, skip declaring them in my YAML, and instead flag that these `unit.*` keys still need to be added.

I'm re-checking the collision logic — `ui_strings` accumulates keys from ui.yaml first, then across topic files in sorted order (inland-mobility, residential-heat, tertiary-heat), flagging duplicates as errors. It occurs to me that a sibling agent working on a related topic file will hit this exact same collision dilemma, and the task setup seems to anticipate that shared files need coordinated handling.

So `unit` is actually meant to double as display text, which makes turning it into a long internal key a mild abuse. A better compromise is choosing raw unit strings that are naturally residential-specific, short, and still honest as display text — like "m² of home per person," "% of the housing stock per year," or "°C less on the home thermostat" — each of which can serve as a reasonable fallback while translations handle the polished versions.

For the "kWh per person per year" case, I realize this isn't uniquely residential since tertiary hot water uses the same phrasing, so I'm considering ways to disambiguate — either by spelling it out fully to differ from a slashed version, or making it explicit like "kWh of hot water/person/year" so the key stays unique while displaying cleanly as "kWh/pers./an" and equivalents.

Working through the remaining units similarly: cooling in kWh/m²/year is already unique, household-based cooking energy needs a small tweak to stay distinct since tertiary has no per-household equivalent, and "% of home heat" is naturally unique. So the plan settles on keeping raw unit strings explicit rather than relying on subtle spelling differences.

But then I reconsider — maybe there's a cleaner approach. Instead of declaring a unit translation for something like kWh/person/year, I could just let it fall back to the untranslated string, since "kWh/person/year" reads fine as-is in French and Dutch. I'm checking whether that fallback actually holds up across each language.

But comparing that against explicit, uniquely-named keys with proper localized phrasing, the explicit approach clearly produces a better result, so I'm going with that. I want to make sure these keys are unmistakably tied to housing context by including words like "home" or "residential" in each one, to avoid any collision risk with sibling topics.

Finalizing phrases like hot water per person per year, cooling per square meter per year, cooking per household per year, and share of home heating — all clearly scoped to home/housing so no other module could accidentally produce the same string. Now I want to verify how these strings actually get used in the cards and reveal displays — whether they go through the same unit-translation function — so I should check those files directly.

## reasoning

Since all display paths route through the translation unit function, I don't need to worry about raw units leaking through as long as translations are declared. Now I'm working out the remaining lever details before writing the module, starting with floor area per person: 2019 reference of 54.6946 m² dropping to a 2050 target of 49.2251 m², tracked via the res_m2_per_person history series.

I'm checking the slider bounds (40-65 range, 0.5 step) land the reference and target values properly within range, and setting up a proportional impact model tied to the pro_RS_sur_spe variable in section 1.2.1, with lower being better. For the supporting facts, I'm digging into why floor area per person rose from 53.2 in 2000 to 54.7 in 2019 despite shrinking households — pulling household-size figures (2.310 people/household in 2019, projected 2.183 by 2050, a 5.5% drop) to explain that area per household actually needs to fall faster than area per person to hit the target.

I'm working out that the household-size projection is a Statbel demographic figure I can present as pre-answer context, while the resulting -14.9% per-household consequence belongs in the justification/reveal section. I still need to find verified European benchmark numbers for floor area per capita, and I'm drafting a tangible comparison (126.4 m² for an average 2.3-person household) before checking the debate/caution notes in section 6.

For the renovation rate metric, I'm trying to figure out the right reference value—the 2019 observed rate of 3.0673%—but I realize the buildings notebook doesn't actually contain that historical series, only a rough "around 3%" figure and a different renovation intensity metric. This is a gap I need to resolve since the verification script expects the reference value to match the historical 2019 figure closely.

Following the guidance that unavailable figures should still be sourced from the notebook's prose rather than invented, I'm looking at how inland_mobility.py handles a similar situation—declaring reference values quoted directly from prose text with citations, using a pattern like naming the variable with a clear comment about what it represents and where it's cited. So I'm considering defining a similar prose-sourced reference value of 3.0% for the renovation rate, tied to the "around 3%" language in the buildings notebook's section 2.1.1.

But then I notice the historical 2019 value computes to 3.0673, which creates a 0.067 gap against the 3.0 reference—about 2%, exceeding the transport verify script's 0.05 tolerance (though that check doesn't apply to buildings). I'm weighing whether to keep the reference at 3.0 as the prose's stated 2000-2023 average, accepting that the 2019 chart point will sit slightly off the tick mark, but reasoning this discrepancy would be invisible given the chart's 2.5-3.9 range and is defensible since the reference represents a multi-year average rather than the single 2019 value.

I'll add an assertion cross-checking that the improvement rate and multiplier constants stay consistent, and reconsider whether I can cite the observed 3.07% figure directly — since it's sourced from JRC-IDEES-2023 with a URL, I can treat it like other externally sourced facts in the YAML prose.

Checking the arithmetic: with the base rate at 3.0, an acc of 1 corresponds exactly to a rate of 3.0, giving zero contribution — consistent. I'm also verifying the slider math: target is acc × 3.0 = 6.0 %/yr, and given slider bounds of 1.5 to 10 with 0.1 steps, I calculate the edge distance ratio comes to about 0.4706, roughly 0.029 off from the center.

That's too close to dead-center, so I need to avoid it — I'm testing alternate slider bounds like min 2, max 12, step 0.25, which gives an edge ratio of exactly 0.4, a comfortable 0.1 away from center, with the reference value 3.0 still safely within range and one decimal place of precision. Now I'm weighing whether a max of 12%/yr renovation rate is realistic — that would mean cycling through the entire building stock in about 8 years, which is aggressive but plausible as the ambitious end of the scale, so I'm cross-checking it against Belgium's actual long-term renovation strategy targets.

I need to be precise here since the JRC's 3% "renewal rate" figure blends light renovation and new construction, which differs from the EU's stricter ~1% "deep renovation rate" definition — mixing these up would be a factual error. I'm now testing other bound combinations to dodge the dead-center trap: min 2/max 10 lands exactly at 0.5 (bad), min 1.5/max 10 gives 0.4706 (still too centered), while min 2.5/max 11 yields 0.4118, which clears the threshold and keeps the 3.0 reference value valid — though I'm noticing this range doesn't leave much room below today's actual rate.

Now I'm working out the scale labels ("1,0" to "9,0 % du parc/an") and computing the impact of a linear shift toward a 6.0% target rate, factoring in current renovation rate, sufficiency share, and 2050 building area — landing on roughly -2.52 TWh.

I'm checking the total contribution formula at the reference rate of 3.0, confirming it nets to zero as expected, then evaluating it at the target of 3.0 to get about 37.93 TWh. I'm also sanity-checking the resulting building energy intensity at that rate, working through the reference value minus the accumulated reduction times the sufficiency factor, landing around 46.4 kWh/m².

Now I'm scaling that intensity across the total floor area to get roughly 28.78 TWh, then adding contributions from other sectors (services, shower, cooking) to reach a total of 37.93 TWh, which matches my earlier calculation. I'm noting the history variable name, marking the direction as "better: up", and identifying the model variable and section reference for the renovation rate slider.

I'm now moving to the third lever, the home thermostat setpoint reduction — setting its unit, reference and target values, and verifying the slider's edge symmetry calculation checks out with the reference value falling within range and one decimal place of precision, before starting to define its linear-shift impact model.

For this thermostat variable, I compute the slope as roughly -1.73 TWh per °C of setpoint change and verify that both endpoints (2°C and 0°C reduction) give consistent totals matching my direct calculation of about 33.81 TWh. Since there's no historical Belgian data on indoor thermostat setpoints (only energy consumption data exists), I note that history is absent, mark the direction as "better: up," and record the model variable name.

I also realize `d_temp` gets overwritten elsewhere, so I need to derive it from `suf_RS_tes_sht` instead, and I confirm the "up" direction is correct since target 2 exceeds reference 0, meaning higher thermostat offset is more ambitious.

Next I move to the "hot-water" lever, checking its units (kWh per person per year), reference and target values, and the corresponding historical data series.

Working through the slider parameters, I compute the edge distance and confirm the reference value falls within range with zero decimal places since the step is 10.

For impact I verify the proportional scaling to `shw2050` and total, note the direction is "down", and record the model variable `trg_RS_tes_shw`. I also flag a caution fact since the historical series shows a plausible rising trend from 2000-2019 but then an odd drop by 2023 that looks more like a data break than real behavior.

For the tangible illustration, I want to avoid directly echoing négaWatt's own shower scenario since that would be too much of a spoiler, so I'm reworking it into a neutral example — heating a fixed quantity of water by a set temperature difference — to convey the physics without giving away the scenario's specific assumptions.

Now I'm moving to cooling: checking the reference and target values against slider bounds, confirming the 2019 history point lines up, and working out where the target sits along the slider's range.

Actually with a 0.1 step the displayed values feel too coarse — 0.44 rounds to just one decimal. I'm switching to a 0.05 step, which gives two decimal places and 120 positions on the slider, so "1.52" and "0.44" display more precisely.

Setting the slider's max to 6 kWh/m²/yr feels sensible given a fully air-conditioned Belgian home might use roughly 5-10 kWh/m², so 6 represents a "much faster deployment" scenario relative to today's baseline. Checking the proportional impact calculation with the target value of 1.5213, scaled to 0.944 giving a total of 30.358.

Looking at the growth trend more closely: cooling energy went from 0.045 in 2000 to 0.44 in 2019 to 1.01 in 2023, so it grew tenfold by 2019 then more than doubled again in just four years. I'm also noting that cooling currently represents only 0.5% of a home's thermal energy use (0.273/51.72), while Belgian tertiary buildings already use 39 times more cooling energy per square meter than residential buildings — a useful structural comparison since both are in the same units.

I'm weighing whether to reference a tertiary-sector variable for the 17.2 kWh/m² figure, but since the brief says to avoid building on TS_ variables, I'll instead cite that number directly from the JRC-IDEES source as external sourced prose rather than pulling it in as a module dependency — that keeps things clean. I still need benchmarks on EU air-conditioner ownership (around 19%, lower in Belgium) and should flag that climate change is expected to raise cooling degree days significantly per the CORDEX projections.

For the cooking factor, there's no exported history series, but I recall the notebook prose mentions a slight historical decline in cooking energy use per household between 2000 and 2023, so I can declare that as a sourced prose constant instead of pulling from a data series.

Now checking the slider bounds for the cooking value: with min 120 max 400, the target lands almost dead center, which fails the off-center requirement. I try min 100/max 400, then min 150/max 400, and that last range finally pushes the distance from center just past the 0.05 threshold.

I settle on min 150, max 420, step 10, giving span 270 and an edge distance of about 0.41, comfortably satisfying the off-center rule, with the reference value still in range and no decimal places needed. A max of 420 kWh/household/year represents an 85% increase and 150 a 34% decrease, both of which feel plausible for this variable.

For the "better" direction, I'm treating up as better since more home cooking implies less reliance on processed food — a deliberate, slightly surprising framing that should work well with the reveal's bolder/shyer labeling. The scaled impact comes out to roughly 1.504 for the cooking variable, bringing the total to about 30.358, and this maps to variable pro_RS_tes_cok in section 2.1.2, with a declining historical trend as supporting context.

I'm now working out the supporting facts: cooking makes up about 2.17% of home thermal energy, I still need a benchmark source for processed/eaten-out food shares in Europe, and for the tangible comparison I'm trying to frame 226 kWh/household/year (~0.62 kWh/day) against something concrete like oven runtime, while being careful to label it as simple arithmetic rather than a modeled assumption.

Reconsidering whether cooking belongs under the residential-heat topic — it's part of the thermal-services block and, while small (1.5 of 30.4 TWh), it's a genuine sufficiency lever with a counter-intuitive sign that keeps the total coherent, so I'll keep it as one of the seven shown levers. Moving to the district-heat lever now — checking its reference value (0.2501), target (15.0), and slider range (0 to 45, step 1), noting the span hits an edge case.

I'm working out the right step size and decimal precision so the reference value doesn't display as a misleading "0 %" — testing step 0.5 for one decimal place, which gives cleaner labels like "0,3 %" for the reference and "15,0 %" for the target while keeping the edge case at 0.3333 valid.

I'm also confirming there's no exported district-heating time series in the project data, so history should be marked absent, and gathering the supporting facts: residential district heating was just 0.25% of heat demand in 2019 versus benchmarks like Denmark's ~65% and Sweden's ~50%, while noting this shift only changes the heat source, not the underlying demand.

For the target figure, I'm weighing Belgium's dispersed settlement pattern against sourced studies — Lund et al. 2025 puts technical potential at up to 45%, and PATHS2050/EnergyVille assumes 13% for buildings, which is close enough to the 15% target that I need to be careful how I phrase the comparison to avoid tipping off the answer.

Checking the spare cooking-gas lever now: the gas share of home cooking energy, roughly 27% → 2%. With a slider range of 0 to 35, the edge distance is too small and triggers an error, but I can't set the minimum negative for a share value. Adjusting the max down helps the edge ratio but then the current value falls outside the range and produces a validation note — since this lever isn't actually shown or played, I think I can just skip it.

I confirm there's no reasonable slider range that fits both the current 26.77 reference and a 2% target without going negative, which is nonsensical, so I'll skip exporting this slider and just note it in the report. Now I'm moving to double-check the impact total against the verification script's buildings-agnostic behavior, then starting to draft the model block JSON for the residential heating topic.

I realize I should just pull `df_tes_RS_tot` and `df_ees_RS_tot` directly from globals rather than recomputing them, since those DataFrames already contain the relevant columns indexed by year.

I'm now double-checking whether reindexing `divider_series` actually mutates `df_SUF`'s underlying index or just the Series copy—this matters for avoiding accidental side effects downstream.

I'll test with `.loc[y]` using integer year keys. Now I need to gather the actual numbers for each lever's facts block — for floor-area I'm noting 2019 values like m² per person (54.69), per household (126.35), household size (2.31), along with target values and percent changes, being careful to flag which figures are spoilers (targets) versus observed reconstructions (2019 baseline data).

Space heating and cooling together account for 42.9 TWh in 2019, about 83% of the home's thermal energy, since they're the services that scale with floor area. For renovation-rate I'm collecting the historical rate (3.0%), reference intensity (68.2), historical improvement (-0.458), and marking the target improvement, target efficiency intensity, and acceleration factor as spoilers since they represent future goals rather than observed data.

For thermostat effects I'm noting the 7% consumption change per degree Celsius, marking the 2°C target and 14% demand cut as spoilers, and pulling reference temperatures from source [3]: 15°C as an acceptable slow-heat threshold, 12°C as the lower threshold, and 19°C as the 2022 lower bound — then deriving roughly 1.726 TWh per degree Celsius from the target intensity.

Applying that 7% figure to 2019's baseline of 42.613 TWh gives about 2.98 TWh per degree Celsius on today's housing stock — I'm thinking through how to phrase this tangibly, maybe as roughly 261 kWh per person per year (dividing by Belgium's 11.43M population), avoiding unsourced comparisons like transit electricity use.

For hot water I'm cataloging the figures: 63.7 litres/day in 2019 versus a 50.2 litres/day target (a 21.2% reduction), flagging the shower/other energy breakdowns and target-defining values as spoilers.

Now I'm working through the physics of heating 100 litres from 15°C to 40°C, which comes to about 2.90 kWh, alongside domestic hot water's 2019 figures — 7.716 TWh total, representing 14.9% of thermal share — and a 2000 baseline of 530.6 needing external sourcing. For cooling, I'm noting the 2019 reference value of 0.4363 against a 1.5213 target, and puzzling over a historical rate parameter of 0.035 that seems to represent the scenario simply carrying forward the historical trend.

I'm deciding not to expose the 0.035 rate directly since combined with the target it would let someone reverse-engineer the answer, and there's also a review flag suggesting the actual observed average might be closer to 0.042, making that figure suspect anyway — so I'll frame the trend fact around the observed series level instead.

Now I'm working through the cooking sector figures: setting the 2019 reference value, the 2050 target, marking the percent change as a spoiler, and noting the historical trend plus thermal energy shares for that category.

For district heating, I'm figuring out the 2019 reference and 2050 target, and calculating heat demand by summing space heating and hot water figures for both years — flagging that the target figure may itself be a spoiler since it encodes other scenario assumptions.

Now I'm gathering the external comparison points I'll need with proper citations: the Lund et al. potential percentage, the EnergyVille 2050 pathway figure, and a list of "elsewhere in Europe" benchmarks to source — floor area per person, hot water demand per capita, cooling/AC penetration, district heating shares in Denmark/Sweden/Belgium, EU renovation rates, and cooking/food-away-from-home shares.

I'll do the actual web research afterward, but first I want to write and test the residential_heat.py module so the pipeline works end-to-end, running the buildings notebook headless to verify the numbers before layering in the benchmark facts, YAML, and final build/browser test.

I'm also noting more variables to track, plus a warning to avoid the overwritten `d_temp` and instead derive values from `suf_RS_tes_sht`. Now I'm ready to start writing the module itself, keeping in mind the `make_lever` function's positional arguments.

## reasoning

I need to run the notebook carefully since the export cell will trigger my module and write output files that need to be restored afterward. I'll run it headless with my own output path, likely in the background since it may take a while.

Since nbconvert runs with the notebook's directory as the working directory by default, and the notebook expects the repo root as CWD, that should work fine — let me go ahead and run it.

## reasoning

Jinja2 isn't available in the default kernel, but the sibling files were regenerated successfully today, so maybe a different environment or kernel was used. I should check environment.yml and available kernels rather than just installing jinja2 myself.

I'm considering options besides modifying the user's environment: setting `post_process = False`, though the style calls are all gated inside that flag; or checking whether another Python environment has jinja2 importable, which seems like the cleanest path forward.

## reasoning

Jinja2 should be available in base Python 3.11, so I'm confused why the notebook failed—maybe the kernelspec points to a different environment. I should check which kernel the notebook actually specifies and verify whether jinja2 exists in the data_processing env, since the traceback points to base miniconda's pandas.

Actually, `import_optional_dependency` returns None both on ImportError and on version mismatches, so pandas might require a newer jinja2 than what's installed. I should check the exact pandas version requirement and test the import directly to confirm.

## reasoning

The notebook ran through to the export cell but my assertion failed there. I'm suspecting it's related to how the index type or year values are being handled — possibly the divider_series index was cast to float earlier, or the target years array has a type mismatch with the integer key I'm using to look up values, causing the lookup to fail or behave unexpectedly.

Rather than reasoning through it further, I think the fastest path is to just debug directly — inject a diagnostic cell into the notebook execution to print the actual values and types at the point of failure, or run it with errors allowed so I can inspect the output notebook afterward.
