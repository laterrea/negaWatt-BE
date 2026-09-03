# Recovered work — tertiary-heat (lead)

Agent `ad1276c0668c07349`, stopped by the session limit before it could report.
Everything below is recovered verbatim from its transcript, in order. **Nothing here has been checked** — treat every figure as unverified until re-sourced.

13 entries, of which 0 web results.

---

## reasoning

I'm looking at the renovation rate history: 2010 was 2.42%, 2019 was 1.42%, and 2023 was 1.32%, which roughly averages to the notebook's cited 1.7% figure over 2000-2023.

For the tertiary sector model, I'm identifying the real degrees of freedom. Floor area per person is one lever, going from 19.86 to 17.88 m²/person, tracked by `ter_m2_per_person`. Space heating has two more: the renovation/efficiency acceleration factor bringing demand down to 71.54 kWh/m² by 2050, and a 1°C setpoint reduction with a 0.07 sensitivity coefficient, yielding a final 66.53 kWh/m². I'm now trying to figure out how the tangible units for these levers should be expressed, and how the residential-heating sibling model handled the analogous case.

I'm weighing whether to expose these as separate, real degrees of freedom rather than a composite lever, since the workshop needs exact reveals that keep scenarios internally consistent -- combining renovation efficiency and setpoint reduction into a single "heat per m²" figure risks conflating two distinct decisions.

That framing captures the model exactly, since space heating TWh equals 2050 intensity times 2050 surface area, making that slider a valid standalone degree of freedom. But I worry this double-counts with a separate thermostat lever unless I split things differently — maybe keep the intensity lever tied purely to renovation rate rather than temperature setpoints, so the two levers stay independent and the workshop structure remains clean.

Actually the model's acceleration factor applies to the demand reduction rate itself, not the renovation rate directly, and the notebook's own math shows a 5× acceleration implies an 8.5%/yr renovation rate — physically absurd since that would renovate the entire stock over two and a half times in 31 years.

Mixing the 1.42% observed-series basis with the 1.7% average basis compounds the confusion, and I'm wary of presenting such a wild number to participants given the basis-mixing warning.

A cleaner path uses the existing history-tracked levers: floor area per person (19.86 → 17.88 m²/person, tied to `pro_TS_sur_spe`), and total heating demand per m² (95.41 → 66.53 kWh/m²/yr, tied to the renovation-rate/temperature-difference pair), which sits on the same basis as the curve itself. A third lever around thermostat setpoint would risk double-counting with the heating-demand lever, so I need to define it differently.

Looking at the tertiary sector's remaining degrees of freedom, I still have cooling deployment fraction, hot water, catering, and electricity to account for beyond floor area, heating efficiency, and heating sufficiency — so the third lever should probably draw from cooling or one of these other categories instead.

Converting 77.819 ktoe using the 11.63 GWh/ktoe factor gives about 905 GWh, or roughly 79.2 kWh/person once divided by population. Applying the same 0.476*31 adjustment brings the street lighting target down to about 64.4 kWh/person, and I'm moving on to building lighting at 150 kWh/person.

For refrigeration, converting 297.768 ktoe gives 3463 GWh, or about 303 kWh/person as the reference value against a 150 kWh/person target. Commercial refrigeration follows similarly, converting 179.674 ktoe to roughly 182.9 kWh/person with a target of 155.5, while miscellaneous just uses the reference value directly.

For ICT, converting 248.983 ktoe yields about 253.4 kWh/person against a target of 278.7. I'm now weighing how to present the levers—the brief wants 6-8 items in intuitive units like m² per person, kWh/m², and °C, which means thermostat setpoint needs to be its own lever separate from heating efficiency, since they multiply together to produce the 2050 intensity. But I'm puzzled about how "space-heating intensity trajectory" fits as a distinct item alongside renovation rate, since it seems like it should overlap with what efficiency and setpoint already capture.

Re-checking the brief's indicative list of levers—floor area per person (falling ~10%), renovation rate paired with space-heating intensity trajectory, thermal services split into heating/cooling/hot water/catering, and specific electrical uses like ventilation and lighting—I'm settling on treating the intensity trajectory as the combined 2050 kWh/m² figure covering both efficiency gains and setpoint, with the thermostat as its own separate °C lever. To avoid double-counting against the observed historical curve, I'll define the heating-per-m² lever using the `ter_heat_per_m2` history series as the consistent baseline.

Adding a separate thermostat lever risks double-counting against that combined intensity figure, so I'm weighing whether to instead define the intensity lever on an efficiency-only basis—but that would misalign with the observed historical curve's endpoint, breaking consistency with D19. The cleaner path seems to be defining two levers that each stay on a clean, non-overlapping basis: one for total 2050 space-heating intensity and another using its own history series.

The efficiency lever as a level would give 71.5 kWh/m² by 2050 from renovation alone, and the sufficiency lever of 1°C off the thermostat gives -7%. But there's a mismatch: the observed curve for heat-per-m² ends at 95.4 in 2019 and 78.6 in 2023, yet an "efficiency only" slider endpoint wouldn't align with that trajectory — it'd represent a counterfactual curve instead, which is the basis confusion D19/D30 explicitly forbid. Maybe defining the lever as a rate of improvement rather than an absolute level would resolve this.

I decide to use the combined heating intensity as the lever itself, going from 95.41 to 66.53 kWh/m²/yr, since it's exactly the endpoint of the drawn curve, it's the quantity the model actually multiplies by floor area making the reveal arithmetic exact, and the two sub-knobs of renovation and thermostat setting become the explanation given at reveal time.

The renovation rate ties directly to the notebook's own numbers: the 2000–2023 average was around 1.7%/yr with a 2019 value of 1.42%/yr, and the notebook proposes multiplying this fivefold to reach the target reduction. So `acc_TS_tes_sht_ren = 5` literally reflects the notebook's stated proportionality between renovation rate and improvement rate, giving me a tangible %/yr lever with a clear reference point.

Using the 2019 baseline of 1.42%/yr, a fivefold increase yields roughly 7.1%/yr — a striking figure that I want to sanity-check, but the mapping itself (accel = rate/ref_rate) stays exact and self-consistent with the curve's own endpoint.

That's still defensible as a debate point since the notebook has a field for that exactly. Let me check the review document for what it actually says about the renovation rate before deciding — I could alternatively express the efficiency lever directly through 2050 kWh/m² without attaching the historical curve, but that loses the best chart, so I'm leaning toward keeping the floor-area lever with its history and international comparison as lever one in a seven-lever plan.

Next comes heating intensity combining renovation and thermostat effects, cooling intensity which needs to rise less as the key story point, and hot water per person where the target equals the reference value—I need to confirm that's handled properly since `make_lever` normally requires the target strictly inside the slider range.

For the "better" direction on flat hot-water demand, defaulting to "up" is semantically wrong since holding steady or lowering is the ambitious path, so I should explicitly pass "down" — this frames a nice question about how far sufficiency can push hot-water demand down, with the honest reveal being that négaWatt essentially holds it flat. Then for catering energy, I'm working out the reference value in kWh per person from the ktoe figure, landing around 289 kWh/person with a target at 1.2× that, roughly 347 kWh/person — though no history series is available for this one.

I'm cataloguing lighting (303 → 150, a major efficiency cut) and ICT (253 → 279, a slight increase) as per-person electrical uses, both lacking history data. That brings the count to seven degrees of freedom, but four lack history and three are fairly abstract per-person electrical categories — I'm wondering whether the brief's grouping of "specific electrical uses" as one item means I should combine them, though that doesn't really work since each is a distinct model input that can't be merged.

Since summing the six electrical sub-uses (ventilation, lighting, IT, refrigeration, etc.) into one lever wouldn't be a genuine degree of freedom unless I fix their internal split, I'm dropping that idea and instead sketching an alternative eight-lever plan: keeping floor area per person as a lever with history, splitting heating into a renovation rate lever (with history) and a separate thermostat setpoint lever...

For the thermostat lever, checking the edge margins on a 0–3.5°C slider with target at 1°C gives roughly 29%, comfortably above the 12% threshold, but I'm second-guessing whether a delta-only setpoint (with 2019 as the zero reference) is intuitive without an absolute baseline to anchor it.

I shouldn't invent a 2019 tertiary setpoint, so the absolute °C lever isn't available and a delta-based one would violate D14. I'll go with Plan A: use the combined 2050 intensity in kWh/m² as the heating lever, keeping the −1°C setpoint as justification and a reveal-true structural fact, and I'll flag this °C deviation explicitly in my report since the brief wanted a °C unit.

Considering the renovation rate itself as the lever instead — its effect on heating TWh isn't proportional but more of an affine relationship, which would need a linear-shift impact type rather than simple proportionality. I need to check how impact.js handles that kind before deciding, though I'm wary since the brief mentions both renovation rate and heating intensity trajectory as potential levers, and they overlap since one drives the other.

I'm settling on a cleaner structure: use the combined 2050 heat-per-m2 intensity as the main heating lever, and treat the renovation rate as background context rather than a separate lever — a reveal-only fact showing how today's 1.4%/yr compares to what the scenario implies. That frees up a slot, so now I'm thinking through which electrical end-uses would work well as the remaining levers.

Building lighting stands out as a strong candidate — cutting from 303 to 150 kWh/person is a tangible, relatable halving that people can picture in shops, offices, and schools. Street lighting is even better, since Belgium's motorway lighting is a well-known "visible from space" quirk, and there's a real, checkable history of Flanders and Wallonia switching or dimming it that makes this a very concrete, structurally-grounded lever.

I'm also weighing commercial refrigeration (182.9 → 155.5, a tangible fridge-doors-vs-open-cabinets story) against ICT, which actually needs to rise (253.4 → 278.7), making it an interesting counter-example lever. Ventilation and miscellaneous stay flat, so they're not useful as levers, and things like district heating share or gas share in catering are carrier/efficiency shifts rather than true demand changes, so they don't really fit the workshop's demand-based impact model.

For the final set, I'm settling on floor area per person, heating and cooling intensity per m², hot water, catering, lighting, street lighting, and refrigeration as the core seven or eight levers, prioritizing the ones with clear historical trends and international comparisons.

Street lighting stays as the most tangibly Belgian item. Catering also feels solid since it's sourceable and relatable — canteens, restaurants, school kitchens. I'm weighing ICT against refrigeration: ICT's data-centre claim risks being unsourceable, whereas refrigeration (supermarket fridge-door regulations) is concrete and verifiable, so I lean toward keeping refrigeration despite the brief mentioning IT.

Now I'm counting out eight levers to show — floor-area, heat, cooling, hot-water, catering, building lighting, street lighting, refrigeration — with export ventilation, misc, ICT, dhn-share, and catering-gas-share held as unshown spares, giving eight shown plus five spare. That feels like a complete export, but I'm second-guessing whether eight is too many for a workshop balance of 1 m²/person, 2 kWh/m² (heating, cooling), and 5 kWh/person items. I'm considering trimming to seven by cutting refrigeration, though I don't want to lose street lighting either.

Given the "6 to 8 shown levers" guidance, I'm settling on seven: floor area, space heating, cooling, hot water, catering, building lighting, and street lighting, with refrigeration, ICT, ventilation, and misc as unshown spares. I'm weighing refrigeration against catering as the more tangible story, and mapping out how each of the seven contributes: floor area as the opener showing international spread, space heating as the big renovation-driven lever, cooling as the counter-trend story, and hot water as a lever that helps contain the overall trend.

I want to check the actual TWh weight of each service in 2050 to confirm the story matches the numbers, so I'm planning to compute these values from the notebook's totals to see which services truly matter most.

Then I need to look into the impact model to understand exactly how proportional, inverse, linear-shift, and negligible effects work, along with the total, scaled, slope, and vTarget parameters. I also want to check the build script's validation rules and ui.yaml for unit strings like m²/person or kWh/m²/year, since topics may need their own interface translations.

## reasoning

I notice the caveat text hard-codes "the other seven assumptions," which is a shared string — this only reads correctly if there are exactly 8 shown levers total, since it implies "seven others." This actually argues for using 8 shown levers rather than editing that shared string myself; I should flag this instead of modifying ui.yaml directly.

For units, I need to declare `unit.m²/person`, `unit.kWh/m²/year`, and `unit.kWh/person/year` in my own topic YAML per D35, so I should check how the build script handles topic-level strings by reading through build_workshop_content.py.

## reasoning

I'm noting other key rules: topics support unit strings, spoiler checks scan for scenario-referencing phrases, target values can't stand alone in pre-answer fields, facts need kind/source with optional url, historyAbsent pairs with historyNote, at least 3 pre-answer facts are needed, and lever order follows the YAML. Now I'm moving on to check verify_work.

py and shop_export.py for their validation logic, and I want to grep play.js for how it handles leverage.caveat and unit formatting with the tangible `{value}` placeholder.

## reasoning

The verify_workshop_export.py script only checks transport-specific files, so it should pass regardless of my topic as long as transport data isn't broken—since I'm not touching that notebook, it should be fine unless a sibling's changes to another notebook overwrote a shared overrides file.

I also want to check where the "seven"/"sept" caveat string from ui.yaml actually gets displayed—grepping play.js didn't show usage, so I need to check play.html and search across the website directory more broadly, along with how spark.js handles history rendering and null values.

For getting actual numbers, rather than running the full notebook through nbconvert right away, I'm thinking I should first write a quick standalone script mimicking the notebook's population_dict logic to compute values, since I'll need to verify the notebook executes properly anyway but want to avoid colliding with a sibling process.

The cleanest approach is probably to execute the macro notebook plus the buildings notebook cells in-process using IPython's InteractiveShell rather than modifying the notebook or relying on `%run` magic, which needs a real IPython environment—I'll write a script that loads the notebook JSON and runs each code cell's source programmatically to extract the values I need before designing the sliders.

## reasoning

I notice the shared caveat string says "the other seven assumptions" — which only makes sense if there are 8 levers total, so that's actually a useful pragmatic signal pointing me toward showing 8 levers, even though I shouldn't design around a hardcoded string. I'll flag this as something the shared file should ideally make parameterized rather than fixed.

Now let me actually run the numbers by executing the notebook in-process with IPython.

## reasoning

Now I'm compiling the tertiary sector demand figures for 2019 versus 2050, breaking out thermal uses like space heating, cooling, hot water, and catering, then moving into electrical end-uses like ventilation and street lighting.

I finish totaling electrical demand (11.353 TWh in 2050) and grand total tertiary demand, noting the striking fact that despite all efficiency levers, total demand only drops about 10% because population growth and rising cooling, catering, and ICT needs offset the gains. I then start cross-checking each lever's assumed value against historical data points—floor area per person, heating and cooling intensity, hot water, catering, and lighting—confirming several match the 2019 baseline exactly.

I continue working out the remaining lever values for street lighting, refrigeration, ventilation, miscellaneous, ICT, district heating share, and catering gas share, checking edge cases where the reference equals the target (like hot water, where ref and target are both 305.045) to make sure the slider constraints and "better direction" logic still hold up properly.

For other levers total should be 40.168 TWh so all sit on a common scale. I'm working through each lever's impact definition: floor-area affects both space heating and cooling proportionally to surface area, and heat-per-m2 scales directly with the space heating demand of 14.989 TWh.

The remaining levers—cooling, hot-water, catering, building lighting, street lighting, and refrigeration—are all straightforward proportional relationships against their respective demand shares, each with clear scaled values and target percentages. Now I want to think about leverage, meaning how much each slider's full range would swing the total outcome.

Turning to the verifier script, I confirm it only checks against inlandTwh for topics that have that key, which buildings don't, so it skips validation there and only reads the transport lever file—no issue for my current work.

For the sliders themselves, I need each target to sit at least 12% from either end of its range, keep the reference value inside the range, and avoid landing the target too close to dead center. Starting with floor area per person: reference is 19.86, target 17.88, and I'm considering a range of 12–26 m²/person using international comparisons (France ~15.3, EU ~22.5, Germany ~42) to calculate where the target falls within that span.

I'll settle on 13–27 with a 0.25 step, since including Germany's 42 would stretch the range too far — this window comfortably covers scenarios from telework-driven space reduction to expansion for schools and hospitals.

For heat use per m², I'm checking 40–110 as the range: it places the target at roughly 38% of the span, keeps enough distance from center, and still includes the reference value of 95.41.

I tested a tighter alternative, 35–105, but that puts the target almost exactly at the midpoint, so I'm sticking with 40–110 for the heat metric before moving to the cooling metric, which needs to increase from 17.22 to 24.90.  Trying ranges 8–44 and 5–42 both land the target too close to center, but 10–45 gives a good offset of about 42.6%.  10 as a lower bound makes sense too — it implies cooling demand actually drops below today's level through passive design measures. Then for hot water demand, the reference and target are both 305.0, and extrapolating the observed 2023 trend of 323.9 forward to 2050 also lands near 305.

9, so I'm checking a few range options: 150–420 gives a span of 270 and positions the target at about 57.4% through the range, which balances well between not being too centered and keeping the edges reasonable.

For catering, the reference is 289.1 and target is 347.0. Extending the trend line to 2050 gives roughly 422, so I'm testing a range of 180–450 which captures both the reference and trend point while keeping the target position around 62%.

For building lighting, reference is 302.9 and target is 150, and I'm starting to test a range of 80–330 to position the target around 28% through the span.

I'll widen it slightly to 80–340 for headroom since lighting only trends downward with LED adoption, landing target at about 27%.

For street lighting, reference is 79.17 and target is 64.41, a modest 19% decrease, and I'm thinking about what a reasonable range floor would be since zero isn't a credible endpoint.

Using range 10–100 gives target at roughly 60.5%, edge distances checking out fine. Notably, the observed historical trend of −0.476/year, if simply continued, lands almost exactly at 64.4 — so this lever requires no extra effort beyond business-as-usual, which is worth flagging honestly. I'm weighing whether street lighting deserves its own workshop screen given it's only about 2% of total energy, but it's such a recognizable everyday Belgian energy touchpoint that it might still be worth including.

Now moving to refrigeration: with ref 182.8 and target 155.4, I test range 90–210 first but that puts the target too close to center; switching to range 80–210 gives roughly 58%, which fits better.

Settling on 8 categories to show: floor-area, heat, cooling, hot-water, catering, building-lighting, street-lighting, refrigeration—all tangible degrees of freedom—while keeping ventilation, miscellaneous, ICT, dhn-share, and catering-gas-share as unshown spares. I briefly reconsider whether ICT (topical due to data centres) should replace refrigeration, but the JRC's ICT category actually refers to office IT equipment, which risks being misleading.

I decide refrigeration stays since the "supermarket chiller doors" story is a solid, real EU/Belgian policy fact, and I'll keep ICT as a spare instead. That means 8 levers times 4 facts each is 32 facts to source carefully, and at 4 minutes per question that's 32 minutes total, matching the transport topic's format—so that's acceptable. I also confirm that in the model, floor-area only drives heating and cooling, while hot water, catering, and lighting scale differently.

Now I'm computing the swing for each lever: floor area contributes the largest share of the total at roughly 40%, heating intensity per square meter comes in close behind at about 39%, cooling adds around 20%, and hot water contributes a smaller share around 8%.

I'm continuing through the remaining levers, finding catering and building lighting each around 8-8.5%, while street lighting turns out to be the smallest contributor at under 3%, and refrigeration lands around 4%. With heating and floor area clearly dominating and cooling as a strong third, I'm now moving on to structuring the model configuration block, following the same pattern used for inland mobility.

Building the model config...

I also want to add consistency checks that cross-reference the model's numbers against the prose figures stated earlier in the notebook, starting with the total tertiary surface area figure from 2019.

For cell 67, I need to check the 66.5 kWh/m² figure against trg*suf, then look at the ninefold increase from 1.8 to 17.2 kWh/m² between 2000 and 2019, asserting ref_TS_tes_scl ≈ 17.2 and noting the 2000 baseline of 1.8 as a separate constant since the module can't verify the historical ratio directly.

I'll set ref_TS_tes_scl_2000 to 1.78, sourced from the exported history series I verified, and confirm 17.2228/1.78 gives roughly 9.68, matching the prose's stated 9.7 factor. Then for the annual growth rate of +0.743 kWh/m²/year reaching one-third and 24.9 kWh/m² by 2050, I'll assert trg_TS_tes_scl comes out to approximately 24.9.

There's also a copy-paste error in cell 67 referring to "residential" air conditioning when it should say tertiary, which is worth flagging in the report. Cell 69's comment cites 2019 and 2023 values of 17.3 and 16.4 kWh/m² that differ slightly from the history data's 17.22 and 16.14 — likely just a different data vintage, minor enough to note but not a real bug. Cell 70's claim of holding sanitary hot water demand at 305.0 kWh/person checks out against ref_TS_tes_shw, and I'm now looking at cell 72's stated increase of +4.297 kWh/person/year.

For cell 72, this trend applies to the tertiary sector with pro_TS_tes_cat matching a further 20% increase by 2050, and I can use +4.297 as a prose constant. Cell 74 has an untranslated French sentence about district heating covering 2.4% of tertiary demand in 2019, which matches ref_TS_tes_dhn at 2.42% — worth flagging as an inconsistency in language. Now checking cell 80's stated average decline of -0.476 kWh/person/year from 2000 to 2023 against its target value.

Cells 82, 84, and 88 check out fine against their stated projections. I'm noticing cell 60 has copy-paste artifacts from the residential section — it references household surface area and showering as the main driver for hot water, but neither applies correctly to the tertiary sector, so these are minor documentation slips worth reporting. I also spot that d_cons_temp gets redefined in cell 65 with the same value as the residential section, which could be another duplication issue to flag.

More concerning is that d_temp appears to be a shared global variable set to 1 in tertiary but likely 2 in residential — since these modules read from globals() and the notebook runs sequentially, whichever cell executes last determines the value both modules actually use. This is a real trap for the sibling agent since residential could silently pick up the wrong d_temp value. I want to verify by checking the residential cells around d_cons_temp and the related accumulator definitions.

For my own calculations, I should use suf_TS_tes_sht directly (computed already as 0.93) rather than recomputing from d_temp and d_cons_temp, since that stored value is TS-specific and safe from this ordering issue. If I need to report the temperature figure explicitly, I should back it out algebraically from suf_TS_tes_sht and d_cons_temp rather than trusting the shared d_temp variable, even though luck of execution order happens to make it correct here.

I'm also thinking about lever id collisions — since a sibling module is handling residential, I need to make sure I don't reuse the same lever ids they're using, since export_topics will raise an error on duplicates within a sector.

I'll prefix all my lever and history keys with `ter-`/`ter_` to keep them distinct from their `res-`/`res_` ones. But I'm realizing the shared unit strings in the YAML (like `unit.kWh/m²/year`) could still collide if both sectors declare the same unit key, which would cause a build error I need to watch for.

If both topics declare the same unit key, whichever lands second breaks the build — a real coordination failure. The ui.yaml comment implies unit strings belong there, but §12 forbids touching it, and D35 exists specifically so new topics don't need shared-file edits, so I need to find a resolution that can't collide.

I check whether `T.unit(raw)` lets me namespace keys per topic, but it just concatenates "unit." with the raw string, so there's no way to isolate it without editing i18n.js. Trying to pick unit strings that happen to be unique is fragile and not a real solution — the safer path is to avoid declaring these unit keys in my topic YAML at all and instead flag the conflict clearly in my report.

I decide the honest move is to note that `unit.m²/person`, `unit.kWh/m²/year`, and `unit.kWh/person/year` need to be added to the shared `ui.yaml`, since both buildings topics require them and the per-topic mechanism collides on shared keys. The page will still function with the raw English fallback in the meantime, which keeps things usable while I flag the coordination issue rather than editing shared files myself — though I want to check if there's a way to avoid needing units the residential topic will also want, so my deliverable doesn't ship with untranslated fallback text.

Actually, since ui_strings is a single flat dict merged across topics, a unit string declared by one topic serves every topic that references it — so declaring the same key twice only errors if both topics actually declare it, not if only one does. That means the safest approach is to not declare these shared unit strings myself, and instead flag that they should be added once (ideally by the residential topic, or wherever makes sense), so my levers just pick them up automatically if the sibling declares them.

A build failure is worse than a cosmetic English fallback, but my test only runs against my own topic right now — the real risk is at integration time. I'll decide not to declare the shared unit keys myself and instead clearly report the exact lines to add to ui.yaml, so the integrated build passes no matter what the sibling does and the fix is trivial to apply centrally. Still, let me reconsider whether there's a topic-unique unit string that works just as well for participants.

Trying alternate spellings like "m² per person" instead of "m²/person" just creates inconsistency across topics in the UI rather than solving the collision risk, so that's not a good path. I'm settling on: don't declare the shared units myself, just report them — though let me check once more whether any of my units (m²/person, kWh/m²/year, kWh/person/year) could plausibly be unique to tertiary rather than overlapping with residential's likely set.

Since none of my units are actually unique, I'm confirming there's no `strings:` block needed for units in my YAML. But before finalizing, I want to verify how visible this fallback behavior actually is in the rendered output, given how much the brief cares about French typography — so I'm going to check the unit-handling function in i18n.js directly, along with the residential cells that might already define these keys.

## reasoning

I confirm `d_temp` gets reassigned between the residential and tertiary cells, so relying on it globally is a trap — the residential module actually needs `suf_RS_tes_sht`, not the shared `d_temp`/`d_cons_temp` variables. I'll flag this and use `suf_TS_tes_sht` for my own consistency check.

Rather than asserting against the shared, order-dependent `d_temp`/`d_cons_temp`, I decide to hard-code the prose constants directly (1°C temperature drop, 7% consumption per degree) and assert that `suf_TS_tes_sht` matches the computed value — this avoids any dependency on the fragile shared globals while still validating against the source document.

Now I'm moving on to check how the i18n fallback handles the `unit()` function.

Weighing whether to declare colliding unit keys in my YAML: the doc's principle of failing loudly over silent drift argues for not declaring them, since a build error at integration is safer than untranslated units slipping through. I briefly consider picking unit strings that naturally avoid collision by matching to distinct quantities, but land on the same conclusion — don't declare, and let the fallback handle it.

Now I need to gather external facts for each lever — trend, structure, benchmark, and tangible examples. For floor area I already have EU27, France, Germany, and Netherlands figures from the JRC-IDEES history, so that lever's covered without web research. For heating intensity per square meter, I'll need to look at Odyssee's service-sector data and EPBD requirements to build the European comparison.

I'm considering cooling degree days as a benchmark for the cooling lever, since Eurostat tracks how Belgium's cooling needs have risen over time, though I need real figures rather than rough estimates. For hot water, I'm less certain — JRC-IDEES would have per-person tertiary consumption but it isn't exported, so I may need to fall back on something like hospital hot-water usage as an illustrative benchmark instead.

For catering, Eurostat's household budget survey data on restaurant and hotel spending share gives me a solid comparison between Belgium and the EU average. For lighting, I have a strong citable fact in the EU's 2023 phase-out of fluorescent tubes under RoHS, alongside the earlier halogen ban and single lighting regulation. For street lighting, I'm thinking of Flanders' well-known practice of switching off motorway lighting overnight, a fact famous enough to have been noted from satellite imagery.

I'm noting that Wallonia may have followed with a similar policy around 2019, though I still need to verify that, and I should compare against the Netherlands, which keeps its motorways lit. For refrigeration, the EU's F-gas regulation and Ecodesign rules requiring doors on commercial chillers and energy labels for direct-sale cabinets since 2021 give me solid citations.

I'm planning to move into actual web research now using search tools to confirm these facts, weighing whether to handle this myself or consider delegating parts of it, though my instructions suggest I should keep this work within my own scope rather than passing off the whole assignment.

I'm deciding that a narrow research subtask could be reasonable as long as I personally verify every fact with exact numbers and URLs, so I'm planning to run a handful of targeted searches myself since accuracy here really matters. I'm thinking about parallelizing: writing the Python module with the numbers I already have while a subagent handles background research, then merging the verified facts into my YAML output afterward. To guard against fabricated numbers from a subagent, I'll have it return only quotes with URLs, which I'll then double-check myself, and I'm now drafting a precise research prompt for that purpose.

## reasoning

The renovation rate checks out at 1.73%, matching the notebook's claim. But the heat demand reduction figure is trickier -- the notebook cites -0.154 kWh/m²/year while my calculations show -0.133 for the full slope and -0.341 endpoint-to-endpoint, so I'm trying to figure out where that -0.154 number actually comes from.

I'm noticing the heating series is extremely noisy due to weather variation -- 81.6 in 2000, spiking to 114.5 in 2004, then down to 78.6 in 2023 -- so any linear trend here is fairly weak and worth flagging as a caveat. For cooling, the 2019/2000 ratio of 9.68 matches the notebook's "x9.7" claim nicely, but the growth rate is more ambiguous: endpoint-to-endpoint gives 0.624 kWh/m²/year while the notebook cites 0.743 as the 2000-2023 historical rate, which suggests they might be using an OLS regression slope rather than simple endpoints. I want to verify this by computing the actual OLS slope for both series.

Hot water demand looks like it's climbed steadily to 323.9 in 2023 (up 6.2% from 2019), matching the notebook's claim of continuous increase since 2000 aside from a 2019-2020 dip, and floor area rose 16.1% from 2000 to 2019 and kept climbing to 20.07 by 2023.

I still need to check the tertiary catering trend the notebook cites (+4.297 kWh/person/year from 2000-2023), but there's no exported history series for it in the current file, so I'll need to search the aux notebook to see if it computes that series internally even if it doesn't export it.

## reasoning

I'm checking that the OLS slopes match the notebook's values exactly — the heating and cooling per-m2 trends for 2000-2023 both line up with the reported figures.

An important nuance stands out: the target formula anchors the trend from the actual 2019 level rather than the fitted line's 2019 value, and 2019 was a mild winter, so that's worth flagging as a caution. Even more striking, the heating intensity trend from 2000-2019 alone was actually rising (+0.231/yr), and it's only the 2022-2023 crisis years that pull the full-period slope negative — a nuance worth surfacing since it complicates the notebook's framing.

I'm also weighing that the notebook attributes the 2000-2023 decrease to efficiency measures, but the decrease is really driven by the 2021-2023 energy-crisis drop (a behavioral effect, not insulation), which the notebook itself acknowledges elsewhere — that's a solid caution fact for the workshop. Separately, the renovation rate trend is negative and 2019's value sits below the long-run mean, which changes the 5× acceleration target depending on which baseline you use. Now I want to check the auxiliary notebook's cells for the tertiary series computation to verify the catering trend and confirm the scaling inconsistency mentioned in the brief.

## reasoning

I want to confirm the catering trend value of +4.297 kWh/person/year by recomputing the OLS slope myself, so I need to find the population data array from the auxiliary notebook to convert catering energy into per-person terms. I should also double-check the tertiary surface array while I'm at it.

The tertiary surface figure of 227,073,149.76 m² checks out against the buildings notebook's cell 24 value using the same 1e3 scaling, confirming that's the correct interpretation. Now I want to compute the catering OLS slope and verify the hot-water slope I calculated earlier (around 3.0817), since the notebook only describes it qualitatively as continuously increasing rather than quoting a number—so I need to locate the population_dic_STATBEL data.

## reasoning

That +4.2971 figure matches the notebook's value, and catering rose 48.9% from 2000 to 2019. Since catering lacks an exported history series but the data exists in the aux notebook, I could supply a hand-curated history block in the YAML for the catering lever, since the schema allows levers to point at hand-curated series.

I could compute the 24-point series from JRC-IDEES ktoe and Statbel population data, marking the source clearly in the YAML. I'm also wondering whether the aux notebook has tertiary EES series for lighting and refrigeration that I could similarly curate, which would give all 8 levers curves — though I need to weigh whether hand-transcribing JRC-IDEES numbers is appropriate versus having the aux notebook export them properly.

Actually, the aux notebook already computes the catering series internally, it just isn't exported as a history series — so the real fix belongs there, not in curated YAML. I'm weighing curated YAML (gets the curve now, but risks drifting from the notebook) against marking history as absent with a note recommending the notebook export be added, which is more honest and consistent with how I handled a similar situation in the transport topic.

I'll mark catering, lighting, street lighting, and refrigeration as missing history and flag that these four series would be worth adding to the aux notebook's export, though losing curves for half the levers is a real cost worth weighing further. Let me check whether the aux notebook actually holds the electrical tertiary series I'd need to reference in that report.

Checking which no-curve levers deserve inclusion on tangibility grounds: catering, building lighting, and street lighting all earn their place through relatable, everyday imagery. Refrigeration is borderline, so I'm weighing whether including it pushes toward 8 shown levers or whether dropping it to 7 keeps a cleaner "the other seven" framing.

I'll settle on 8 levers, since it keeps the caveat text accurate and gives each lever a distinct story, and I'll note the 4 missing history series separately for follow-up. Street lighting fits since it's genuinely part of JRC-IDEES' tertiary/services electricity accounting, even if it reads oddly under a "heating and cooling" label — I can frame the topic lead to clarify that street lighting is counted within the broader service sector.

Now I'll check the aux notebook for the EES series names to reference in the report before writing the module itself.

## reasoning

I see that the aux notebook has all the tertiary EES arrays computable but not exported, so I could report four more series that'd be cheap to add. I also notice the scaling inconsistency between cells is already documented elsewhere, which satisfies the brief's requirement to flag it if I touch that area.

For the four no-curve levers, I want to quote observed trend facts sourced from the notebook, but I'm running into a namespace issue — my module can only populate the `facts` block from the buildings notebook's globals, not the aux notebook's arrays. Looking at how inland_mobility handles this, I see it declares reference constants directly in its own module with inline citations, which is the pattern I should follow here.

For street lighting, I can actually derive the -0.476 kWh/person/year figure from the target/reference values already in the code rather than needing an external source. But for building lighting's observed 2000/2010/2023 values, I don't have those in the model code, so I'll need to declare them as module-level constants similarly.

That's a solid check — the transcribed 2019 value should reproduce the model's ref value exactly, giving a strong validation point. Now I'm working through the arithmetic to convert building lighting energy in ktoe to kWh per person across 2000, 2010, 2019, and 2023, using the population figures for each year and the standard ktoe-to-GWh conversion factor.

For 2010 and 2023 I get 362 and 231 kWh/person respectively, confirming the 2019 figure matches the reference value of 302.94, which validates my transcription. This shows building lighting peaked around 2010 and has been declining since — down 36% from peak to 2023, with the 2050 target requiring another 35% cut from there. I'm now doing the same conversion for street lighting figures across the same years.

Refrigeration confirms the notebook's stable/decreasing trend after 2010, catering rose about 49% from 2000 to 2019, and ventilation grew then leveled off, matching the 72.88 figure. I'm wondering if there's a more efficient way to verify these numbers than transcribing each one by hand.

I'm thinking I could define these directly as module constants in kWh/person for the levers without a history curve, with comments noting the ktoe source, then assert the 2019 value against the model in the same style as the inland_mobility pattern. Since levers with a curve don't need transcription but these four do need 2-3 observed points for their trend facts, I'm now weighing whether to also give them curated history blocks in the YAML using the same data since I'm transcribing the numbers anyway.

Adding the full series into the YAML would give participants the actual curve, a real UX win, at the same drift risk as just the summary points — and the build already validates shape. Since these are measured JRC-IDEES-2023/Statbel figures rather than projections, I'm deciding to add curated history blocks for the four levers lacking exported series, sourced accordingly.

But now I'm reconsidering — this could be over-engineering and diverge from the repo's own precedent, where transport left four levers uncurved because that data genuinely didn't exist. Here the data does exist, so curating it duplicates work, which cuts against principle 8's "numbers from the notebook, prose from a hand-edited file" and makes me question whether D15's exception really applies here.

Since catering, lighting, and refrigeration all have series that exist in the aux notebook, D15's stated condition for `historyAbsent` isn't actually met, so I should follow the repo's rule with `historyAbsent: true` and a `historyNote` for those four, then report the four export additions — this stays consistent with the documented design and avoids hand-typing 96 numbers into a YAML a reviewer would need to verify. I'm second-guessing the count of four levers without curves though — dropping refrigeration would break the caveat string, so I need to check how `play.js` actually renders when `historyAbsent` is set before deciding.

Q8 confirms this degrades gracefully as an accepted pattern: show the 2019 anchor against the range of possible answers with a note explaining the missing curve. So the final shape is 8 levers shown, 4 with generated curves and 4 with `historyAbsent: true` plus a participant-facing note — I need to word that note carefully in French, avoiding internal jargon like "not yet exported," and should check how inland-mobility.yaml phrases its own historyNotes to match the right register.

Once I settle that wording, I can move on to writing the module itself.

## reasoning

I'll use the same phrasing for the historyNote register, noting it's not available as a historical series in the project's data. Now I'm starting the module, making sure each lever's facts dict captures every placeholder number from the YAML and spoilers lists the keys reflecting négaWatt's own choices.

Writing module setup...

Now I'm laying out the full list of context variables needed for this module — reference, target, and progression time series covering surface, heating, cooling, hot water, catering, district heating networks, ventilation, lighting, elevators, refrigeration, and other electrical service categories.

I'm noting that `tes_TS_tot` and `ees_TS_tot` are dicts indexed like years, and that cell 94 adds extra keys to them before the corresponding DataFrames get built, so by the time the export cell runs, `df_tes_TS_tot` and `df_ees_TS_tot` are safe to use for the TWh values.

I need to make sure the totals sum only the four base thermal services and six electrical columns, since heat_ihs/heat_dhn duplicate space heating and hot water, and cooking_* duplicates catering -- avoiding double-counting when I compute the 2050 tertiary end-use demand total.

Now I'm working out the details for the `ter-floor-area` lever: floor area per person in m², reference value 19.864 dropping to a target of 17.878, with a slider from 13 to 27 in 0.25 steps and one decimal of precision. I'm noting the history key, proportional impact scaling against 2050 heating+cooling of 20.599, and starting to write out the reductionPct fact.

I'm also considering whether to pull in the residential floor-area-per-person figure (54.7 m²) as a contrasting data point, since it comes from a sibling variable I should mostly avoid touching — reading it for context should be safe since it's not an edit, but I want to be careful not to create unintended coupling.

Since the YAML placeholders only resolve against the lever's facts block, I'll declare the benchmark values as module constants sourced to JRC-IDEES-2023, noting they mirror the history series the data notebook exports — that's the honest, checkable path rather than trying to reach into build-script internals.

Actually, looking at how other facts in this codebase handle observed history — like the "35.6 km/day in 2008" trend fact — the established pattern is to write those historical values as literals directly in the YAML with a cited source, and only use placeholders for the specific value tied to the current target year.

Now I'm drafting the remaining facts fields: totalMm2_2019, heatCoolTwh with the 2050 target as the spoiler, and phrasing for trend, structure, and benchmark strings that weave in the literals and placeholders together, checking that number formatting stays clean since these values fall under 1000.

For benchmark, I'm comparing Belgium's tertiary floor space per inhabitant against the EU27, France, Netherlands, and Germany, noting it sits between France and the EU average. For the tangible comparison, I want something relatable — thinking through options like a small studio flat, a supermarket, or a single-car garage to represent 19.9 m² per inhabitant.

Now I'm moving on to the second fact about space heating in service-sector buildings, setting up the unit as kWh/m²/year with a reference value of 95.4146 and a target of 66.5365, and configuring a slider ranging from 40 to 110.

Testing decimal options: step=1 with decimals=1 shows ugly ".0" endings, while step=0.5 changes the answer granularity awkwardly. I'll settle on step=1, decimals=0 so refValue displays as "95" and target as "67," keeping things clean even if some precision is lost for the trend fact.

I need to verify the export script doesn't flag buildings, and confirm the 2019 history value (95.4147) matches the lever's refValue (95.4146) closely enough. Then I'm calculating the scaled impact from vTarget 66.5365 to about 14.989, and starting to draft the facts array with the efficiency-only 2050 level at 71.5.

Now I'm sorting which lever values count as spoilers versus neutral observations: renovAccel is a spoiler, the historical improvement rate itself is an observation but the accelerated scenario product is a spoiler, the degree-drop setpoint is a spoiler choice, while the degree-saving percentage from the literature isn't, and the sufficiency percentage is borderline since it derives the same way as the degree-saving figure.

I'm also noting the mean renovation rate (1.7, verified against the full history at 1.7254) should be declared as a plain prose constant with a comment, and drafting the wording for the trend narrative — describing the weather-noisy series with specific yearly values like 81.6 in 2000, a peak of 114.5 in 2004, and lower figures by 2019.

and 2023, while emphasizing the fitted trend is only a slight annual decline. I'm also framing the falling renovation-rate story (2019 vs 2023 values), a caution about useful heat versus boiler/distribution losses and weather swings, and a tangible takeaway about thermostat savings per degree.

For the tangible school example, I want to convert a 1,000 m² school's annual heat demand into a relatable comparison against average household heating — using the residential per-m² figure and average household floor area to derive kWh per household, then dividing the school's total demand by that to express it as "equivalent to X homes."

I'm settling on reading just `ref_RS_sur_spe` and `ref_RS_tes_sht`, both stable 2019 observations, to avoid coupling risk with the sibling module — as long as variable names don't change, `need()` will work fine.

Now moving to the third indicator, `ter-cooling-per-m2` for service-sector space cooling: unit kWh/m²/year, reference 17.2228 rising to a target of 24.9005, with a slider from 10-45 in 0.5 steps showing one decimal. I'm noting the growth factor of 9.7x from the notebook's prose as a supporting fact.

I'm verifying the growth calculation checks out at roughly 9.68, rounding to 9.7 as stated, and cross-referencing the notebook's prose value of 1.8 kWh/m² starting point against the more precise exported figure of 1.7798, declaring a constant with a comment citing the source. I'm also noting the observed deployment rate of 0.743 as a fitted trend from 2000-2023, distinguishing it from the deploymentShare spoiler value of 1/3, and projecting forward to 2050 using that trend.

I'm confirming this projection is a counterfactual ("if the rate continued") rather than an actual scenario value, so it's safe to present as a fact — continuing the observed trend would yield about 40 kWh/m² by 2050. I'm also cataloging other reference points: cooling energy figures for 2019 and target values, plus the historical trajectory showing growth from 1.8 in 2000 stalling out around 15.4 by 2021.

I'm noting that cooling might be flagged as the one service where sufficiency conflicts with climate goals. Moving to hot water in service-sector buildings, I'm setting up the parameters: target of 305.045 kWh/person/year, a slider range from 150-420, and impact scaling of 3.844. I'm weighing whether to cite the observed 2000-2023 slope (3.45) or the OLS trend (3.08), leaning toward quoting the observed endpoint values directly since the notebook doesn't specify a preferred rate for this metric.

Extrapolating forward with the OLS rate of 3.08/year from the 2019 value of 305 gives roughly 400 kWh/person by 2050, while the endpoint slope of 3.45 would suggest closer to 412 — I'm leaning toward the OLS figure for consistency with how other services are calculated. I'm second-guessing whether running my own regression on the exported data counts as inventing a number, though it seems justified since it mirrors what the auxiliary notebook does for other rate calculations.

That's actually a striking reveal — even holding intensity flat, total hot-water demand still climbs from 3.49 to 3.84 TWh purely from population growth. I'm noting the trend facts (244.5 kWh/person in 2000 up to 323.9 in 2023, a 32% climb), flagging that hospitals and care homes are likely the biggest users but need a source before claiming rank, and working out a tangible comparison: converting 305 kWh/person/year into an equivalent number of 5-minute showers heated from 12 to 40°C.

I'm refining that: since tertiary hot water isn't really showers, a cleaner comparison is against residential hot water use (674.9 kWh/person in 2019), giving "for every 100 kWh of hot water at home, another 45 kWh gets heated in offices, hospitals, hotels, sports halls and restaurants" — a clean 45.2% ratio using existing notebook values. I'm also noting the caution that flat per-person intensity still means +10% total demand due to population growth, then moving to the next data point: catering, measured in kWh/person/year, with reference value 289.141 and target 346.970, slider at 180.

I'm working out the catering figures — a 20% change, a verified trend rate of 4.297, and projecting that trend forward to 2050 gives roughly 422.3 kWh/person/year as a counterfactual. I'm also cross-checking against residential cooking energy use (226.55 kWh/household in 2019), converting to a per-person basis using household size, which comes out to about 98.1 kWh/person for contrast.

Comparing home cooking (98.1 kWh/person, ~96.4 ktoe) against tertiary catering (289.1 kWh/person, ~284.2 ktoe) reveals a striking gap — nearly three times more energy is spent cooking outside the home than inside it, which strikes me as the standout fact for this lever, though I want to sanity-check whether that residential figure is really plausible for Belgium.

Working through the math, 96 ktoe converts to roughly 226 kWh per household per year, which seems low but reasonable for useful cooking energy, while the 284 ktoe for catering translates to about 3,305 GWh, which makes sense given how energy-intensive commercial kitchens are with open equipment and long operating hours. I'll cite these as the model's own figures sourced from nW-BE/JRC-IDEES, and I'm noting that the residential prose separately shows a +15% cooking increase target versus the tertiary sector's +20% — different trajectories worth flagging. I'm also tracking the catering TWh figures (3.305 current, 4.372 target) and the gas share numbers (40.4% today dropping toward a 5.0% target) as additional observations to weave in.

I'm now thinking through the framing for the trend narrative — something like consistent per-person growth in out-of-home cooking energy since 2000, using the 194 kWh/person baseline as a literal figure sourced from JRC-IDEES data. I also want to build a home-vs-away comparison showing that cooking outside the home now uses roughly three times the energy per person as cooking at home, and I'm considering a tangible benchmark — maybe framing 289 kWh/person against something relatable like a professional gas range's typical usage — to make the number land better with readers.

For the tangible comparison, I'm settling on translating 289 kWh/person into total sector energy: roughly 3.3 TWh across 11.4 million people, comparable to the sector's total lighting usage — a nice internal comparison since both come from the same notebook dataset (3.305 TWh cooking vs 3.463 TWh lighting). Then I move to the next indicator on building lighting in the service sector, setting up its reference and target values with a slider around 80.

Now I'm working out the lighting reduction scenario: computing the target energy scaled to 1.890 TWh, the reduction percentage at about 50.5%, and cross-checking against household-level lighting consumption converted from ktoe to kWh, landing around 328.8 kWh per household or roughly 142.3 kWh per person.

Comparing that to service-sector lighting per person of about 302.9 kWh, I note it's more than double the home figure — a useful structural point. I'm also compiling trend data across years (2000, 2010, 2019, 2023) showing LED adoption cutting about a third off the 2010 peak, though I should verify these hand-computed values in code, plus considering benchmarks like the EU fluorescent-tube phase-out and tangible comparisons like LED-vs-fluorescent efficiency gains, though these need sourcing.

For the tangible fact, I'm weighing whether to frame 3.46 TWh as equivalent to roughly a million Belgian households' annual electricity, but that requires an unverified household consumption figure, so instead I'll go with a cleaner comparison: Belgium's commercial/institutional lighting used 3.5 TWh in 2019, more than commercial refrigeration and ventilation combined — all figures I can pull directly from the notebook.

Next I'm looking at street lighting, measured in kWh per person per year, with a reference of 79.171 and a target of 64.415, using a slider from 10 to 100. The notebook models a reduction rate of 0.476 kWh/person/year based on the observed 2000-2023 trend, so I'm noting that this rate itself is an empirical observation rather than a scenario choice — the real assumption is continuing that trend forward.

Continuing, I want to verify that 0.476 matches the fitted slope from the historical data, and I'm working out the reduction percentage (18.6%) and street lighting's small share of total sector consumption (about 2% in 2019), along with the trajectory from 82 kWh/person in 2000 up to a 2010 peak of 89, down to 68 by 2023, plus checking on Belgium's lit motorway structure as a pending research item.

Now I'm setting up the commercial refrigeration section — comparing to European benchmarks and framing the "switch off all street lights" scenario as roughly 2% of service sector demand, then moving to refrigeration's parameters (182.795 kWh/person/year reference, 155.376 target, 15% reduction) and its 2019/target TWh values.

I calculate that home refrigeration works out to about 213.5 kWh/person, actually higher than commercial refrigeration's 182.8 — a notable contrast worth highlighting. I'm drafting facts around the historical trend (118 kWh/person in 2000, rising 58% to 187 by 2010, then flattening to current levels), the home-vs-shop structure, and EU Ecodesign labeling benchmarks for chilled display cases.

For the tangible fact, I'm considering framing refrigeration's 2019 footprint against ICT's comparable size, and for the lower-priority spare levers (ventilation, misc, ICT, district heating, catering gas), I'm working out slider ranges that keep the target comfortably within the edge margins even when target equals reference.

Continuing to tune slider ranges for ter-ict, ter-district-heating, and ter-catering-gas, adjusting bounds so the edge ratios avoid dead-centre positions and stay within acceptable thresholds while keeping reference values in range.

I'll export only ventilation, misc, and ICT as spares since those are genuine demand-side degrees of freedom that just didn't make the 8-screen cut, while district heating and catering-gas stay off the list since they're carrier choices already covered elsewhere and don't change useful demand — I'll note that reasoning in the module docstring. Spare impacts remain proportional to their scaled 2050 TWh values.

Now I'm defining the model block's scope description...

I'm cross-checking the reference values against the notebook's prose — verifying that the per-person surface figure (~19.9 m²) and total surface area (227.073 Mm²) match, though I notice the prose's unit labeling for "227,073 Mm²" is actually inconsistent by a factor of 1000, since that figure really represents thousands of m².

The same unit confusion applies to the residential total (625,236), even though the underlying code correctly computes it in m². This mirrors the same 1e3/1e6 unit mixup flagged earlier for the aux notebook, so it's worth reporting as a documentation error. I'm also checking the thermal envelope surface heat transfer values, roughly 95.4 for reference and 71 for target.

Continuing through the numeric checks for scaled and showcase heat transfer coefficients, plus the category proportion and district heating network figures across sections 3.1.1 and 3.1.2.

I'm working through more lever definitions and district heating values, then planning a self-consistency check comparing my scaled figures against the notebook's total TES output table to confirm the space heating TWh computation matches.

I want to add per-service assertions verifying the proportional impact assumption, confirm the 2050 population figure, and write a small verification script to check my hand-computed kWh/person values against the observed data points.
