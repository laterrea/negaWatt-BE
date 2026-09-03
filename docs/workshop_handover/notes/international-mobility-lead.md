# Recovered work — international-mobility (lead)

Agent `a2b61c99b31f72dea`, stopped by the session limit before it could report.
Everything below is recovered verbatim from its transcript, in order. **Nothing here has been checked** — treat every figure as unverified until re-sourced.

11 entries, of which 0 web results.

---

## reasoning

I found a discrepancy in Q4: cells 28 and 29 disagree on the intra-EU aviation shift split — the prose says 25% to high-speed rail and 20% to conventional, but the code reverses these values. This is exactly the kind of inconsistency I need to flag, so I'll note it for my report without altering the notebook itself.

Looking further, I want to check cells 43, 44 for modal shares, 58-64 for aviation carrier shares, and 82, 83 for aviation energy figures. I also notice cell 23 has a formatting issue where "150.0833 people" appears to be a mangled version of 1,150,833 — working through the math confirms this checks out as roughly 10% of the population making long-haul trips, and the 2050 reduction scenario similarly checks out at about 5.5% of a larger future population.

I'm now weighing which framing works best as a communicative lever: percentage of population able to make an 18,000 km round trip annually, years between long-haul trips per person, or total round trips per lifetime. The lifetime framing (roughly 8 trips now versus 4.5 in 2050) feels most intuitive, though I need to trace how this connects back to the -40% aviation reduction parameter in the model and how it feeds into the modal share calculations.

The lever needs to anchor to a real historical curve, so I want to check whether `aviation_km_day` in the history data covers total aviation or just extra-EU travel. Let me look through the remaining aviation-related cells and the history file to pin this down.

## reasoning

I'm noting that the 2019 occupancy figures for intra-EU and extra-EU kerosene flights appear as raw literals embedded in specific cells rather than as named exported variables, which creates a tension with the reporting principle that undocumented numbers should be flagged rather than fabricated — even though these values technically exist within the notebook itself.

But I realize I can recover them from the notebook's global state: `occupancy_PM_avi_intra`, `occupancy_PM_avi_extra`, `cons_fuel_PM_avi_intra`, and `cons_fuel_PM_avi_extra` are DataFrames already assigned at module level, so I can read the 2019 values directly without re-declaring anything. Now I need to verify these are accessible at the relevant export cell and check how the export function and helper functions like `make_lever` and `write_lever` handle them.

## reasoning

The aviation_km_day series is already available as history data, so I can use it directly for the international mobility levers. I'm defining the degrees of freedom: a -40% reduction in extra-EU air travel per person and a -50% shift away from intra-EU flights, with reference values dropping from 1.812 pkm/person in 2019 to 1.087 pkm/person by 2050.

I'm also mapping where shifted intra-EU traffic goes—conventional trains getting +25%, high-speed rail +20%, coaches +5%—plus load factor multipliers for both intra-EU (×1.25, raising passengers from 121.927 to 152.4) and extra-EU (×1.17, from 187.817 to 219.7), fuel efficiency adjustments for aircraft-km, and a hydrogen share lever for intra-EU flights. That gives me 8 potential levers, near the top of the acceptable range, and I need to be careful not to duplicate the `pro_PM_spe` variable already claimed by the inland mobility sector since lever IDs must stay unique.

I'm considering collapsing the rail/coach split into a single slider expressing what percentage of shifted air-km goes to rail versus coach, since rail (conventional plus high-speed) makes up 90% of the split—but the energy difference between rail and coach by 2050 is small enough that this lever might not carry much real leverage.

Instead I want to find levers that feel intuitive to a non-expert, starting with extra-EU long-haul flying. I'm working out that 2019 extra-EU travel of 1812 pkm/person/yr, given a 9,000 km one-way (18,000 km round trip) flight, works out to roughly one trip every ten years per person, though I need to settle on the right lifetime framing the notebook uses, something like once every 18 years.

Using Belgian life expectancy of 82 years as the reference, that comes to about 8.26 long-haul return trips in 2019 versus 4.95 in 2050 — the ratio stays 0.6 regardless of how I scale the unit, so the choice is mostly about which framing feels most honest and intuitive, maybe trips per decade instead.

Actually the population share angle seems more tangible: roughly 10.1% of people making one long-haul return trip yearly in 2019, dropping to about 5.5% in 2050. The gap between those percentages reflects two things layered together — the population growing from 11.4M to 12.5M, and the per-person travel intensity falling by 40%.

So the notebook's cell scaled the 2050 figure off the absolute Gpkm rather than the per-person intensity, which understates the real per-person rate — properly scaling gives 1,087.2 pkm/person, 0.0604 trips per person, roughly 6.04% of the population taking a trip per year, or about one trip every 16.6 years. I should just run the notebook directly to confirm the population figures rather than reasoning through it further.

I should pull exact values straight from `df_PM` for accuracy, matching how inland_mobility.py handles this. Thinking about units now — "% of population who could take one long-haul trip a year" is picturable but risks misreading on a slider. I actually prefer "long-haul return trips per person per lifetime" (8.3 → 5.0) since it's more tangible, framing it as "how many times in a life can you fly to Shanghai and back?" — and it matches how the notebook already frames things.

But converting to a lifetime figure means picking a life expectancy value, like Statbel's 82.0 years, which the notebook itself doesn't compute — that seems to run against the principle that undocumented numbers belong in the notebook, not invented in prose. Then again, inland_mobility.py does declare its own reference constants for prose-only figures, like fuel energy content, so there may be precedent for this.

Following that pattern, since the 9,000 km round-trip distance already appears in the notebook prose, defining a lifetime-scale reference constant with a clear source comment would fit the same convention. I'm now weighing whether "trips per lifetime" is really the best unit for the slider lever, considering the range and step size, but worrying that a population-average lifetime figure might be confusing without more context.

Actually, going with the notebook's own framing seems stronger here — anchoring the lever on the 9,000 km round trip and asking how often someone makes one over a lifetime feels more grounded and true to the source material.

Using an exact lifetime figure risks stating an unverified number, so I'm reconsidering — expressing it per decade instead avoids needing any external life-expectancy constant while staying concrete: roughly one long-haul return trip per decade in 2019 versus 0.6 in 2050.

Actually, using life expectancy as an external statistic is risky since revisions would shift the lever's values, whereas defining a fixed round 80-year lifetime is cleaner since it's a definition rather than a claim needing sourcing. I'll go with "over an 80-year lifetime" and mention Belgian life expectancy (~82) as a supporting fact if I can verify it, showing the 80 is a conservative round number.

For the first lever, long-haul flights per lifetime, I'm working out the slider range and reference values, checking that the reference point sits comfortably away from the edges and isn't dead-center, with a range of 2 to 14 trips and a reference around 8.26.

Actually, I'm reconsidering the upper bound given historic aviation growth trends of 3-5% per year, which suggests extending the max to 16 to allow room for growth scenarios, and confirming the target still clears the edge-margin threshold within that revised span.

Moving to the second lever for short-haul intra-EU flights, I'm looking at how the proportional impact coefficient applies to the target module.

For intra-EU air travel per person, the 2019 baseline sits around 1364 pkm/person dropping to roughly 640 by 2050, so I'm weighing whether the slider unit should be raw distance per person or something more tangible like short-haul return trips per year, estimating typical trip distances like Brussels-Barcelona to calibrate that conversion.

I'm now considering km/person/day instead, since the aviation_km_day history series (8.70 km/day in 2019) already combines intra- and extra-EU travel, and I can't split that without touching the data_aux notebook, which isn't allowed. This suggests using the combined series directly for a single aviation lever rather than trying to isolate just the intra-EU component.

Mixing bases would be a cardinal sin, so I can't just express lever 2 as a share going from 0 to 50%, since that construction has no historical curve. I'm hunting for a natural share basis, like intra-EU air's share of medium-distance international travel, rather than an artificial zero-start metric.

I keep hitting a wall—there's no clean international-rail split in the model to build this share from. Expressing total air travel per person as a single degree of freedom doesn't work either, since it's actually determined by two separate long-haul and short-haul aviation variables plus an overall parameter, so a single slider would create ambiguity.

Then I consider splitting them into two levers without a history curve, but that breaks consistency between the trend fact and the question basis. So I'm now wondering whether JRC-IDEES actually reports intra-EU and extra-EU flight distances separately for 2000–2023, which would let me supply a proper per-topic history series instead.

I want to check the data_aux notebook's history export cell and the local data files to see if that split is available, and look at build_workshop_content.py for how the history/historyAbsent/curated-series schema is defined.

## reasoning

Now I know the YAML curated history block schema requires label, unit, source, url, x, y, and sets curated to true. This means I can supply a curated history series with a proper source — if JRC-IDEES data is available locally, I could compute the intra/extra split for 2000–2023 myself, giving verified real curves for both aviation levers rather than guessed numbers.

I'm checking what data is present locally, looking at the data_aux notebook's history export cell and the data directory.

## reasoning

I realize the aviation figures are only totals with no intra/extra split available anywhere in the repo, so I'd need the actual JRC-IDEES Transport workbook to compute that split. Fetching it from the web would require explicit user permission per my safety rules on downloading files, so I can't just go grab it myself.

I'm also weighing whether Eurostat's flight/passenger data could substitute, but that's on a different basis than passenger-km, so it doesn't really solve the problem cleanly. The most honest path forward is probably to keep the two demand levers separate but mark both with `historyAbsent: true` and a note explaining that JRC-IDEES only publishes the combined intra+extra series, which the model splits using 2019 modal shares.

Inventing a curated series scaled from the combined curve would mean fabricating data, so that's not honest either. The `historyAbsent` path is already an accepted, working pattern used by four of the eight inland levers — it shows the 2019 anchor against the range of possible answers with a note explaining why there's no curve, so applying that same approach to the aviation demand levers is legitimate. Still, I want at least one lever with a real curve, so I'm considering whether either of the two aviation demand DOFs — extra-EU long-haul or intra-EU short-haul — could be framed around total aviation instead.

I'm working out that redefining the levers as total air km per person plus the intra-EU share would let me derive both intra and extra travel exactly from those two combined values, checking the arithmetic against the −45.6% drop from 2019's 8.70 km/day.  However, the intra-EU share metric feels awkward and unintuitive as a workshop question — the 2019 figure of 42.9% dropping to 37.1% by 2050 isn't a compelling lever, and I want the framing to fit the km/day unit the brief already specifies.

I need to reconcile two brief instructions that push opposite ways: using an intuitive per-trip unit versus using aviation_km_day as the strong trend curve for the slider. Maybe the resolution is to keep the trip-based unit for the levers themselves while using the aviation_km_day series as curated history data converted into that same unit.

Since Belgian aviation data lumps short- and long-haul together into one series, I don't think I can cleanly split extra-EU history from it, so I'll likely need to mark that lever as lacking a direct history series and note why.

I could try expressing trips per person per year, but I'd need a reference distance and I don't want to invent one from thin air — Eurostat's avia_paoc data could yield an average intra-EU trip distance but that's not in the repo. A cleaner alternative would be defining a standard return trip of a fixed round-number distance, like 2,000 km, rather than picking a specific city pair.

For short-haul, maybe I should frame it differently — as a modal shift where intra-EU flights move to train and coach instead. The Brussels–Barcelona plane vs. night train comparison feels tangible. I'm checking whether the model actually tracks this shift, and it looks like it does: those diverted flights go into high-speed rail and conventional rail categories in the 2050 baseline.

I could use intra-EU air-km as a share of air-plus-high-speed-rail km — 2019 comes out to about 91% flown. But projecting this to 2050 gets messy since high-speed rail also carries domestic trips, so it's not a clean comparison.

I'm dropping that approach and going back to keeping the model simple and faithful, expressing intra-EU aviation demand using a clearly-defined short-haul reference distance rather than a share metric. Let me reconsider the full set of levers from scratch.

For the long-haul flights lever, I'm setting the unit as long-haul return trips per lifetime using a 9,000 km reference distance like Brussels–Shanghai, with 2019 values around 8.26/8.05 depending on life expectancy assumption and 2050 values dropping to roughly 4.95/4.83, framing the question around how far sufficiency can bring long-haul flying down. For the short-haul flights lever, I'm working out a percentage-based unit tied to intra-EU air travel.

Trying "per decade" for both feels awkward since long-haul dips below 1. I consider whether the short-haul lever could instead be framed as a share of intra-EU air travel shifted to train and coach, though I recall the reasoning against percentage-based sliders since they start at zero with no real curve — but that concern was specifically about the car case where a share alternative existed, so maybe it doesn't fully apply here.

Ultimately I decide the clearest, most discussable framing for short-haul is a share metric after all: the percentage of intra-EU air travel still flown in 2050, dropping from 100% to 50%, since that's the most intuitive unit for non-experts even if it departs from the per-decade framing used elsewhere.

Trying per-lifetime framing instead — long-haul drops from 8.05 to 4.83 trips, and short-haul (2,000 km return) goes from 54.5 to 25.6, which feels like a more intuitive, readable unit than per-decade.

Actually the JRC-IDEES intra-EU average might skew lower than 2,000 km round trip since it includes short hops, so I should be careful about the framing. Better to present it explicitly as an equivalent conversion of the km budget into one standard trip — Brussels–Barcelona as a stand-in for "the same air-kilometre allowance," not a literal count of trips — mirroring how the notebook handles the long-haul case.

I'm now reconsidering whether I'm overanalyzing this, and re-reading the brief's actual ask: express each metric in a picturable unit, and the notebook's own cell 23 framing — a 9,000 km round trip, how many people could make one per year, how often over a lifetime — is the strongest candidate, possibly serving as the lever unit itself. That framing essentially reduces to something like a percentage of the population able to take one such trip annually.

I compute this for long-haul (10.1% → 6.0%) and short-haul using a 2,000 km standard trip (68.1% → 32.0% by 2019/2050). It's a clean, consistent unit needing no lifetime constant, but I worry it risks being misread as "this is literally what percent of Belgians flew," when it's really an equivalent budget figure, not an actual travel rate.

I decide instead to settle on "return trips per lifetime (80 years)" as the framing for both categories, since that reads more clearly as a budget rather than an actual yearly rate — though I briefly reconsider mixing frames before just committing to move forward. I finalize my lever list, starting with a long-haul-flights lever measured in long-haul return trips per lifetime.

I set the short-haul-flights lever similarly, in short-haul return trips per lifetime, and I start reasoning through how its removed kilometers should shift into rail and coach travel rather than simply scaling energy demand the way the long-haul lever does.

The basket splits into train-conventional at 0.25, train-high-speed at 0.20, and bus/coach at 0.05, and I want to verify the coach allocation matches cell 35's bus&coach output. There's also a rail-vs-coach destination split (roughly 90% to rail) for the freed short-haul traffic, but I think I'll treat that as a low-leverage background lever rather than something surfaced to the user, similar to occupancy assumptions.

Next I'm turning to the intra-EU load factor lever: applying a 1.25 target ratio to the baseline of 121.927 gives about 152.4 passengers per aircraft for intra-EU flights, measured against the 2019 baseline.

Since hydrogen aircraft stay fixed at 100 seats and make up only about 5% of intra-EU flights in 2050, I'll frame the lever specifically around kerosene occupancy, with energy scaling inversely for just that portion of the intra-EU fleet. Next I'm looking at the long-haul equivalent, "plane-load-extra": 1.17 × 187.817 gives roughly 219.7 passengers per long-haul plane, so I need to figure out a sensible slider range for that.

I'm second-guessing having two separate load-factor levers since it feels a bit repetitive alongside the other 6-8 levers, but they represent genuinely distinct degrees of freedom, so merging them would misrepresent the model even if it means spending more time on a technical point.

That gives me 7 candidates, but noticing (c)/(d) and (e)/(f) are near-duplicate pairs, so four of seven slots are eaten by aircraft technicalities alone — weak workshop design. I'm testing an alternative: collapsing load factor and fuel burn into a single combined metric, kWh per passenger-kilometre, calculating 2019 intra-EU comes out to roughly 0.566 kWh/pkm from the raw fuel intensity and occupancy figures, and starting the same conversion for 2050.

For extra-EU I get 2019 at about 0.358 kWh/pkm and 2050 at roughly 0.257, a 28% drop. But combining fuel and occupancy into one number means the group's answer would map onto two separate model inputs without a clean unique split, which seems to conflict with the requirement that no group enter a self-inconsistent scenario — though on reflection it's not actually inconsistent since the energy result multiplies out exactly.

That derived kWh/pkm value is a product, not a degree of freedom, and the reveal probably needs to be on an actual input rather than an output. Since it's a one-to-many mapping without a defined split rule, this feels too risky to use directly.

I should focus on what's most pedagogically valuable and settle on seven levers: extra-EU long-haul flight demand as a headline item with a huge swing, intra-EU short-haul flights covering both demand and modal shift to rail as another headline, load factor per long-haul plane going from 188 to 220 seats filled, and energy intensity per long-haul aircraft-km.

Rounding out the set: passengers per short-haul plane rising from 122 to 152, energy per short-haul aircraft-km actually getting worse as a surprising twist, and hydrogen share of short-haul flights growing from 0 to 5% as the "technology won't save us" example. This gives a clean symmetric structure — a long-haul trio, a short-haul trio, plus hydrogen — that's easy to explain since each flight type gets the same three types of levers.

I need to justify why short-haul fuel per aircraft-km rises 9.4% from 2000 to 2023 despite efficiency efforts — it seems this follows the observed trend of bigger aircraft on intra-EU routes carrying more passengers, with occupancy rising nearly 50% over the same period, so fuel consumption per km increases even as per-passenger efficiency might improve.

But I realize I should double-check my units here: since redu_fuel is measured per aircraft-km rather than per passenger-km, the honest tangible metric for fuel would be something like litres of kerosene burned per kilometre flown. Converting from kWh/km using jet fuel's energy density and specific gravity, I work out roughly 7.19 litres per kilometre.

Dividing by typical passenger counts gives compelling tangibles: about 5.9 litres per 100 passenger-km for intra-EU flights — comparable to a petrol car with a single occupant — and about 3.73 litres per 100 passenger-km for long-haul flights. I should be careful about consistency though, since a similar inland car-energy fact used a slightly different conversion reference for kWh per litre of petrol.

Actually, better to sidestep kerosene conversion entirely and just compare energy per passenger-km directly: the notebook's own petrol car figure (5.798 kgoe/100km → 0.674 kWh/km, divided by 1.22 people ≈ 0.553 kWh/pkm) lines up almost exactly with an intra-EU flight's 0.566 kWh per passenger-km — a clean comparison using only numbers already in the notebook.  Long-haul flights come in lower at 0.358 kWh/pkm, which is a genuinely surprising fact. I realize the intensity function pulls kWh/pkm directly from the TWh tables for each mode, so I can compute plane-intra, plane-extra, and car figures straight from the 2019 model tables rather than doing manual conversions. Now I'm turning to the hydrogen lever — the share of short-haul flights using it goes from 0% in 2019 to 5% by 2050.

I need to check the slider bounds since the edge margin looks tight relative to the minimum threshold, so I should verify the LEVER_MIN_EDGE_MARGIN value before proceeding. I also want to work out the energy impact of hydrogen share by comparing hydrogen aircraft fuel consumption against kerosene, converting kg of hydrogen to kWh using the standard energy density.

Working the numbers, hydrogen comes out at 0.433 kWh per passenger-km versus kerosene's 0.476, meaning hydrogen is actually slightly more efficient per pkm, so the technology-share lever has a tiny leverage on final energy — really it's about which fuel gets used, not how much energy is consumed overall. This feels like a solid finding worth surfacing, especially since it ties into the broader SAF/hydrogen discussion the review flagged as missing from the model.

I'll keep this as one of the shown levers, and it also gives me a good place to note the caution about SAF being entirely absent from the model, plus a useful fact: 5% of intra-European flights amounts to less than 2% of total flights. That brings me to seven shown levers plus the one hidden one (destination split) — within my 6-8 target, so I'll keep it at seven rather than trimming down to six. Now I want to double-check the load factor levers before finalizing.

That's a strong trend: intra-EU passengers per flight from Belgium rose from 87.5 in 2000 to 130.8 in 2023, extra-EU from 154.0 to 205.5, both traceable to JRC-IDEES via the notebook comment. Notably, the 2050 target of 152.4 is only +16.5% above the already-observed 2023 value of 130.8, since the model's 2019 anchor (121.9) sits below the latest actual observation — worth flagging in the justification.

I considered building a curated three-point history series (2000/2019/2023) for the chart, but three points would look thin and I'm not sure how spark.js/play.js would render it. Simpler and safer to just put those numbers in the trend fact text and mark historyAbsent: true, with a note explaining JRC-IDEES only provides passengers-per-aircraft data for select years.

I'm now worried that across the aviation levers, none might end up with a real chartable curve, which contradicts the brief's promise that aviation_km_day gets a real slider curve — I need to reconsider whether that data could instead work for the long-haul lever, though the unit basis differs.

Because short-haul flights can be replaced by trains but long-haul can't, sufficiency shifts the mix toward long-haul travel. So lever 1 becomes total air travel per person per day, dropping from 8.70 to 4.73 km/day, with a real historical curve to back it (2000: 5.52, 2019: 8.70, 2023: 7.66) — a strong trend with the biggest swing. I consider expressing this in trip-equivalents instead but that mixes short and long haul incorrectly, so I try framing it as km per person per year instead.

I keep weighing the trade-offs and land on prioritizing the design principle that the slider should represent the 2050 endpoint of a real historical curve, since that removes the need to explain what's being measured. This mirrors an earlier decision to define a lever based on having actual measured data even if the scope is less intuitive — so I'll define this demand lever as total air travel per person in km/person/day, matching the unit used for the ground-travel lever for a clean parallel.

I also want to preserve the vivid trip framing by using it in the "tangible" field, which renders dynamically as the participant moves the slider — showing something like a relatable comparison (e.g. a round-trip flight equivalent) built from the resolved value and facts.

Let me check play.js for how it computes valuePerDay/valuePerYear at runtime, and look at spark.js to understand historyAbsent and curated behavior. I want to reconsider one more framing that might satisfy both goals: a lever specifically around long-haul flights.

Comparing this to splitting flying into air-km/day plus a long-haul-share percentage, that second option feels awkward — asking citizens to decide what share of flying should be long-haul in 2050 isn't a natural policy question. It's a derived metric that would confuse people rather than clarify.

The real advantage of keeping long-haul and short-haul as separate levers is that it preserves the actual debate happening in Europe right now: how much long-haul flying is acceptable, and how much short-haul travel could shift to rail — echoing France's short-flight ban and night-train push. Collapsing that into a share metric would lose this pedagogical connection entirely, and I can lean on the existing historyAbsent pattern to handle the missing curve while still citing the combined observed trend clearly.

Good, that confirms play.html handles the no-history case fine. Now I'm nailing down the exact units for the long-haul lever: 18,000 km round trip per destination, with 2019 extra-EU per-capita distance around 1812 km/person converting to roughly 0.1 trips/year, dropping to 0.06 by 2050 under a 40% reduction, which I'll then scale to an 80-year lifetime.

I want to phrase this as "long-haul return trips per lifetime" and describe one trip clearly in the subtitle, referencing routes like Brussels–Shanghai or Brussels–Los Angeles averaged across the population over an 80-year life. For the slider, I'm working out that a range of 2 to 16 with 0.25 steps puts the 4.832 target and 8.053 reference comfortably within bounds, and calculating that this step size needs one decimal place of precision for display.

I'm checking whether the max of 16 trips per lifetime makes sense — it works out to about +99% growth over 2019 levels by 2050, which is a plausible "business as usual" endpoint given historical growth rates, while the minimum of 2 represents a 75% reduction. Now I'm turning to short-haul flights and realizing I need a similarly tangible unit, weighing options like km per person per day or percentage of today's short-haul flying retained by 2050, and trying to land on something clean.

I try dividing total pkm by passenger count but the bases don't quite match cleanly, so that math is too shaky to rely on. I'll go with the safer approach: express short-haul flights as "return trips to a destination 1,000 km away," and I can anchor that with a real example like Brussels–Barcelona at 1,080 km.

I'll verify this distance computationally using published airport coordinates — it's checkable arithmetic, and Brussels–Barcelona fits nicely since it's the actual route being discussed for night trains. Now I'm wondering if "Brussels–Barcelona returns/lifetime" is too unwieldy as a unit string for the UI, so I should check how ui.yaml handles unit translation strings.

I'm planning to batch several file reads together — ui.yaml, the play/spark modules, the export verification script, and the levers file — to be efficient. I also want to kick off the notebook execution early since it's slow, but I realize my module needs to exist first before the export step will pick it up, so I should draft the module before running anything, then use nbconvert to execute the notebook and test it.

For faster iteration than re-running nbconvert each time, I'm considering executing the notebook's code cells directly in a Python namespace (skipping magic commands and the export cells themselves), keeping the state in-process so I can quickly re-call my module's build function without a full re-execution each time.

## reasoning

I'm noticing that the shared caveat string hard-codes "the other seven assumptions," which assumes 8 levers total — but my topic only has 7, so the correct phrasing would differ. Since the rules say I can't override a key already defined elsewhere, this looks like something I need to flag as an issue rather than fix myself, though I want to first check whether play.js generates this string dynamically.

## reasoning

I see that `historyNote` serves as the chart note when there's no series, while `historyAbsent` alone doesn't affect rendering — so the YAML needs both fields together. I'm also noting that `play.leverage.caveat` hard-codes "the other seven assumptions" as a static string, which is a real issue for a 7-lever topic that I should flag, suggesting the maintainer make it generic instead.

Now I want to set up a fast-iteration harness by executing the notebook's code cells, excluding the two export cells, into a namespace so I can test changes quickly.

## reasoning

Cell 44 only runs display code under a post_process flag, so I'll patch cell 6 to force post_process=False and skip that block entirely. I also noticed the macro notebook writes to website/data/macro.js via an export cell, which is a side effect I need to account for.

Since macro.js was already showing as modified in git status before I started, I'll snapshot its current content now so I can restore it exactly afterward, since generated files shouldn't count as deliverables but I want to avoid unnecessary churn. Time to set post_process=False.

## reasoning

Now I have the key figures: extra-EU passenger aviation drops 40% in energy use per person by 2050 while intra-EU falls over 53% due to compounded reductions, bringing total passenger aviation energy demand down from about 16.25 TWh to roughly 7.34 TWh.

For air freight, I'm checking whether it falls within my topic's scope. Looking at the numbers, the modal shares for intra- and extra-EU air freight stay essentially fixed across both years, and the energy intensity per tonne-km is identical in 2019 and 2050 — so the small 0.8% drop in total freight TWh is purely a population/tkm-per-person effect, not a genuine efficiency lever. That tells me air freight has no independent degree of freedom in this scenario; its only real driver is the general freight-transport-intensity assumption, which belongs to inland mobility rather than aviation specifically.

Since the inland-mobility module explicitly excludes intra/extra-EU planes and covers only domestic freight, air freight isn't claimed there — it falls under my topic instead, so I should include it in my scope. Now let me check the freight aviation cells directly.

## reasoning

I notice air freight parameters like fuel reduction and payload targets for intra/extra flights are all left at 1.00, meaning the scenario assumes no change for these DOFs even though they're defined as inputs.

Testing whether this could work as a lever: with target equal to reference (both 100% of 2019), the slider math technically checks out against the edge constraints, and "négaWatt assumes zero improvement in air freight" would actually be a striking reveal about model gaps. But it overlaps with the maritime bunkers point I'm already planning to make, so I need to decide if it's worth including alongside that.

Given the goal of showing 6-8 well-chosen levers rather than a long list, I'm now working through which seven to settle on.  Long-haul and short-haul flight demand, load factors and fuel efficiency for each, plus one more — deciding between air-freight energy and a hydrogen-flights lever as the seventh, since hydrogen's jump from 0% to 5% of intra-EU flights offers a compelling, unexpected slider too.

I'm computing the actual energy saved by hydrogen substitution and finding it's tiny — about 0.017 TWh, negligible compared to the 7.3 TWh topic total — which makes it a striking finding but not practical for a snappy slider question.

But I realize this misses something important: hydrogen at low shares barely shifts final energy demand but massively changes what fuel infrastructure has to be built, and that nuance can't be captured in a simple energy readout. Turning instead to air freight fuel, I calculate a swing of roughly 1.88 TWh across a plausible slider range — a much more meaningful lever to build a question around.

Actually payload-per-aircraft feels like a fiction since so much air freight rides in passenger bellies. The cleaner, more honest lever is energy per tonne-kilometre for long-haul air freight, computing 2019 at roughly 0.732 kWh/tkm and holding that flat for 2050 as the baseline.

I keep circling back to whether combining fuel reduction and payload into a single "% of 2019" slider is the right pattern — mirroring how inland car-energy levers work. I'm checking the slider range (55–130, with target at 100 giving an edge case of 0.40) and noting it's unusual for a lever where reference equals target, since that makes "better" default to "up" by convention.

I'll need to override the direction explicitly since lower energy is actually better here, and confirm the reveal logic and versusPlain phrasing handle this correctly. I'm also weighing whether framing this as "négaWatt didn't model air-freight energy intensity" makes for honest, useful workshop content — it's a real gap worth surfacing, and it pairs naturally with the maritime-bunkers caveat.

Now I'm debating whether to settle on 7 levers or 8, since 8 would let me use the "other seven assumptions" phrasing literally — though that alone feels like a weak justification. Given the workshop runs 30-40 minutes at roughly 4 minutes per lever, and the inland version already established 8 as the working shape, I'm leaning toward keeping 8 for consistency, while still weighing each candidate on whether it makes a genuinely good discussion question.

Going through the eight: long-haul and short-haul flights both stand out as strong, politically charged, high-leverage topics; short-haul load and fuel are solid with real trend data, the fuel one being counter-intuitive since négaWatt expects it to worsen; long-haul load mirrors the short-haul case, while long-haul fuel is notable as the only lever expecting real technical progress. Hydrogen flights are politically interesting but carry almost no energy leverage, and air freight exposes a modeling gap while still offering decent leverage — so hydrogen looks like the one worth dropping given its near-zero impact.

Rather than fully cutting hydrogen, though, I'm leaning toward keeping it as a hidden spare lever and folding the "technology won't save us" point into the caution notes on the fuel levers, where the SAF-absence critique already fits naturally. That preserves the insight while trimming a weak screen, landing on seven shown levers total — though I'm now second-guessing whether having four aircraft-technical levers is too many, even though they pair up logically by segment.

Actually, the symmetry between the two load-factor levers seems worth keeping, since it lets the reveal highlight négaWatt's asymmetric demands on long-haul versus short-haul fuel burn while asking both to fill more seats. I'm also reconsidering the freight lever, since its 2019-to-2050 target of no change raises an interesting question about how far freight's energy efficiency could actually fall.

Every group answering "lower" here means every group looks bolder than négaWatt, which makes for a pointed reveal and surfaces air freight's substantial energy footprint that would otherwise stay hidden — plus it's the natural place for the maritime caveat about freight and ports. I'm also noting a wrinkle: the reduction factor for aviation fuel only applies to extra-EU air freight, while intra-EU freight uses a separate factor, so I need to account for both pieces.

Since extra-EU dominates air freight's tonne-kilometres and energy use, I'll frame lever #8 as "long-haul air freight" and use the extra-EU energy-per-tonne-km reduction as the honest, natural label for it rather than the payload variable.

But now I'm second-guessing this — since that target exactly equals the reference with no underlying assumption, exposing it as a lever might feel strange, since the justification would just have to admit there's no real assumption behind it. I want to check how the reveal logic handles a case where target and reference are identical before committing to this.

Meanwhile I'm scanning the notebook for other aviation-related degrees of freedom to find one with a cleaner, more genuine assumption behind it — checking things like the aviation fuel reduction and shift variables, hydrogen adoption targets, and occupancy rates to see which ones actually carry real modeling assumptions worth exposing.

## reasoning

I'm settling on a final set of seven levers to show, starting with long-haul and short-haul flight frequency and passenger load per plane, and working through their corresponding parameter values.

Filling in the remaining levers, I list load and fuel targets for short-haul and freight, but pause on air-freight-fuel since its target equals its reference value. That makes me second-guess whether it belongs versus a hydrogen-flights lever — the brief flags aviation-only scope as a gap and notes SAF is missing from the model entirely, so a lever with no real change might be a weak choice since there'd be no meaningful justification for its value.

Comparing to `hydrogen-flights`, I find it has a genuinely argued négaWatt assumption with real numbers and sources, and its reveal — that technology contributes almost nothing and demand-side action matters most — is a much stronger takeaway. The live leverage readout there is near-zero everywhere, but the impact code handles that gracefully by showing "negligible effect" messaging, which actually reinforces the lesson rather than undercutting it, even though it does risk previewing the reveal slightly. Weighing both options, I lean toward picking one of these two.

I can also fold air freight into a structure fact on the long-haul-fuel lever, noting its energy use and constant efficiency assumption. So my final seven levers are the six plus hydrogen-flights, leaving the intra-EU rail-shift split and air-freight efficiency inputs hidden but still passing validation — checking the destination-split slider gives 90% and the FT-fuel slider edge case checks out at 0.43.

I'm reconsidering whether to add hidden levers at all — since they cost nothing and document degrees of freedom, I'll add two: the short-haul-to-rail split and extra-EU air-freight fuel, both unshown, verifying the 50-100 slider edge case works out to 0.2, and making sure the "better" direction is set correctly as "down" for the fuel lever despite target equaling reference.

Now I'm finalizing intra-EU trip numbers, using 2019 and 2050 pkm-per-person figures and computing the great-circle distance between Brussels and Barcelona airports from their coordinates to establish a reference round-trip distance, then dividing per-capita travel by that distance to get trips per year and scaling over an 80-year horizon — landing around 50 trips for 2019 and roughly 24 for 2050, which feels like a reasonable range for setting the slider bounds.

Since the true average intra-EU trip is likely shorter than 1,080 km, the actual number of trips would be higher — so I need to frame this explicitly as "equivalent" Brussels–Barcelona round trips rather than actual counted journeys, phrasing it as "exprimé en allers-retours Bruxelles–Barcelone (2 160 km), équivalent." I'll keep this reference distance and double-check it, then reconsider the lifetime framing once more for the short-haul comparison.

For consistency with the other lever, I want both expressed as lifetime totals rather than mixing annual and lifetime framing, though I realize the tangible field template only supports value, valuePerDay, and valuePerYear at runtime — so I can't dynamically render something like "one trip every N years" since there's no built-in divisor for that.

I'll settle on a fixed-form sentence using {value} directly, framing it as trips across an 80-year lifetime, which is clean and consistent. Then I recompute the long-haul-flights numbers at 18,000 km for 2019 and 2050, getting values around 8.1 and 4.8 on the slider, and I verify the edge position works out to roughly 0.2023 with a reference value of 8.

Now I need to trace through the lever mechanics for long-haul-flights—it's proportional, with the target value scaling extra-EU passenger travel to about 3.524 TWh in 2050, and I need to figure out what "total" means for this topic, likely the combined 2050 international-mobility demand across passenger aviation and air freight.

I'm checking whether air freight belongs in that total since it's part of international aviation—leaning toward including it, which puts the base around 10.87 TWh. But I realize the contribution formula subtracts the reference evaluation from the actual value, which cancels out the fixed and total terms for proportional and inverse lever types, so the total doesn't actually matter for computing contributions.

Now I'm working through the second lever, short-haul flights, which uses a linear-shift model where the kilometers removed from intra-EU air travel get redistributed across rail and coach modes based on fixed percentage splits.

I'm calculating the emissions intensity difference: one lever unit equals a return trip, converting that to per-person kilometers per year, then computing the weighted basket intensity across conventional rail, high-speed rail, and coach versus the plane intensity, and multiplying by population to get the total gigawatt-hour impact per unit.

This roughly matches: 9.1245 Gpkm of extra flying at 0.4736 minus what it displaces from rail/coach nets to about +3.64 TWh, aligning with the −26.8 × 0.13574 = −3.638 calculation.

But I need to be careful here — the reference value the lever uses is 50.5, the 2019 level, which is correct. Though there's a subtlety: the actual pre-shift 2050 baseline for intra-EU is 1279.86 pkm/person after the global reduction is applied, not 1363.92, so holding at the 2019 level isn't quite the right comparison.

That said, the lever value doesn't need to exactly equal `pro_PM_spe_avi_srt` — since it's defined on the outcome and that outcome maps one-to-one with the input variable, any slider position still resolves to a unique input value, just like how the inland car-share lever works on outcome share rather than directly on the underlying parameter. I do think the model variable documentation should still reference `pro_PM_spe_avi_srt` with a clarifying note about this mapping.

Now moving to the third lever, `long-haul-load`, which uses an inverse relationship with a target value of 219.746.  I'm working through the scaling: extra-EU passenger energy in 2050 comes to 3.524 TWh assuming all kerosene, against a reference of 187.817. Testing slider ranges from 150–260 with a step of 1 and no decimals, I check the edge case and it validates. I consider whether 260 seatload is realistic for long-haul aircraft — comparing typical widebody capacities, it seems like a plausible aggressive upper bound, so I decide to extend the range slightly to 150–270 for more headroom, rechecking the edge calculation to confirm it still holds.

Settling on a step of 5 instead of 1 gives a coarser but sufficient granularity of 24 positions, similar to the range used for the inland lever, and I like that it prevents participants from landing exactly on the reference target since that marker gets drawn separately anyway. With decimals set to 0, I move on to the fourth lever, long-haul fuel share, setting it up as a proportional lever with a target of 84% of 2019 levels and checking the reference and slider bounds for consistency.

Now I'm double-checking that "% of 2019" as a unit qualifies as tangible enough for the guidelines, since inland's car-energy lever already sets precedent for this format. I'm also weighing whether a more physically grounded unit like litres or kilograms of kerosene per km would be clearer, but that requires converting through kWh-per-km figures for extra-EU flights, which adds complexity I'm not sure is worth it.

Given the LHV uncertainty, maybe I should sidestep it entirely and use kWh per aircraft-kilometre instead — that's the notebook's own unit, needs no extra conversion constant, and is still tangible: a long-haul flight uses 67 kWh for every kilometre flown, comparable to an electric car's energy use over 400 km. 2019 comes to 67.278 kWh/km, dropping to 56.514 kWh/km by 2050.

Checking the slider setup at 40–90 with step 1, the edge calculation works out fine and it's proportional. For intra-EU flights the numbers actually rise from 69.056 to 72.508 kWh/km — with a 45–95 slider the edge also checks out — which makes for a genuine surprise: efficiency gains don't bring short-haul fuel use down by 2050, négaWatt actually pushes it up 5%. I'm weighing whether kWh/km is intuitive enough for a non-expert, though framing it as "a short-haul flight burns about 69 kWh per kilometre" helps make it tangible.

I'm settling on using kWh per km flown for both fuel levers since it's more concrete than "% of 2019" and makes the intra/extra-EU comparison direct — 69 vs 67 kWh/km per aircraft-km are nearly identical, meaning the real difference comes down to occupancy rather than aircraft efficiency.

Now I need to work through a subtlety: for intra-EU flights, the fuel-efficiency lever only affects the kerosene-powered portion of the fleet (69.056 → 72.508 kWh/km), since hydrogen aircraft stay fixed at 43.333 kWh/km. So the scaled impact value has to be based on just the kerosene share of intra-EU 2050 TWh — 3.645, not the full 3.819 — and I want to verify that figure lines up correctly before locking it in.

I calculate the kerosene TWh directly from Gpkm × consumption/occupancy and get 3.6446, confirming it matches. The proportional relationship to consumption and inverse relationship to occupancy both check out, so for the short-haul-load lever the scaled value should be 3.6446, with vTarget 152.409 and ref 121.927.

Working through the slider bounds for load factor, I settle on 85 to 190 as the range so it still captures the 2000 value, and the edge computation comes out to 0.358, which checks out. I also weigh whether 190 passengers per average flight is realistic given typical A320neo/A321neo seating, and decide it's a reasonable upper bound assuming larger aircraft.

Now moving to the short-haul-fuel lever, which is proportional with vTarget 72.508 and ref 69.

056—I catch that since the target exceeds the reference, the direction would default to "up," but that's backwards since lower fuel use is actually more ambitious. I need to explicitly pass better="down" for this one, and I should apply the same fix to long-haul-fuel where the target is below the reference.

For the hydrogen-flights lever, I'm testing slider ranges to make sure the edge case ratio clears the 0.12 threshold — trying 0–40, 0–35, and 0–30 configurations, since a 0–50 range fails the check. Settling on 0–30 with step 1 seems reasonable, since some groups might argue for very ambitious hydrogen adoption targets.

But I realize this exposes a deeper tension: when the target sits near the low end of a plausible range, the slider bounds themselves start hinting at whether négaWatt's stance is weak, even when the guard condition technically passes. That's pushing me to reconsider swapping this lever for `air-freight-fuel` instead, where the target sits more comfortably within a wider range.

Weighing it out: `air-freight-fuel` has real leverage — a meaningful energy swing, a comfortable slider position, and an honest, pointed reveal about the scenario's assumption gap, plus it brings a third of the topic's freight volume into view. There's also a third possibility worth considering: a shift-destination lever around rail's share of freed short-haul traffic, with its own slider range to check against the edge guard.

I ran the numbers on that rail-share lever though — comparing the rail energy basket against coach across the freed traffic pool — and the swing comes out to something like 0.045 TWh, which is essentially nothing, so I'm dropping it as a shown lever and keeping it hidden. That settles #7 as `air-freight-fuel`, shown.

Now `hydrogen-flights` should also become shown=False, with its message folded into the facts and the fuel-lever debate. But I want to double-check an edge case: what happens when the reference value equals the target value? I need to trace through how the lever direction logic, the verdict rendering, and especially the spoiler check handle that — since if ref and target are both 100, any fact text mentioning "100%" could accidentally trip the spoiler filter since it would match the rendered target value.

Since the runtime placeholder stays unresolved in the built text, I could technically avoid triggering the spoiler check just by not writing "100" anywhere in the prose, but that feels fragile. A cleaner fix is to reframe `air-freight-fuel` using kWh per tonne-kilometre instead of "% of 2019", so I'm working out the 2019 baseline conversion using the kgoe-to-kWh factor.

This is a real design tension for any lever where target equals ref. Let me reconsider hydrogen-flights: ref 0, target 5, decimals 0, which is easier to avoid quoting standalone but has negligible leverage anyway. Maybe six shown levers is actually the right count to settle on.

Six feels right: the two demand levers plus the four aircraft levers (load × fuel for each haul type) form a coherent set covering how much we fly, how full the plane is, and how thirsty it is — each with real argued value and comfortable slider ranges, while gaps like SAF, hydrogen, and freight get covered through caution facts and debate text as the brief wants. So I'll settle on six shown levers plus a few shown=False spares.

I notice the leverage caveat text says "the other seven assumptions" which should read "five" with six levers — that's a bug worth flagging, best fixed by making it generic rather than hardcoding a number. I should also make sure my own intro text says "six" not "eight" since I control that copy directly.

Now I want to precisely compute the numbers with a script including great-circle distances, and pin down 2019 values, slider ranges for the two demand levers, and external benchmark facts. Looking back at the long-haul-load lever, the notebook comment shows 2000 at 154.0 passengers and 2023 at 205.5 — a +33.7% increase — with 2019 at 187.8. The target of 219.7 is only +6.9% above the observed 2023 figure, which makes for a weak "objective" framing and a good debate point.

Similarly for intra-EU load: 2000 was 87.5, 2019 121.9, 2023 130.8, and the target of 152.4 is +16.5% over 2023 — more debate material.

For fuel, intra-EU went from 578.8 kgoe/100km in 2000 to 633.0 in 2023 (+9.4%), while the model's 2019 value is 593.771 kgoe/100km and the target (×1.05) comes out to 623.5 kgoe/100km — actually below the observed 2023 figure. So négaWatt's 2050 intra-EU fuel-per-aircraft-km target is technically an improvement on 2023 despite being framed as +5% on 2019, which is a nice nuance for the reveal. I'm converting these kgoe/100km figures into kWh/km for consistency.

Extra-EU fuel tells a different story though: 2000 was 937.0 kgoe/100km (109.0 kWh/km), 2023 dropped to 629.5 (73.2 kWh/km, a −32.8% decline per the notebook), but the model's 2019 value is only 578.489 kgoe/100km — meaning 2023 actually came in higher than 2019. That's a real inconsistency worth flagging.

I'll convert all these kgoe/100km figures using the notebook's kgoe-to-kWh factor of 11.63, and declare them as source-commented `ref_*` constants at the top of build(), matching the pattern used in inland_mobility.py. Now I need to gather at least one comparable benchmark fact from elsewhere in Europe for each lever.

For long-haul flights, I'm thinking of citing the Gössling & Humpel finding that 1% of the world's population accounts for 50% of aviation emissions as a strong equity data point, plus benchmarks like France's 2023 domestic flight ban or Sweden's flygskam movement for the international comparison.

For short-haul flights, I want Eurostat data comparing rail vs air modal share, or the share of intra-EU routes under 750km with viable rail alternatives, and Austria's minimum ticket pricing as a policy example.

For load factors, I'm considering IATA's 2019 average load factor of about 82%, cross-referenced with Airbus's published seating capacity for an A320neo, to estimate typical passengers per flight.

For fuel efficiency figures, I could pull from the EEA/EASA European Aviation Environmental Report which tracks fuel burn per passenger-kilometer trends over time.

Given the amount of ground to cover, I'm planning to read through the Eurostat benchmark fetching script first, then run targeted web searches to verify facts, compute the final figures, and write the module and YAML config. I'll launch a subagent to gather and verify the external benchmark data in parallel while I build out the module myself.

Before dispatching that subagent, I want to read fetch_eurostat_benchmarks.py so I can brief it properly on what's already available.
