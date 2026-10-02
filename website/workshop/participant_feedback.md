# Participant feedback — workshop notes

Collected from the free-text note field under each slider (`ws_answers.condition_text`),
read through `api/results.php` on 2026-10-02 for all four topics, with no date filter.

Every note comes from the **22 September 2026** sitting. Each topic got its notes from a
single group: *The Filippekes* (inland mobility), *Group 1* (international mobility) and
*Chaleur logement* (residential heat). Tertiary heat has no notes.

Each note is tagged:
- **[interface]**: criticism of how the question or screen is put to participants
- **[information]**: a fact the group missed when answering
- **[measure]**: a policy or behavioural idea, i.e. content rather than criticism

The group's answer is shown in brackets.

---

## Inland mobility (group "The Filippekes", 7 notes on 8 questions)

| Question | Answer | Note | Tag |
|---|---|---|---|
| `ground-km-day`: daily motorised travel | 28.5 km/day | "What is the tendency for teleworking? We lack information: what are these kilometres for? (work? leisure? education? holidays? just random driving around?)" | information |
| `car-share`: car share of motorised km | 51 % | "Very dependent on urban planning; goes slow to change it." | measure |
| `car-energy`: car energy per km vs 2019 | 73 % | "Hard to think about the 3 factors at the same time, and come up with a number." | interface |
| `bike-km-day`: cycling level | 5 km/day | "Would it also make more sense to split: kilometres versus trip frequency?" | interface |
| `freight-tkm`: freight demand | 5400 tkm/pers/yr | "More borrowing from neighbours, less (online) shopping. Question: how many of these tonne-km are transporting goods that we need for survival / a good living standard?" | measure + information |
| `truck-share`: road share of tkm | 65 % | "How much capacity does our rail network have? How much can we shift onto it?" | information |
| `truck-load`: average truck load | 14.1 t | "Is the main question here 'can we / how to reduce empty runs'? Would be good to make the question sharper." | interface |

No note on `car-occupancy`.

**What follows from these notes**
- `car-energy` asks the group to combine three factors in one number. Split the question, or show the factors as separate anchors with a worked example.
- `truck-load` should be sharper: say plainly whether empty runs are the main lever.
- `ground-km-day` needs a breakdown of km by purpose (work, leisure, education, holidays) and a fact on telework.
- `freight-tkm` needs an information card on what the tonne-km carry: what kinds of goods, and which are essential.
- `truck-share` needs a fact on spare rail capacity.
- `bike-km-day`: consider splitting distance from trip frequency, or at least giving both as anchors.

## International mobility (group "Group 1", 4 notes on 7 questions)

| Question | Answer | Note | Tag |
|---|---|---|---|
| `long-haul-flights`: long-haul trips per lifetime | 3 | "The metric of flights/lifetime is hard to grasp. Perhaps /5 years." | interface |
| `short-haul-flights`: short-haul trips per lifetime | 10 | "Investments in rail, progressive taxes for flying, cap number of flights per person, stop airports expansion." | measure |
| `hydrogen-flights`: share of intra-EU km on hydrogen | 5 % | "If China does it, it will come…" | measure |
| `air-freight`: air freight per inhabitant | 60 tkm/inh/yr | "Missing now: the tonnes of what kinds of goods (pharma vs flowers vs…)." | information |

No notes on `long-haul-load`, `short-haul-load` or `plane-fuel`.

**What follows from these notes**
- The per-lifetime unit is hard to grasp. Restate it per 5 or 10 years in the `tangible` line (or the unit itself), and apply the same change to `short-haul-flights`.
- `air-freight` needs a breakdown of air freight by type of goods.

## Residential heat (group "Chaleur logement", 2 notes on 8 questions, in French)

| Question | Answer | Note | Tag |
|---|---|---|---|
| `floor-area`: floor area per person | 50 m² | "Mesures politique : créer un statut colocataire." *(Policy: create a legal status for flatmates.)* | measure |
| `renovation-rate`: share of stock renovated per year | 4.6 %/yr | "Mesure politique : imposer un minimum PEB pour pouvoir louer ET plafonner les loyers." *(Policy: require a minimum EPC to rent out a home AND cap rents.)* | measure |

Neither note criticises the interface. The group used the field for policy measures, and
other groups may do the same; the reveal could show these notes next to the group dots.

## Tertiary heat

No notes. The tertiary topic has no stored answers at all (4 groups, 1 October 2026).

---

## Data caveat

On **1 October 2026**, 15 groups were created across the four topics (Groep 1–5), and
**none of them has a single stored answer**. A group's `created_at` equals its `updated_at`,
so nothing was ever saved for it. Either that sitting used the paper cards only, or the
answers did not reach the server. Worth checking before relying on the database for that day.

Test groups ("Test 1 Arthur B", "ced", the 2 October "Group 1") have answers but no notes.
