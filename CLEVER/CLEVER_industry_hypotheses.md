# CLEVER Industry Decarbonisation Hypotheses — Structured Reference

> Source: negaWatt / CLEVER, *Establishment of energy consumption convergence corridors to 2050 — Industrial sector* (June 2022), file `2206-Convergence-corridors-Industry.md`.
> Cross-checked against `CLEVER_final-report.md` for feedstock / carrier figures (noted where used).
> This file is a faithful, quantitative extraction. All numeric values preserved from the source. Inferences or unclear points are flagged with `> NOTE:`.

---

## 0. Core methodology (the 3-step logic and FEC definition)

For each industrial **branch**, the CLEVER vision builds the 2050 corridor in 3 steps:

1. **Sufficiency** — scale material demand down (adjust nature and amount of demand to deliver the service with minimum material). Directly reduces **production** and hence energy consumption.
2. **Circularity** — optimise product lifecycle: more durable design + longer use (reduce material demand → reduce production) + higher recycling rates (shift from raw to recycled materials, generally **less** energy intensive). Affects both **production** and **energy intensity**.
3. **Efficiency** — reduce **energy intensity** of production through new technologies and fuel/material substitution.

**Two levels of analysis** (see Table 2):

- **Detailed** (heavy sectors: cement, steel, pulp & paper, chemicals [ammonia, HVC, other], glass): build a **production corridor** (index 2050, % of 2015) AND an **energy-intensity corridor** (MWh/kt). Their product gives the **FEC corridor**.
  - **FEC = Production (kt) × Energy intensity (MWh/kt)**.
- **Simplified** (lighter sectors: food, non-ferrous metals, "others", and chemicals where subsector data is missing): build a **direct FEC index corridor** (% of 2015).

Two corridor families exist in CLEVER: **energy consumption corridors** (this note) and **energy carrier corridors** (CO₂ intensity of energy use; available on demand, not in this note).

---

## 1. Table 2 — Summary table of proposed corridors

Detailed approach = production corridor + intensity corridor; simplified = direct FEC corridor.

| Industrial sector | Production (index 2050, % of 2015 value) — sufficiency + circularity | Energy intensity (MWh/kt) — circularity + efficiency | FEC (index 2050, % of 2015 value) |
|---|---|---|---|
| **Cement** (detailed) | 52 – 99 | 560 – 800 | 31 – 64 |
| **Steel** (detailed) | 74 – 92 | 2060* – 2690* | 42* – 52* |
| **Pulp & paper** (detailed, indicative) | 58 – 110 | 1890* – 3780* | 31* – 64* |
| **Chemicals** (aggregated / simplified) | — | — | 70 – 75 |
| **Chemicals – Ammonia** (detailed) | 58 – 80 | 1580 – 2500 | — |
| **Chemicals – HVC** (detailed) | 59 – 98 | 3140 – 5680 | — |
| **Chemicals – Others** (detailed/simplified) | — | — | 69 – 89 |
| **Glass** (detailed) | 61 – 95 | 700 – 2190 | 23 – 68 |
| **Food** (simplified) | — | — | 42 – 64 |
| **Non-ferrous metals** (simplified) | — | — | 39 – 87 |
| **Others** (simplified) | — | — | 63 – 85 |

`*` corridors with exceptions for some countries (see sector disclaimers — depend on national recycled share / P/C ratio).

> NOTE: The glass intensity lower bound is given as `700` in Table 2 but `697` / `690` in the glass section text; cement intensity Table-2 lower bound is `560` but the cement text cites efficient values "around 600 MWh/kt". Pulp & paper intensity upper bound is `3780` in Table 2 but `3880`/`3888` ("3.888") in text. These are rounding/typo discrepancies in the source.

---

## 2. Table 1 — Baseline scenarios used

| Organism | Scenario name | Scale | Reference label |
|---|---|---|---|
| EU Calculator | Module on industry: key behaviours pathway | Europe | **EUCALC** |
| Climact, ECF, ClimateWorks | EU CTI 2050 — Shared effort scenario | Europe | **EU CTI 2050** |
| Fraunhofer ISI, ICF | Pathways to deep decarbonisation of Industry | Europe | **FhISI** |
| Material Economics | Industrial Transformation 2050 | Europe | **Material Economics** |
| negaWatt association | negaWatt 2022 scenario | France | **negaWatt** |
| ReINVENT Decarbonisation | Climate innovation in the (paper) industry: demand scenario | Germany | **ReINVENT** |
| Umwelt Bundesamt | Resource-Efficient Pathways towards GHG-Neutrality (RESCUE), GreenSupreme scenario | Germany | **RESCUE** |

Other sources cited: **Dechema** (2017, low-carbon energy & feedstock for chemical industry), **IEA** (2013 chemical roadmap), **Odyssee** database (Enerdata, EU energy consumption), **MODEIRE/Pepito** (French project, route-specific energy intensities), **Toktarota et al. 2020** (steel TRL).

> NOTE: ReINVENT is labelled "Germany" in Table 1, but in the pulp & paper section it is described as a Demand-Management scenario for the paper industry; treat ReINVENT as the paper-industry demand scenario.

---

## 3. Sector-by-sector hypotheses

### 3.1 Cement (detailed)

**Overview.** Clinker (produced at ~1450 °C) drives most energy use and CO₂. EU28 demand >182 Mt (2019). EU ≈ 5% of world cement; cement is the **largest industrial GHG-emitting sector in Europe**. Main consuming sectors: new buildings (50%), civil engineering / new infrastructure (30%), maintenance (20%).

**Corridor.** FEC 2050 = **31% (FhISI) – 64% (RESCUE & negaWatt)** of 2015. Material Economics ~mid-corridor; EU CTI 2050 more ambitious (−84%, excluded from corridor — different P/C ratio 0.8).
- P/C ratios (2017): Germany 1.18, France 1.02, Europe 1.1; assumed constant to 2050.

**Production corridor: 52 – 99 (index 2050).** Sufficiency + circularity = **38% to 48% reduction** of cement demand.
- Sufficiency drivers: demographic change (lower cement/capita), wood & carbon-concrete construction, more cohabitation → fewer dwellings, lower tertiary/industrial building growth, fewer new engineering structures (favour renovation), **reduction of road network**. Infrastructure trends → 30–70% cut in EU cement demand by 2050.
- Material efficiency: cement losses ≈ 15% of building materials wasted; smarter design / less over-specification / end-to-end optimisation → Material Economics finds **up to 65% less cementitious material**.
- Material Economics −38%, negaWatt −48%, RESCUE −48%; FhISI low reduction (renovation + infrastructure investment).

**Circularity specifics:** concrete recycling **14% to 65%**.
- Cement not easily re-melted; recovery via cement-fines recycling + reuse of structural elements → less clinker / polymer cement demand.
- Recycling rate ~5% (2015) → FhISI 14% (2050), EU CTI 2050 34%, Material Economics up to 65%.

**Energy-intensity corridor: 560 – 800 MWh/kt.** Efficiency reduces intensity **7% to 30%**.
- RESCUE & negaWatt ≈ 600 MWh/kt by 2050; FhISI ≈ 650 MWh/kt; Material Economics 804 MWh/kt (upper bound).
- **Technology/innovation: 4–18% intensity reduction.** Wet clinker → dry clinker; polymer cement (~10% of production), low-carbon / re-carbonating cements (lower process temps & thermal demand). FhISI: innovative cements substitute ~50% of production by 2050.
- **Material & fuel substitution: 3–12% intensity reduction.** Clinker substitutes (GGBS, PFA, Pozzolana, limestone). EU CTI 2050: polymer cement leaves only 66% clinker; RESCUE: unburnt limestone additives. In buildings: concrete substituted 10–40% by timber (cross-laminated) + 10% by insulation (HVC); in infrastructure 2.5% by insulation (HVC).
- **Fuel switch:** EU CTI 2050 — 46% of fossil fuels → biomass; RESCUE GreenSupreme — coal abandoned by 2040, renewable gas + electricity; thermal efficiency +10% via waste heat / efficient kilns.

---

### 3.2 Steel (detailed)

**Overview.** EU production 153 Mt (2020), >500 sites, €125 bn GVA/yr. Traditional route: iron-ore reduction with coke in blast furnaces (BF-BOF). Recycled (scrap) steel = **41% of EU28 production (2019)**. Recycled route is **3–4× less energy intensive** than primary.
- Route intensities (MODEIRE): **Primary** 5000 MWh/kt (2015) → 4060 MWh/kt (2050); **Recycled** 1500 MWh/kt (2015) → 1020 MWh/kt (2050).
- Main consuming sectors (2015, Eurofer): construction & infrastructure 35%, automotive 20%, mechanical engineering 15%.

**Corridor caveat.** Built for countries near EU-average recycled rate (**39%**, i.e. 30–50%: France, Germany, **Belgium**, Sweden, Poland…). Higher (Italy, Spain) or lower (Netherlands, UK) recycled rates handled separately for intensity & FEC, but same production assumptions.

**FEC corridor (recycled rate 30–50%): 42% (Material Economics) – 52% (negaWatt)** of 2015. FhISI & RESCUE mid-corridor; EU CTI 2050 −72% (excluded, P/C 0.78). P/C ratios (2015): Germany ~1.06, France ~1.03, Europe ~1.0.

**Production corridor: 74 – 92 (index 2050).** Sufficiency + circularity = **8% to 25% reduction** (FhISI −8%, negaWatt −26%, RESCUE −25%, Material Economics −16%).
- Construction: changing building surface + wood penetration (timber substitutes ~10%); building lifetime +40% via modularity; less over-specification cuts steel 20–30%; waste reduced ~5% (today 15–50% wasted).
- Transport: need falls 23% (negaWatt) – 33% (Material Economics); modal shift + car-sharing (63% of cars shared); Material Economics: steel in transport could fall 75% (33% lighter vehicles, 15% remanufacturing, occupancy 1.93/car, +94% car lifetime). High-strength steel cuts vehicle weight 25–39%. Substitution: ~10% steel → carbon fibre (HVC) and ~10% → aluminium for cars/trucks; 25% → carbon fibre for planes.

**Energy-intensity corridor (recycled 30–50%): 2060 – 2690 MWh/kt.** Efficiency + circularity. negaWatt & RESCUE 2690; FhISI & Material Economics 2060. Split: primary route −15% to −30%; secondary route −15% to −45%.
- **Circularity — recycled (EAF) share rises to 50–77%** (vs 40% in 2015). Primary route (BF-BOF / H-DRI) falls to **23–50%** of crude steel (vs 60% in 2015). EAF = 100% scrap; ≤25% scrap added in primary oxygen converter. Scrap availability grows (up to 2/3 of new steel).
- **Technology & fuel substitution: 20–30% intensity reduction.** 2015 mix: ~60% BF-BOF (coal), ~40% EAF (electricity). Key levers: replace heat generation + ore reduction. **DRI** (only 0.4% EU in 2015) via **H-DRI** (hydrogen-based direct reduction, from NG or electrolysis water) and **DR electrolysis** (pure electricity); plus **HIsarna** (early stage). Scenario specifics:
  - EU CTI 2050: HIsarna replaces ~10% of BF-BOF by 2050.
  - Material Economics: ~35% of primary steel via H-DRI by 2050.
  - FhISI (4aMix80): 80% of conventional BF production substituted by H-DRI + electrolysis steel (available after 2030).
  - RESCUE & negaWatt: BF route **completely replaced** by H-DRI; RESCUE achieves changeover by **2040**.
- **Material & fuel substitution: 10–15% intensity reduction.** Remaining coal: EU CTI 2050 substitutes 2.5% of coke by gas; charcoal (biomass) replaces 10–15% of coal in BOF plants by 2050.
- **Hydrogen note:** energy for hydrogen production in the steel process **is included in the energy-intensity** figure.

---

### 3.3 Pulp & Paper Industry (PPI) (detailed, indicative)

**Overview.** Mature, stagnating demand, high recycling. Virgin wood pulp = mechanical (28% of EU virgin pulp, weaker papers) + chemical (72%, high-quality). EU is net paper exporter; production = 46% virgin + 54% recovered paper.
- Route intensities (MODEIRE): **Primary pulp** 5400 MWh/kt (2015) → 3300 (2050); **Recycled pulp** 460 MWh/kt (2015) → 280 (2050). Recycled pulp ≈ 10× less energy intensive than primary.
- Primary-pulp/paper ratio varies <40% (France, Germany, Belgium) to >90% (Sweden, Finland) → corridor is **indicative**.

**FEC corridor (indicative): 31% (FhISI) – 64% (RESCUE)** of 2015. negaWatt (France) & ReINVENT (Demand-Management) ~mid. P/C ratios (2015): Germany ~1.14, EU ~1.22, France 0.90 (assumed constant to 2050). All countries use the same production assumptions.

**Production corridor: 58 – 110 (index 2050).** Sufficiency + circularity = **12% to 42% reduction** (negaWatt −12%, RESCUE −42%, ReINVENT −32%; **FhISI +10%** growth from e-commerce).
- 2016 paper uses: packaging 50%, graphic 37%, sanitary/household 8%, special 5%.
- Packaging paper **+12% to +30%** by 2050 (digitalisation logistics + substitution of plastic packaging); graphic paper **−20% to −49%** (digital replacement, end of abusive advertising). Packaging:graphic ratio → 2:1 by 2050. Sanitary/special +1% to +5%.
- Material efficiency (lightweighting, composition): **8% to 17%** saving.

**Energy-intensity corridor (indicative): 1890 – 3780 MWh/kt.** Efficiency + circularity = **11% to 53% reduction**.
- FhISI, negaWatt, RESCUE: 2920 → **1890 MWh/kt**; ReINVENT lower gain at **3880 MWh/kt** (Table 2: 3780).
- Drying = up to 70% of fossil energy use in EU PPI.
- **Circularity — paper recycling rate 62% → 80%.** Paper recyclable 4–8 times; EU recycling 62% (2016); recovered-paper share 54% (2015). Recycled paper ≈ 2× less energy intensive (3.11 TWh vs 6.04 TWh). 80% target ≈ fibres used ~5× (ReINVENT, aligned with negaWatt & RESCUE). negaWatt keeps recovered-paper share constant; FhISI & RESCUE +10% to +20% recovered-fibre share.
- **Technology & innovation: 5–40% intensity reduction.** Impulse / steam-air impingement drying (~10% by 2050, market ~2025); black-liquor gasification + green-liquor reuse (~5%); enzymatic pre-treatment for mechanical pulp (~20%); **deep eutectic solvents** (up to 40%, market ~2030–2035); waste-heat recovery + heat pumps.
- **Fuel substitution: 6–13% intensity reduction.** 2015 mix (Odyssee): biomass 37%, electricity 31%, natural gas 20%, liquid/solid fossil 12%. Fossils phased out / very low by 2050 (2030 for ReINVENT). FhISI & negaWatt: **electricity >40%** of FEC; ReINVENT & RESCUE: modern biofuels (black liquor) first carrier ~38%. Sludge biogas can supply 5–10% at recovered-fibre mills.

---

### 3.4 Chemicals — overview & aggregated corridor

**Overview.** Olefins, aromatics, methanol and ammonia ≈ 50% of global chemicals energy demand (IEA top-18). EU production volumes: **HVC ~40%, ammonia ~10%, rest of chemicals ~50%**. In 2015 chemicals = EU's **most energy-intensive** sector and **3rd-largest emitter**.

**Methodology.** Detailed corridors for **ammonia** and **HVC** (production + intensity) where data allows; **other chemicals** treated separately; an **aggregated** chemicals corridor built by combining the detailed analyses when subsector data is lacking.

**Aggregated FEC corridor: 70% (FhISI: 75.0) – 75%** … `> NOTE: source text states "between 75.0 for FhISI and 70.3 for negaWatt", so the corridor is FEC 2050 = 70.3% (negaWatt) to 75.0% (FhISI) of 2015.` RESCUE ~mid-corridor.

---

### 3.5 Chemicals: Ammonia (detailed)

**Overview.** EU produced ~17.2 Mt (2015), consumption ~17.5 Mt; 42 plants across 17 countries. Top producers: Germany 17%, Poland 16%, Netherlands 13% of EU capacity. ~82% of global ammonia → fertilisers (nitrogen 72%, potassium 16%, phosphorous 12%; fertiliser mix: 46% (ammo)nitrates, 22% urea, 13% UAN). Remaining 18% → industrial applications.
- P/C ratio (2015): France ~0.93, EU ~0.98 (assumed constant to 2050).

**Production corridor: 58 – 80 (index 2050).** Sufficiency + circularity = **20% to 32% reduction**.
- FhISI −20% (index 80), negaWatt −26%… `> NOTE: text says negaWatt and Material Economics reduce 26% to 42%; the production index lower bound 58 corresponds to ~−42% (Material Economics/negaWatt). Reductions of "20% for FhISI to 42% for negaWatt" are also stated later.`
- **Carrier switch in production:** negaWatt & Material Economics → **100% hydrogen-based ammonia** by 2050 (vs methane today); FhISI → **74% hydrogen, 26% methane-based**.
- Demand drivers: food-waste reduction (90 Mt/yr wasted in EU; Material Economics −70% by 2050); **synthetic fertiliser reduction**: FhISI −40%, Material Economics −45%, negaWatt −50%. Agro-ecology (legume rotations fixing N, inter-crop cover), shift to organic fertilisers.

**Energy-intensity corridor: 1580 – 2500 MWh/kt** (excluding feedstock). negaWatt ~**1580 MWh/kt** by 2050 (low due to hydrogen switch), confirmed by Dechema hydrogen-based trajectory; methane-based BPT also improves but remains more emitting. **Including feedstock, intensity increases** (see Annex 2).
- **Innovation / material substitution:** fertiliser efficiency + precision agriculture → up to **10%** intensity reduction.
- **Technology & fuel substitution:** Haber-Bosch (H₂ + N₂); H₂ usually from steam methane reforming (NG most common EU feedstock). Low-carbon H₂ via **water electrolysis**: alkaline (mature) and PEM. Per FhISI, electrolysis uses 11% less energy than steam cracking → **8% reduction** of total process; Solid Electrolyte membrane electrolysis → 12% better efficiency → **9% chain saving** (market entry 2025). If all EU plants matched best-plant efficiency, energy could fall **20%**.
- **Energy balance (Material Economics / Dechema):** today's process = **8.9 MWh NG (fuel + feedstock) + 2.1 MWh electricity** per tonne ammonia; electrolysis route ≈ **9.1 MWh electricity** per tonne. Biogas can also replace NG as fuel + feedstock.

---

### 3.6 Chemicals: High Value Chemicals (HVC) (detailed)

**Overview.** Plastics produced via **steam cracking of naphtha & ethane** (naphtha ≈ ¾ of EU feedstock). HVC = olefins (ethylene, propylene, butadiene) + aromatics (benzene, toluene, xylene = BTX). HVC = **>60% of chemicals-industry energy consumption**. 5 main polymers (PE, PP, PS, PVC, PET) ≈ 75% of EU use.
- EU plastics production ~60 Mt, consumption ~51 Mt (2015). Applications: packaging 40%, building/construction 20%, automotive 10%, electrical/electronic 6%.
- EU HVC production ~55 Mt (Dechema): ethylene 22 Mt, propylene 17 Mt, BTX 16 Mt. Methanol low (~2.5 Mt, FhISI) but rising (MTO/MTA).
- P/C ratio (2015): France ~1.32, EU ~1.33 (assumed constant to 2050).

**Production corridor: 59 – 98 (index 2050).** Sufficiency + circularity = **2% to 17% reduction**.
- FhISI −2% (index 98), negaWatt ~−40% (index ~59/60).
- **Carrier switch in production:** negaWatt → **60% of production switched to hydrogen via methanol** (MTO/MTA) by 2050 (vs naphtha/ethane); FhISI → **79% hydrogen via methanol, 21% naphtha/ethane**.
- Demand drivers: ~40% of plastics "single-use"; reuse of bags/bottles; car-sharing cuts materials up to 50%; biggest potential in B2B packaging. Plastics production drop **−23% (Material Economics) to −29% (negaWatt)**. Material-efficient design cuts packaging mass 20%. Fibre-based substitution: up to 20% of packaging plastics; ≥5% aggregate for bio-composites.

**Energy-intensity corridor: 3140 – 5680 MWh/kt** (excluding feedstock). negaWatt ~**4470 MWh/kt** by 2050 — **higher** than 2015 because hydrogen (MTO/MTA) route is more energy intensive; including feedstock it increases further (Annex 2).
- **Innovation/technologies:** BAT on conventional ethylene cracking → ~2% efficiency; catalytic cracking / pyrolysis / partial oxidation → ~15% savings (demonstration). Plastics recycling techs (pyrolysis, gasification, depolymerisation, solvolysis) — low energy-efficiency potential.
- **Fuel substitution:** major switch = **gasification → methanol from water electrolysis (MTO / MTA)**. negaWatt 60% switch, FhISI 79%. **Hydrogen feedstock demand is very large**: per Dechema ~**21.7 GWh_elec/kt olefin** and **41 GWh_elec/kt BTX**; negaWatt France needs **83 TWh electricity by 2050** for hydrogen-based HVC feedstock (a major limit). EUCALC: electrification up to 40%. Biomass switch 10–20% (EUCALC, Material Economics): bio-ethanol/methanol/biogas/bio-naphtha.
- **Circularity:** plastics recycling could cut intensity up to **10%**. ~30% of plastic waste collected; actual recycling <10%. Collection rate → 50% by 2050 (−8% raw ethylene per FhISI). Mechanical recycling 15–26% by 2050; chemical recycling emerging; together recirculation **50% to 62%** of production.

---

### 3.7 Chemicals: Other Chemicals (detailed/simplified)

**Definition.** All chemicals except ammonia and HVC — consumer chemicals (paints, inks, varnishes, glues, explosives, solvents, pharmaceuticals) and specialty chemicals (soaps, detergents). ~50% of chemicals-industry FEC. P/C ratio (2015): France 1.08, EU 1.10.

**FEC corridor: 69% – 89%** of 2015.
- **No sufficiency** assumed (heterogeneous, low share): per-capita volumes constant; production evolves with EU population only.
- Based on negaWatt subsector FEC reductions (without sufficiency), Figure 19: examples −29%, −30%, −31%, −57%, −33%, −11%, −30%, −25/26%, −33%, −32%, −31%.
- Corridor selection rule:
  - French-like distribution → **69%** of 2015 FEC (negaWatt aggregate).
  - Styrene/MVC/Nylon salt larger share than France → **75%** minimum.
  - Nylon & salt largest subsector → **89%** minimum.
  - Lack of data → **89%** minimum.

---

### 3.8 Glass (detailed)

**Overview.** Products: packaging glass, flat glass, utility/special/crystal/commercial glass, mineral & textile glass fibres. Energy-intensive; lower GHG weight. Main GHG levers: more cullet use + waste-heat recovery + electric melting.
- Route intensities (MODEIRE/ESvidrio): **Primary glass** 3500 MWh/kt (2015) → 2000 (2050); **Recycled glass** 2500 MWh/kt (2015) → 1300 (2050); recycled ≈ 30% less intensive.

**FEC corridor: 23.8% (negaWatt) – 68.2% (FhISI)** of 2015 (RESCUE 27.0). Table 2: 23 – 68.

**Production corridor: 61 – 95 (index 2050).** Sufficiency + circularity = **5% to 39% reduction** (FhISI 95.0, RESCUE 84.3, negaWatt 61.1).
- Cullet share ~40% today → RESCUE 45% (2030), 54% (2040), 69%; negaWatt recycled share 41% (2014) → **63% (2050)**.
- FhISI: only slight decrease (material efficiency, bio-fibre substitution, reuse).

**Energy-intensity corridor: 697 – 2190 MWh/kt** (Table 2: 700 – 2190). negaWatt & RESCUE down to **690–1330 MWh/kt** (RESCUE 1326); FhISI **2193 MWh/kt** (upper bound).
- **Electrification of furnaces** is the key lever. RESCUE GreenSupreme: −80% specific thermal demand vs 2010, switch to electric furnaces; from 2030 no new oil furnaces; fully-electric baths: 10% (2030), 30% (2040), **100% (2050)**. FhISI: **80% of conventional glass production → electric melting by 2050**.

---

### 3.9 Food (simplified)

**FEC corridor: 42% (RESCUE) – 64%** of 2015. negaWatt aggregate 58%; 64% = subsectors with minimal reduction. Only negaWatt & RESCUE (national) study food; negaWatt gives all subsectors (A alcohol → H sugar).
- National choice via negaWatt subsector data (Figure 24, reductions e.g. −48%, −46%, −34%, −52%, −44%, −35%, −42%): strong activity in strains / fruit & veg / oils → nearer upper bound.

**Key assumptions.** GHG entirely energy-related; energy-intensive processes = heating (cooking, boiling, baking, drying) + cooling.
- Sufficiency: production decrease from healthier/faster diets, regional sustainable products, lower self-sufficiency.
- Efficiency complementary; FEC declines by 2050.
- **Electrification**: RESCUE — full switch to renewable electricity for heating & cooling avoids GHG entirely (needs framework conditions).

---

### 3.10 Non-Ferrous Metals (NFM) (simplified)

**Overview.** Aluminium (largest), then copper, zinc, lead, precious metals. EU NFM FEC ≈ **6× lower than steel**. Recycling far less energy intensive: **copper recycling = 36%** of primary energy; **aluminium recycling = 5%**. Primary aluminium ≈ **14 MWh/tonne** electricity.

**FEC corridor: 39.3% (negaWatt) – 87.3% (FhISI)** of 2015 (RESCUE 44.7). Table 2: 39 – 87.

**Key assumptions.**
- **Secondary (recycled) production rise** is the main lever: RESCUE GreenSupreme → 90% by 2050 (vs 56% in 2010 base year); negaWatt → aluminium secondary 85% by 2050 (vs 55% in 2014 base year).
- **Electrification:** gas-fired smelting → electric induction furnaces by 2050. FhISI: NG demand ÷3 by 2050. RESCUE: electricity share → 65% for secondary metals/semi-finished (2030–2050); primary metal electricity share constant at 85%.
- Other measures: waste/residual heat reuse, energy management systems, regeneratively-produced reducing agents, **inert anodes** in primary aluminium.
- Production: slight decrease (RESCUE, negaWatt); FhISI slower aluminium growth (material efficiency, reuse, longer lifetimes outweighing steel→aluminium substitution).

---

### 3.11 Others (metallurgy, machinery, electronics, etc.) (simplified)

**FEC corridor: 63% – 85%** of 2015. Represents ~¼ of industrial FEC but much lower GHG share. Based on negaWatt subsector data (Figure 26; reductions e.g. −46%, −43%, −74%, −43%, −48%, −37%, −32%, −15%, −47%).
- Corridor selection rule:
  - French-like distribution → **63%** (negaWatt aggregate).
  - Mechanical/Electricity/Textile larger share → **68%** minimum.
  - Electronics largest subsector → **85%** minimum.
  - Lack of data → **85%** minimum.

---

## 4. Annex 1 — Methodology for integrating assumptions into trajectories

- **Total industry FEC (2050)** = sum of 2050 consumption of each sector with a story.
- **Reference / base year = 2015** (historical data). National trajectories use a **linear slope between 2015 historical data and the 2050 target**. (Partners with deeper knowledge may model a more detailed trajectory.)
- **Index convention:** `index 2050 = 1 − (% reduction by 2050)`.

**Detailed analysis (steel, cement, paper, glass, chemicals subsectors with data).**
Partners set assumptions on (a) **production reduction** (index 2050) and (b) **energy efficiency / intensity** (kWh/t) by 2050, respecting the European corridors. Using 2015 national production data:

```
FEC_2050 (sector) = Production_2015 (kt) × ProductionIndex_2050 × EnergyIntensity_2050 (MWh/kt)
```

> NOTE: The exact formula is given as an image (`imageFile23.png`) in the source and is not transcribed in the markdown. The expression above reconstructs it from the prose: FEC = production (in tonnes, scaled by the 2050 index from the 2015 base) × energy intensity (kWh/t → MWh/kt) in 2050. Verify against the original PDF figure if precision is required.

**Simplified analysis (food, NFM, others, and chemicals without subsector data).**
Where reasoning in tonnes is complex or inappropriate (variable/heterogeneous production units, e.g. food mixing tonnes of fruit, tonnes of meat, litres of beer), partners assume the **FEC reduction in 2050 vs 2015 directly**, respecting the corridor:

```
FEC_2050 (sector) = FEC_2015 (sector) × FECIndex_2050        where FECIndex_2050 = 1 − (% reduction by 2050)
```

> NOTE: The simplified formula is also given as an image (`imageFile24.png`); reconstruction above follows the prose.

---

## 5. Annex 2 — Ammonia & HVC including feedstock (non-energy use)

The Industry note's headline intensities for ammonia and HVC **exclude feedstock** (non-energy use). Footnote 51/53 states feedstock is treated later in the project:

- **Methane and naphtha** for ammonia and HVC production are accounted as a **primary energy requirement**.
- **Hydrogen** for low-emission ammonia / HVC processes is accounted as a **decarbonised electricity requirement** (i.e. counted toward electricity demand).
- The "including feedstock" energy-intensity graphs are Figures 27 & 28.

> NOTE: In the converted markdown, Annex 2 (Figures 27 & 28) contains only image placeholders / empty tables — the numeric "including-feedstock" curves are not transcribed. Key implication retained from the body text:
> - Ammonia intensity **excluding** feedstock falls (negaWatt ~1580 MWh/kt) but **rises when feedstock is included** (hydrogen feedstock is large).
> - HVC intensity **rises even excluding** feedstock (negaWatt ~4470 MWh/kt due to MTO/MTA hydrogen route) and **rises further with feedstock**.
> - Reference figures: today's ammonia = 8.9 MWh NG (fuel+feedstock) + 2.1 MWh elec per t; electrolytic ≈ 9.1 MWh elec/t. HVC hydrogen feedstock ≈ 21.7 GWh_elec/kt olefin, 41 GWh_elec/kt BTX; negaWatt FR needs 83 TWh elec by 2050 for HVC feedstock.

**Corroborating figures from `CLEVER_final-report.md` (downstream model — EU27/EU30):**
- Chemical-industry **feedstock = 650 TWh (2019)** at EU27 level; **reduced 22% to 480 TWh by 2050**.
- By 2050 **all ammonia from green hydrogen**; olefins mainly via **MTO using hydrogen-based methanol**; **hydrogen = 78% of chemical-industry feedstocks** by 2050.
- Hydrogen feedstock for non-energy use is **accounted as H₂ in FEC for non-energy use** (final-report footnote 67); Power-to-Methanol (olefins) + Power-to-Ammonia ≈ 415 TWh H₂-equivalent in 2050 (EU30).
- Energy use (separate from feedstock): ~**190 TWh hydrogen as energy use** in industry by 2050; industry electricity share **32% (2019) → 64% (2050)**; gas **31% → 10%**; coal phased out before 2040; industry GHG **−92%** by 2050.

> This separation (energy FEC vs non-energy feedstock consumption) is exactly the split the downstream model relies on: ammonia/HVC "energy" intensities exclude feedstock, while methane/naphtha (primary energy) and hydrogen (as decarbonised electricity / H₂ for non-energy use) are tracked as feedstock.

---

## 6. Policy measures (brief)

**Sufficiency policies** (scale down production):
- Dimensional downscaling of goods (e.g. smaller cars — SUVs are ~40% heavier than average car).
- Consumer information on sustainability/reparability; eco-design & energy labelling; CO₂ & life-cycle footprint labelling; unified **Digital Product Passport**.
- Minimum environmental requirements / cap on product GHG; VAT incentives for low-impact products.
- Ban ads promoting high-energy-consuming goods.
- Mutualisation/sharing (car-sharing, home-sharing).
- Behaviour/diet change (less meat → less fertiliser/chemicals demand; public-transport & soft mobility → less car/steel production).
- Anti soil-sealing / zero net land take by 2050 (reduces construction-material demand).

**Circularity policies** (lifecycle tracking + recycling):
- EU-wide value-chain tracking (Digital Product Passport).
- Eco-design requirements; ban planned obsolescence (minimum lifetime floor); extended legal guarantees; "precycling" (restrict non-recyclable materials).
- Reuse targets, deposit systems, second-hand; **"right to repair"** (spare parts available 5–10 years); reparability/recyclability labelling.
- Highest end-of-life recycling rates; limit waste exports; **minimum recycled-content rates** in production; tax on raw-material use.
- Relocalisation to reinforce recycling-production synergies and cut carbon footprint.

**Efficiency policies** (lower energy/carbon intensity):
- Strong **EU ETS** + **CBAM** (make efficiency investment cost-effective; protect against carbon-intensive imports).
- Fuel substitution away from fossil fuels (e.g. ban on fossil-fuel generation tech by 2040; clear certification for H₂/biogas via REDII).
- Material substitution toward less energy/carbon-intensive materials (e.g. wood in buildings — negaWatt FR: 95% of new individual homes in wood by 2050, 80% by 2030).
- Technological gains via R&D financing (EU Innovation Fund from ETS revenues).

> Box 1: sufficiency/circularity/efficiency expected to have a positive or neutral employment impact (job redistribution, reuse/repair economy, European reindustrialisation).
