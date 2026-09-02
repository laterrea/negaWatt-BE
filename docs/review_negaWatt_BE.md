# Review of the negaWatt-BE Demand-Side Model

**Reviewers' note:** This document provides a critical review of the negaWatt-BE demand-side notebooks as of 2026-08-17. It covers the general approach, per-notebook comments, methodological concerns, improvement pathways, and alternative data sources.

---

## 1. General Assessment

### 1.1. Strengths

The negaWatt-BE project is a commendable and ambitious effort to build a transparent, bottom-up demand-side scenario for Belgium's energy transition (2019–2050). Several aspects stand out positively:

- **Transparency and openness:** All assumptions are explicitly stated in the notebooks with inline references and comments. The authors honestly flag their own uncertainties with `> **Comment:**` blocks — a rare and valuable practice.
- **Coherent framework:** The sufficiency-first approach (demand reduction before efficiency, before supply-side solutions) is methodologically sound and aligned with the original French négaWatt philosophy.
- **Appropriate reference year:** Using 2019 instead of 2020 to avoid COVID-19 distortions is a well-justified choice.
- **Modular architecture:** The notebook chain (`macro` → `buildings` / `transports` → PyPSA) allows independent development and clear separation of concerns.
- **Rich data foundation:** Heavy reliance on JRC-IDEES 2023 provides a consistent, peer-reviewed data backbone across sectors.
- **Useful visualisation:** Horizontal stacked bar charts and styled DataFrames give immediate visual feedback on scenario trajectories.

### 1.2. Overarching Concerns

Despite these strengths, several cross-cutting issues limit the robustness and credibility of the current model:

1. **No uncertainty or sensitivity analysis.** Every parameter is a single deterministic value. There is no exploration of what happens if renovation rates are halved, car occupancy reaches only 1.5 instead of 2.0, or district heating stalls at 5%.
2. **Uniform "-10%" assumptions.** The same round number (-10%) appears for residential surface per person, tertiary surface per person, passenger mobility intensity, and freight transport intensity. While convenient, this uniformity suggests the assumptions are placeholders rather than evidence-based projections.
3. **Linear interpolation everywhere.** Almost all trajectories between 2019 and 2050 use `linear_growth()`. Real-world transitions are rarely linear — they exhibit inertia, acceleration, saturation, or policy-induced step changes. The S-curve and bell-curve functions in `sub_functions.py` are used only for carrier shares, not for demand trajectories.
4. **Incomplete coverage.** Maritime bunkers are entirely absent ("To be done!"). Agriculture still enters PyPSA-Eur from the CLEVER dashboard. There is no climate trajectory for heating and cooling. The "Short List of Sufficiency Assumptions" sections in the buildings and transports notebooks are now populated via the website-export cells.
5. **Industry is documented, not modelled bottom-up in PyPSA.** `nW_BE_demand_model_industry.ipynb` reconstructs the CLEVER industrial FEC for Belgium (see the Industry notebook section of `README.md`). Remaining demand-side gaps: agriculture still from the CLEVER dashboard, maritime bunkers still "To be done!", no climate trajectory.
6. **Validation gap.** While `nW_BE_demand_data_aux.ipynb` provides historical series, there is no systematic back-testing or validation of the model against a known year (e.g., checking that 2019 reconstructed demand matches actual Eurostat totals within a tolerance).

---

## 2. Per-Notebook Review

### 2.1. `nW_BE_demand_model_macro.ipynb`

**Purpose:** Defines population, household projections, and physical constants.

**Comments:**
- Clean and concise. This is the most mature notebook.
- Population projections from the Federal Planning Bureau and Statbel are appropriate.
- The conversion factors (`ktoe_to_GWh`, `kgoe_to_kWh`, `kgh2_to_kWh`, `kgLNG_to_kWh`) are clearly defined and consistent with standard values.
- Minor point: `kgh2_to_kWh = 33.33` corresponds to the lower heating value (LHV) of hydrogen. This is the correct convention for energy system modelling, but it would be worth stating explicitly that all energy values are LHV-based to avoid confusion when comparing with sources that use HHV (39.4 kWh/kg).
- The `df_SUF` DataFrame is built up incrementally across notebooks via `%run`. While this works, it creates implicit dependencies that can be fragile.

**Suggestions:**
- Add a simple sanity check: print total population in 2050 and compare with the Planning Bureau's central projection.
- Consider storing `df_SUF` in a shared `.csv` or `.parquet` file to decouple notebook execution order.

---

### 2.2. `nW_BE_demand_model_buildings.ipynb`

**Purpose:** Residential and tertiary building energy demand (thermal + electrical services).

#### 2.2.1. Residential Sector

**Space heating** is the largest single demand component and receives appropriately detailed treatment:
- The assumption of doubling the historical renovation rate (-0.916 kWh/m²/year) is ambitious but defensible if Belgium implements its Long-Term Renovation Strategy.
- The SlowHeat assumption (-2°C setpoint → -14% demand) is interesting and well-referenced to the SlowHeat book. However, achieving widespread thermostat reduction requires behavioural change at scale, which is historically difficult. The 14% figure assumes a uniform response across the population.
- The combined target of ~34.2 kWh/m² by 2050 is aggressive. For reference, the Flemish PATHS2050 scenario targets ~40–50 kWh/m² depending on the pathway. The gap should be acknowledged and justified.

**Space cooling** uses a simple linear extrapolation of the historical deployment rate (+0.035 kWh/m²/year). This may underestimate future cooling demand given:
- Climate change projections for Belgium (CORDEX scenarios suggest +30–50% increase in cooling degree days by 2050).
- The "rebound" effect from improved insulation (overheating risk increases in well-insulated buildings).

**Hot water** reduction of -21% (based on 5-minute showers at 38°C + 10L at 60°C for other uses) is physically well-founded but represents an optimistic behavioural target. No evidence is provided that this level of sufficiency has been achieved at scale in any comparable country.

**Cooking:** The +15% increase in residential cooking (more home cooking) is a plausible sufficiency assumption. The shift to almost fully electric cooking (2% gas remaining) by 2050 is consistent with current EU policy trends.

**Electrical appliances:** Individual targets per category are generally well-motivated by efficiency standards (EU Ecodesign). However:
- The ICT projection (+10%) may underestimate the growth of home servers, gaming, and smart home devices.
- The "other appliances" category is held constant (not +5%) and remains a catch-all that deserves decomposition.

**District heating at 15%** by 2050 is a policy aspiration. For context:
- Belgium's current district heating share is ~2.4% (one of the lowest in the EU).
- Even Denmark, which started expanding DHN in the 1970s, took ~40 years to reach 60%. Going from 2.4% to 15% in 30 years is plausible but requires massive investment.
- The Lund et al. reference supports the concept but does not directly support 15% for Belgium.

**Carrier distribution:** The decomposition into `heat-ihs`, `heat-dhn`, `cold`, `electricity`, `fuel-gas`, `fuel-bio` is clean and appropriate for PyPSA input.

#### 2.2.2. Tertiary Sector

- The authors themselves flag that "Less effort has been spent on this section" for electrical uses, and that normalisation should be improved. This is an honest assessment.
- **Normalisation inconsistency:** Residential demands are normalised per household or per person, while some tertiary demands use per m² and others per person. A fully consistent normalisation by m² of heated floor area would be more appropriate for the tertiary sector.
- **Surface reduction of -10% per person** is marked with "Should further motivate this!" — this is indeed a critical gap. Tertiary surface trends depend heavily on telework adoption, commercial real estate dynamics, and public service rationalisation, none of which are discussed.
- **Tertiary heating target (~66.5 kWh/m²):** The assumption of 5× the historical improvement rate is extremely aggressive. The historical rate includes the effect of relatively easy early gains; future improvements on already-renovated buildings face diminishing returns. In addition, the slow renovation cycles (20-40 year lifetimes for commercial building components) makes this rate change hard to achieve without forced early retirement of assets.
- **Catering (+20%):** In the tertiary sector, this seems inconsistent with an overall sufficiency narrative. If the goal is demand reduction, increasing catering energy does not align well.
- **Ventilation held constant:** In well-insulated, airtight buildings, mechanical ventilation actually *increases* in importance (and energy use). Holding it constant may underestimate future demand.

#### 2.2.3. Code Quality

- Repetitive code patterns (the same plotting code is duplicated ~8 times). This should be refactored into a reusable plotting function.
- The `post_process` flag is a good pattern to separate computation from visualisation, but it is not consistently applied.
- Variable naming (`ref_RS_tes_sht`, `trg_TS_ees_blt`) uses an internally consistent convention but is cryptic without the accompanying markdown. A small glossary or naming convention document would help.

---

### 2.3. `nW_BE_demand_model_transports.ipynb`

**Purpose:** Passenger mobility and freight transport energy demand.

This is the largest and most complex notebook (~2558 lines, 143 cells). It is also the least complete, with several TODO items and unresolved comments.

#### 2.3.1. Passenger Mobility

**Global mobility intensity (-10%):**
- The authors note that the French négaWatt scenario assumes -23% but do not verify this claim. The -10% for Belgium is described as more conservative but still requires justification. What behavioural changes or policies drive this reduction? Telework? Urban densification? Higher transport costs?

**Modal shifts:**
- The -30% modal shift away from cars is the centrepiece of the passenger scenario. The distribution (10% to bus, 8% to train, 7% to cycling, etc.) is detailed but the evidence base is thin. Modal shift of this magnitude has not been achieved in any EU country within 30 years without transformative infrastructure investment.
- **Tram/metro tripling** is flagged by the authors as "likely too high." Indeed, tripling tram/metro usage requires not just more vehicles but entirely new lines, which take 10–15 years to plan and build.
- **Cycling at 7% modal share** implies a Dutch/Danish level of cycling culture. Given Belgium's hilly Walloon geography and dispersed settlement patterns, this is optimistic for a national average.
- **Intra-EU aviation (-50%)** with redistribution to rail is consistent with the European "ban short-haul flights" policy trend, but the 50% figure exceeds most policy proposals (which target flights under 2.5h train journey, representing ~20–30% of intra-EU flights).
- **Extra-EU aviation (-40%)** is a very strong demand reduction. International aviation demand has historically grown by 3-5% per year. Achieving an absolute 40% *reduction* would require either a carbon tax making flying prohibitively expensive or strong regulatory rationing.

**Car occupancy from ~1.2 to 2.0:**
- This is perhaps the single most ambitious assumption in the entire model. The EU average has been stagnant at ~1.2–1.5 for decades despite various carpooling initiatives. Reaching 2.0 nationally would require either:
  - Mandatory carpooling policies (no precedent in EU democracies), or
  - A fundamental shift in mobility culture on a scale not observed anywhere.
- This assumption alone reduces car energy consumption by ~40%. If it fails to materialise, the entire transport scenario is significantly off.

**Carrier shares:**
- The near-complete electrification of cars (~95.5% BEV by 2050) is aggressive but within the range of recent scenarios (e.g., IEA Net Zero 2050 projects ~60% BEV sales share by 2030, reaching >85% by 2050).
- **Bus electrification at 90%** with 5% H2 is reasonable. The corrected BEV consumption (2 kWh/km vs JRC-IDEES's 3.06 kWh/km) is well justified with a literature reference.
- **Aviation remaining 95–100% kerosene** (with only 5% H2 for intra-EU) is realistic given the state of Sustainable Aviation Fuels (SAF). However, SAF (bio-kerosene, e-kerosene) is entirely absent from the model, which is a significant omission given EU ReFuelEU mandates (6% SAF by 2030, 70% by 2050).

**Energy consumption calculations:**
- The conversion chain (modal share → Gpkm → fuel consumption/occupancy → TWh) is clear and consistent.
- The PHEV 50/50 electric/gasoline split is a reasonable simplification.
- The -25% fuel consumption reduction for cars (speed limits, eco-driving, smaller cars) is plausible but the individual effects are not disaggregated. The author's own comment acknowledges this.

#### 2.3.2. Freight Transport

- **25% modal shift from heavy-duty trucks to rail (15%) and inland waterways (10%):** This is an ambitious target. Belgian rail freight has been *declining* in recent years. The Vision Rail 2040 target and Federal Planning Bureau projections should be checked for consistency.
- The **discrepancy in truck tkm** between JRC-IDEES (52.4 Gtkm) and Statbel (34.8 Gtkm) is a fundamental data quality issue (60% difference!) that is flagged but unresolved. This affects the absolute magnitude of the freight energy demand.
- **Train freight tkm discrepancy:** Similarly, the difference between JRC-IDEES and Eurostat for the rail modal share (18.3% vs 12.1%) is substantial. Which is correct matters enormously for the scenario.
- **Navigation carriers:** The introduction of ammonia (17.5%), methanol (35%), and hydrogen (7.5%) for inland and coastal navigation by 2050 is consistent with IMO/DNV projections but represents emerging technologies with high uncertainty. The same consumption rate (kWh/km) is assumed for all carriers in navigation, which ignores significant efficiency differences between diesel, ammonia IC engines, and H2 fuel cells.
- **Heavy-duty truck BEV consumption at 130 kWh/100km** is based on current models (Renault E-Tech T). Real-world consumption with Belgian road gradients and cold weather may be 15–25% higher.
- **Cargo bikes** are mentioned in a comment ("We should integrate alternative modes, such as cargo-bikes") but not modelled. For urban last-mile freight, this is an increasingly relevant mode.
- **Maritime bunkers** are entirely absent ("To be done!"). For a port country like Belgium (Antwerp-Bruges is Europe's 2nd largest port), this is a major gap.

#### 2.3.3. Code Quality

- Same repetitive plotting pattern as in buildings — should be factored out.
- The boilerplate for converting carrier × mode DataFrames to TWh is copy-pasted ~15 times with minor variations. This is error-prone and should be a function.
- Some commented-out lines (e.g. the original JRC-IDEES bus BEV consumption) are helpful for traceability but should be more systematically documented.

---

### 2.4. `nW_BE_demand_data_aux.ipynb`

**Purpose:** Historical time series for validation and trend analysis.

**Comments:**
- Useful for context but currently disconnected from the main model. There is no automated validation step that compares model outputs against these historical series.
- The comparison of JRC-IDEES vs ODYSSEE tertiary floor areas across multiple countries is insightful and highlights non-trivial data quality issues (discrepancies of 30-50% for some countries).
- The polynomial fits (`np.polyfit`) are used only for trend analysis, not for projections. This is appropriate.
- The renovation rate analysis is valuable and should be linked more explicitly to the buildings notebook assumptions.

---

### 2.5. `nW_BE_demand_model_sub_functions.py`

**Purpose:** Projection and formatting utility functions.

**Comments:**
- Well-structured and reasonably documented.
- The `s_curve_growth()` function uses `scipy.stats.norm.cdf` for S-curves, which is mathematically sound.
- `b_curve_with_control_value()` uses `scipy.optimize.fsolve` to find the control parameter — robust but requires good initial guesses.
- `accelerated_growth()` and `strong_acceleration_growth()` use polynomial exponents (2 and 3) — ensure these don't generate overshoots for near-boundary inputs.
- Missing: no unit tests. Given the critical role of these functions, a test suite with known input/output pairs would significantly improve confidence.

---

## 3. Methodological and Data Issues

### 3.1. The Sufficiency Assumption Aggregation Problem

Each subsector applies its own sufficiency, efficiency, and electrification assumptions *independently*. There is no check for *systemic consistency*:
- If people telecommute more (reducing transport), their residential heating demand increases. This cross-sectoral effect is not captured.
- If car occupancy doubles, ride-sharing services may increase total pkm (rebound effect). This is not modelled.
- If buildings are deeply renovated, the construction sector's embodied energy increases. This is outside the current scope but worth flagging.

### 3.2. The Reference Year Problem

Although 2019 is well justified to avoid COVID effects, some 2019 values may themselves be atypical:
- 2019 was a mild winter in Belgium (HDD ~2100 vs long-term average ~2400). This affects the space heating reference.
- Rail transport was already declining in 2019 due to service disruptions (SNCB strikes). Using this as a baseline for modal shift targets may skew the results.

### 3.3. Policy Assumptions vs. Physical Modelling

The model conflates policy targets with physical outcomes. For example:
- "90% BEV trucks by 2050" is a policy aspiration, not an energy-physical constraint. What if BEV truck costs remain 50% higher than diesel? What if charging infrastructure lags?
- These policy assumptions should be clearly separated from engineering/physical parameters to enable proper scenario analysis.

---

## 4. Improvement Pathways

### 4.1. Short-Term (Low Effort, High Impact)

1. **Complete the remaining TODO sections:** Maritime bunkers. The short lists of sufficiency assumptions in buildings and transports are now filled by the website-export cells.
2. **Add a validation cell** at the end of each notebook that compares 2019 reconstructed demand with Eurostat/JRC-IDEES totals and prints the percentage deviation.
3. **Refactor repetitive code:** Extract the repeated plotting code and the carrier-shares-to-TWh conversion boilerplate into functions in `sub_functions.py`.
4. **Resolve data discrepancies:** The truck tkm and train tkm inconsistencies between JRC-IDEES, Statbel, and Eurostat need to be resolved or at least bounded.
5. **Add SAF (Sustainable Aviation Fuels)** to the aviation carrier mix, per ReFuelEU Aviation mandates.

### 4.2. Medium-Term (Moderate Effort)

6. **Sensitivity analysis:** Identify the 5–10 most impactful parameters (car occupancy, renovation rate, modal shift percentage, district heating share, heavy-truck electrification rate) and perform a tornado-diagram or Monte Carlo analysis.
7. **Non-linear demand curves:** Replace `linear_growth()` with `s_curve_growth()` or custom logistic curves for demand parameters where saturation or acceleration is expected (e.g., renovation rates accelerate after policy implementation, then slow near completion).
8. **Cross-sectoral coupling:** Implement simple feedback loops (e.g., telework increases residential heating but decreases commuting).
9. **Climate change integration:** Use CORDEX or Copernicus climate projections to adjust heating and cooling degree days to 2050, rather than assuming constant climate.
10. **Normalise tertiary sector consistently** by m² of heated floor area, not by population.

### 4.3. Long-Term (High Effort)

11. **Industry reconstruction exists** (`nW_BE_demand_model_industry.ipynb`); remaining work is a bottom-up Belgian module or a clearer bridge into PyPSA, plus agriculture (still from the CLEVER dashboard).
12. **Implement scenario comparison:** Define at least two scenarios (e.g., "sufficiency" vs. "technology-only") to explore the sensitivity of total demand to behavioural vs. technological levers.
13. **Automated pipeline:** Replace the `%run` chain with a proper build system (e.g., `papermill` or `nbconvert` with parameterisation) to improve reproducibility and enable batch runs.
14. **Open-data packaging:** Publish input data and scenario parameters as structured `.csv`/`.yaml` files with metadata, enabling external users to modify assumptions without editing notebooks.
15. **Unit testing:** Write `pytest` tests for all functions in `sub_functions.py`, especially edge cases (zero growth, negative targets, years outside range).

---

## 5. Alternative and Complementary Data Sources

| Domain | Source | Description | URL / Reference |
|--------|--------|-------------|-----------------|
| **Buildings** | EU Building Stock Observatory | Detailed building stock data for all EU member states, including renovation rates, energy performance certificates, and building typologies | https://energy.ec.europa.eu/topics/energy-efficiency/energy-efficient-buildings/eu-building-stock-observatory_en |
| **Buildings** | TABULA/EPISCOPE | Building typology data for EU countries including Belgium, with reference buildings and renovation scenarios | https://episcope.eu/building-typology/ |
| **Buildings** | Flanders Energy Agency (VEA) | Regional building stock and EPC data for Flanders | https://www.energiesparen.be/ |
| **Buildings** | ODYSSEE-MURE | Energy efficiency indicators and policies database for EU countries, useful for cross-validation and trend analysis | https://www.odyssee-mure.eu/ |
| **Transport** | ICCT (International Council on Clean Transportation) | Vehicle efficiency data, fleet composition, and emission standards analysis | https://theicct.org/ |
| **Transport** | European Alternative Fuels Observatory (EAFO) | Real-time data on alternative fuel vehicle registrations, charging infrastructure, and fuel deployment across EU | https://alternative-fuels-observatory.ec.europa.eu/ |
| **Transport** | Eurostat Transport Statistics | Harmonised transport data including modal split, vehicle-km, passenger-km, and ton-km for all EU members | https://ec.europa.eu/eurostat/web/transport/overview |
| **Transport** | Belgian Federal Mobility Survey (MONITOR) | Belgian-specific mobility behaviour data, trip purposes, mode choices | https://mobilit.belgium.be/fr/mobilite-durable/enquetes-et-resultats |
| **Transport** | ENTSOE Transparency Platform | Electricity consumption data for rail transport and system-level validation | https://transparency.entsoe.eu/ |
| **Transport** | ITF (International Transport Forum) | Transport outlook data and modal shift analyses | https://www.itf-oecd.org/ |
| **Transport / Maritime** | IMO Fourth GHG Study | Global and regional maritime emissions and fuel mix projections | IMO (2020), Fourth IMO GHG Study 2020 |
| **Transport / Maritime** | Port of Antwerp-Bruges statistics | Detailed port traffic and energy data for Belgium's main port | https://www.portofantwerpbruges.com/ |
| **Climate** | Copernicus Climate Data Store | Future climate projections (CORDEX) for heating/cooling degree days in Belgium | https://cds.climate.copernicus.eu/ |
| **Cross-sector** | IEA World Energy Outlook / Net Zero by 2050 | Global and European scenario benchmarks for demand reduction and electrification | https://www.iea.org/ |
| **Cross-sector** | Climact CLEVER scenario | European sufficiency-based scenario (similar philosophy to négaWatt) with Belgium disaggregation | https://clever-energy-scenario.eu/ |
| **Cross-sector** | EnergyVille PATHS2050 | Belgian energy transition pathways; directly comparable and should be used for benchmarking | https://perspective2050.energyville.be/ |
| **Cross-sector** | Federal Planning Bureau — Energy/Transport Outlook | Official Belgian government projections for energy and transport demand | https://www.plan.be/ |

---

## 6. Specific Criticisms

### 6.1. The Car Occupancy Assumption Is Unrealistic

Projecting car occupancy from 1.2 to 2.0 by 2050 is the single most impactful and least supported assumption in the model. This represents a ~67% increase in a parameter that has been essentially flat across all EU countries for 30+ years. No country has achieved sustained occupancy above 1.6 through policy alone. If this assumption is relaxed to 1.5 (already ambitious), the car energy demand increases by roughly 33% compared to the scenario.

**Recommendation:** Present results with at least two occupancy scenarios (1.5 and 2.0) and discuss the policy mechanisms required for each.

### 6.2. The -10% Blanket Assumptions Lack Differentiation

Using -10% for residential surface/person, tertiary surface/person, mobility intensity, and freight intensity creates an artificial sense of coherence. In reality:
- Residential surface per person in Belgium has been *increasing* (trend toward smaller households, not smaller homes). The -10% assumption reverses a multi-decade trend without justification.
- Freight intensity is driven by GDP structure and trade patterns, which are largely outside domestic policy control.

**Recommendation:** Replace each -10% with a value derived from sector-specific analysis, even if the resulting numbers are similar.

### 6.3. Missing Rebound Effects

Efficiency improvements often lead to increased consumption (Jevons paradox / rebound effect). For example:
- More efficient cars → cheaper driving → more driving.
- Better insulated buildings → higher indoor temperatures.
- LED lighting → more lights left on.

The model implicitly assumes zero rebound, which is optimistic. Even a modest 10–20% direct rebound would meaningfully increase total demand.

### 6.4. No Cost or Investment Dimension

The scenario defines a physically possible endpoint but says nothing about economic feasibility:
- What investment is needed for the renovation rates assumed?
- What does the rail infrastructure expansion cost?
- Is the district heating expansion economically viable in Belgium's dispersed settlement pattern?

While this may be outside the scope of a demand model, flagging the investment implications would strengthen the work.

### 6.5. Incomplete Treatment of Emerging Technologies

- **Vehicle-to-Grid (V2G):** With ~95% BEV car fleet, V2G could significantly affect electricity demand profiles. Not modelled.
- **Heat pumps:** Implied by the shift from gas to electricity in heating, but never explicitly quantified (COP assumptions, peak electricity demand implications).
- **Green hydrogen:** Used as a carrier in transport but the efficiency chain (electrolysis → compression → fuel cell) is not made explicit. The well-to-wheel efficiency of H2 trucks is ~25–30%, vs. ~75–80% for BEV. This affects the total primary energy demand.
- **Synthetic fuels (e-fuels):** Entirely absent despite EU regulatory frameworks that may mandate their use in aviation and maritime.

### 6.6. Bug: Tertiary Catering Uses 2019 Population for 2050

In the buildings notebook (and the `nW_BE.py` transcription), the tertiary catering demand for 2050 used the 2019 population:
```python
'catering': linear_growth(2019, ref_TS_tes_cat * df_SUF["population [person]"][2019] * 1e-9,
                          2050, trg_TS_tes_cat * df_SUF["population [person]"][2019] * 1e-9, years),
```
The 2050 endpoint used `[2019]` instead of `[2050]`. Other thermal services correctly use `[2050]` for the endpoint. This underestimated 2050 catering demand by ~10% (the population growth between 2019 and 2050). **Corrected 2026-08-17** in the notebooks and `nW_BE.py`.

---

## 7. References Used in the Notebooks

The notebooks cite the following main references:

1. **JRC-IDEES 2023** — Joint Research Centre Integrated Database of the European Energy System. Mantzos, L. et al. (2023). https://data.jrc.ec.europa.eu/collection/id-00681
2. **Eurostat Energy Balances** — https://ec.europa.eu/eurostat/web/energy/database
3. **Statbel** — Belgian Statistical Office, demographic and transport data. https://statbel.fgov.be/
4. **Federal Planning Bureau** — Belgian demographic and economic projections. https://www.plan.be/
5. **SlowHeat** — Wallenborn, G. et al. Reference book on reduced heating setpoints.
6. **Lund, H. et al.** — 4th Generation District Heating (4GDH), Energy (2014).
7. **EnergyVille PATHS2050** — Belgian energy transition scenarios. https://perspective2050.energyville.be/
8. **EU Ecodesign Impact Accounting** — Appliance efficiency standards and projections.
9. **SPF Mobility / MONITOR surveys** — Belgian Federal Mobility Surveys (2017, 2019).
10. **Association négaWatt (France)** — Original négaWatt scenario methodology. https://www.negawatt.org/
11. **Vision Rail 2040** — Belgian rail transport development strategy.
12. **TML Electrification Study** — Transport & Mobility Leuven, electrification of Belgian transport.
13. **JRC Heavy-Duty Decarbonisation (2025)** — Decarbonising European heavy-duty transport.
14. **DNV Maritime Forecast to 2050 (2024)** — Maritime fuel mix projections.
15. **ReFuelEU Aviation Regulation** — EU 2023/2405 on sustainable aviation fuels.

---

## 8. Summary of Recommendations (Priority Order)

| # | Priority | Recommendation |
|---|----------|----------------|
| 1 | **Critical** | Fix the tertiary catering population bug (2019 vs 2050; ≈10 %, not ≈4 %) |
| 2 | **Critical** | Complete maritime bunkers section |
| 3 | **Critical** | Resolve JRC-IDEES vs Statbel/Eurostat freight data discrepancies |
| 4 | **High** | Add validation cells comparing 2019 model vs actual Eurostat totals |
| 5 | **High** | Present car occupancy as a scenario variable (1.5 vs 2.0) |
| 6 | **High** | Add SAF to aviation carrier mix |
| 7 | **High** | Perform sensitivity analysis on top-10 parameters |
| 8 | **Medium** | Replace blanket -10% assumptions with justified values |
| 9 | **Medium** | Refactor repetitive plotting and conversion code |
| 10 | **Medium** | Add climate change impact on heating/cooling degree days |
| 11 | **Medium** | Normalise tertiary sector consistently by floor area |
| 12 | **Medium** | Discuss rebound effects, even qualitatively |
| 13 | **Low** | Add unit tests for `sub_functions.py` |
| 14 | **Low** | Agriculture still from CLEVER dashboard; no climate trajectory |
| 15 | **Low** | Implement automated notebook pipeline |

---

*Review updated 2026-08-17 from the notebooks in this repository. All page/line references correspond to the notebook files.*
