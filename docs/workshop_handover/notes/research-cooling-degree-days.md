# Recovered work — research-cooling-degree-days

Agent `a3d889210c96fe033`, stopped by the session limit before it could report.
Everything below is recovered verbatim from its transcript, in order. **Nothing here has been checked** — treat every figure as unverified until re-sourced.

25 entries, of which 19 web results.

---

## WebSearch: Eurostat cooling degree days dataset nrg_chdd_a Belgium

Web search results for query: "Eurostat cooling degree days dataset nrg_chdd_a Belgium"

Links: [{"title":"Heating and cooling degree days - statistics Statistics Explained","url":"https://ec.europa.eu/eurostat/statistics-explained/SEPDF/cache/92378.pdf"},{"title":"Cooling and heating degree days by country - monthly data","url":"https://fgeerolf.com/data/eurostat/nrg_chdd_m.html"},{"title":"Eurostat: Degree days | CoolLIFE - Knowledge Hub","url":"https://knowledgehub.coollifeproject.eu/eurostat-degree-days/"},{"title":"nrg chdd esms","url":"https://ec.europa.eu/eurostat/cache/metadata/fr/nrg_chdd_esms.htm"},{"title":"nrg chdd esms","url":"https://ec.europa.eu/eurostat/cache/metadata/en/nrg_chdd_esms.htm"},{"title":"climate.discomap.eea.europa.eu","url":"https://climate.discomap.eea.europa.eu/arcgis/rest/services/UAMV/CoolingDegreeDays/MapServer/info/iteminfo"}]

Based on the search results, here's what I found about the Eurostat cooling degree days dataset:

## Dataset Overview

The Cooling degree day (CDD) index is a weather-based technical index designed to describe the need for the cooling (air-conditioning) requirements of buildings.

## Data Collection and Methodology

HDD and CDD are derived from meteorological observations of air temperature, interpolated to regular grids at 25 km resolution for Europe. Calculated gridded HDD and CDD are aggregated and subsequently presented on NUTS-3 level.

The dataset includes monthly data as published by the Joint Research Centre's AGRI4CAST Resources Portal, and annual data are calculated as sum of monthly data by Eurostat.

## Belgium Data

Belgium is included in the dataset, which covers multiple European countries including Austria, Bulgaria, Croatia, Cyprus, Czech Republic, Denmark, Estonia, Finland, France, Germany, Greece, Hungary, Ireland, Italy, Latvia, Lithuania, Luxembourg, Malta, Netherlands, Poland, Portugal, Romania, Slovakia, Slovenia, Spain, and Sweden.

The nrg_chdd_a dataset is the annual version of this cooling and heating degree days data from Eurostat, with Belgium being one of the covered countries.


REMINDER: You MUST include the sources above in your response to the user using markdown hyperlinks.

## WebSearch: Eurostat Statistics Explained "Heating and cooling degree days - statistics" methodology base temperature

Web search results for query: "Eurostat Statistics Explained "Heating and cooling degree days - statistics" methodology base temperature"

Links: [{"title":"Heating Degree Days. A tale of two(at least) methodologies | by Diana Vieira Fernandes | Energy in Buildings | Medium","url":"https://medium.com/energy-in-buildings/heating-degree-days-3fb700491356"},{"title":"Heating and cooling degree days - statistics Statistics Explained","url":"https://ec.europa.eu/eurostat/statistics-explained/SEPDF/cache/92378.pdf"},{"title":"The need for air conditioning is increasing in the EU | 2021/05/31","url":"https://iifiir.org/en/news/the-need-for-air-conditioning-is-increasing-in-the-eu"},{"title":"System and method for managing utility resources based on utility service points","url":"https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/8671037"},{"title":"System and method for managing utility resources based on utility service points","url":"https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/8131609"},{"title":"System and method for managing utility resources based on normalized utility usage","url":"https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/8359215"},{"title":"Eurostat Heating Degree Days - THERMOS: Help","url":"https://tool.thermos-project.eu/help/heating-degree-days.html"},{"title":"cooling degree days","url":"https://www.eea.europa.eu/data-and-maps/figures/cooling-degree-days"}]

Based on the search results, I found information about the methodology for heating and cooling degree days according to Eurostat's approach:

## Methodology and Base Temperature

Eurostat calculates HDD as (18 °C — Tm) x d if Tm is lower than or equal to 15 °C (heating threshold) and zero if Tm is greater than 15 °C, where Tm is the mean outdoor temperature over a given period of d days.

For cooling degree days, a base temperature of 22 °C is considered as representative for assessing the energy demand at the pan-European scale.

The key difference in Eurostat's approach is that the definition of the base temperature in buildings relates to the energy balance of the building and systems. More specifically, cooling degree days are computed from the outdoor air temperature as cumulated daily deviation above a given base temperature threshold from April to September.

I was able to retrieve the Eurostat Statistics Explained PDF document on "Heating and cooling degree days - statistics" (listed as item 2 in the results), though the excerpt shown contains primarily data analysis rather than detailed methodology sections. For the complete methodology details, you may want to access the full PDF directly at the Eurostat source.


REMINDER: You MUST include the sources above in your response to the user using markdown hyperlinks.

## WebFetch: https://ec.europa.eu/eurostat/cache/metadata/en/nrg_chdd_esms.htm

# HDD and CDD Methodology Summary

## Formulas and Base Temperatures

**Heating Degree Days (HDD):**
- Formula: "If Tm ≤ 15°C Then [HDD = ∑i(18°C - Tim)] Else [HDD = 0]"
- Base temperature threshold: 15°C
- Reference indoor temperature: 18°C
- Example: Daily mean of 12°C yields HDD of 6

**Cooling Degree Days (CDD):**
- Formula: "If Tm ≥ 24°C Then [CDD = ∑i(Tim - 21°C)] Else [CDD = 0]"
- Base temperature threshold: 24°C
- Reference indoor temperature: 21°C
- Example: Daily mean of 26°C yields CDD of 5

## Calculation Process

"These calculations are executed on a daily basis, added up to a calendar months and subsequently to calendar years."

## Data Source and Resolution

- Origin: Joint Research Centre's AGRI4CAST Resources Portal (monthly data republished by Eurostat)
- Grid resolution: 25 km across Europe
- Spatial reporting: NUTS-3 regions
- Statistical basis: "Observations from about 3000 weather stations across Europe"

## Time Coverage

- HDD/CDD available since January 1975 (Cyprus from January 1978)
- Update frequency: Annual

## Dataset Codes

The document references the general dataset identifier "nrg_chdd" but does not list specific variant codes like nrg_chdd_a or nrg_chdd_m.

## reasoning

The Eurostat API works via curl, and I've pulled Belgium's HDD and CDD annual data from nrg_chdd_a. I'm working out the indexing scheme: with dimensions ordered as freq, unit, indic_nrg, geo, time, I can calculate the flat array index as indic_idx*46 + time_idx, where HDD=0 and CDD=1, and years span 1980 to 2025.

Now I'm mapping out the CDD values year by year from 1980 through 2019, seeing the cooling degree days fluctuate significantly across decades, with notably high values in years like 1994, 2003, and 2019.

Continuing to check the remaining years through 2025, and confirming which index range corresponds to CDD versus HDD by comparing typical magnitudes — HDD in the 2400-3300 range fits Belgium's heating needs, while CDD staying under 50 makes sense for its climate. I'm noting that 2020 stands out with the highest CDD value at 48.74, followed closely by 2003 and 2006.

Now I want to double-check by querying CDD data separately and pulling in the regional dataset for additional confirmation, plus review how the Eurostat formula's temperature thresholds make CDD values so spiky year to year.

## reasoning

I've got the Belgium CDD/HDD data now, with CDD values notably small due to the 24°C threshold quirk. I'm planning to compute decadal averages to show warming trends, and want to cross-check the regional NUTS2 dataset and possibly the monthly figures for 2022 to confirm consistency.

The key caveat worth flagging is that Eurostat's CDD definition gives tiny values for Belgium compared to alternative base-18°C definitions used elsewhere, which yield much higher numbers—this distinction matters for the workshop. I should also start looking into parts (b) and (c) of the task.

## WebFetch: https://ec.europa.eu/eurostat/statistics-explained/index.php?title=Heating_and_cooling_degree_days_-_statistics

The server returned HTTP 404 Not Found.

The response body was not retrieved. If this URL requires authentication, use an authenticated tool (e.g. `gh` for GitHub, or an MCP-provided fetch tool) instead of WebFetch.

## WebSearch: eurostat statistics-explained "Heating and cooling degree days" 2025 cooling degree days EU record

Web search results for query: "eurostat statistics-explained "Heating and cooling degree days" 2025 cooling degree days EU record"

Links: [{"title":"Heating and Cooling degree days in EU, 1979-2024 (Eurostat 2024;... | Download Scientific Diagram","url":"https://www.researchgate.net/figure/Heating-and-Cooling-degree-days-in-EU-1979-2024-Eurostat-2024-Database-available_fig1_405120457"},{"title":"Heating of buildings decreasing, cooling increasing - Products Eurostat News - Eurostat","url":"https://ec.europa.eu/eurostat/en/web/products-eurostat-news/-/ddn-20220531-1"},{"title":"Heating and cooling degree days - statistics Statistics Explained","url":"https://ec.europa.eu/eurostat/statistics-explained/SEPDF/cache/92378.pdf"},{"title":"The need for air conditioning is increasing in the EU | 2021/05/31","url":"https://iifiir.org/en/news/the-need-for-air-conditioning-is-increasing-in-the-eu"},{"title":"heating and cooling degree days in europe","url":"https://statista.com/statistics/1434419/heating-and-cooling-degree-days-in-europe"},{"title":"Back Heating of buildings decreasing, cooling increasing","url":"https://ec.europa.eu/eurostat/web/products-eurostat-news/-/ddn-20210311-1"},{"title":"heating buildings decreasing cooling increasing","url":"https://build-up.ec.europa.eu/en/resources-and-tools/publications/heating-buildings-decreasing-cooling-increasing"}]

I found relevant information from Eurostat statistics on heating and cooling degree days. Here's what the search results show:

## Key Findings on Cooling Degree Days

Cooling degree days (CDD) values in 2022 were almost four times higher compared to 1979 (37), reaching 140, indicating that the needs for cooling (air conditioning) in a given building increased over the last decades.

The average CDD value for the EU is 75.

## Country-Specific Data (2022)

Malta had the highest CDD (580.4), followed by Cyprus (579.9) and Greece (274).

## Trend Analysis

For cooling degree days, the trend observed shows an increase across time. After 2001, only a few years (2002, 2004, 2005 and 2014) were below average.

The search results show Eurostat data through 2022. While I found comprehensive data on EU cooling degree day trends and records by country, the search results don't contain 2025-specific data or a record-breaking year announcement for 2025. The most recent Eurostat Statistics Explained article I found covers data through 2022.


REMINDER: You MUST include the sources above in your response to the user using markdown hyperlinks.

## WebFetch: https://ec.europa.eu/eurostat/statistics-explained/SEPDF/cache/92378.pdf

# Analysis of Heating and Cooling Degree Days Content

I apologize, but the provided content appears to be a corrupted or improperly rendered PDF file containing primarily binary/compressed data rather than readable text. The document structure shows PDF formatting metadata, but the actual article content is not legible in the format provided.

To accurately report on:
1. HDD and CDD formulas with temperature thresholds
2. Dataset codes
3. EU-level CDD values and record years
4. Belgium-specific data
5. Data source and publication date

I would need access to a properly readable version of the Eurostat Statistics Explained article on heating and cooling degree days. 

**Recommendation:** Please provide either:
- A text-extracted version of the PDF
- A direct link to the article
- The HTML or plain-text version from Eurostat

This would allow me to extract the specific technical formulas, numerical values, and source citations you've requested with accuracy.

[Binary content (application/pdf, 4.1MB) also saved to /home/sylvain/.claude/projects/-home-sylvain-svn-negaWatt-BE/0dead3ce-a8ac-4bab-be29-da0ba920056d/tool-results/webfetch-1788363277584-03foxc.pdf]

## WebFetch: https://ec.europa.eu/eurostat/en/web/products-eurostat-news/-/ddn-20220531-1

# Eurostat Heating and Cooling Degree Days Report

**Publication Date:** 31 May 2022

**Dataset Code:** nrg_chddr2_a

## EU-Wide Figures

**Heating Degree Days (HDD):**
- 1979: 3,510 degree days
- 2021: 3,126 degree days
- Change: 11% decrease

**Cooling Degree Days (CDD):**
- 1979: 37 degree days
- 2021: 100 degree days
- Change: Nearly tripled

## Definitions

"Weather-based technical indexes designed to describe the energy requirements of buildings in terms of heating or cooling."

## Highest CDD Countries (1979-2021 Average)

1. **Cyprus:** 577 degree days
2. **Malta:** 574 degree days
3. **Greece:** 272 degree days

## Lowest CDD Countries

- Ireland: 0.03 degree days
- Sweden: 0.37 degree days
- Denmark: 0.91 degree days
- Finland: 1.36 degree days

Note: The article does not provide 2022 data; reporting ends with 2021 figures.

## reasoning

This confirms the methodology I suspected: CDD uses a 24°C mean daily threshold with 21°C as the base for summing degree-days, while HDD uses a 15°C threshold with an 18°C base. My understanding matches the source exactly, so now I'm checking the dataset codes referenced alongside this definition.

The data comes from the JRC AGRI4CAST Resources Portal at 25 km grid resolution, and I'm noting the EU CDD figures: 37 in 1979, 140 in 2022, with a 1979-2022 average around 75. I'm also spotting a discrepancy — the article's timeframe starts in 1979, but the current database seems to begin in 1980 for country-level annual data, and my API pull showed a slightly different 2022 value (136.46 vs. 140), likely due to a data revision. I'll need to check the regional dataset for the 1979 figure before moving on to researching AC penetration in the service sector and cooling demand growth trends in the EU.

For sources I'm considering the IEA's Future of Cooling report, JRC studies, EU building stock observatory data, EEA indicators, and Odyssee-Mure, particularly for cooling energy intensity figures in the service sectors of Spain, Italy, and Greece. I'll run these searches in parallel.

## WebSearch: JRC "space cooling" Europe service sector floor area equipped air conditioning share percentage report

Web search results for query: "JRC "space cooling" Europe service sector floor area equipped air conditioning share percentage report"

Links: [{"title":"Estimation of European Union residential sector space cooling potential","url":"https://publications.jrc.ec.europa.eu/repository/handle/JRC101855"},{"title":"Space Cooling Market in Europe: Assessment of the Final Energy Consumption for the Year 2016","url":"https://doi.org/10.3390/su14052667"},{"title":"Estimation of European Union service sector space cooling potential - ScienceDirect","url":"https://www.sciencedirect.com/science/article/pii/S0301421517307498"},{"title":"Estimation of European Union service sector space cooling potential","url":"https://publications.jrc.ec.europa.eu/repository/handle/JRC106654?mode=full"},{"title":"Status Quo of the Air-Conditioning Market in Europe: Assessment of the Building Stock","url":"https://www.mdpi.com/1996-1073/10/9/1253"},{"title":"Estimation of European Union residential sector space cooling potential - ScienceDirect","url":"https://www.sciencedirect.com/science/article/pii/S030142151630653X"},{"title":"Space Cooling Technology in Europe","url":"https://heatroadmap.eu/wp-content/uploads/2018/11/HRE4_D3.2.pdf"}]

I found several relevant JRC reports on space cooling in Europe. The search results include studies on both the residential and service sectors. Here are the key findings:

**Service Sector Space Cooling:**
The total potential space cooling demand of the EU was estimated to be 174 TWh for the service sector in an average year. A georeferenced approach based on comparing Cooling Degree Days and space cooling consumption in the USA was used to establish the potential service space cooling demand in NUTS-3 regions of EU.

**Overall Energy Consumption:**
Space cooling is responsible for a significant portion of EU electricity consumption in households (nearly 5%) and even more in the service sector (~13%).

The search results also indicate that data describing the actual space cooling market in Europe includes quantity of SC units, equivalent full-load hours, installed capacities, seasonal energy efficiency values as well as cooled floor area per AC type and/or sector.

However, I didn't find specific information about the percentage of floor area equipped with air conditioning in the service sector. If you need more specific statistics about the share of floor area equipped with air conditioning, you may want to access the full JRC reports directly through the European Commission's publications repository.


REMINDER: You MUST include the sources above in your response to the user using markdown hyperlinks.

## WebSearch: Odyssee-Mure indicator "cooling" energy consumption service sector kWh/m2 Spain Italy Greece air conditioned

Web search results for query: "Odyssee-Mure indicator "cooling" energy consumption service sector kWh/m2 Spain Italy Greece air conditioned"

Links: [{"title":"EU Heat Supply & Energy Efficiency: Key Indicators for Sustainable Heating and Cooling | ODYSSEE-MURE","url":"https://www.odyssee-mure.eu/publications/policy-brief/european-heat-supply.html"},{"title":"Measuring & selecting policies to complement energy efficiency policies | Policy brief | ODYSSEE-MURE","url":"https://www.odyssee-mure.eu/publications/policy-brief/energy-sufficiency.html"},{"title":"Energy Efficiency Trends in Buildings in Europe | Policy brief | ODYSSEE-MURE","url":"https://www.odyssee-mure.eu/publications/policy-brief/buildings-energy-efficiency-trends.html"},{"title":"Air conditioning per dwelling | ODYSSEE-MURE","url":"https://www.odyssee-mure.eu/publications/efficiency-by-sector/households/unit-consumption-air-conditioning.html"},{"title":"Introduction to the Odyssee-Mure Project | ODYSSEE-MURE","url":"https://www.odyssee-mure.eu/project.html"},{"title":"Definition of ODEX indicators in ODYSSEE data base","url":"https://www.indicators.odyssee-mure.eu/odex-indicators-database-definition.pdf"},{"title":"Energy efficiency indicators for heat supply sector","url":"https://www.odyssee-mure.eu/publications/policy-brief/european-heat-supply.pdf"},{"title":"Energy Efficiency Trends & Policies | ODYSSEE-MURE","url":"https://www.odyssee-mure.eu/"}]

I found relevant search results from the ODYSSEE-MURE project. Here are the key findings related to your query:

## Overview
ODYSSEE-MURE provides comprehensive monitoring of energy consumption and efficiency trends as well as evaluation of energy efficiency policy measures by sector for all 27 EU Member States, including Spain, Italy, and Greece.

## Cooling Energy Consumption
Malta, Cyprus, Spain and Italy have the most cooling degree days, which indicates these countries have significant cooling needs.

For air conditioning in the household sector, air conditioning is significant in Cyprus, Malta, Greece, Croatia and Italy (from 8% to 29% of household electricity consumption). Additionally, average per-dwelling consumption for air conditioning increased from 21 kWh/household in 2000 to 87 kWh/household in 2023.

## Service Sector Data
For thermal uses in the service sector, the indicator is the unit consumption (toe) per m² (buildings surface area) when floor area data is available.

The search results provide general ODYSSEE-MURE data on cooling energy consumption and building energy efficiency metrics. For more specific data on cooling energy consumption in the service sector measured in kWh/m², you may need to access the ODYSSEE-MURE interactive database directly at odyssee-mure.eu, which contains detailed country-specific indicators.


REMINDER: You MUST include the sources above in your response to the user using markdown hyperlinks.

## WebSearch: IEA "The Future of Cooling" 2018 European Union share of commercial buildings floor area air conditioners cooling electricity

Web search results for query: "IEA "The Future of Cooling" 2018 European Union share of commercial buildings floor area air conditioners cooling electricity"

Links: [{"title":"IEA © OECD/IEA 2018 The Future of Cooling","url":"https://asiacleanenergyforum.adb.org/wp-content/uploads/2018/06/Melanie-Slade-The-Future-of-Cooling-Opportunities-for-Energy-Efficient-Air-Conditioning.pdf"},{"title":"The Future of Cooling | OECD","url":"https://www.oecd.org/en/publications/the-future-of-cooling_9789264301993-en.html"},{"title":"(PDF) The Future of Cooling: Opportunities for energy efficient air conditioning","url":"https://www.researchgate.net/publication/363611579_The_Future_of_Cooling_Opportunities_for_energy_efficient_air_conditioning"},{"title":"The future of cooling - new report from IEA - HPT - Heat Pumping Technologies","url":"https://heatpumpingtechnologies.org/future-cooling-new-report-iea/"},{"title":"The future of cooling : opportunities... - English - IEA (International Energy Agency) - 2018","url":"https://iifiir.org/en/fridoc/the-future-of-cooling-opportunities-for-energy-efficient-air-conditioning-4787"},{"title":"IEA © OECD/IEA 2018 The Future of Cooling:","url":"https://www.esmap.org/sites/default/files/events-files/Int%20Conf%20Sust%20Cooling/Kathleen%20Gaffney_IEA%20The%20Future%20of%20Cooling_Optimized.pdf"},{"title":"The Future of Cooling – Analysis - IEA","url":"https://www.iea.org/reports/the-future-of-cooling"},{"title":"Digest - The IEA Future Of Cooling Report: Opportunities For Energy Efficient Air Conditioning — Positive Energy","url":"https://positiveenergy.pro/building-science-blog/2018/7/16/digest-the-iea-future-of-cooling-report-opportunities-for-energy-efficient-air-conditioning"},{"title":"IoT-Driven Building Energy Management Systems (BEMS) for Net Zero Energy Buildings: Concept, Integration and Future Directions","url":"https://arxiv.org/pdf/2602.20453"}]

I found search results for the IEA "The Future of Cooling" 2018 report. However, the search results don't contain the specific statistic you're looking for about the European Union's share of commercial buildings floor area with air conditioners and their cooling electricity consumption. 

The results include general information from the report, such as:

- Air conditioners and electric fans account for nearly 20% of total electricity used in buildings around the world today
- ACs sold in Japan and the European Union are typically 25% more efficient than those sold in the United States and China
- The European Union represents 6.0% of global residential AC sales

To find the specific data about EU commercial buildings floor area with air conditioners and cooling electricity share, you may need to access the full IEA report directly or search for more specific sections of the report that contain regional breakdowns by building type.


REMINDER: You MUST include the sources above in your response to the user using markdown hyperlinks.

## WebFetch: https://heatroadmap.eu/wp-content/uploads/2018/11/HRE4_D3.2.pdf

I apologize, but the content provided appears to be a corrupted or improperly formatted PDF file. The text shown consists primarily of PDF encoding/compression data and binary streams rather than readable content about the Heat Roadmap Europe report on space cooling technology.

To extract the specific information you're requesting—namely:
1. Share of cooled floor area in the service sector (EU and by country)
2. Cooling energy consumption/intensity data
3. Total final energy for space cooling
4. Reference year and authorship

I would need access to a properly decoded or displayed version of the document. I recommend:
- Downloading the original PDF directly from the Heat Roadmap Europe website
- Opening it with a standard PDF reader to verify readability
- Re-uploading the content if needed

This would allow me to accurately locate and extract the specific data points you've requested while adhering to the citation standards you've outlined.

[Binary content (application/pdf, 1.4MB) also saved to /home/sylvain/.claude/projects/-home-sylvain-svn-negaWatt-BE/0dead3ce-a8ac-4bab-be29-da0ba920056d/tool-results/webfetch-1788363341479-m598tg.pdf]

## reasoning

This HRE4 table is useful — Belgium's specific cooling demand for service buildings is 52.7 kWh/m² and residential is 10.8 kWh/m², with Austria at 72.2, Bulgaria at 85.1, and UK at 43.4, though this extraction only pulled four countries so there's likely a fuller table elsewhere. I'm noting this figure is demand per cooled floor area, not total floor area.

I also see 2050 EU28 penetration shares of 22.8% for residential and 42.7% for service, but I still need the 2015 figures. I should search for tables with cooled floor area in Mm² and penetration rates, check the country datasheets and results section, and grep for more kWh/m² occurrences to find the complete specific demand table.

## reasoning

This is exactly what I need for parts (b) and (c). I found the HRE4 D3.2 data showing cooled surface area shares in the service sector for 2015: Belgium at 23.8%, EU28 average 21.3%, with Italy, Spain, and Greece much higher at 73.5%, 72.7%, and 87.1% respectively, while France sits at 22.5%.

I'm also gathering the 2050 projections (Belgium rising to 50.3%, EU28 to 42.7%) and residential figures for 2015 (Belgium 1.5%, EU28 6.3%). There's a specific demand table in kWh per m² of cooled area with values for Belgium, Austria, Bulgaria, and the UK, but it looks like only four rows extracted so far — I need to recover Spain, Italy, and Greece from that table, likely by trying different pdftotext extraction modes on the relevant page around 20-21. I also want to confirm the key caveat that this specific demand figure is normalized per cooled floor area, not total floor area, since Section 4.3.3 explicitly states that.

## reasoning

I'm compiling the service-specific cooling demand figures by country for 2015, noting big variation from Belgium's low 52.7 kWh/m² up to Cyprus's 241.3, and starting to gather the residential cooling demand numbers as well.

I want to cross-check these modeled per-cooled-area figures against actual measured data, so I'm planning to pull total cooling supply (TWh) from country datasheets and cross-reference with Odyssee-Mure, the JRC study on EU service sector cooling potential, EU BSO data, IEA's Future of Cooling report, and EEA cooling degree day indicators. Starting with the Belgium datasheet and service sector totals.

## WebFetch: https://www.sciencedirect.com/science/article/pii/S0301421517307498

The server returned HTTP 403 Forbidden.

The response body was not retrieved. If this URL requires authentication, use an authenticated tool (e.g. `gh` for GitHub, or an MCP-provided fetch tool) instead of WebFetch.

## WebFetch: https://publications.jrc.ec.europa.eu/repository/handle/JRC106654

# Estimation of European Union Service Sector Space Cooling Potential

**Authors:** Jakubcionis Mindaugas; Carlsson Johan

**Publication Year:** 2017

**Journal:** Elsevier Science Ltd

**DOI:** 10.1016/j.enpol.2017.11.012

**Total Potential Space Cooling Demand:** 174 TWh for EU service sector in an average year

**Specific Cooling Demand:** Not provided in the available content

**Methodology:** "A georeferenced approach based on comparing Cooling Degree Days and space cooling consumption in the USA" was employed, using the United States as a reference point due to its mature, nearly saturated cooling market. The analysis examined NUTS-3 regions across EU member states.

**Figures:** Only the total estimate (174 TWh) is presented in the excerpt provided.

**Additional Note:** The document indicates this research could support "EU MSs' Comprehensive Assessments related to the Article 14 of the Energy Efficiency Directive."

## WebSearch: EU Building Stock Observatory cooling service sector non-residential air conditioned floor area share indicator

Web search results for query: "EU Building Stock Observatory cooling service sector non-residential air conditioned floor area share indicator"

Links: [{"title":"Status Quo of the Air-Conditioning Market in Europe: Assessment of the Building Stock – DOAJ","url":"https://doaj.org/article/7a0e70cfd16f421388288db251c89655"},{"title":"Status Quo of the Air-Conditioning Market in Europe: Assessment of the Building Stock","url":"https://www.mdpi.com/1996-1073/10/9/1253"},{"title":"(PDF) Status Quo of the Air-Conditioning Market in Europe: Assessment of the Building Stock","url":"https://www.researchgate.net/publication/319244880_Status_Quo_of_the_Air-Conditioning_Market_in_Europe_Assessment_of_the_Building_Stock"},{"title":"Status Quo of the Air-Conditioning Market in Europe - Hotmaps Project","url":"https://www.hotmaps-project.eu/energies-status-quo-air-conditioning-market-europe-assessment-building-stock/"},{"title":"Towards Hydrogen Autarky? Evaluating Import Costs and Domestic Competitiveness in European Energy Strategies","url":"https://arxiv.org/pdf/2510.04669"},{"title":"GLOBUS: Global building renovation potential by 2070","url":"https://arxiv.org/pdf/2406.04133"},{"title":"European Building Stock Analysis","url":"https://www.datocms-assets.com/31538/1643788710-ebsa_web_2.pdf"}]

The search results provide relevant information about air-conditioning in the EU building stock, particularly regarding cooling services and floor area distribution. Here are the key findings:

## Key Results

Research on the European air-conditioning market has collected data describing the actual space cooling market in Europe including cooled floor area per AC type and sector.

**Service Sector Share:** The percentage of total cooled floor area is much higher in the service sector (greater than 60%) compared to households (less than 40%).

**Energy Consumption:** Space cooling is responsible for a significant portion of EU electricity consumption in households (nearly 5%) and even more in the service sector (~13%).

**Sector Breakdown:** The wholesale and retail sector is the most energy consuming sector for AC application, while offices, hotels and bars, and health sectors each consume more than 20 TWh/y for AC purposes.

The research addresses the knowledge gap regarding non-residential air-conditioned floor area by providing a comprehensive bottom-up assessment of the EU building stock across different service sectors including wholesale and retail, offices, education, health, hotels and bars.


REMINDER: You MUST include the sources above in your response to the user using markdown hyperlinks.

## WebFetch: https://www.mdpi.com/1996-1073/10/9/1253

The server returned HTTP 403 Forbidden.

The response body was not retrieved. If this URL requires authentication, use an authenticated tool (e.g. `gh` for GitHub, or an MCP-provided fetch tool) instead of WebFetch.

## WebSearch: Odyssee-Mure "air conditioning" service sector tertiary electricity consumption cooling trend Europe 2024 indicator

Web search results for query: "Odyssee-Mure "air conditioning" service sector tertiary electricity consumption cooling trend Europe 2024 indicator"

Links: [{"title":"Status Quo of the Air-Conditioning Market in Europe: Assessment of the Building Stock","url":"https://doi.org/10.3390/en10091253"},{"title":"EU Heat Supply & Energy Efficiency: Key Indicators for Sustainable Heating and Cooling | ODYSSEE-MURE","url":"https://www.odyssee-mure.eu/publications/policy-brief/european-heat-supply.html"},{"title":"Heating consumption per m2 | Heating energy consumption | ODYSSEE-MURE","url":"https://www.odyssee-mure.eu/publications/efficiency-by-sector/households/heating-consumption-per-m2.html"},{"title":"Energy intensity of service sector in EU | ODYSSEE-MURE","url":"https://www.odyssee-mure.eu/publications/efficiency-by-sector/services/service-energy-intensity-eu.html"},{"title":"Air conditioning per dwelling | ODYSSEE-MURE","url":"https://www.odyssee-mure.eu/publications/efficiency-by-sector/households/unit-consumption-air-conditioning.html"},{"title":"Introduction to the Odyssee-Mure Project | ODYSSEE-MURE","url":"https://www.odyssee-mure.eu/project.html"},{"title":"European Union energy efficiency & Trends policies | European Union profile | ODYSSEE-MURE","url":"https://www.odyssee-mure.eu/publications/efficiency-trends-policies-profiles/european-union.html"},{"title":"EU Heating Energy | Heating energy consumption by energy source | ODYSSEE-MURE","url":"https://www.odyssee-mure.eu/publications/efficiency-by-sector/households/heating-energy-consumption-by-energy-sources.html"},{"title":"Odyssee-mure","url":"https://www.odyssee-mure.eu/publications/efficiency-by-sector/households/energy-consumption-by-end-use.html"}]

I've performed the web search for your query about Odyssee-Mure data on air conditioning, service sector tertiary electricity consumption, and cooling trends in Europe for 2024. Here are the relevant findings:

## Key Results:

**Air Conditioning Trends:**
In 2023, air conditioning represented only 2.4% of household electricity consumption in the EU, though per-dwelling consumption is rising from 21 kWh/household in 2000 to 87 kWh/household in 2023, driven by growing use of air conditioning appliances. Air conditioning is particularly significant in Cyprus, Malta, Greece, Croatia and Italy, representing 8% to 29% of household electricity consumption.

**Service Sector Energy Intensity:**
The increasing energy intensity in some EU countries, particularly Greece and Portugal, is probably linked to tourism and a larger diffusion of air conditioning.

**About ODYSSEE-MURE Project:**
ODYSSEE-MURE provides comprehensive monitoring of energy consumption and efficiency trends as well as evaluation of energy efficiency policy measures by sector for all 27 EU Member States, the EU as a whole, Switzerland and other Energy Community Contracting Parties.

The search results show the project's focus on sectoral energy consumption monitoring including the services/tertiary sector, with air conditioning emerging as an increasingly important indicator, particularly in southern European countries.


REMINDER: You MUST include the sources above in your response to the user using markdown hyperlinks.

## WebFetch: https://www.odyssee-mure.eu/publications/efficiency-by-sector/services/service-energy-intensity-eu.html

# Service Sector Energy Intensity Indicators

## Key Metrics Reported

The document provides the following numeric indicators:

**EU-27 Level:**
- Energy intensity decreased at "-2.5%/year" since 2010
- Energy intensity increased at "+0.7%/year" until 2010

**Country-Specific Trends (2000-2023):**
- 5 Member States showed increasing energy intensity overall
- Romania and Cyprus experienced rises "over 1% per year"
- Luxembourg, Greece, Bulgaria, and Croatia saw increases through 2019, followed by decreases "exceeding 3% per year" after 2019

## Energy Intensity Units

The document does not specify the actual units of measurement (e.g., koe/€, kWh/m²) or provide absolute values for Belgium, Spain, Italy, or Greece individually.

## Additional Context

- "Energy consumption at normal climate" adjustment is applied
- **Data Currency:** Last updated November 2025
- No mention of air conditioning or cooling's role in service sector energy consumption

## Data Access

Specific numeric values and country breakdowns are reportedly available through interactive tools (Energy intensity visualization), though the values themselves are not displayed in this page excerpt.

## WebSearch: "air conditioned floor area" OR "cooled floor area" Europe service sector Mm2 offices retail hotels Energies 2017 Jakubcionis OR "Status Quo of the Air-Conditio

Web search results for query: ""air conditioned floor area" OR "cooled floor area" Europe service sector Mm2 offices retail hotels Energies 2017 Jakubcionis OR "Status Quo of the Air-Conditioning Market""

Links: [{"title":"(PDF) Status Quo of the Air-Conditioning Market in Europe: Assessment of the Building Stock","url":"https://www.researchgate.net/publication/319244880_Status_Quo_of_the_Air-Conditioning_Market_in_Europe_Assessment_of_the_Building_Stock"},{"title":"(PDF) Status Quo of the Air-Conditioning Market in Europe: Assessment of the Building Stock","url":"https://www.academia.edu/66332526/Status_Quo_of_the_Air_Conditioning_Market_in_Europe_Assessment_of_the_Building_Stock"},{"title":"Status Quo of the Air-Conditioning Market in Europe: Assessment of the Building Stock","url":"https://www.mdpi.com/1996-1073/10/9/1253"},{"title":"Status Quo of the Air-Conditioning Market in Europe: Assessment of the Building Stock – DOAJ","url":"https://doaj.org/article/7a0e70cfd16f421388288db251c89655"},{"title":"Status Quo of the Air-Conditioning Market in Europe - Hotmaps Project","url":"https://www.hotmaps-project.eu/energies-status-quo-air-conditioning-market-europe-assessment-building-stock/"},{"title":"Space Cooling Market in Europe: Assessment of the Final Energy Consumption for the Year 2016","url":"https://doi.org/10.3390/su14052667"},{"title":"(PDF) Assessing the market for air conditioning systems in European buildings","url":"https://www.academia.edu/87358042/Assessing_the_market_for_air_conditioning_systems_in_European_buildings"},{"title":"www.oeko.de Contribution of Renewable Cooling to the Renewable Energy","url":"https://www.oeko.de/fileadmin/oekodoc/Renewable-Cooling.pdf"}]

Perfect! I found the paper you're looking for. The study "Status Quo of the Air-Conditioning Market in Europe: Assessment of the Building Stock" was published in Energies 2017, 10(9), 1253, and collected data describing the actual space cooling market in Europe including cooled floor area per AC type and/or sector.

The research addresses the shortfall of data for the services sector (wholesale and retail, offices, education, health, hotels and bars) by exploring cooled floor area per AC type and sector using a bottom-up approach.

The results indicate that space cooling is responsible for a significant portion of EU electricity consumption in households (nearly 5%) and even more in the service sector (~13%).

The paper is available at the MDPI Energies journal at https://www.mdpi.com/1996-1073/10/9/1253, where you can access the full publication with the specific Mm2 (million square meters) data for offices, retail, and hotels in the European service sector.


REMINDER: You MUST include the sources above in your response to the user using markdown hyperlinks.

## WebFetch: https://www.iea.org/reports/the-future-of-cooling

You've hit your session limit · resets 10:10pm (Europe/Paris)
