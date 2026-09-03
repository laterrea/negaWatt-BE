# Recovered work — research-eating-out

Agent `a4d58fe200b1bee73`, stopped by the session limit before it could report.
Everything below is recovered verbatim from its transcript, in order. **Nothing here has been checked** — treat every figure as unverified until re-sourced.

17 entries, of which 9 web results.

---

## WebFetch: https://ec.europa.eu/eurostat/statistics-explained/index.php?title=Household_consumption_by_purpose

# Household Consumption on Restaurants and Accommodation Services (CP11)

## EU Average
The webpage indicates that restaurants and accommodation services represented **9.2% of total EU household expenditure in 2024**, or "4.8% of GDP."

## Country-Specific Data
Unfortunately, the article does not provide the specific breakdown for individual countries you requested (Belgium, Spain, Ireland, Greece, Portugal, Cyprus, Malta, Austria). The document focuses on EU aggregates and only discusses country variations for three categories: housing, food and beverages, and transport.

## Reference Information
- **Reference Year:** 2024
- **Data Extraction Date:** 20 November 2025
- **Eurostat Dataset Code:** nama_10_cp18

The article notes that restaurants and accommodation services showed notable growth, with "the largest increases in shares between 2023 and 2024 were recorded for restaurants and accommodation services (+0.2 pp)" and experienced significant price increases of "+5.1%" compared to 2023.

To obtain country-specific data for the nations you mentioned, you would need to access the underlying Eurostat database directly rather than this summary article.

## reasoning

The API is returning empty for BE/CP11/PC_TOT/2024 — I'm noticing the dataset nama_10_co3_p3 uses the older COICOP 1999 classification labeled "Restaurants and hotels," while the Statistics Explained article's 9.2% EU figure actually comes from nama_10_cp18, which uses COICOP 2018 where CP11 means "Restaurants and accommodation services." That mismatch likely explains why Belgium has no data under this unit.  I should switch to querying nama_10_cp18 directly and check its dimension structure before running more queries.

## WebSearch: Eurostat dataset nama_10_co3_p3 COICOP 2018 household final consumption expenditure by purpose restaurants accommodation services percentage of total

Web search results for query: "Eurostat dataset nama_10_co3_p3 COICOP 2018 household final consumption expenditure by purpose restaurants accommodation services percentage of total"

Links: [{"title":"Household consumption by purpose - Statistics Explained - Eurostat","url":"https://ec.europa.eu/eurostat/statistics-explained/index.php?title=Household_consumption_by_purpose"},{"title":"Household final consumption expenditure by purpose (COICOP 2018) (nama_10_cp18)","url":"https://ec.europa.eu/eurostat/cache/metadata/en/nama_10_cp18_esms.htm"},{"title":"Eurostat/nama_10_co3_p3 on DBnomics","url":"https://db.nomics.world/Eurostat/nama_10_co3_p3"},{"title":"Final consumption expenditure of households by consumption purpose (COICOP 3 digit)","url":"https://fgeerolf.com/data/eurostat/nama_10_co3_p3.html"},{"title":"ec.europa.eu","url":"https://ec.europa.eu/eurostat/statistics-explained/SEPDF/cache/49480.pdf"},{"title":"final consumption expenditure of households restaurants hotels eurostat data","url":"https://tradingeconomics.com/euro-area/final-consumption-expenditure-of-households-restaurants-hotels-eurostat-data.html"},{"title":"final consumption expenditure of households restaurants hotels eurostat data","url":"https://tradingeconomics.com/european-union/final-consumption-expenditure-of-households-restaurants-hotels-eurostat-data.html"}]

Based on the search results, I found information about the Eurostat dataset nama_10_co3_p3 and household consumption data related to restaurants and accommodation services.

The nama_10_co3_p3 dataset contains information on household consumption by purpose using the COICOP classification system. Final consumption expenditure of households components are estimated according to COICOP 2018, which is an international reference classification system for household expenditure.

Regarding restaurants and accommodation services specifically, in 2023, 'Restaurants and accommodation services' accounted for 9.1% of total EU household expenditure (or 4.7% of GDP). More recent data shows that comparing with 2023, restaurants and accommodation services increased by 5.1%.

The search results also provide country-specific data on restaurants and accommodation services as a percentage of total household consumption expenditure, with variations ranging from around 6-7% in some countries to over 20% in Malta.


REMINDER: You MUST include the sources above in your response to the user using markdown hyperlinks.

## reasoning

Belgium 2024 shows CP11 at 6.4% of total household consumption expenditure, with CP111 "food and beverage serving services" at 5.9% and CP112 "accommodation services" at just 0.6% — this cleanly separates eating out from hotel stays, which is exactly what I need. I should now pull the same breakdown for EU27 and other countries to compare.

I'm noting that COICOP 2018's CP11 "restaurants and accommodation services" has a slightly different scope than COICOP 1999's "restaurants and hotels" — worth flagging. I also want to verify the EU27_2020 average, since the Statistics Explained article cited 9.2% for 2024, much higher than Belgium's 6.4%, and check whether 2025 data is available in the nama_10_cp18 dataset given that nama_10_co3_p3 already had 2025 figures.

## said

Belgium 2024: CP11 = 6.4%, with CP111 (food & beverage serving) = 5.9%. Now let me pull all countries and check for 2025 data.

## reasoning

2024 has the most complete data with both Belgium and EU27 figures, letting me cross-validate against the earlier article. EU27 shows CP11 at 9.2% (matching the Statistics Explained figure), with CP111 at 7.1% and CP112 at 2.1%, while Belgium comes in lower at 6.4% overall. I'm also noting the highest-ranking countries like Malta, Croatia, Cyprus, and Ireland for comparison.

For "eating out" specifically, CP111 (food and beverage serving services) is the cleanest metric: Belgium 5.9%, EU27 7.1%, with Ireland and Spain notably higher. I'm concerned that Croatia, Cyprus, Malta, and Montenegro's high CP112 accommodation figures might reflect the "domestic concept" of national accounts, which includes non-resident tourist spending rather than just resident household expenditure — I need to verify whether this dataset uses domestic or national concept, since that distinction matters significantly for tourism-heavy economies.

I want to check the Eurostat metadata to confirm this, since Malta's 21.1% figure is likely driven largely by inbound tourism rather than domestic consumption patterns. I'll also verify the Belgium figures through the data browser, then move on to researching Belgian horeca establishments and Sciensano's data on eating out habits, working through these in parallel.

## said

Comprehensive data obtained. Let me verify the concept (domestic vs national) in the metadata and start on the Belgian sources.

## WebFetch: https://ec.europa.eu/eurostat/cache/metadata/en/nama_10_cp18_esms.htm

# Dataset Concept and Coverage

**Concept Used:** DOMESTIC

**Exact Statement:**
"Data in nama_10_cp18 are based on the domestic concept, i.e. including expenditures of non-residents on the domestic territory and excluding residents expenditures outside of the domestic territory."

The document clarifies that while a national concept version exists (covering resident household consumption at home and abroad), the COICOP 2018 breakdown uses the domestic approach.

**Statistical Unit:** Households as resident units of the economic territory with a centre of predominant economic interest

**Coverage:** Final consumption expenditure of the household sector, broken down by 13 main COICOP 2018 categories (food, housing, transport, health, education, etc.)

**Reference Period:** Calendar year for annual data, with information requested at t+9 months after the reference period

## WebSearch: Statbel horeca nombre d'établissements Belgique restaurants cafés statistiques entreprises

Web search results for query: "Statbel horeca nombre d'établissements Belgique restaurants cafés statistiques entreprises"

Links: [{"title":"4.000 entreprises du secteur Horeca créées en 10 ans | Gondola","url":"https://www.gondola.be/fr/news/4000-entreprises-du-secteur-horeca-creees-en-10-ans"},{"title":"Etablissements par classe de taille et secteur d'activité - Iweps","url":"https://www.iweps.be/indicateur-statistique/etablissements-classe-de-taille/"},{"title":"Horeca - Le Moniteur Belge du fonds de commerce et de l'entreprise","url":"https://www.lemoniteur.be/documentation/horeca-135.html"},{"title":"Restaurant Business: Aperçu de secteur · Evolution Media Group","url":"https://www.evolution.be/horeca/restaurant-business-apercu-secteur/?lang=fr"},{"title":"Horeca workers | Statbel","url":"https://statbel.fgov.be/en/themes/datalab/horeca-workers"},{"title":"Horeca en chiffres : impressionnant","url":"https://www.misterhoreca.be/news-Horeca-en-chiffres--impressionnant--19.aspx"},{"title":"nombre etablissements secteur restauration tradition par taille france","url":"https://fr.statista.com/statistiques/963220/nombre-etablissements-secteur-restauration-tradition-par-taille-france"}]

Voici les résultats de la recherche web concernant les statistiques Statbel sur le secteur horeca en Belgique :

## Nombre d'établissements horeca en Belgique

En dix ans, plus de 4.000 entreprises de restauration se sont ajoutées, passant de 22.575 fin 2013 à 26.748 fin 2023.

Des données plus anciennes montrent que fin 2015, la Belgique totalisait 57 120 commerces horeca, ce qui inclut les hôtels, restaurants et débits de boissons.

## Importance du secteur

La restauration est l'un des secteurs qui connaît la plus forte croissance dans notre pays. De plus, les établissements horeca sont nombreux en Belgique et le secteur constitue une des activités économiques les plus importantes du pays.

## Données sur les travailleurs

En ce qui concerne l'emploi, début 2020, 113,744 personnes travaillaient dans des établissements servant de la nourriture (restaurants) et 21,541 dans des établissements servant des boissons (cafés et bars).

Ces statistiques montrent une dynamique positive du secteur horeca belge au cours de la dernière décennie.


REMINDER: You MUST include the sources above in your response to the user using markdown hyperlinks.

## reasoning

I should note that nama_10_cp18 uses domestic concept, which explains why Malta, Croatia, and Cyprus show high accommodation shares from inbound tourism, while Belgium's lower share partly reflects Belgians spending abroad. I want to dig into Statbel's primary sources for horeca establishment numbers, checking their workers datalab page and business demography/VAT registration data.

I should also check the Sciensano food consumption survey for out-of-home eating data and browse Statbel's open data portal for relevant statistics.

## WebFetch: https://statbel.fgov.be/en/themes/datalab/horeca-workers

# Belgian Horeca Sector Data

I cannot extract the requested information about the Belgian horeca sector from the provided content. The webpage shown appears to be a CAPTCHA verification page containing:

- A security test prompt asking users to prove they're human
- An encoded image (Red dot GIF)
- A bottle icon image
- A support ID number (3644462762287095953)
- A submit button

**No data about Belgian horeca establishments, worker numbers, sector breakdowns, reference years, or NACE codes is present in this document.**

To obtain this information, you would need to consult sources such as:
- Belgian statistical office (Statbel)
- Chamber of Commerce databases
- Horeca sector association reports
- Eurostat databases using NACE codes (55-56 for accommodation and food service)

## WebSearch: Sciensano enquête de consommation alimentaire repas hors domicile pourcentage Belgique voedselconsumptiepeiling buitenshuis eten

Web search results for query: "Sciensano enquête de consommation alimentaire repas hors domicile pourcentage Belgique voedselconsumptiepeiling buitenshuis eten"

Links: [{"title":"Les habitudes alimentaires en Belgique : un décalage persistant avec les recommandations en vigueur | sciensano.be","url":"https://www.sciensano.be/fr/coin-presse/les-habitudes-alimentaires-en-belgique-un-decalage-persistant-avec-les-recommandations-en-vigueur"},{"title":"Enquête de consommation alimentaire 2022-2023: Rapport de synthèse sur la consommation alimentaire et le respect des recommandations alimentaires dans la population belge | sciensano.be","url":"https://www.sciensano.be/fr/biblio/enquete-de-consommation-alimentaire-2022-2023-rapport-de-synthese-sur-la-consommation-alimentaire-et"},{"title":"Consommation d’aliments préparés en-dehors du domicile | sciensano.be","url":"https://www.sciensano.be/fr/resultats-de-lenquete-nationale-de-consommation-alimentaire-2022-2023/habitudes-alimentaires/consommation-daliments-prepares-en-dehors-du-domicile"},{"title":"Résultats de l'Enquête nationale de consommation alimentaire 2022-2023 | sciensano.be","url":"https://www.sciensano.be/fr/resultats-de-lenquete-nationale-de-consommation-alimentaire-2022-2023"},{"title":"Enquête de consommation alimentaire 2022-2023 | sciensano.be","url":"https://www.sciensano.be/fr/enquete-de-consommation-alimentaire-2022-2023"},{"title":"ENQUÊTE DE CONSOMMATION ALIMENTAIRE 2014-2015","url":"https://www.sciensano.be/sites/default/files/resume_rapport1_fr.pdf"},{"title":"Enquête nationale de consommation alimentaire | sciensano.be","url":"https://www.sciensano.be/fr/projets/enquete-nationale-de-consommation-alimentaire-0"},{"title":"MINISTÈRE DE LA RÉGION WALLONNE Observatoire de la Consommation Alimentaire","url":"https://www.gembloux.ulg.ac.be/economie-et-developpement-rural/wp-content/uploads/sites/34/2018/10/2011_OCA_Evolution-des-d%C3%A9penses-pour-la-restauration-hors-domicile_2011.pdf"}]

Based on the search results, here is information from the Sciensano food consumption survey regarding meals eaten outside the home in Belgium:

In Belgium, 21% of the population eats meals with table service at least once a week, and 7% order delivery meals at least once a week. 

More specifically, the results come from the third edition of the food consumption survey which took place in 2022-2023. The proportion of people consuming delivery meals at least once a month is higher in Brussels than in Flanders and Wallonia.

Additionally, from the earlier 2014-2015 survey, dinner is consumed at home in 86% of cases, which indicates that 14% of dinners are consumed outside the home.

The survey also notes that there is a real need for data on this subject to analyze the impact of eating outside the home on food intake and health, and such data are important for monitoring the nutritional quality and energy intake of foods and beverages prepared outside the home.


REMINDER: You MUST include the sources above in your response to the user using markdown hyperlinks.

## WebFetch: https://www.sciensano.be/fr/resultats-de-lenquete-nationale-de-consommation-alimentaire-2022-2023/habitudes-alimentaires/consommation-daliments-prepares-en-deho

timeout of 60000ms exceeded

## reasoning

Eurostat SBS shows 50,738 enterprises in Belgium's NACE I56 "Food and beverage service activities" for 2024, which is higher than the 26,748 restaurants figure since I56 also covers event catering, other food service, and beverage serving activities like cafés and bars. I need to break this down further by the 4-digit codes, and note that this figure counts enterprises specifically.

Statbel typically also tracks "établissements" (local units) separately, so I should verify against their VAT-registered enterprises dataset and try to pull Statbel's open data directly, while also retrying the Sciensano lookup in parallel.

## reasoning

I'm pulling together Belgium's 2024 Eurostat SBS figures for accommodation and food service enterprises, breaking down hotels, restaurants, and catering into their sub-categories.

I want to cross-check this against Statbel's own published figure, but they seem to be blocking access with a CAPTCHA. I'll try Sciensano again and look into the Statbel open data portal (bestat.statbel.fgov.be) as alternative sources.

## WebSearch: Statbel "unités d'établissement" horeca 2024 nombre restaurants débits de boissons Belgique statistiques TVA

Web search results for query: "Statbel "unités d'établissement" horeca 2024 nombre restaurants débits de boissons Belgique statistiques TVA"

Links: [{"title":"Etablissements par classe de taille et secteur d'activité - Iweps","url":"https://www.iweps.be/indicateur-statistique/etablissements-classe-de-taille/"},{"title":"Lentement mais sûrement, le monde de la restauration reprend des couleurs","url":"https://www.horecawallonie.be/sites/default/files/2025-06/L%27Echo%2021%20juin%202025.pdf"},{"title":"3 Tableau de bord Santé du secteur Horeca et marché de l’emploi Avril 2024","url":"https://horecaformabepro.be/wp-content/uploads/2024/05/Tableau-de-bord-Avril.pdf"},{"title":"Horeca workers | Statbel","url":"https://statbel.fgov.be/en/themes/datalab/horeca-workers"},{"title":"Statistiques sur les unités établissements","url":"https://statbel.fgov.be/fr/themes/datalab/statistiques-sur-les-unites-etablissements"},{"title":"Statistiken zu Tourismus & Gastronomie in Belgien","url":"https://de.statista.com/map/europa/belgien/tourismus-gastronomie"}]

I found some relevant information from the search results. Here are the key findings:

**Horeca Statistics in Belgium (2024):**

According to Statbel figures compiled by hub.brussels, 187 new horeca activities were launched in the Brussels Capital Region in the first quarter of 2024.

**Employment in Horeca:**

Horeca employed 67,098 people in Wallonia in the third quarter of 2024 across all types of contracts.

**Establishment Size:**

There were 89,246 establishments with at least one salaried employee in Wallonia as of June 30, 2024, and 36.9% of establishments with fewer than 20 workers are active in commerce and horeca sectors.

**Business Trends:**

At the beginning of 2024, the number of horeca companies remained relatively stable.

However, the search results did not provide specific numbers about the total number of restaurants and beverage establishments ("débits de boissons") across all of Belgium, or detailed breakdown by TVA data for 2024. For more comprehensive and specific statistics on these aspects, you may want to directly consult Statbel's official databases on "unités d'établissement" for the complete 2024 figures.


REMINDER: You MUST include the sources above in your response to the user using markdown hyperlinks.

## WebFetch: https://www.sciensano.be/fr/resultats-de-lenquete-nationale-de-consommation-alimentaire-2022-2023/habitudes-alimentaires/consommation-daliments-prepares-en-deho

timeout of 60000ms exceeded
