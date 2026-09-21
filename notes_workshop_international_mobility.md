# Workshop review — international mobility · progress tracker

Review comments by Sylvain Quoilin, worked through sequentially.
Status: ⬜ todo · 🟡 in progress · ✅ done · ⛔ blocked / dropped (with reason)

Files expected to be touched: `website/workshop/content/international-mobility.yaml`,
`website/workshop/content/ui.yaml`, `workshop_levers/international_mobility.py`,
`scripts/build_workshop_content.py`, `nW_BE_demand_model_transports.ipynb`.

Rebuild after each batch:
```bash
python scripts/build_workshop_content.py --check
```
Levers-module changes additionally need the transport notebook re-run.

---

## Task list (from the raw notes, reorganised)

### Q1 — `long-haul-flights` (Voyages long-courriers)

| # | Task | Status |
|---|---|---|
| 1.1 | Fiche "Tendance passée": remplacer le texte par un graphique | ✅ courbe JRC-IDEES 2000-2023 tracée (nouveau `series:` dans le build, voir I.3) |
| 1.2 | Enlever la fiche "Contexte belge" | ✅ retirée ; l'ancrage « 1 habitant sur 10 » déplacé dans le sous-titre |
| 1.3 | Titre de la fiche benchmark → "Ailleurs" (les chiffres sont mondiaux) | ✅ via le nouveau `label:` (I.2) |
| 1.4 | Enlever la fiche "Concrètement" | ✅ |
| 1.5 | Ajouter une fiche: part de la population mondiale n'ayant jamais pris l'avion | ✅ deux fiches : l'échelle de revenus de l'ICCT (76 % n'ont pris aucun vol en 2019) **avec graphique**, et une mise en garde sur le « 80 % n'ont jamais volé », qui ne repose sur rien |

### Q2 — `short-haul-flights` (Voyages court-courriers)

| # | Task | Status |
|---|---|---|
| 2.1 | Fiche "Tendance passée": graphique au lieu du texte (comme 1.1) | ✅ même courbe |
| 2.2 | Fiche "Contexte belge": préciser que la comparaison voiture vaut pour un seul occupant | ✅ voir la note ci-dessous — la base était déjà « par passager », la fiche le dit maintenant |
| 2.3 | Ajouter une fiche sur le potentiel de report modal avion → train | ✅ deux fiches, deux graphiques — voir ci-dessous |
| 2.4 | Courbe de Lorenz de la répartition des vols, si les données le permettent | ✅ **oui, les données existent** — courbe tracée (Royaume-Uni, Büchs & Mattioli 2021) |

### Q3 — `long-haul-load` (Passagers par vol long-courrier)

| # | Task | Status |
|---|---|---|
| 3.1 | Titre → "Remplissage des vols long-courriers en 2050" | ✅ « Quel remplissage des vols long-courriers viser en 2050 ? » (forme interrogative gardée pour rester homogène avec les autres cartes — dis-moi si tu préfères ta formulation littérale) |
| 3.2 | Enlever la fiche "Contexte belge" | ✅ |
| 3.3 | Clarifier que la consommation moyenne par avion est constante dans la feuille de calcul; quantifier l'effet "avions plus gros = plus de conso" et justifier qu'il est faible — sinon revoir la formulation du levier | ✅ fiche « Ce que le calcul ne relie pas » sur les deux questions de remplissage, chiffrée sur nos propres séries — voir l'analyse ci-dessous |
| 3.4 | Données sur le taux de remplissage moyen actuel + expliquer le passage taux de remplissage ↔ passagers/vol | ✅ deux fiches par question : « Sièges × remplissage » (l'identité + IATA 82,6 % en 2019, 83,4 % en 2024, + graphique EUROCONTROL des sièges par vol) et « Jusqu'où peut-on remplir ? » (Ryanair ~96 % sur l'année) |

### Q4 — `long-haul-fuel` (Consommation par km-avion, long-courrier)

| # | Task | Status |
|---|---|---|
| 4.1 | Trouver des données historiques; les tracer; si la tendance est visible sur le graphique, retirer la fiche "Tendance passée" | ✅ **série annuelle 2000-2023 trouvée et vérifiée** (voir « la trouvaille » plus bas). Le levier porte maintenant une vraie courbe observée au-dessus du curseur |
| 4.2 | Enlever la fiche "Contexte belge" | ✅ |
| 4.3 | Renommer "Comparaison internationale" → "Décarbonation de l'aviation"; insister sur les 3 % qui correspondent au levier | ✅ renommée ; les 3 % sont mis en évidence — avec une nuance, voir la note |
| 4.4 | Enlever la fiche "Concrètement" | ✅ |

### Q5 — `short-haul-load` (Passagers par vol court-courrier)

| # | Task | Status |
|---|---|---|
| 5.1 | Déplacer la question juste avant/après Q3 (levers similaires) | ✅ ordre de jeu : long-courrier ×2, remplissages ×2, consommation, hydrogène |
| 5.2 | Mêmes remarques que Q3 (3.2, 3.3, 3.4) | ✅ fiche « contexte belge » retirée (son unique information de portée est passée dans le sous-titre) ; titre aligné sur celui de Q3 ; fiches 3.3 et 3.4 ajoutées aux deux questions |
| 5.3 | Enlever la fiche "À manier avec prudence" | ✅ |

### Q6 — `short-haul-fuel` (Consommation par km-avion, court-courrier)

| # | Task | Status |
|---|---|---|
| 6.1 | Fusionner avec Q4 (questions quasi équivalentes) | ✅ un seul levier `plane-fuel`, et la fusion est **exacte**, pas approchée — voir ci-dessous |
| 6.2 | Reprendre les remarques 4.1 à 4.4 sur la question fusionnée | ✅ |

### Q7 — `hydrogen-flights`

Rien à faire. ✅

### Q8 — Question supplémentaire: report modal du fret aérien

| # | Task | Status |
|---|---|---|
| 8.1 | Vérifier si le report modal du fret aérien est une hypothèse de la feuille de calcul | ✅ **non** — §3.1.4 dit explicitement « we do not, at this stage, envisage any modal shift », et §3.1 porte déjà ton commentaire « We should maybe consider reductions of aviation intensity, and other modal shifts » |
| 8.2 | Si oui (ou si documentable): nouveau levier + fiches | ✅ levier `air-freight` (notebook §3.1.4 + module + 7 fiches), avec tendance historique tracée, « Ce qui voyage par avion », « Liège, un des plus gros d'Europe » et « Le dossier Alibaba » |

### Infrastructure

| # | Task | Status |
|---|---|---|
| I.1 | Vérifier le support des images dans les fiches | ⛔ **les images ne sont pas supportées** : une fiche ne peut porter qu'un graphique SVG en ligne (`chart:`). Rien dans `build_workshop_content.py`, `play.js`, `cards.js` ni `reveal.js` ne rend une image. Aucun point de cette revue n'en a réellement besoin — tout ce qui suit est fait en graphique. Dis-moi si tu veux que j'ajoute le support d'images (photo du hub de Liège, par ex.) |
| I.2 | Étiquette de fiche personnalisée (`label:`) — nécessaire pour 1.3 et 4.3 | ✅ ajoutée : `label: {fr,nl,en}` sur n'importe quelle fiche, contrôlée comme les autres chaînes (trois langues, pas d'astérisques, pas de spoiler) |
| I.4 | Écran de fin : graphique ± des effets | ✅ activé pour ce thème aussi (`summaryChart:` dans le YAML) — les sept leviers ont tous une réponse exploitable. Une ligne à supprimer si tu n'en veux pas ici |
| I.3 | Tracer une série historique dans une fiche sans recopier les nombres | ✅ ajouté : `chart: {kind: line, series: <clé de history_transport.js>, from:, to:}` — la courbe est lue au build, donc la règle « aucun nombre du modèle tapé à la main » vaut aussi pour les graphiques de séries |

---

## La trouvaille qui a débloqué Q3, Q4, Q5 et Q6

Le classeur **JRC-IDEES-2023 `Transport` pour la Belgique** était présent sur la
machine (téléchargé lors d'une session précédente). Il contient, par année de 2000 à
2023 et séparément pour l'intra-européen et le long-courrier :

* les voyageurs-kilomètres et les kilomètres-avion → le **remplissage** (pkm/vkm) ;
* la « vehicle-efficiency - effective (kgoe/100 km) » → la **consommation par
  kilomètre d'avion** ;
* les tonnes-kilomètres, les tonnes et les vols du **fret aérien**.

Ses valeurs 2019 reproduisent **exactement** les quatre ancrages sur lesquels §2.3
démarre ses projections (593,771 et 578,489 kgep/100 km ; 121,927 et 187,817
passagers par vol). Les séries sont donc bien celles du modèle, et pas une source
concurrente.

Conséquences, toutes vérifiées par assertion dans le notebook auxiliaire :

* six nouvelles séries observées dans `website/data/history_transport.js` ;
* **les questions 3, 4 et 5 ne sont plus « sans historique »** : le curseur prolonge
  désormais une courbe mesurée, comme le veut le principe de l'atelier ;
* les fiches « tendance passée » ne récitent plus des nombres, elles montrent la courbe.

Les valeurs sont recopiées dans le notebook auxiliaire sous forme de tableaux
`np.array`, comme toutes les autres séries JRC de ce fichier, avec la feuille et la
ligne exactes en commentaire. **Si tu préfères que le classeur lui-même entre dans le
dépôt** (838 ko, à côté des deux classeurs JRC-IDEES-2021 déjà versionnés), dis-le et
je remplace les tableaux par une lecture du fichier.

## La fusion des deux questions « consommation » (6.1)

Un seul levier `plane-fuel`, en kWh par kilomètre d'avion, **moyenne pondérée par les
kilomètres-avion** des deux types de vol : 68,2 en 2019 → 63,7 chez négaWatt (−6,7 %).

La fusion est **exacte et non approchée**, et c'est ce qui la rend légitime : l'énergie
vaut « kilomètres-avion × consommation par kilomètre », donc la moyenne pondérée par les
kilomètres-avion *est* l'énergie kérosène totale divisée par les kilomètres-avion totaux.
Quelle que soit la répartition entre intra et extra, la demande du thème est
proportionnelle à cette moyenne. Le calcul d'impact du curseur reste donc exact, et la
réponse de négaWatt est un nombre unique bien défini.

Ce que la fusion cache — et qu'une fiche `reveal` dit explicitement — c'est que le
scénario, lui, fixe deux valeurs très différentes : long-courrier −16 %, intra-européen
**+5 %**.

## 3.3 — l'indépendance entre remplissage et consommation, chiffrée

Ta remarque est exacte : `occu_trgt_PM_avi_*` et `redu_fuel_PM_avi_*` sont deux réglages
**indépendants** dans §2.3. Augmenter le nombre de passagers par vol ne touche pas la
consommation par kilomètre d'avion. C'est juste si les passagers occupent des sièges vides,
faux s'il faut de plus gros appareils.

Ce que disent nos propres séries (JRC-IDEES, 2000 → 2019, en évitant les années COVID) :

| | passagers par vol | consommation par km d'avion |
|---|---|---|
| intra-européen | 87,5 → 121,9 (**+39,3 %**) | 67,3 → 69,1 kWh (**+2,6 %**) |
| long-courrier | 153,8 → 187,8 (**+22,0 %**) | 109,0 → 67,3 kWh (**−38,3 %**) |

Autrement dit : sur vingt ans, les avions ont beaucoup grandi et la consommation par
kilomètre n'a quasiment pas bougé (court-courrier) ou s'est effondrée (long-courrier). Le
progrès technique a absorbé l'effet de taille. **L'approximation du modèle est donc faible
sur la trajectoire historique**, et la fiche le dit avec ces chiffres.

Ce que ces nombres ne prouvent pas, en revanche, c'est que l'effet de taille *pur* soit
petit : ils mesurent l'effet de taille **net** du progrès technique. Un A321 consomme bien
plus par kilomètre qu'un A319. La recherche en cours (consommation par siège selon la taille
de l'appareil) doit permettre de donner aussi cet ordre de grandeur-là ; si elle montre un
effet important, la formulation du levier devra être revue — c'est exactement l'alternative
que tu posais.

Note liée, qui touche la calibration : **`occu_trgt` est calibré sur 2000→2023**, comme les
consommations. Or 2023 est une année de rattrapage où les avions étaient anormalement pleins.
Sur 2000→2019 la tendance est de +39,3 % (intra) et +22,0 % (extra) au lieu de +49,5 % et
+33,4 %. La règle « la moitié de la tendance » donnerait donc **+19,7 % et +11 %** au lieu des
+25 % et +17 % retenus. La cible de remplissage est, elle, rendue *plus* ambitieuse par le
choix de 2023 — l'inverse de ce qui se passe sur la consommation.

## Ce qui mérite d'être revu dans le calcul lui-même

1. **La règle « la moitié de la tendance 2000→2023 » est faussée par le COVID.**
   2023 n'est pas une année normale : la consommation par kilomètre d'avion y est encore
   très au-dessus de 2019 (633 contre 594 kgep/100 km en intra, 629 contre 578 en extra).
   Mesurée jusqu'en 2019, la même règle donnerait **−19 % pour le long-courrier** (au lieu
   de −16 %) et **+1,3 % pour l'intra-européen** (au lieu de +5 %). Autrement dit, la
   dégradation admise sur le court-courrier est pour moitié un artefact de l'année choisie.
   Idem pour le remplissage (`occu_trgt`), calibré sur 2000→2023.
2. **Les deux leviers `occupancy` et `cons_fuel` sont indépendants dans le modèle**, alors
   qu'ils ne le sont pas dans la réalité (c'est ta remarque 3.3). Quantification en cours.
3. **Le fret aérien n'est piloté par rien** et pèse pourtant **3,53 TWh sur les 10,87 TWh**
   du thème en 2050, soit un tiers — contre 3,57 TWh en 2019. C'est le seul poste de tout
   l'atelier que le scénario laisse quasi inchangé, et §3.1 le signale déjà en commentaire.

## Q8 — ce qui a été ajouté, et ce qui a été vérifié

**Dans le carnet de calcul** (§3.1.4) : deux paramètres, `pro_FT_spe_avi` (variation de
l'intensité de fret aérien en plus de la baisse générale) et `sft_FT_rel_avi_to_trn` (la part
reportée sur le rail), **tous deux à zéro**. Les exports sont identiques octet pour octet
avant et après : le scénario n'a pas bougé d'un chiffre. Ce qui change, c'est qu'une
hypothèse implicite est devenue visible et manipulable.

Pourquoi le rail comme destination, et pas le bateau : le bateau n'existe pas dans ce modèle
de demande. Le seul mode maritime présent est le cabotage (0,19 tkm/personne), et le
transport maritime international n'y est pas du tout. Envoyer les tonnes vers le cabotage
aurait été pire que vers le rail : son intensité dans le modèle est de 0,46 kWh/tkm, ce qui
est très au-dessus de ce que consomme un porte-conteneurs — chiffre à vérifier un jour, il
est probablement faux. Le rail, lui, est représenté correctement (0,04 kWh/tkm) et le
corridor ferroviaire Chine-Europe existe réellement. La fiche « à manier avec prudence » dit
noir sur blanc que la vraie alternative est le bateau et qu'il est hors modèle.

**Vérification à valeur non nulle.** J'ai fait tourner une copie du notebook avec
`pro_FT_spe_avi = -0,30` et `sft_FT_rel_avi_to_trn = +0,30` :

| | référence | avec −30 % |
|---|---|---|
| fret aérien | 313,7 tkm/pers. | 219,6 |
| énergie fret aérien 2050 | 3,53 TWh | 2,47 |
| total mobilité internationale 2050 | 10,87 TWh | 9,81 |
| total mobilité intérieure 2050 | 23,076 TWh | 23,111 (le rail encaisse les tonnes) |

Aucun contrôle interne du modèle ne s'est plaint, et les parts modales somment toujours à
100 %. Les fichiers exportés ont été restaurés à leur valeur de référence après le test.

Note au passage : mon propre contrôle de cohérence des parts modales était **trop strict**
(1e-9) et se déclenchait sur l'arrondi de `df_SUF["FT intensity"]`. Tolérance portée à 1e-3,
avec le commentaire qui explique pourquoi.

### Les fiches de la question fret aérien

| fiche | source |
|---|---|
| tendance 2000-2023, **graphique** | JRC-IDEES-2023 |
| « Peu de tonnes, beaucoup d'énergie », **graphique** des quatre intensités | le modèle |
| « En vols et en tonnes » : 18 148 vols, 715 094 t, ~60 kg par habitant et par an | JRC-IDEES-2023 |
| « Ce qui voyage par avion », **graphique** : moins de 1 % du commerce mondial en tonnes, ~35 % en valeur ; express 21 % / fret général 79 % | Boeing, World Air Cargo Forecast 2022 |
| « Liège, un des plus gros d'Europe » : un des cinq aéroports de l'UE au-dessus du million de tonnes, 24 334 vols de fret, +20,4 % en 2024 | Eurostat |
| « Le dossier Alibaba » : accord de décembre 2018, 75 M€, 220 000 m² ; le e-commerce transfrontalier vole dans plus de 80 % des cas, 131 milliards de colis par an | IATA (le chiffre vérifiable) + chiffres annoncés par Alibaba, attribués comme tels dans le texte |
| « à manier avec prudence » : le modèle envoie les tonnes vers le rail, la vraie alternative est le bateau et il est hors modèle ; une partie du fret voyage en soute des vols passagers | le modèle |
| *(révélation)* « Un angle mort assumé » | §3.1 et §3.1.4 |

Ce que la recherche n'a **pas** pu établir, et qui n'est donc écrit nulle part : une
comparaison rigoureuse tonnes-par-rail contre tonnes-par-avion sur le même corridor
Chine-Europe, et une étude chiffrant la part du fret aérien réellement transférable. Les
chiffres du hub Cainiao (emplois, tonnage à terme) ne sont pas auditables : seuls les
75 M€ et les 220 000 m² du communiqué de 2018 sont repris, et le texte dit qu'ils viennent
d'Alibaba.

Une note de cohérence pour plus tard : l'étude CE Delft pour l'AEE donne, en gCO₂e/tkm,
un rapport avion/route d'environ **6 pour 1** (834 contre 137), là où nos intensités
énergétiques donnent **2 pour 1** (0,90 contre 0,46). Les deux ne mesurent pas la même chose
(carbone contre énergie finale, et l'AEE applique un facteur de forçage radiatif), mais
l'écart est assez grand pour mériter un coup d'œil — notre chiffre routier de 0,46 kWh/tkm
en 2019 correspond à environ 60 l/100 km pour un camion moyen, ce qui est beaucoup. La
piste la plus probable est le carburant vendu en Belgique à des camions en transit, compté
dans l'énergie mais pas dans les tonnes-kilomètres belges.

## Deux écarts assumés par rapport à la consigne

**4.1 — « retirer la fiche Tendance passée ».** Le levier fusionné porte désormais la courbe
observée au-dessus du curseur, donc la fiche récitant les trois points a disparu. Mais je ne
l'ai pas simplement supprimée : elle est remplacée par **deux** fiches à graphique, « Le
long-courrier s'est amélioré » et « L'intra-européen, non ». Raison : la courbe du levier est
la *moyenne* des deux types de vol, et cette moyenne cache le fait le plus intéressant du
levier — le long-courrier a gagné 38 % pendant que le court-courrier se dégradait. La moyenne
seule donnerait l'impression d'une amélioration lente et régulière, qui n'existe dans aucun
des deux sous-ensembles.

**5.3 — « enlever la fiche À manier avec prudence ».** Retirée, dans les deux questions de
remplissage (l'équivalente existait aussi sur la question 3 et est partie avec). Mais une
nouvelle fiche `caution` prend sa place, « Ce que le calcul ne relie pas » : c'est
exactement le point 3.3 que tu demandais de clarifier et de chiffrer, et il appartient à la
catégorie « à manier avec prudence ». L'ancienne disait la même chose sans chiffres et
ajoutait « aucune série comparable n'existe pour d'autres pays », devenu à moitié faux
depuis que les séries observées sont là.

## Ce que le build refuse toujours (tests négatifs rejoués)

Les trois nouveautés du build (étiquette de fiche, courbe par `series:`, assouplissement du
garde-fou « ne pas imprimer la valeur négaWatt ») ont été testées à l'envers, en cassant
volontairement le fichier :

| essai | résultat |
|---|---|
| un graphique dans l'unité du levier dont un point vaut la valeur négaWatt | refusé |
| une clé `spoilers` citée dans une fiche d'avant-réponse | refusé |
| une étiquette `label:` sans le néerlandais | refusé |
| `series:` pointant vers une série inexistante | refusé |
| la valeur négaWatt imprimée avec un `%` **sur un levier exprimé en %** | refusé |

L'assouplissement ne porte donc que sur ce qu'il visait : un pourcentage cité dans une phrase
alors que le levier se compte en voyages, en passagers ou en kWh.

## Longueur des fiches imprimées

Même remarque que pour la mobilité intérieure : les fiches s'allongent. Mesuré dans le
navigateur à la largeur de la colonne d'impression, mais avec les tailles de police de
l'écran (donc surestimé ; la feuille de style d'impression réduit la question à 11 pt et
les légendes à 6,5 pt). La cible A5 est 128 mm.

| question | fiches avant réponse | graphiques |
|---|---|---|
| voyages long-courriers | 4 | 2 |
| **voyages court-courriers** | **7** | **4** |
| remplissage long-courrier | 5 | 1 |
| remplissage court-courrier | 5 | 1 |
| consommation des avions | 4 | 2 |
| hydrogène | 4 | 0 |
| fret aérien | 4 | 2 |

Seule la question 2 dépasse nettement : elle porte maintenant l'inégalité (deux fiches +
courbe de Lorenz) *et* le report modal vers le train (deux fiches + deux graphiques). Si tu
veux la raccourcir, la fiche « Concrètement » (kWh et litres par aller-retour) est la plus
dispensable : le même calcul figure sur la question « consommation des avions ».

## Correction du désaccord cellules 28/29 (2026-09-21)

Tu as demandé de trancher, pas seulement d'aligner. **C'est le code qui avait tort**, et il a
été corrigé : `sft_PM_rel_avi_srt_to_trn_spd = +0.25` (grande vitesse) et
`sft_PM_rel_avi_srt_to_trn_cnv = +0.20` (train classique), comme l'annonçait le texte de §2.1.

### Pourquoi

1. **La distance moyenne d'un vol intra-européen au départ de Belgique est de 1 150 km** en
   2019 (1 216 km en 2023). Source : JRC-IDEES-2023, classeur *Transport* pour la Belgique,
   feuille `TrAvia_act`, ligne « Distance travelled per flight », International
   Intra-EEAwCHUK. À cette distance, la question n'est pas de savoir si un train existe, mais
   s'il va vite.
2. **Le rail ne prend des passagers à l'avion que quand il est rapide.** L'AEE, passant la
   littérature en revue, situe la domination du rail sous 2 h 30 de trajet et une part de
   moitié ou plus jusqu'à environ 3 h 30 ; et son tableau de liaisons montre que toutes les
   grandes victoires du rail sont sur ligne à grande vitesse — Paris-Lyon 3,4 millions de
   voyageurs par le rail contre 0,64 par avion, Madrid-Barcelone 3,9 contre 2,47,
   Amsterdam-Paris 2,0 contre 1,40 — tandis que les liaisons desservies en voie **classique**
   perdent nettement à distance comparable : Berlin-Vienne (603 km) 1,05 million par avion
   contre 20 à 50 mille par train, Budapest-Francfort (917 km) 0,66 million contre 1 à 5 mille.
   Source : AEE, *Transport and Environment Report 2020 — Train or plane?*, p. 17 et tabl. 5.1
   p. 50 —
   https://www.eea.europa.eu/en/analysis/publications/transport-and-environment-report-2020/transport-and-environment-report-2020/@@download/file
3. **Pas de pénalité énergétique à craindre.** Le matériel à grande vitesse le plus récent
   (Alstom AGV, 300 km/h) est à 0,033 kWh par siège-km, soit autant qu'un Pendolino à
   200 km/h et mieux que les TGV et Eurostar des années 1990 : la masse réduite et le nombre
   de places compensent la vitesse. Source : ATOC pour Greengauge21, *Energy consumption and
   CO2 impacts of high speed rail*, tabl. 1 —
   https://www.greengauge21.net/wp-content/uploads/Energy-Consumption-and-CO2-impacts.pdf
   C'est cohérent avec le modèle, qui donne 0,0704 kWh/pkm à la grande vitesse contre 0,0764
   au train classique en 2050.

### Trois raisons de ne pas aller au-delà de 25 points

Elles sont dans le notebook aussi, parce qu'elles nuancent vraiment la décision :

1. **Le réseau ne tient pas ses promesses.** La Cour des comptes européenne qualifie le réseau
   européen à grande vitesse de « patchwork inefficace », avec des trains circulant en moyenne
   à **45 % de la vitesse de conception** de la ligne, et 36 % sur le maillon transfrontalier
   Figueres-Perpignan. D'où le contre-exemple qui compte : Barcelone-Paris, 945 km et pourtant
   liaison TGV, c'est 2,5 millions de voyageurs par avion contre 20 mille par train. Source :
   Cour des comptes européenne, rapport spécial 19/2018, §VI et §40-48 —
   https://www.eca.europa.eu/Lists/ECADocuments/SR18_19/SR_HIGH_SPEED_RAIL_EN.pdf
2. **Ce que l'Europe a légiféré est plus modeste que la grande vitesse.** Le règlement RTE-T
   révisé (UE) 2024/1679 impose **160 km/h** sur le réseau central et central étendu pour
   2040, et le raccordement au rail longue distance de tout aéroport de plus de 12 millions de
   passagers. Source : briefing du service de recherche du Parlement européen —
   https://www.europarl.europa.eu/RegData/etudes/ATAG/2025/769545/EPRS_ATA(2025)769545_EN.pdf
3. **La Belgique elle-même ne mise pas tout sur la grande vitesse.** *Vision Rail 2040* compte
   3 615 km de lignes principales et seulement **214 km à grande vitesse**, et décrit son offre
   internationale future comme un mélange assumé de « relations TGV, trains internationaux
   classiques ou trains de nuit », en citant Berlin, Hambourg, Zurich, Bâle, Copenhague, Rome
   et Milan — soit 600 à 1 200 km, exactement la bande incertaine. Source : SPF Mobilité et
   Transports, *Vision Rail 2040*, 6 mai 2022, p. 9 et p. 16 —
   https://mobilit.belgium.be/sites/default/files/publicaties%20en%20statistieken/20220506_vision_rail_2040_-_versionlongue_fr.pdf

Le train classique garde tout de même 20 points : les liaisons les plus courtes de
l'intra-européen, celles qu'aucune ligne rapide ne dessert, et les trains de nuit.

### Ce que ça change

| | avant | après |
|---|---|---|
| mobilité intérieure 2050 | 23,076 TWh | **23,072 TWh** |
| panier d'arrivée du report | 0,075 kWh/pkm | **0,074 kWh/pkm** |
| `data/energy_totals_overrides.csv` | — | quatre lignes `rail` modifiées de 0,004 à 0,006 TWh |

C'est marginal en énergie (la grande vitesse est très légèrement plus efficace par
voyageur-kilomètre dans le modèle, 0,0704 contre 0,0764), mais l'hypothèse est maintenant
cohérente et défendable.

### ⚠ À répercuter ailleurs

Les trois forks PyPSA-Eur portent encore l'ancien couple, à **`scripts/nW_BE.py`, lignes
103-104** :

```
/home/sylvain/svn/pypsa-eur_negawatt/scripts/nW_BE.py
/home/sylvain/svn/pypsa-eur_sufficiency/scripts/nW_BE.py
/home/sylvain/svn/pypsa-wal/scripts/nW_BE.py
```

Tant qu'ils ne sont pas mis à jour, leur contrôle de cohérence contre
`data/energy_totals_overrides.csv` échouera sur les quatre lignes `rail`. Je ne les ai pas
touchés : ce sont d'autres dépôts, et le choix du moment t'appartient. La fiche de l'atelier
qui signalait l'incohérence a été remplacée par une fiche de révélation qui explique le choix
et le source.

---

## Log

- Start: read the topic YAML, the lever module and the inland-mobility review notes
  (`notes_workshop.md`) for the conventions established there.
- Tous les points de la liste sont traités. Rien n'est commité, rien n'est déployé.

### État final

Toujours 7 questions, mais pas les mêmes : deux fusionnées en une, une ajoutée.
**41 fiches, dont 36 avant réponse**, **13 graphiques** (il y en avait 0), et
**quatre leviers sur sept prolongent désormais une courbe mesurée** au lieu de déclarer
« pas d'historique » — ils étaient zéro.

Vérifié après la dernière reconstruction :
- `jupyter nbconvert --execute` sur le notebook transport et sur le notebook auxiliaire — sortie 0
- `python scripts/build_workshop_content.py` — propre, aucun avertissement
- `python scripts/verify_workshop_export.py` — 327/327
- `python scripts/test_workshop_helpers.py` — 8/8
- `python scripts/test_workshop_api.py --base http://127.0.0.1:8787` — 59/59
- balayage du bundle construit : aucun `{placeholder}` non résolu, aucun astérisque littéral,
  aucune fiche sans source, aucune valeur négaWatt imprimée avant la réponse
- cinq tests négatifs sur les nouveaux garde-fous du build : tous refusés comme prévu
- pages parcourues dans le navigateur en FR, NL et EN : jeu, fiches imprimables, écran de
  mise en commun, écran de fin avec le graphique ±
- exports du notebook identiques octet pour octet avant/après l'ajout des paramètres de
  §3.1.4, et test à valeur non nulle réussi

### Fichiers touchés

| fichier | quoi |
|---|---|
| `website/workshop/content/international-mobility.yaml` | tout le contenu : fusion, réordonnancement, 17 fiches nouvelles ou réécrites, 11 graphiques, `summaryChart` |
| `website/workshop/content/ui.yaml` | « les sept autres hypothèses » → « toutes les autres » (faux dès que le thème n'a pas huit leviers) |
| `workshop_levers/international_mobility.py` | levier `plane-fuel` fusionné, levier `air-freight` nouveau, courbes historiques rattachées, une dizaine de clés `facts` nouvelles |
| `nW_BE_demand_model_transports.ipynb` | §3.1.4 : deux paramètres de fret aérien à zéro ; §2.1 : prose du report modal alignée sur le code |
| `nW_BE_demand_data_aux.ipynb` | six séries observées de l'aviation, 2000-2023, vérifiées contre les ancrages de §2.3 |
| `scripts/build_workshop_content.py` | `label:` de fiche, `series:` de graphique, garde-fou « valeur négaWatt » affiné sur les pourcentages |
| `website/assets/js/workshop/{play,cards,reveal}.js` | affichage de l'étiquette de fiche |
| `docs/workshop_module.md` | D43 à D47, §15, question ouverte Q4 close |
| `README.md` | compte des questions, `series:`, `label:` |
