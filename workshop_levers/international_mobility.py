"""International mobility: intra-EU and extra-EU aviation.

Six levers, all of them genuine degrees of freedom of the transport notebook's
aviation block, so the reveal is exact and no group can enter a
self-inconsistent scenario:

    long-haul-flights   return trips of 18 000 km per lifetime  (pro_PM_spe_avi_lng)
    short-haul-flights  return trips of 2 168 km per lifetime   (pro_PM_spe_avi_srt)
    long-haul-load      passengers per long-haul flight         (occu_trgt_PM_avi_extra)
    short-haul-load     passengers per short-haul flight        (occu_trgt_PM_avi_intra)
    plane-fuel          kWh per aircraft-km, fleet average      (redu_fuel_PM_avi_intra
                                                                 + ..._extra)
    hydrogen-flights    % of intra-EU flights on hydrogen       (end_PM_avi_srt)

`plane-fuel` is one question for two model parameters. The two were asked
separately until the 2026 review and were very nearly the same question: both
start within 3 % of each other (69.1 and 67.3 kWh per aircraft-km in 2019) and
both ask "how much energy does an aeroplane need to fly a kilometre in 2050".
Merging them is exact rather than approximate, because energy is aircraft-km
times fuel per aircraft-km: the aircraft-km-weighted average *is* total kerosene
energy over total kerosene aircraft-km, so the topic's demand is proportional to
the merged value however the scenario splits it between the two haul types. What
the merge does hide is that the scenario splits it very unevenly -- intra-EU
+5 %, extra-EU -16 % -- and a reveal-only fact says so.

Inland mobility is a separate topic (`inland_mobility.py`) out of the same
notebook. Lever ids are unique across the sector by construction.

**The lever unit is a trip, not a kilometre.** Aviation is the one mode where
"km per person per day" is meaningless — nobody flies 8.7 km a day — and where
the distribution across the population is the whole political question. Section
2.1's own note does the conversion: at 2019 volumes, all of Belgium's long-haul
flying amounts to about a tenth of the population making one 18 000 km return
trip in the year. This module exports that arithmetic so the content can put it
on a card, and the workshop asks for trips per lifetime.

Two scope facts the content states explicitly, because the model does not
represent them and a group will ask:

  * **maritime bunkers are absent.** JRC-IDEES compiles them and the notebook
    discusses the maritime fuel mix, but no international shipping *demand* is
    modelled: freight carries only `navigation-inland` and `navigation-coastal`,
    the latter 0.19 tkm/person. Nothing here invents a maritime lever.
  * **air freight became a lever in the 2026 review** (`air-freight`), because it
    is 3.5 of this topic's 10.9 TWh in 2050 and nothing in the scenario pointed at
    it. Its tonne-km per person carry `pro_FT_spe` -- inland mobility's
    `freight-tkm` lever -- times `pro_FT_spe_avi`, which section 3.1.4 gained for
    this purpose and which is zero. The two topics therefore overlap on the
    general freight reduction: a group here moves air freight alone, a group in
    the inland workshop moves the freight total including air. They are never
    played together, and neither lever's `scaled` energy includes the other's, so
    nothing is double-counted.

Three of the six now carry an observed curve. The 2026 review added the aviation
detail of the JRC-IDEES-2023 *Transport* workbook to `nW_BE_demand_data_aux.ipynb`
(occupancy and fuel per aircraft-km, intra and extra, 2000-2023), whose 2019
values reproduce exactly the four anchors section 2.3 projects from. So
`long-haul-load`, `short-haul-load` and `plane-fuel` extend a measured line, as
the design intends. The two demand levers still cannot: their unit is trips per
lifetime, and no measured series splits Belgian flying between the two haul types
in those terms — they declare `historyAbsent` with that reason, and their trend
card plots `aviation_km_day`, the combined series, with its basis named.
See docs/workshop_module.md Q6/Q8.

Everything else is read from quantities the notebook has already computed — this
module adds no assumptions of its own. It also cross-checks the model against the
figures written in the notebook's prose and code comments, so that editing either
the text or the code without the other fails loudly here rather than feeding a
wrong number into a workshop.

See docs/workshop_module.md for the design, and
website/workshop/content/international-mobility.yaml for the wording.
"""
from nW_BE_demand_model_sub_functions import make_lever, mode_totals_twh

from . import need

TOPIC = "international-mobility"
SECTOR = "transport"
ORDER = 20

NOTEBOOK = "../notebooks/nW_BE_demand_model_transports.html"


def build(ctx):
    (years, population_dict, df_SUF, df_PM, df_FT,
     df_PM_TWh_all, df_FT_TWh_all, df_PM_avi_srt_TWh, df_PM_car,
     ref_occu_PM_car,
     ref_PM_mod_spe, trg_PM_mod_spe,
     ref_FT_mod_spe, trg_FT_mod_spe,
     pro_FT_spe, pro_FT_spe_avi, sft_FT_rel_avi_to_trn,
     pro_PM_spe_avi_lng, pro_PM_spe_avi_srt, red_PM_rel,
     sft_PM_rel_avi_srt_to_trn_cnv, sft_PM_rel_avi_srt_to_trn_spd,
     sft_PM_rel_avi_srt_to_cch,
     occu_trgt_PM_avi_intra, occu_trgt_PM_avi_extra,
     redu_fuel_PM_avi_intra, redu_fuel_PM_avi_extra,
     end_PM_avi_srt,
     occupancy_PM_avi_intra, occupancy_PM_avi_extra,
     cons_fuel_PM_avi_intra, cons_fuel_PM_avi_extra,
     kgoe_to_kWh) = need(
        ctx, 'years',
        'population_dict',
        'df_SUF',
        'df_PM',
        'df_FT',
        'df_PM_TWh_all',
        'df_FT_TWh_all',
        'df_PM_avi_srt_TWh',
        'df_PM_car',
        'ref_occu_PM_car',
        'ref_PM_mod_spe',
        'trg_PM_mod_spe',
        'ref_FT_mod_spe',
        'trg_FT_mod_spe',
        'pro_FT_spe',
        'pro_FT_spe_avi',
        'sft_FT_rel_avi_to_trn',
        'pro_PM_spe_avi_lng',
        'pro_PM_spe_avi_srt',
        'red_PM_rel',
        'sft_PM_rel_avi_srt_to_trn_cnv',
        'sft_PM_rel_avi_srt_to_trn_spd',
        'sft_PM_rel_avi_srt_to_cch',
        'occu_trgt_PM_avi_intra',
        'occu_trgt_PM_avi_extra',
        'redu_fuel_PM_avi_intra',
        'redu_fuel_PM_avi_extra',
        'end_PM_avi_srt',
        'occupancy_PM_avi_intra',
        'occupancy_PM_avi_extra',
        'cons_fuel_PM_avi_intra',
        'cons_fuel_PM_avi_extra',
        'kgoe_to_kWh')

    _NB = NOTEBOOK
    _Y0, _Y1 = years[0], years[-1]
    _pop = {y: float(population_dict[y]) for y in (_Y0, _Y1)}

    # --- Reference values quoted only in this notebook's prose or comments ------
    # Named here so the workshop can export them, and so that editing either the
    # text or the model trips the assertions below instead of drifting silently.
    # Section numbers refer to the markdown headings of the transport notebook.
    ref_trip_lng_km    = 18000.0  # 9 000 km each way: BRU-Shanghai / BRU-LA  -- §2.1
    ref_life_years     = 80.0     # "over a lifetime", same note              -- §2.1
    # The short-haul yardstick is a presentation choice, not a model assumption:
    # the notebook defines no reference short-haul trip. Brussels-Barcelona return,
    # great-circle 1 084 km each way. Stated as such on the card.
    ref_trip_srt_km    = 2168.0
    ref_earth_km       = 40075.0  # equatorial circumference, for the tangibles
    # Jet A-1: 43.0 MJ/kg lower heating value at a density of 0.80 kg/litre.
    ref_kwh_per_l_kero = 9.56
    # Observed 2000 / 2019 / 2023 anchors, from the code comments of section 2.3.
    ref_intra_fuel_2000 = 578.8   # kgoe/100 aircraft-km, intra-EU              -- §2.3
    ref_intra_fuel_2019 = 593.771 # kgoe/100 aircraft-km, the frame's 2019 start -- §2.3
    ref_intra_fuel_2023 = 633.0   # kgoe/100 aircraft-km, intra-EU              -- §2.3
    ref_intra_occu_2000 = 87.5    # passengers per intra-EU flight              -- §2.3
    ref_intra_occu_2023 = 130.8   # passengers per intra-EU flight              -- §2.3
    ref_extra_fuel_2000 = 937.0   # kgoe/100 aircraft-km, extra-EU              -- §2.3
    ref_extra_fuel_2019 = 578.489 # kgoe/100 aircraft-km, the frame's 2019 start -- §2.3
    ref_extra_fuel_2023 = 629.5   # kgoe/100 aircraft-km, extra-EU              -- §2.3
    ref_extra_occu_2000 = 154.0   # passengers per extra-EU flight              -- §2.3
    ref_extra_occu_2023 = 205.5   # passengers per extra-EU flight              -- §2.3
    # IATA, Commitment to Fly Net Zero by 2050, as quoted in section 2.3.
    ref_iata_saf_pct    = 65.0
    ref_iata_newtech_pct = 13.0
    ref_iata_ccs_pct    = 19.0
    ref_iata_ops_pct    = 3.0
    # Airbus ZEROe, as described in section 2.3.
    ref_zeroe_seats     = 100.0
    ref_zeroe_range_km  = 1850.0
    # Air freight volumes, JRC-IDEES-2023 Transport workbook for Belgium, sheet
    # "TrAvia_act" (2019). Tonnes and tonnes per flight are not computed by the
    # model -- they are named here so a card can say what the tonne-kilometres
    # look like in freight terms, and they come from the same workbook as the
    # tonne-kilometres the model does use.
    # Passenger load factor, IATA "Industry Statistics" fact sheet: the share of
    # available seats actually sold. External to the model, and the only bridge
    # between "passengers per flight", which the model uses, and "how full the
    # aircraft is", which is what a participant can picture.
    ref_load_factor_2019 = 82.6   # % of seats, world, full year 2019
    ref_load_factor_2024 = 83.4   # % of seats, world, full year 2024
    ref_air_frt_tonnes_2019   = 715094.0   # tonnes carried in departing flights
    ref_air_frt_t_per_flight  = 39.4       # tonnes per departing freight flight
    ref_air_frt_flights_2019  = 18148.0    # departing freight flights

    # --- Scope ------------------------------------------------------------------
    _AVI = ["plane-intra EU", "plane-extra EU"]
    _pm_twh = mode_totals_twh(df_PM_TWh_all)
    _ft_twh = mode_totals_twh(df_FT_TWh_all)

    def _twh(tbl, mode, year):
        return float(tbl.loc[mode, year]) if mode in tbl.index else 0.0

    def _act(df, mode, unit, year):
        return float(df.loc[(mode, unit), year])

    def _intensity(tbl, df, mode, unit, year):
        """kWh per pkm (or tkm): TWh/Gpkm is already kWh/pkm."""
        act = _act(df, mode, unit, year)
        return _twh(tbl, mode, year) / act if act else 0.0

    _pax_twh = {y: sum(_twh(_pm_twh, m, y) for m in _AVI) for y in (_Y0, _Y1)}
    _frt_twh = {y: sum(_twh(_ft_twh, m, y) for m in _AVI) for y in (_Y0, _Y1)}
    _topic_twh = {y: _pax_twh[y] + _frt_twh[y] for y in (_Y0, _Y1)}
    _TOT = _topic_twh[_Y1]
    # Whole-sector context, so the content can say what share aviation becomes.
    _all_pm = {y: sum(_twh(_pm_twh, m, y) for m in _pm_twh.index) for y in (_Y0, _Y1)}
    _all_ft = {y: sum(_twh(_ft_twh, m, y) for m in _ft_twh.index) for y in (_Y0, _Y1)}
    _transport_twh = {y: _all_pm[y] + _all_ft[y] for y in (_Y0, _Y1)}

    # The two demand levers read the model's own dictionaries, not df_PM. df_PM's
    # "pkm/person" row is a display reconstruction from the rounded percentage
    # shares, so it differs from the exact value in the fourth digit (1812.12
    # against 1812.20 for long-haul in 2019) and the -40% relation would not hold
    # on it. The intensities below keep using df_PM, because there the Gpkm row and
    # the TWh table are built from the same rounded shares and so are consistent.
    _lng_pkm = {_Y0: float(ref_PM_mod_spe["plane-extra EU"]),
                _Y1: float(trg_PM_mod_spe["plane-extra EU"])}
    _srt_pkm = {_Y0: float(ref_PM_mod_spe["plane-intra EU"]),
                _Y1: float(trg_PM_mod_spe["plane-intra EU"])}
    _air_pkm = {y: _lng_pkm[y] + _srt_pkm[y] for y in (_Y0, _Y1)}
    for _m, _exact in (("plane-extra EU", _lng_pkm), ("plane-intra EU", _srt_pkm)):
        for _y in (_Y0, _Y1):
            _shown = _act(df_PM, _m, "pkm/person", _y)
            assert abs(_shown - _exact[_y]) < 0.5, (
                f"{_m} in {_y}: df_PM shows {_shown:.3f} pkm/person but the model "
                f"dictionary says {_exact[_y]:.3f}; more than display rounding")

    # Powertrain rows: intra-EU flies on kerosene plus a hydrogen share, extra-EU
    # on kerosene only.
    def _row(frame, powertrain, year):
        return float(frame.loc[powertrain, year])

    _srt_kero_twh = {y: float(df_PM_avi_srt_TWh.set_index("Powertrain")
                              .loc["liquid-kerosene", y]) for y in (_Y0, _Y1)}
    _occ_srt = {y: _row(occupancy_PM_avi_intra, "liquid-kerosene", y) for y in (_Y0, _Y1)}
    _occ_lng = {y: _row(occupancy_PM_avi_extra, "liquid-kerosene", y) for y in (_Y0, _Y1)}
    _fuel_srt = {y: _row(cons_fuel_PM_avi_intra, "liquid-kerosene", y) for y in (_Y0, _Y1)}
    _fuel_lng = {y: _row(cons_fuel_PM_avi_extra, "liquid-kerosene", y) for y in (_Y0, _Y1)}
    _h2_fuel = _row(cons_fuel_PM_avi_intra, "hydrogen", _Y1)
    _h2_occ = _row(occupancy_PM_avi_intra, "hydrogen", _Y1)

    # --- Consistency checks: the model vs. the figures written in the prose -----
    # Long-haul takes the -40% straight off its 2019 per-person value.
    assert abs(_lng_pkm[_Y1] - (1 + pro_PM_spe_avi_lng) * _lng_pkm[_Y0]) < 1e-6, (
        f"long-haul pkm/person is {_lng_pkm[_Y1]:.4f} in {_Y1} but "
        f"(1{pro_PM_spe_avi_lng:+.2f}) x {_lng_pkm[_Y0]:.4f} gives "
        f"{(1 + pro_PM_spe_avi_lng) * _lng_pkm[_Y0]:.4f} -- section 2.1")
    # Short-haul takes the -50% shift *on top of* the general mobility reduction
    # that red_PM_rel spreads over every mode except long-haul.
    assert abs(_srt_pkm[_Y1] - (1 + pro_PM_spe_avi_srt) * red_PM_rel
               * _srt_pkm[_Y0]) < 1e-6, (
        f"short-haul pkm/person is {_srt_pkm[_Y1]:.4f} in {_Y1} but "
        f"(1{pro_PM_spe_avi_srt:+.2f}) x red_PM_rel({red_PM_rel:.5f}) x "
        f"{_srt_pkm[_Y0]:.4f} gives "
        f"{(1 + pro_PM_spe_avi_srt) * red_PM_rel * _srt_pkm[_Y0]:.4f} -- section 2.1")
    # The whole short-haul reduction is a modal shift, so the destinations add up.
    _srt_dest = {"train-conventional": sft_PM_rel_avi_srt_to_trn_cnv,
                 "train-high speed":   sft_PM_rel_avi_srt_to_trn_spd,
                 "bus&coach":          sft_PM_rel_avi_srt_to_cch}
    assert abs(sum(_srt_dest.values()) + pro_PM_spe_avi_srt) < 1e-12, (
        "the short-haul modal-shift destinations no longer add up to the shift "
        f"itself ({sum(_srt_dest.values()):.3f} vs {-pro_PM_spe_avi_srt:.3f}) "
        "-- section 2.1")
    # Occupancy and fuel-per-aircraft-km are the multipliers times their 2019 start.
    # These four come out of `linear_growth()`, which rounds its series to three
    # decimals, so the tolerance below is that rounding and not a licence to drift.
    for _label, _obs, _mult, _sec in (
            ("intra-EU occupancy", _occ_srt, occu_trgt_PM_avi_intra, "2.3"),
            ("extra-EU occupancy", _occ_lng, occu_trgt_PM_avi_extra, "2.3"),
            ("intra-EU fuel/km", _fuel_srt, redu_fuel_PM_avi_intra, "2.3"),
            ("extra-EU fuel/km", _fuel_lng, redu_fuel_PM_avi_extra, "2.3")):
        assert abs(_obs[_Y1] - _mult * _obs[_Y0]) < 1e-3, (
            f"{_label} is {_obs[_Y1]:.4f} in {_Y1} but {_mult} x {_obs[_Y0]:.4f} "
            f"gives {_mult * _obs[_Y0]:.4f} -- section {_sec}")
    # The 2019 fuel figures quoted in the code comments, converted.
    for _label, _kgoe, _obs in (("intra-EU", ref_intra_fuel_2019, _fuel_srt),
                                ("extra-EU", ref_extra_fuel_2019, _fuel_lng)):
        assert abs(_obs[_Y0] - _kgoe / 100.0 * kgoe_to_kWh) < 1e-3, (
            f"{_label} 2019 fuel use is {_obs[_Y0]:.4f} kWh/km but section 2.3 "
            f"quotes {_kgoe} kgoe/100 km = {_kgoe / 100.0 * kgoe_to_kWh:.4f}")
    assert abs(_occ_lng[_Y0] - ref_extra_occu_2023) > 1.0, (
        "the extra-EU 2019 occupancy now equals the 2023 figure quoted in the "
        "comments; one of the two is wrong")

    # 2019 fleet-average car occupancy = total pkm / total vkm, i.e. the pkm-weighted
    # *harmonic* mean of the per-powertrain occupancies -- the same arithmetic as
    # inland_mobility.py, repeated here because the short-haul card compares the
    # plane with the car *per passenger* and has to say on what car it is based.
    _occ_car_2019 = 100.0 / sum(float(df_PM_car.loc[_pt, _Y0]) / _occ
                                for _pt, _occ in ref_occu_PM_car.items())
    assert 1.0 < _occ_car_2019 < 2.0, (
        f"fleet-average car occupancy came out at {_occ_car_2019:.4f} -- section 2.1 "
        f"quotes about 1.22; one of the two is wrong")

    # --- Derived lever quantities ----------------------------------------------
    def _trips_per_life(pkm_per_year, trip_km):
        return pkm_per_year / trip_km * ref_life_years

    _lng_trips = {y: _trips_per_life(_lng_pkm[y], ref_trip_lng_km) for y in (_Y0, _Y1)}
    _srt_trips = {y: _trips_per_life(_srt_pkm[y], ref_trip_srt_km) for y in (_Y0, _Y1)}
    # Section 2.1's own equity arithmetic: how many people the year's long-haul
    # flying would carry on one 18 000 km return trip each, and what share of the
    # population that is.
    _lng_people = {y: _lng_pkm[y] * _pop[y] / ref_trip_lng_km for y in (_Y0, _Y1)}
    _lng_pct_pop = {y: 100.0 * _lng_people[y] / _pop[y] for y in (_Y0, _Y1)}
    _lng_years_per_trip = {y: ref_trip_lng_km / _lng_pkm[y] for y in (_Y0, _Y1)}
    # Aircraft-kilometres behind the 2019 long-haul flying, for the tangible card.
    _lng_vkm_2019 = _lng_pkm[_Y0] * _pop[_Y0] / _occ_lng[_Y0]

    # --- The fleet average behind the single fuel question ----------------------
    # Energy = aircraft-km x fuel per aircraft-km, so the aircraft-km-weighted
    # average of the two haul types is *exactly* the total kerosene energy over
    # the total kerosene aircraft-km. That is what makes one question out of two
    # possible without losing anything: whatever the split between intra- and
    # extra-EU, the topic's kerosene demand is proportional to this average.
    # Aircraft-km are read back from the model's own TWh, which is the same
    # arithmetic run backwards (kWh / (kWh/km)), so no new assumption enters.
    _kero_twh = {y: _twh(_pm_twh, "plane-extra EU", y) + _srt_kero_twh[y]
                 for y in (_Y0, _Y1)}
    _kero_vkm = {y: (_twh(_pm_twh, "plane-extra EU", y) * 1e9 / _fuel_lng[y]
                     + _srt_kero_twh[y] * 1e9 / _fuel_srt[y]) for y in (_Y0, _Y1)}
    _fuel_avg = {y: _kero_twh[y] * 1e9 / _kero_vkm[y] for y in (_Y0, _Y1)}
    # It has to sit between the two it averages, or the weights are wrong.
    for _y in (_Y0, _Y1):
        assert min(_fuel_lng[_y], _fuel_srt[_y]) - 1e-6 <= _fuel_avg[_y] \
               <= max(_fuel_lng[_y], _fuel_srt[_y]) + 1e-6, (
            f"the {_y} fleet average {_fuel_avg[_y]:.3f} kWh/km is outside "
            f"[{_fuel_srt[_y]:.3f}, {_fuel_lng[_y]:.3f}] -- check the weights")
    assert abs(_lng_vkm_2019 - _twh(_pm_twh, "plane-extra EU", _Y0) * 1e9
               / _fuel_lng[_Y0]) < 1e6, (
        "aircraft-km from pkm/occupancy and from TWh/(kWh per km) disagree")
    # Energy per passenger for one reference return trip, 2019.
    _kwh_per_lng_trip = _intensity(_pm_twh, df_PM, "plane-extra EU", "Gpkm", _Y0) \
        * ref_trip_lng_km
    _kwh_per_srt_trip = _intensity(_pm_twh, df_PM, "plane-intra EU", "Gpkm", _Y0) \
        * ref_trip_srt_km

    # TWh per extra short-haul trip per lifetime. Exact for a single lever moved
    # alone: the shift is share-preserving on the total and the 2050 intensities
    # do not depend on the shares, so the response is linear.
    def _basket_intensity(dest):
        weight = sum(dest.values())
        return sum(w / weight * _intensity(_pm_twh, df_PM, m, "Gpkm", _Y1)
                   for m, w in dest.items())

    _srt_basket = _basket_intensity(_srt_dest)
    _srt_slope = (ref_trip_srt_km / ref_life_years * _pop[_Y1] / 1e9
                  * (_intensity(_pm_twh, df_PM, "plane-intra EU", "Gpkm", _Y1)
                     - _srt_basket))
    # TWh per percentage point of hydrogen. A hydrogen aircraft carries fewer
    # passengers but uses less energy per passenger-km, so the slope is small and
    # negative: this lever is about *which* energy, not how much.
    _h2_kwh_pkm = _h2_fuel / _h2_occ
    _kero_kwh_pkm = _fuel_srt[_Y1] / _occ_srt[_Y1]
    _h2_slope = (_srt_pkm[_Y1] * _pop[_Y1] / 1e9 / 100.0
                 * (_h2_kwh_pkm - _kero_kwh_pkm))

    def _impact(kind, v_target, scaled=0.0, slope=None):
        """Leverage record read by website/assets/js/workshop/impact.js.

        TWh(vTarget) always equals `total`, the négaWatt 2050 demand for this
        topic, so every lever's readout is on the same comparable scale.
        """
        rec = {"kind": kind, "vTarget": round(float(v_target), 4),
               "total": round(_TOT, 4), "scaled": round(float(scaled), 4)}
        if slope is not None:
            rec["slope"] = round(float(slope), 6)
        return rec

    def _pct(part, whole):
        return round(100.0 * part / whole, 1) if whole else 0.0

    # --- Air freight ------------------------------------------------------------
    # The one freight flow inside this topic. Its tonne-km per person is what the
    # lever asks about; the 2050 value carries the general freight reduction plus
    # whatever section 3.1.4 assumes on top of it (nothing, at the time of writing).
    _air_frt_spe = {_Y0: sum(float(ref_FT_mod_spe[m]) for m in _AVI),
                    _Y1: sum(float(trg_FT_mod_spe[m]) for m in _AVI)}
    assert abs(_air_frt_spe[_Y1] - (1 + pro_FT_spe_avi) * (1 + pro_FT_spe)
               * _air_frt_spe[_Y0]) < 1e-6, (
        f"air freight is {_air_frt_spe[_Y1]:.3f} tkm/person in {_Y1} but "
        f"(1{pro_FT_spe_avi:+.2f}) x (1{pro_FT_spe:+.3f}) x {_air_frt_spe[_Y0]:.3f} "
        f"gives {(1 + pro_FT_spe_avi) * (1 + pro_FT_spe) * _air_frt_spe[_Y0]:.3f} "
        f"-- section 3.1.4")
    _frt_tkm_spe = {y: float(df_SUF.loc["FT intensity [tkm/person]", y])
                    for y in (_Y0, _Y1)}
    _frt_intensity = {m: _intensity(_ft_twh, df_FT, m, "Gtkm", _Y0)
                      for m in ("truck-heavy duty", "train", "navigation-inland")}
    # Air freight as one mode: both haul types together, as the lever asks it.
    _frt_intensity["air"] = (_frt_twh[_Y0] * 1e9
                             / (_air_frt_spe[_Y0] * _pop[_Y0]) if _air_frt_spe[_Y0] else 0.0)

    # --- The levers -------------------------------------------------------------
    _L = []
    def _add(*a, **k):
        _L.append(make_lever(*a, **k))

    _T = "international-mobility"
    _scope_facts = {"topicTwh": round(_topic_twh[_Y0], 1),
                    "freightTwh": round(_frt_twh[_Y0], 2),
                    "freightTwhTarget": round(_frt_twh[_Y1], 2),
                    "transportTwh": round(_transport_twh[_Y0], 0),
                    "topicShareTransport": _pct(_topic_twh[_Y0], _transport_twh[_Y0]),
                    "topicShareTransportTarget": _pct(_topic_twh[_Y1],
                                                      _transport_twh[_Y1])}

    _add("long-haul-flights", _T, "Long-haul return trips per lifetime",
         "long-haul return trips per lifetime",
         _lng_trips[_Y0], _lng_trips[_Y1], ref_year=_Y0, target_year=_Y1,
         slider={"min": 2, "max": 16, "step": 0.25},
         impact=_impact("proportional", _lng_trips[_Y1],
                        scaled=_twh(_pm_twh, "plane-extra EU", _Y1)),
         model={"var": "pro_PM_spe_avi_lng", "section": "2.1",
                "note": "one unit = one 18 000 km return trip over an 80-year life, "
                        "the yardstick section 2.1 itself uses; the model input is "
                        "the change in extra-EU pkm per person"},
         facts=dict(_scope_facts, **{
             "tripKm": ref_trip_lng_km, "lifeYears": ref_life_years,
             "changePct": round(pro_PM_spe_avi_lng * 100, 1),
             "people2019": round(_lng_people[_Y0]),
             "pctPop2019": round(_lng_pct_pop[_Y0], 1),
             "peopleTarget": round(_lng_people[_Y1]),
             "pctPopTarget": round(_lng_pct_pop[_Y1], 1),
             "yearsPerTrip2019": round(_lng_years_per_trip[_Y0], 1),
             "yearsPerTripTarget": round(_lng_years_per_trip[_Y1], 1),
             "shareAirPkm2019": _pct(_lng_pkm[_Y0], _air_pkm[_Y0]),
             "shareAirPkmTarget": _pct(_lng_pkm[_Y1], _air_pkm[_Y1]),
             "lngTwh": round(_twh(_pm_twh, "plane-extra EU", _Y0), 1),
             "lngTwhTarget": round(_twh(_pm_twh, "plane-extra EU", _Y1), 2),
             "vkmMillion2019": round(_lng_vkm_2019 / 1e6),
             "earthLaps2019": round(_lng_vkm_2019 / ref_earth_km),
             "kwhPerTrip2019": round(_kwh_per_lng_trip),
             "litresPerTrip2019": round(_kwh_per_lng_trip / ref_kwh_per_l_kero)}),
         spoilers=["changePct", "peopleTarget", "pctPopTarget",
                   "yearsPerTripTarget", "shareAirPkmTarget", "lngTwhTarget",
                   "topicShareTransportTarget", "freightTwhTarget"],
         notebook=_NB + "#section_2", reference="nW-BE §2.1")

    _add("short-haul-flights", _T, "Short-haul return trips per lifetime",
         "short-haul return trips per lifetime",
         _srt_trips[_Y0], _srt_trips[_Y1], ref_year=_Y0, target_year=_Y1,
         slider={"min": 10, "max": 80, "step": 1},
         impact=_impact("linear-shift", _srt_trips[_Y1], slope=_srt_slope),
         model={"var": "pro_PM_spe_avi_srt", "section": "2.1",
                "note": "one unit = one 2 168 km return trip (Brussels-Barcelona) "
                        "over an 80-year life; the whole reduction is a modal shift, "
                        "and its destinations are fixed at the négaWatt split"},
         facts=dict(_scope_facts, **{
             "tripKm": ref_trip_srt_km, "lifeYears": ref_life_years,
             "shiftPct": round(-pro_PM_spe_avi_srt * 100, 1),
             "toRailCnvPct": round(sft_PM_rel_avi_srt_to_trn_cnv * 100, 1),
             "toRailSpdPct": round(sft_PM_rel_avi_srt_to_trn_spd * 100, 1),
             "toCoachPct": round(sft_PM_rel_avi_srt_to_cch * 100, 1),
             "monthsPerTrip2019": round(12.0 * ref_trip_srt_km / _srt_pkm[_Y0], 1),
             "shareAirPkm2019": _pct(_srt_pkm[_Y0], _air_pkm[_Y0]),
             "srtKwhPkm2019": round(_intensity(_pm_twh, df_PM, "plane-intra EU",
                                               "Gpkm", _Y0), 3),
             "carKwhPkm2019": round(_intensity(_pm_twh, df_PM, "car", "Gpkm", _Y0), 3),
             "carOccupancy2019": round(_occ_car_2019, 2),
             "srtKwhPkmTarget": round(_intensity(_pm_twh, df_PM, "plane-intra EU",
                                                 "Gpkm", _Y1), 3),
             "carKwhPkmTarget": round(_intensity(_pm_twh, df_PM, "car", "Gpkm", _Y1), 3),
             "ratioTarget": round(_intensity(_pm_twh, df_PM, "plane-intra EU", "Gpkm", _Y1)
                                  / _intensity(_pm_twh, df_PM, "car", "Gpkm", _Y1), 1),
             "railKwhPkmTarget": round(_srt_basket, 3),
             "srtTwh": round(_twh(_pm_twh, "plane-intra EU", _Y0), 1),
             "srtTwhTarget": round(_twh(_pm_twh, "plane-intra EU", _Y1), 2),
             "kwhPerTrip2019": round(_kwh_per_srt_trip),
             "litresPerTrip2019": round(_kwh_per_srt_trip / ref_kwh_per_l_kero)}),
         spoilers=["shiftPct", "toRailCnvPct", "toRailSpdPct", "toCoachPct",
                   "railKwhPkmTarget", "srtTwhTarget", "srtKwhPkmTarget",
                   "ratioTarget", "topicShareTransportTarget", "freightTwhTarget"],
         notebook=_NB + "#section_2", reference="nW-BE §2.1")

    _add("long-haul-load", _T, "Passengers per long-haul flight",
         "passengers per long-haul flight",
         _occ_lng[_Y0], _occ_lng[_Y1], ref_year=_Y0, target_year=_Y1,
         slider={"min": 150, "max": 270, "step": 5},
         history="avia_occupancy_extra",
         impact=_impact("inverse", _occ_lng[_Y1],
                        scaled=_twh(_pm_twh, "plane-extra EU", _Y1)),
         model={"var": "occu_trgt_PM_avi_extra", "section": "2.3",
                "note": "passengers actually carried per flight, so it blends the "
                        "load factor with the size of the aircraft; the model cannot "
                        "separate the two"},
         facts=dict(_scope_facts, **{
             "gainPct": round((occu_trgt_PM_avi_extra - 1) * 100, 1),
             "occu2000": ref_extra_occu_2000, "occu2023": ref_extra_occu_2023,
             "histGainPct": round(100 * (ref_extra_occu_2023 / ref_extra_occu_2000 - 1), 1),
             # The same comparison stopped in 2019, so a card can say what the
             # trend was before the pandemic distorted both ends of it -- and the
             # matching change in fuel per aircraft-km, for the card that explains
             # why the model treats the two as independent.
             "histGainPct2019": round(100 * (_occ_lng[_Y0] / ref_extra_occu_2000 - 1), 1),
             "loadFactor2019": ref_load_factor_2019,
             "loadFactor2024": ref_load_factor_2024,
             "seatsImplied2019": round(_occ_lng[_Y0] / (ref_load_factor_2019 / 100.0)),
             "fuelChangePct2019": round(100 * (ref_extra_fuel_2019
                                               / ref_extra_fuel_2000 - 1), 1),
             "kwhKm2000": round(ref_extra_fuel_2000 / 100.0 * kgoe_to_kWh, 1),
             "kwhKm2019": round(_fuel_lng[_Y0], 1),
             "lngTwh": round(_twh(_pm_twh, "plane-extra EU", _Y0), 1),
             "lngTwhTarget": round(_twh(_pm_twh, "plane-extra EU", _Y1), 2),
             "lngKwhPkm2019": round(_intensity(_pm_twh, df_PM, "plane-extra EU",
                                               "Gpkm", _Y0), 3),
             "vkmMillion2019": round(_lng_vkm_2019 / 1e6),
             "earthLaps2019": round(_lng_vkm_2019 / ref_earth_km)}),
         spoilers=["gainPct", "lngTwhTarget", "topicShareTransportTarget",
                   "freightTwhTarget"],
         notebook=_NB + "#section_2", reference="nW-BE §2.3")

    _add("short-haul-load", _T, "Passengers per short-haul flight",
         "passengers per short-haul flight",
         _occ_srt[_Y0], _occ_srt[_Y1], ref_year=_Y0, target_year=_Y1,
         slider={"min": 85, "max": 190, "step": 5},
         history="avia_occupancy_intra",
         impact=_impact("inverse", _occ_srt[_Y1], scaled=_srt_kero_twh[_Y1]),
         model={"var": "occu_trgt_PM_avi_intra", "section": "2.3",
                "note": "applies to the kerosene fleet; the hydrogen aircraft of the "
                        "hydrogen-flights lever is fixed at 100 seats"},
         facts=dict(_scope_facts, **{
             "gainPct": round((occu_trgt_PM_avi_intra - 1) * 100, 1),
             "occu2000": ref_intra_occu_2000, "occu2023": ref_intra_occu_2023,
             "histGainPct": round(100 * (ref_intra_occu_2023 / ref_intra_occu_2000 - 1), 1),
             "histGainPct2019": round(100 * (_occ_srt[_Y0] / ref_intra_occu_2000 - 1), 1),
             "loadFactor2019": ref_load_factor_2019,
             "loadFactor2024": ref_load_factor_2024,
             "seatsImplied2019": round(_occ_srt[_Y0] / (ref_load_factor_2019 / 100.0)),
             "fuelChangePct2019": round(100 * (ref_intra_fuel_2019
                                               / ref_intra_fuel_2000 - 1), 1),
             "kwhKm2000": round(ref_intra_fuel_2000 / 100.0 * kgoe_to_kWh, 1),
             "kwhKm2019": round(_fuel_srt[_Y0], 1),
             "srtTwh": round(_twh(_pm_twh, "plane-intra EU", _Y0), 1),
             "srtKeroTwhTarget": round(_srt_kero_twh[_Y1], 2),
             "srtKwhPkm2019": round(_intensity(_pm_twh, df_PM, "plane-intra EU",
                                               "Gpkm", _Y0), 3),
             "zeroeSeats": ref_zeroe_seats}),
         spoilers=["gainPct", "srtKeroTwhTarget", "topicShareTransportTarget",
                   "freightTwhTarget"],
         notebook=_NB + "#section_2", reference="nW-BE §2.3")

    _add("plane-fuel", _T, "Fuel per aircraft-km", "kWh per aircraft-km",
         _fuel_avg[_Y0], _fuel_avg[_Y1], ref_year=_Y0, target_year=_Y1,
         slider={"min": 40, "max": 95, "step": 1}, decimals=1,
         history="avia_fuel_pax",
         impact=_impact("proportional", _fuel_avg[_Y1], scaled=_kero_twh[_Y1]),
         model={"var": "redu_fuel_PM_avi_intra + redu_fuel_PM_avi_extra",
                "section": "2.3",
                "note": "one question for the two model parameters. The value is the "
                        "aircraft-km-weighted average of the intra- and extra-EU fuel "
                        "use per aircraft-km, so the topic's kerosene demand is exactly "
                        "proportional to it whatever the split between the two. Energy "
                        "in the tank, whatever the fuel: which fuel it is (kerosene, "
                        "SAF, e-fuel) is left to PyPSA"},
         facts=dict(_scope_facts, **{
             "kwhKmAvg2019": round(_fuel_avg[_Y0], 1),
             "kwhKmAvgTarget": round(_fuel_avg[_Y1], 1),
             "avgChangePct": round(100 * (_fuel_avg[_Y1] / _fuel_avg[_Y0] - 1), 1),
             # the two haul types the average is made of
             "kwhKmIntra2019": round(_fuel_srt[_Y0], 1),
             "kwhKmExtra2019": round(_fuel_lng[_Y0], 1),
             "kwhKmIntra2023": round(ref_intra_fuel_2023 / 100.0 * kgoe_to_kWh, 1),
             "kwhKmExtra2023": round(ref_extra_fuel_2023 / 100.0 * kgoe_to_kWh, 1),
             "kwhKmIntra2000": round(ref_intra_fuel_2000 / 100.0 * kgoe_to_kWh, 1),
             "kwhKmExtra2000": round(ref_extra_fuel_2000 / 100.0 * kgoe_to_kWh, 1),
             "kgoeIntra2000": ref_intra_fuel_2000, "kgoeIntra2023": ref_intra_fuel_2023,
             "kgoeExtra2000": ref_extra_fuel_2000, "kgoeExtra2023": ref_extra_fuel_2023,
             "histChangeIntraPct": round(100 * (ref_intra_fuel_2023
                                                / ref_intra_fuel_2000 - 1), 1),
             "histChangeExtraPct": round(100 * (ref_extra_fuel_2023
                                                / ref_extra_fuel_2000 - 1), 1),
             "histChangeIntra2019Pct": round(100 * (ref_intra_fuel_2019
                                                    / ref_intra_fuel_2000 - 1), 1),
             "histChangeExtra2019Pct": round(100 * (ref_extra_fuel_2019
                                                    / ref_extra_fuel_2000 - 1), 1),
             # what the scenario does to each of them separately
             "changeIntraPct": round((redu_fuel_PM_avi_intra - 1) * 100, 1),
             "changeExtraPct": round((redu_fuel_PM_avi_extra - 1) * 100, 1),
             "fuelTargetIntra": round(_fuel_srt[_Y1], 1),
             "fuelTargetExtra": round(_fuel_lng[_Y1], 1),
             # per passenger and per trip, 2019
             "occuIntra2019": round(_occ_srt[_Y0]), "occuExtra2019": round(_occ_lng[_Y0]),
             "lngKwhPkm2019": round(_intensity(_pm_twh, df_PM, "plane-extra EU",
                                               "Gpkm", _Y0), 3),
             "srtKwhPkm2019": round(_intensity(_pm_twh, df_PM, "plane-intra EU",
                                               "Gpkm", _Y0), 3),
             "kwhPerLngTrip2019": round(_kwh_per_lng_trip),
             "litresPerLngTrip2019": round(_kwh_per_lng_trip / ref_kwh_per_l_kero),
             "kwhPerSrtTrip2019": round(_kwh_per_srt_trip),
             "litresPerSrtTrip2019": round(_kwh_per_srt_trip / ref_kwh_per_l_kero),
             "iataSaf": ref_iata_saf_pct, "iataNewTech": ref_iata_newtech_pct,
             "iataCcs": ref_iata_ccs_pct, "iataOps": ref_iata_ops_pct,
             "lngTwh": round(_twh(_pm_twh, "plane-extra EU", _Y0), 1),
             "srtTwh": round(_twh(_pm_twh, "plane-intra EU", _Y0), 1),
             "keroTwh": round(_kero_twh[_Y0], 1),
             "keroTwhTarget": round(_kero_twh[_Y1], 2)}),
         spoilers=["kwhKmAvgTarget", "avgChangePct", "changeIntraPct", "changeExtraPct",
                   "fuelTargetIntra", "fuelTargetExtra", "keroTwhTarget",
                   "topicShareTransportTarget", "freightTwhTarget"],
         notebook=_NB + "#section_2", reference="nW-BE §2.3")

    _add("air-freight", _T, "Air freight per person", "tkm/person/year",
         _air_frt_spe[_Y0], _air_frt_spe[_Y1], ref_year=_Y0, target_year=_Y1,
         slider={"min": 0, "max": 600, "step": 10},
         history="air_freight_tkm_person",
         impact=_impact("proportional", _air_frt_spe[_Y1], scaled=_frt_twh[_Y1]),
         model={"var": "pro_FT_spe_avi", "section": "3.1.4",
                "note": "tonne-kilometres flown per inhabitant. The scenario assumes "
                        "nothing of its own here: air freight simply follows the general "
                        "freight reduction, so its 2050 value is (1+pro_FT_spe) times "
                        "2019. What a group takes off it goes to rail "
                        "(sft_FT_rel_avi_to_trn), the one alternative the model carries"},
         facts=dict(_scope_facts, **{
             "airTkm2019": round(_air_frt_spe[_Y0], 1),
             "airTkmTarget": round(_air_frt_spe[_Y1], 1),
             "airChangePct": round(100 * (_air_frt_spe[_Y1] / _air_frt_spe[_Y0] - 1), 1),
             "frtTkm2019": round(_frt_tkm_spe[_Y0]),
             "airShareTkm2019": _pct(_air_frt_spe[_Y0], _frt_tkm_spe[_Y0]),
             "airShareTkmTarget": _pct(_air_frt_spe[_Y1], _frt_tkm_spe[_Y1]),
             "airKwhPerTkm2019": round(_frt_intensity["air"], 2),
             "truckKwhPerTkm2019": round(_frt_intensity["truck-heavy duty"], 2),
             "railKwhPerTkm2019": round(_frt_intensity["train"], 3),
             "navKwhPerTkm2019": round(_frt_intensity["navigation-inland"], 2),
             "airTonnes2019": ref_air_frt_tonnes_2019,
             "airTPerFlight": ref_air_frt_t_per_flight,
             "airFlights2019": ref_air_frt_flights_2019,
             "freightTwhShareTopicTarget": _pct(_frt_twh[_Y1], _topic_twh[_Y1]),
             "shiftToRailPct": round(sft_FT_rel_avi_to_trn * 100, 1)}),
         spoilers=["airTkmTarget", "airChangePct", "airShareTkmTarget",
                   "freightTwhShareTopicTarget", "shiftToRailPct",
                   "topicShareTransportTarget", "freightTwhTarget"],
         notebook=_NB + "#section_3", reference="nW-BE §3.1.4")

    _add("hydrogen-flights", _T, "Hydrogen share of intra-EU flights",
         "% of intra-EU flights on hydrogen",
         0.0, float(end_PM_avi_srt), ref_year=_Y0, target_year=_Y1,
         slider={"min": 0, "max": 35, "step": 1},
         # Small and negative on purpose: a hydrogen aircraft carries fewer
         # passengers but needs less energy per passenger-km, so swapping kerosene
         # for hydrogen barely moves the *demand* this topic measures. What it
         # changes is which energy carrier has to be produced, which the demand
         # model does not represent. That is the point of the lever, not a defect.
         impact=_impact("linear-shift", float(end_PM_avi_srt), slope=_h2_slope),
         model={"var": "end_PM_avi_srt", "section": "2.3",
                "note": "share of intra-EU passenger-km flown on hydrogen in 2050, "
                        "ramped from zero with an inflection in 2045"},
         facts=dict(_scope_facts, **{
             "zeroeSeats": ref_zeroe_seats, "zeroeRangeKm": ref_zeroe_range_km,
             "h2KwhPkm": round(_h2_kwh_pkm, 3),
             "keroKwhPkm2019": round(_intensity(_pm_twh, df_PM, "plane-intra EU",
                                                "Gpkm", _Y0), 3),
             "iataSaf": ref_iata_saf_pct, "iataNewTech": ref_iata_newtech_pct,
             "iataCcs": ref_iata_ccs_pct, "iataOps": ref_iata_ops_pct,
             "srtTwh": round(_twh(_pm_twh, "plane-intra EU", _Y0), 1),
             "shareAllFlightsPct": round(float(end_PM_avi_srt)
                                         * _srt_pkm[_Y1] / _air_pkm[_Y1], 1),
             "swingTwh": round(abs(_h2_slope) * 35.0, 2)}),
         spoilers=["shareAllFlightsPct", "topicShareTransportTarget",
                   "freightTwhTarget"],
         notebook=_NB + "#section_2", reference="nW-BE §2.3")

    # --- Shared model quantities (context + the leverage arithmetic) -------------
    _ws_model = {
        "scope": "international mobility: intra-EU and extra-EU aviation, passengers "
                 "and freight. Maritime bunkers are not represented in the demand "
                 "model, so shipping is available neither as a mode nor as a "
                 "destination for a modal shift.",
        "refYear": _Y0, "targetYear": _Y1,
        "population": {str(y): round(_pop[y]) for y in (_Y0, _Y1)},
        "topicTwh": {str(y): round(_topic_twh[y], 3) for y in (_Y0, _Y1)},
        "passengerAviationTwh": {str(y): round(_pax_twh[y], 3) for y in (_Y0, _Y1)},
        "airFreightTwh": {str(y): round(_frt_twh[y], 3) for y in (_Y0, _Y1)},
        "transportTwh": {str(y): round(_transport_twh[y], 3) for y in (_Y0, _Y1)},
        "airPkmPerPerson": {str(y): round(_air_pkm[y], 1) for y in (_Y0, _Y1)},
        "airKmPerDay": {str(y): round(_air_pkm[y] / 365.0, 3) for y in (_Y0, _Y1)},
        "longHaulPkmPerPerson": {str(y): round(_lng_pkm[y], 1) for y in (_Y0, _Y1)},
        "shortHaulPkmPerPerson": {str(y): round(_srt_pkm[y], 1) for y in (_Y0, _Y1)},
        "longHaulTripsPerLife": {str(y): round(_lng_trips[y], 3) for y in (_Y0, _Y1)},
        "shortHaulTripsPerLife": {str(y): round(_srt_trips[y], 3) for y in (_Y0, _Y1)},
        "modeTwh": {str(y): {m: round(_twh(_pm_twh, m, y), 3) for m in _AVI}
                    for y in (_Y0, _Y1)},
        "modeIntensity": {str(y): {m: round(_intensity(_pm_twh, df_PM, m, "Gpkm", y), 5)
                                   for m in _AVI} for y in (_Y0, _Y1)},
        "shiftBasketIntensityTarget": round(_srt_basket, 5),
        "referenceTrip": {"longHaulKm": ref_trip_lng_km,
                          "shortHaulKm": ref_trip_srt_km,
                          "lifeYears": ref_life_years},
    }

    return {"levers": _L, "model": _ws_model}
