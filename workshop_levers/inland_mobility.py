"""Inland mobility: domestic passenger ground transport and domestic freight.

Eight levers, played in the proof-of-concept workshop. Two more (bus and train
occupancy) are exported but not shown.

Everything here reads quantities the transport notebook has already computed —
this module adds no assumptions of its own. It also cross-checks the model
against the figures quoted in the notebook's prose, so that editing either the
text or the code without the other fails loudly here rather than feeding a wrong
number into a workshop.

See docs/workshop_module.md for the design, and
website/workshop/content/inland-mobility.yaml for the wording.
"""
from nW_BE_demand_model_sub_functions import make_lever, mode_totals_twh

from . import need

TOPIC = "inland-mobility"
SECTOR = "transport"
ORDER = 10

NOTEBOOK = "../notebooks/nW_BE_demand_model_transports.html"


def build(ctx):
    (years, population_dict, df_SUF, df_PM, df_FT,
     df_PM_TWh_all, df_FT_TWh_all, df_PM_car, ref_occu_PM_car,
     pro_PM_spe, pro_PM_spe_car, occu_trgt_PM_car, redu_fuel_PM_car,
     sft_PM_rel_car_to_bus, sft_PM_rel_car_to_trn_cnv, sft_PM_rel_car_to_byc,
     sft_PM_rel_car_to_trm_met, sft_PM_rel_car_to_mot, sft_PM_rel_car_to_ped,
     pro_FT_spe, pro_FT_spe_trk_hvy,
     sft_FT_rel_trk_hvy_to_trn, sft_FT_rel_trk_hvy_to_nav_ild,
     pyld_trgt_FT_trk_hvy, ref_pyld_FT_trk_hvy,
     occu_trgt_PM_bus_cch, occu_trgt_PM_trn_cnv, kgoe_to_kWh) = need(
        ctx, 'years',
        'population_dict',
        'df_SUF',
        'df_PM',
        'df_FT',
        'df_PM_TWh_all',
        'df_FT_TWh_all',
        'df_PM_car',
        'ref_occu_PM_car',
        'pro_PM_spe',
        'pro_PM_spe_car',
        'occu_trgt_PM_car',
        'redu_fuel_PM_car',
        'sft_PM_rel_car_to_bus',
        'sft_PM_rel_car_to_trn_cnv',
        'sft_PM_rel_car_to_byc',
        'sft_PM_rel_car_to_trm_met',
        'sft_PM_rel_car_to_mot',
        'sft_PM_rel_car_to_ped',
        'pro_FT_spe',
        'pro_FT_spe_trk_hvy',
        'sft_FT_rel_trk_hvy_to_trn',
        'sft_FT_rel_trk_hvy_to_nav_ild',
        'pyld_trgt_FT_trk_hvy',
        'ref_pyld_FT_trk_hvy',
        'occu_trgt_PM_bus_cch',
        'occu_trgt_PM_trn_cnv',
        'kgoe_to_kWh')

    _NB = NOTEBOOK
    _Y0, _Y1 = years[0], years[-1]
    _pop = {y: float(population_dict[y]) for y in (_Y0, _Y1)}

    # --- Reference values quoted only in this notebook's prose -------------------
    # Named here so the workshop can export them, and so that editing either the
    # text or the model trips the assertions below instead of drifting silently.
    # Section numbers refer to the markdown headings above.
    ref_occu_PM_car_survey  = 1.20   # SPF Mobilité travel surveys [3,4]      -- §2.1
    ref_occu_PM_car_idees   = 1.22   # JRC-IDEES fleet average [1,2]          -- §2.1
    ref_road_share_tkm_2019 = 66.8   # % of all tonne-km on the road [1,2]    -- §3.1
    ref_ebike_share_km_2025 = 64.0   # % of bicycle-km ridden on e-bikes [4]  -- §2.2.1
    ref_bike_trip_km        = 4.0    # average conventional-bicycle trip [4]  -- §2.2.1
    ref_ebike_trip_km       = 5.0    # average e-bike trip [4]                -- §2.2.1
    ref_pedelec_trip_km     = 10.0   # average speed-pedelec trip [4]         -- §2.2.1
    ref_nav_ild_peak_2017   = 980.0  # inland-waterway peak, tkm/person [1,2] -- §3.1.3
    ref_nav_ild_2023        = 594.0  # inland waterways in 2023, tkm/person   -- §3.1.3
    ref_ft_spe_2023         = 6455.0 # freight intensity in 2023, tkm/person  -- §1.2.2
    ref_pm_spe_ground_min   = 5000.0 # Peeters et al. (2026) lower bound      -- §1.2.1
    ref_pm_spe_ground_max   = 15000.0 # Peeters et al. (2026) upper bound     -- §1.2.1
    ref_rail_share_2040_target = 15.0 # Vision Rail 2040, % of pkm            -- §2.1.3
    ref_kwh_per_litre_petrol = 9.7   # lower heating value of petrol, kWh/litre
    ref_bike_trips_2040_flanders = 30.0 # FietsDNA target, % of trips         -- §2.1.4

    # --- Scope ------------------------------------------------------------------
    # "motorised ground" matches JRC-IDEES road + rail exactly (aviation excluded,
    # walking and cycling excluded); "inland" additionally includes the active modes.
    _PM_MOTORISED = ["two-wheeler", "tram&metro", "bus&coach", "car",
                     "train-conventional", "train-high speed"]
    _PM_ACTIVE    = ["pedestrian", "bicycle"]
    _PM_INLAND    = _PM_MOTORISED + _PM_ACTIVE
    _FT_INLAND    = ["train", "navigation-inland", "navigation-coastal",
                     "truck-heavy duty", "truck-light commercial"]
    _FT_ROAD      = ["truck-heavy duty", "truck-light commercial"]

    _pm_twh = mode_totals_twh(df_PM_TWh_all)
    _ft_twh = mode_totals_twh(df_FT_TWh_all)

    def _twh(tbl, mode, year):
        return float(tbl.loc[mode, year]) if mode in tbl.index else 0.0

    def _act(df, mode, unit, year):
        return float(df.loc[(mode, unit), year])

    def _intensity(tbl, df, mode, unit, year):
        """kWh per pkm (or tkm): TWh/Gpkm is already kWh/pkm, no conversion needed."""
        act = _act(df, mode, unit, year)
        return _twh(tbl, mode, year) / act if act else 0.0

    _pm_mot_twh    = {y: sum(_twh(_pm_twh, m, y) for m in _PM_MOTORISED) for y in (_Y0, _Y1)}
    _pm_inland_twh = {y: sum(_twh(_pm_twh, m, y) for m in _PM_INLAND)    for y in (_Y0, _Y1)}
    _ft_inland_twh = {y: sum(_twh(_ft_twh, m, y) for m in _FT_INLAND)    for y in (_Y0, _Y1)}
    _inland_twh    = {y: _pm_inland_twh[y] + _ft_inland_twh[y]           for y in (_Y0, _Y1)}
    _TOT = _inland_twh[_Y1]

    # --- Consistency checks: the model vs. the figures written in the prose -----
    # 2019 fleet-average car occupancy = total pkm / total vkm, i.e. the pkm-weighted
    # *harmonic* mean of the per-powertrain occupancies. Section 2.1 quotes 1,22.
    _occ_car_2019 = 100.0 / sum(float(df_PM_car.loc[_pt, _Y0]) / _occ
                                for _pt, _occ in ref_occu_PM_car.items())
    assert abs(_occ_car_2019 - ref_occu_PM_car_idees) < 0.005, (
        f"fleet-average car occupancy is now {_occ_car_2019:.4f}, but section 2.1 "
        f"quotes {ref_occu_PM_car_idees} -- update one or the other")

    _car_dest = {"bus&coach":          sft_PM_rel_car_to_bus,
                 "train-conventional": sft_PM_rel_car_to_trn_cnv,
                 "bicycle":            sft_PM_rel_car_to_byc,
                 "tram&metro":         sft_PM_rel_car_to_trm_met,
                 "two-wheeler":        sft_PM_rel_car_to_mot,
                 "pedestrian":         sft_PM_rel_car_to_ped}
    assert abs(sum(_car_dest.values()) + pro_PM_spe_car) < 1e-9, (
        "the car modal-shift destinations no longer add up to the shift itself "
        f"({sum(_car_dest.values()):.3f} vs {-pro_PM_spe_car:.3f}) -- section 2.1.2")

    _trk_dest = {"train":             sft_FT_rel_trk_hvy_to_trn,
                 "navigation-inland": sft_FT_rel_trk_hvy_to_nav_ild}
    assert abs(sum(_trk_dest.values()) + pro_FT_spe_trk_hvy) < 1e-9, (
        "the heavy-truck modal-shift destinations no longer add up to the shift "
        f"({sum(_trk_dest.values()):.3f} vs {-pro_FT_spe_trk_hvy:.3f}) -- section 3.1.1")

    _road_share_tkm = {y: 100.0 * sum(_act(df_FT, m, "tkm/person", y) for m in _FT_ROAD)
                            / float(df_SUF.loc["FT intensity [tkm/person]", y])
                       for y in (_Y0, _Y1)}
    assert abs(_road_share_tkm[_Y0] - ref_road_share_tkm_2019) < 0.1, (
        f"the 2019 road share of tonne-km is now {_road_share_tkm[_Y0]:.2f}%, but "
        f"section 3.1 quotes {ref_road_share_tkm_2019}% -- update one or the other")

    # --- Derived lever quantities ----------------------------------------------
    _mot_pkm  = {y: sum(_act(df_PM, m, "pkm/person", y) for m in _PM_MOTORISED)
                 for y in (_Y0, _Y1)}
    _mot_km_d = {y: _mot_pkm[y] / 365.0 for y in (_Y0, _Y1)}
    _car_share_mot = {y: 100.0 * _act(df_PM, "car", "pkm/person", y) / _mot_pkm[y]
                      for y in (_Y0, _Y1)}
    _bike_km_d = {y: _act(df_PM, "bicycle", "pkm/person", y) / 365.0 for y in (_Y0, _Y1)}
    _ft_spe    = {y: float(df_SUF.loc["FT intensity [tkm/person]", y]) for y in (_Y0, _Y1)}

    def _basket_intensity(dest, df, unit, tbl):
        weight = sum(dest.values())
        return sum(w / weight * _intensity(tbl, df, m, unit, _Y1) for m, w in dest.items())

    def _share_slope(total_activity_per_person, source, dest, df, unit, tbl):
        """TWh change per +1 percentage point of `source`'s share of the total.

        Exact for a single lever moved alone: the modal shift is share-preserving on
        the total, and the 2050 energy intensities do not depend on the shares, so
        the response is linear.
        """
        activity_per_pt = total_activity_per_person / 100.0 * _pop[_Y1] / 1e9
        return activity_per_pt * (_intensity(tbl, df, source, unit, _Y1)
                                  - _basket_intensity(dest, df, unit, tbl))

    _car_share_slope = _share_slope(_mot_pkm[_Y1], "car", _car_dest, df_PM, "Gpkm", _pm_twh)
    _road_share_slope = _share_slope(_ft_spe[_Y1], "truck-heavy duty", _trk_dest,
                                     df_FT, "Gtkm", _ft_twh)
    # Extra bicycle-km are taken out of car-km (that is what sft_PM_rel_car_to_byc does).
    _bike_slope = (_pop[_Y1] * 365.0 / 1e9) * (_intensity(_pm_twh, df_PM, "bicycle", "Gpkm", _Y1)
                                               - _intensity(_pm_twh, df_PM, "car", "Gpkm", _Y1))

    def _impact(kind, v_target, scaled=0.0, slope=None):
        """Leverage record read by website/assets/js/workshop/impact.js.

        TWh(vTarget) always equals `total`, the négaWatt inland-mobility 2050 demand,
        so every lever's readout is on the same comparable scale.
        """
        rec = {"kind": kind, "vTarget": round(float(v_target), 4),
               "total": round(_TOT, 4), "scaled": round(float(scaled), 4)}
        if slope is not None:
            rec["slope"] = round(float(slope), 6)
        return rec

    # --- The levers -------------------------------------------------------------
    _L = []
    def _add(*a, **k):
        _L.append(make_lever(*a, **k))

    _T = "inland-mobility"

    _add("ground-km-day", _T, "Motorised travel per person", "km/person/day",
         _mot_km_d[_Y0], _mot_km_d[_Y1], ref_year=_Y0, target_year=_Y1,
         slider={"min": 15, "max": 42, "step": 0.5},
         impact=_impact("proportional", _mot_km_d[_Y1], scaled=_pm_mot_twh[_Y1]),
         model={"var": "pro_PM_spe", "section": "1.2.1",
                "note": "road + rail only; the -10% total mobility cut is split 45/55 "
                        "with aviation in section 2.1.1"},
         history="ground_km_day",
         facts={"pkmYear2019": round(_mot_pkm[_Y0]), "pkmYearTarget": round(_mot_pkm[_Y1]),
                "changePct": round(100 * (_mot_km_d[_Y1] / _mot_km_d[_Y0] - 1), 1),
                "totalIntensityChangePct": round(pro_PM_spe * 100, 1),
                "aviationChangePct": round(100 * (_act(df_PM, "plane-extra EU", "pkm/person", _Y1)
                                                  + _act(df_PM, "plane-intra EU", "pkm/person", _Y1))
                                           / (_act(df_PM, "plane-extra EU", "pkm/person", _Y0)
                                              + _act(df_PM, "plane-intra EU", "pkm/person", _Y0)) - 100, 1),
                "peetersMin": ref_pm_spe_ground_min, "peetersMax": ref_pm_spe_ground_max},
         spoilers=["changePct", "totalIntensityChangePct", "aviationChangePct", "pkmYearTarget"],
         notebook=_NB + "#section_1", reference="nW-BE §1.2.1")

    _add("car-share", _T, "Car share of motorised travel", "% of motorised km",
         _car_share_mot[_Y0], _car_share_mot[_Y1], ref_year=_Y0, target_year=_Y1,
         slider={"min": 30, "max": 90, "step": 1},
         impact=_impact("linear-shift", _car_share_mot[_Y1], slope=_car_share_slope),
         model={"var": "pro_PM_spe_car", "section": "2.1.2",
                "note": "a -30% shift of car-km; the destinations are fixed at the "
                        "négaWatt split (bus 10, rail 8, bike 7, tram 3, moto 1, walk 1)"},
         history="car_share_motorised",
         facts={"shiftPct": round(-pro_PM_spe_car * 100, 1),
                "busShareMot2019": round(100 * _act(df_PM, "bus&coach", "pkm/person", _Y0)
                                         / _mot_pkm[_Y0], 1),
                "railShareMot2019": round(100 * (_act(df_PM, "train-conventional", "pkm/person", _Y0)
                                                 + _act(df_PM, "train-high speed", "pkm/person", _Y0))
                                          / _mot_pkm[_Y0], 1),
                "tramShareMot2019": round(100 * _act(df_PM, "tram&metro", "pkm/person", _Y0)
                                          / _mot_pkm[_Y0], 1),
                "motoShareMot2019": round(100 * _act(df_PM, "two-wheeler", "pkm/person", _Y0)
                                          / _mot_pkm[_Y0], 1),
                "railShareTarget2040": ref_rail_share_2040_target,
                "toBusPct": round(sft_PM_rel_car_to_bus * 100, 1),
                "toRailPct": round(sft_PM_rel_car_to_trn_cnv * 100, 1),
                "toBikePct": round(sft_PM_rel_car_to_byc * 100, 1)},
         spoilers=["shiftPct", "toBusPct", "toRailPct", "toBikePct"],
         notebook=_NB + "#section_2", reference="nW-BE §2.1")

    _add("car-occupancy", _T, "Car occupancy", "persons/car",
         _occ_car_2019, occu_trgt_PM_car, ref_year=_Y0, target_year=_Y1,
         slider={"min": 1.0, "max": 2.6, "step": 0.05},
         impact=_impact("inverse", occu_trgt_PM_car, scaled=_twh(_pm_twh, "car", _Y1)),
         model={"var": "occu_trgt_PM_car", "section": "2.3.5",
                "note": "pkm-weighted fleet average, i.e. total pkm / total vehicle-km"},
         history="car_occupancy",
         facts={"occupancySurvey": ref_occu_PM_car_survey,
                "occupancyIdees": ref_occu_PM_car_idees,
                "carTwh2019": round(_twh(_pm_twh, "car", _Y0), 1),
                "carTwhTarget": round(_twh(_pm_twh, "car", _Y1), 2),
                "gainPct": round(100 * (occu_trgt_PM_car / _occ_car_2019 - 1), 1)},
         spoilers=["gainPct", "carTwhTarget"],
         notebook=_NB + "#section_2", reference="nW-BE §2.3.5")

    _add("car-energy", _T, "Car energy use per km", "% of 2019",
         100.0, redu_fuel_PM_car * 100, ref_year=_Y0, target_year=_Y1,
         slider={"min": 50, "max": 115, "step": 1},
         impact=_impact("proportional", redu_fuel_PM_car * 100, scaled=_twh(_pm_twh, "car", _Y1)),
         model={"var": "redu_fuel_PM_car", "section": "2.3.5",
                "note": "speed limits, eco-driving and smaller cars are bundled into this "
                        "single figure; section 2.3.5 flags the split as unquantified"},
         facts={"reductionPct": round((1 - redu_fuel_PM_car) * 100, 1),
                "kwhPerKmPetrol2019": round(5.798 / 100 * kgoe_to_kWh, 3),
                "kwhPerKmBev2019": round(1.818 / 100 * kgoe_to_kWh, 3),
                "litresPetrol2019": round(5.798 / 100 * kgoe_to_kWh
                                          / ref_kwh_per_litre_petrol * 100, 1),
                "litresPetrolEqBev2019": round(1.818 / 100 * kgoe_to_kWh
                                               / ref_kwh_per_litre_petrol * 100, 1)},
         spoilers=["reductionPct"],
         notebook=_NB + "#section_2", reference="nW-BE §2.3.5")

    _add("bike-km-day", _T, "Cycling per person", "km/person/day",
         _bike_km_d[_Y0], _bike_km_d[_Y1], ref_year=_Y0, target_year=_Y1,
         slider={"min": 0.5, "max": 8, "step": 0.1},
         impact=_impact("linear-shift", _bike_km_d[_Y1], slope=_bike_slope),
         model={"var": "sft_PM_rel_car_to_byc", "section": "2.1.4"},
         facts={"bikeSharePkmTarget": round(_act(df_PM, "bicycle", "% of total", _Y1), 1),
                "bikeGpkm2019": round(_act(df_PM, "bicycle", "Gpkm", _Y0), 1),
                "toBikePct": round(sft_PM_rel_car_to_byc * 100, 1),
                "shiftPct": round(-pro_PM_spe_car * 100, 1),
                "ebikeShareKm2025": ref_ebike_share_km_2025,
                "bikeTripKm": ref_bike_trip_km, "ebikeTripKm": ref_ebike_trip_km,
                "pedelecTripKm": ref_pedelec_trip_km,
                "flandersTripTarget": ref_bike_trips_2040_flanders},
         spoilers=["bikeSharePkmTarget", "toBikePct", "shiftPct"],
         notebook=_NB + "#section_2", reference="nW-BE §2.1")

    _add("freight-tkm", _T, "Goods moved per person", "tkm/person/year",
         _ft_spe[_Y0], _ft_spe[_Y1], ref_year=_Y0, target_year=_Y1,
         slider={"min": 4000, "max": 9000, "step": 100},
         impact=_impact("proportional", _ft_spe[_Y1], scaled=_ft_inland_twh[_Y1]),
         model={"var": "pro_FT_spe", "section": "1.2.2"},
         history="freight_tkm_person",
         facts={"changePct": round(pro_FT_spe * 100, 1), "tkm2023": ref_ft_spe_2023,
                "growth2000to2019Pct": 26.4},
         spoilers=["changePct"],
         notebook=_NB + "#section_3", reference="nW-BE §1.2.2")

    _add("truck-share", _T, "Road share of freight", "% of tonne-km",
         _road_share_tkm[_Y0], _road_share_tkm[_Y1], ref_year=_Y0, target_year=_Y1,
         slider={"min": 30, "max": 85, "step": 1},
         impact=_impact("linear-shift", _road_share_tkm[_Y1], slope=_road_share_slope),
         model={"var": "pro_FT_spe_trk_hvy", "section": "3.1.1",
                "note": "a -25% shift of heavy-truck tonne-km, 15 points to rail and 10 "
                        "to inland waterways; air freight (~5%) stays in the denominator"},
         history="road_share_tkm",
         facts={"shiftPct": round(-pro_FT_spe_trk_hvy * 100, 1),
                "toRailPct": round(sft_FT_rel_trk_hvy_to_trn * 100, 1),
                "toWaterPct": round(sft_FT_rel_trk_hvy_to_nav_ild * 100, 1),
                "navPeak2017": ref_nav_ild_peak_2017, "nav2023": ref_nav_ild_2023,
                "truckKwhPerTkm": round(_intensity(_ft_twh, df_FT, "truck-heavy duty", "Gtkm", _Y1), 4),
                "bargeKwhPerTkm": round(_intensity(_ft_twh, df_FT, "navigation-inland", "Gtkm", _Y1), 4),
                "railKwhPerTkm": round(_intensity(_ft_twh, df_FT, "train", "Gtkm", _Y1), 4),
                "truckKwhPerTkm2019": round(_intensity(_ft_twh, df_FT, "truck-heavy duty", "Gtkm", _Y0), 4)},
         spoilers=["shiftPct", "toRailPct", "toWaterPct"],
         notebook=_NB + "#section_3", reference="nW-BE §3.1")

    _add("truck-load", _T, "Truck payload", "tonnes",
         ref_pyld_FT_trk_hvy, ref_pyld_FT_trk_hvy * pyld_trgt_FT_trk_hvy,
         ref_year=_Y0, target_year=_Y1,
         slider={"min": 11, "max": 18, "step": 0.1},
         impact=_impact("inverse", ref_pyld_FT_trk_hvy * pyld_trgt_FT_trk_hvy,
                        scaled=_twh(_ft_twh, "truck-heavy duty", _Y1)),
         model={"var": "pyld_trgt_FT_trk_hvy", "section": "3.3.3"},
         facts={"gainPct": round((pyld_trgt_FT_trk_hvy - 1) * 100, 1),
                "truckTwhTarget": round(_twh(_ft_twh, "truck-heavy duty", _Y1), 2)},
         spoilers=["gainPct", "truckTwhTarget"],
         notebook=_NB + "#section_3", reference="nW-BE §3.3.5")

    # --- Spare levers: exported for later, not played in the proof of concept ----
    _add("bus-occupancy", _T, "Bus & coach occupancy", "% of 2019",
         100.0, occu_trgt_PM_bus_cch * 100, ref_year=_Y0, target_year=_Y1,
         slider={"min": 80, "max": 170, "step": 5}, shown=False,
         impact=_impact("inverse", occu_trgt_PM_bus_cch * 100,
                        scaled=_twh(_pm_twh, "bus&coach", _Y1)),
         model={"var": "occu_trgt_PM_bus_cch", "section": "2.3.4"},
         notebook=_NB + "#section_2", reference="nW-BE §2.3.4")
    _add("train-occupancy", _T, "Conventional train occupancy", "% of 2019",
         100.0, occu_trgt_PM_trn_cnv * 100, ref_year=_Y0, target_year=_Y1,
         slider={"min": 80, "max": 170, "step": 5}, shown=False,
         impact=_impact("inverse", occu_trgt_PM_trn_cnv * 100,
                        scaled=_twh(_pm_twh, "train-conventional", _Y1)),
         model={"var": "occu_trgt_PM_trn_cnv", "section": "2.3.6"},
         notebook=_NB + "#section_2", reference="nW-BE §2.3.6")

    # --- Shared model quantities (context + the leverage arithmetic) -------------
    _ws_model = {
        "scope": "inland mobility: domestic passenger ground transport + domestic freight "
                 "(aviation excluded)",
        "refYear": _Y0, "targetYear": _Y1,
        "population": {str(y): round(_pop[y]) for y in (_Y0, _Y1)},
        "inlandTwh": {str(y): round(_inland_twh[y], 3) for y in (_Y0, _Y1)},
        "inlandPassengerTwh": {str(y): round(_pm_inland_twh[y], 3) for y in (_Y0, _Y1)},
        "motorisedPassengerTwh": {str(y): round(_pm_mot_twh[y], 3) for y in (_Y0, _Y1)},
        "inlandFreightTwh": {str(y): round(_ft_inland_twh[y], 3) for y in (_Y0, _Y1)},
        "modeTwhTarget": {m: round(_twh(_pm_twh, m, _Y1), 3) for m in _PM_INLAND},
        "freightModeTwhTarget": {m: round(_twh(_ft_twh, m, _Y1), 3) for m in _FT_INLAND},
        "modeIntensityTarget": {m: round(_intensity(_pm_twh, df_PM, m, "Gpkm", _Y1), 5)
                                for m in _PM_INLAND},
        "freightModeIntensityTarget": {m: round(_intensity(_ft_twh, df_FT, m, "Gtkm", _Y1), 5)
                                       for m in _FT_INLAND},
    }

    return {"levers": _L, "model": _ws_model}
