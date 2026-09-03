"""Residential heat: heating, cooling, hot water and cooking in Belgian homes.

Seven levers, all of them genuine degrees of freedom of the buildings notebook's
residential block, so the reveal is exact and no group can enter a
self-inconsistent scenario:

    floor-area       m² of dwelling per person        (pro_RS_sur_spe)
    renovation-rate  % of the housing stock per year  (acc_RS_tes_sht_ren)
    thermostat       °C off the heating setpoint      (suf_RS_tes_sht)
    hot-water        kWh/person/year                  (trg_RS_tes_shw)
    cooling          kWh/m²/year                      (acc_RS_tes_scl_ren)
    cooking          kWh/household/year               (pro_RS_tes_cok)
    district-heat    % of home heat from a network    (trg_RS_tes_dhn)

The tertiary sector is a separate topic (`tertiary_heat.py`) out of the same
notebook, so nothing here reads a `*_TS_*` variable.

Everything is read from quantities the notebook has already computed — this
module adds no assumptions of its own. It also cross-checks the model against
the figures written in the notebook's prose, so that editing either the text or
the code without the other fails loudly here rather than feeding a wrong number
into a workshop.

Two traps in the notebook's namespace, both handled below:

  * `d_temp`, `d_cons_temp`, `share_heat_dhn` and `share_heat_ihs` are assigned
    twice — once in the residential section, then again with tertiary values
    (cells 65 and 75). By the time this module runs they hold the *tertiary*
    numbers. The residential setpoint drop is therefore recovered from
    `suf_RS_tes_sht`, and the carrier split from `ref_RS_tes_dhn` /
    `trg_RS_tes_dhn`, never from the shared names.
  * the notebook never computes the observed renovation rate, only the
    improvement in kWh/m²/year it is attributed to. The rate quoted in the prose
    of section 2.1.1 is named here as a reference constant, exactly like the
    prose values in `inland_mobility.py`.

See docs/workshop_module.md for the design, and
website/workshop/content/residential-heat.yaml for the wording.
"""
from nW_BE_demand_model_sub_functions import make_lever

from . import need

TOPIC = "residential-heat"
SECTOR = "buildings"
ORDER = 30

NOTEBOOK = "../notebooks/nW_BE_demand_model_buildings.html"


def build(ctx):
    (years, population_dict, households_dict, df_SUF,
     df_tes_RS_tot, df_ees_RS_tot,
     ref_RS_sur_spe, ref_RS_sur_hld, pro_RS_sur_spe,
     ref_RS_tes_sht, trg_RS_tes_sht, suf_RS_tes_sht,
     acc_RS_tes_sht_ren, cur_RS_tes_sht_ren,
     ref_RS_tes_scl, trg_RS_tes_scl, acc_RS_tes_scl_ren, cur_RS_tes_scl_ren,
     ref_RS_tes_shw, trg_RS_tes_shw, pro_RS_tes_shw,
     shower_duration, shower_flow_rate, shower_temperature,
     others_volume, others_temperature,
     ref_RS_tes_cok, trg_RS_tes_cok, pro_RS_tes_cok,
     ref_RS_tes_cok_gas, trg_RS_tes_cok_gas,
     ref_RS_tes_dhn, trg_RS_tes_dhn,
     cp_h2o, rho_h2o) = need(
        ctx, 'years',
        'population_dict',
        'households_dict',
        'df_SUF',
        'df_tes_RS_tot',
        'df_ees_RS_tot',
        'ref_RS_sur_spe',
        'ref_RS_sur_hld',
        'pro_RS_sur_spe',
        'ref_RS_tes_sht',
        'trg_RS_tes_sht',
        'suf_RS_tes_sht',
        'acc_RS_tes_sht_ren',
        'cur_RS_tes_sht_ren',
        'ref_RS_tes_scl',
        'trg_RS_tes_scl',
        'acc_RS_tes_scl_ren',
        'cur_RS_tes_scl_ren',
        'ref_RS_tes_shw',
        'trg_RS_tes_shw',
        'pro_RS_tes_shw',
        'shower_duration',
        'shower_flow_rate',
        'shower_temperature',
        'others_volume',
        'others_temperature',
        'ref_RS_tes_cok',
        'trg_RS_tes_cok',
        'pro_RS_tes_cok',
        'ref_RS_tes_cok_gas',
        'trg_RS_tes_cok_gas',
        'ref_RS_tes_dhn',
        'trg_RS_tes_dhn',
        'cp_h2o',
        'rho_h2o')

    _NB = NOTEBOOK
    _Y0, _Y1 = years[0], years[-1]
    _pop = {y: float(population_dict[y]) for y in (_Y0, _Y1)}
    _hld = {y: float(households_dict[y]) for y in (_Y0, _Y1)}

    # --- Reference values quoted only in this notebook's prose -------------------
    # Named here so the workshop can export them, and so that editing either the
    # text or the model trips the assertions below instead of drifting silently.
    # Section numbers refer to the markdown headings of the buildings notebook.
    ref_RS_reno_rate       = 3.0    # % of floor area renewed per year, 2000-2023  -- §2.1.1
    ref_RS_cook_trend      = -1.6   # kWh/household/year, 2000-2023 average        -- §2.1.2
    ref_slowheat_ok_degc   = 15.0   # average indoor temperature judged liveable [3] -- §2.1.2
    ref_slowheat_min_degc  = 12.0   # vigilance threshold [3]                      -- §2.1.2
    ref_setpoint_2022_degc = 19.0   # the floor most households kept in 2022 [3]   -- §2.1.2
    ref_dhn_potential_pct  = 45.0   # techno-economic potential for 2050 [4]        -- §2.1.3
    ref_dhn_paths2050_pct  = 13.0   # EnergyVille PATHS2050, buildings [5]         -- §2.1.3
    ref_pct_per_degc       = 7.0    # % of heat demand saved per -1 °C [3]         -- §2.1.2
    ref_setpoint_drop_degc = 2.0    # the setpoint reduction assumed for 2050      -- §2.1.2

    # --- Scope ------------------------------------------------------------------
    # The four thermal services of the residential sector: everything the seven
    # levers below act on. Appliances (fridge, washing machine, lighting…) are
    # residential too but are not part of this topic, so they stay out of the
    # total against which the leverage readout is calibrated.
    _TES = ["space heating", "space cooling", "sanitary hot water", "cooking"]
    _EES = list(df_ees_RS_tot.columns)

    def _twh(df, column, year):
        return float(df.loc[year, column])

    _tes = {y: {c: _twh(df_tes_RS_tot, c, y) for c in _TES} for y in (_Y0, _Y1)}
    _tes_tot = {y: sum(_tes[y].values()) for y in (_Y0, _Y1)}
    _ees_tot = {y: sum(_twh(df_ees_RS_tot, c, y) for c in _EES) for y in (_Y0, _Y1)}
    _res_tot = {y: _tes_tot[y] + _ees_tot[y] for y in (_Y0, _Y1)}
    _TOT = _tes_tot[_Y1]

    _sur_spe = {y: float(df_SUF["RS specific surface [m²/person]"][y]) for y in (_Y0, _Y1)}
    _sur_tot = {y: float(df_SUF["RS total surface [Mm²]"][y]) for y in (_Y0, _Y1)}       # Mm²
    _sur_hld = {y: float(df_SUF["RS household surface [m²/household]"][y]) for y in (_Y0, _Y1)}
    _hld_size = {y: _pop[y] / _hld[y] for y in (_Y0, _Y1)}

    # --- Consistency checks: the model vs. the figures written in the prose -----
    assert abs(_sur_spe[_Y0] - ref_RS_sur_spe) < 1e-6, (
        "df_SUF's specific residential surface no longer starts at ref_RS_sur_spe")
    assert abs(_sur_hld[_Y0] - ref_RS_sur_hld) < 1e-6, (
        "df_SUF's household surface no longer starts at ref_RS_sur_hld")

    # -2 °C at -7% per degree = the 0.86 factor of section 2.1.2. `d_temp` itself
    # cannot be read: cell 65 reassigns it to the tertiary value of 1 °C.
    _setpoint_drop = (1.0 - suf_RS_tes_sht) / (ref_pct_per_degc / 100.0)
    assert abs(_setpoint_drop - ref_setpoint_drop_degc) < 1e-9, (
        f"the residential setpoint reduction implied by suf_RS_tes_sht is now "
        f"{_setpoint_drop:.3f} °C at {ref_pct_per_degc} %/°C, but section 2.1.2 "
        f"quotes {ref_setpoint_drop_degc} °C -- update one or the other")

    # Space heating is the only service that carries both the renovation and the
    # setpoint assumption, so its 2050 value must be reproducible from the two.
    _sht_twh_eff_only = trg_RS_tes_sht * _sur_tot[_Y1] * 1e-3
    assert abs(_sht_twh_eff_only * suf_RS_tes_sht - _tes[_Y1]["space heating"]) < 1e-6, (
        "space heating in 2050 is no longer trg_RS_tes_sht x suf_RS_tes_sht x floor area; "
        "the levers below would misreport their leverage")
    assert abs(trg_RS_tes_sht - (ref_RS_tes_sht
                                 + acc_RS_tes_sht_ren * cur_RS_tes_sht_ren
                                 * (_Y1 - _Y0))) < 1e-9, (
        "trg_RS_tes_sht is no longer the linear improvement of section 2.1.1")
    assert abs(trg_RS_tes_scl - (ref_RS_tes_scl
                                 + acc_RS_tes_scl_ren * cur_RS_tes_scl_ren
                                 * (_Y1 - _Y0))) < 1e-9, (
        "trg_RS_tes_scl is no longer the linear deployment of section 2.1.2")

    # The renovation lever asks for a rate in % of the stock per year, which the
    # notebook only quotes in prose; the improvement in kWh/m²/year is what the
    # model actually uses. The two are proportional by construction (§2.1.1).
    _reno_rate_target = acc_RS_tes_sht_ren * ref_RS_reno_rate

    # --- Derived lever quantities ----------------------------------------------
    _litres_40c = {y: v / (rho_h2o * cp_h2o * (40.0 - 15.0)) / 365.0
                   for y, v in ((_Y0, ref_RS_tes_shw), (_Y1, trg_RS_tes_shw))}
    _shower_kwh = (shower_duration * shower_flow_rate * rho_h2o * cp_h2o
                   * (shower_temperature - 15.0))
    _others_kwh = others_volume * rho_h2o * cp_h2o * (others_temperature - 15.0)
    _kwh_per_100l_40c = 100.0 * rho_h2o * cp_h2o * (40.0 - 15.0)

    # Space heating and cooling both scale with the floor area; hot water scales
    # with population and cooking with households, so neither moves with m²/person.
    _area_driven_twh = {y: _tes[y]["space heating"] + _tes[y]["space cooling"]
                        for y in (_Y0, _Y1)}
    # Heat a network could carry: space heating + sanitary hot water (cell 57).
    _networkable_twh = {y: _tes[y]["space heating"] + _tes[y]["sanitary hot water"]
                        for y in (_Y0, _Y1)}

    # TWh per extra point of renovation rate, and per degree off the thermostat.
    # Exact for a single lever moved alone: the 2050 intensity is affine in each
    # of the two, and the floor area does not depend on either.
    _reno_slope = (cur_RS_tes_sht_ren * (_Y1 - _Y0) / ref_RS_reno_rate
                   * suf_RS_tes_sht * _sur_tot[_Y1] * 1e-3)
    _setpoint_slope = -(ref_pct_per_degc / 100.0) * _sht_twh_eff_only
    # What one degree is worth on today's housing stock, for the tangible card.
    _degc_twh_2019 = (ref_pct_per_degc / 100.0) * _tes[_Y0]["space heating"]

    def _impact(kind, v_target, scaled=0.0, slope=None):
        """Leverage record read by website/assets/js/workshop/impact.js.

        TWh(vTarget) always equals `total`, the négaWatt 2050 demand for the four
        residential thermal services, so every lever's readout is on the same
        comparable scale.
        """
        rec = {"kind": kind, "vTarget": round(float(v_target), 4),
               "total": round(_TOT, 4), "scaled": round(float(scaled), 4)}
        if slope is not None:
            rec["slope"] = round(float(slope), 6)
        return rec

    def _pct(part, whole):
        return round(100.0 * part / whole, 1) if whole else 0.0

    # --- The levers -------------------------------------------------------------
    _L = []
    def _add(*a, **k):
        _L.append(make_lever(*a, **k))

    _T = "residential-heat"

    _add("floor-area", _T, "Home floor area per person", "m² of home per person",
         _sur_spe[_Y0], _sur_spe[_Y1], ref_year=_Y0, target_year=_Y1,
         slider={"min": 40, "max": 65, "step": 0.5},
         impact=_impact("proportional", _sur_spe[_Y1], scaled=_area_driven_twh[_Y1]),
         model={"var": "pro_RS_sur_spe", "section": "1.2.1",
                "note": "space heating and cooling scale with the floor area; hot water "
                        "scales with population and cooking with households, so they do "
                        "not move with this lever"},
         history="res_m2_per_person",
         facts={"changePct": round(pro_RS_sur_spe * 100, 1),
                "m2PerHousehold": round(_sur_hld[_Y0], 1),
                "m2PerHouseholdTarget": round(_sur_hld[_Y1], 1),
                "householdChangePct": round(100 * (_sur_hld[_Y1] / _sur_hld[_Y0] - 1), 1),
                "householdSize": round(_hld_size[_Y0], 2),
                "householdSizeTarget": round(_hld_size[_Y1], 2),
                "householdSizeChangePct": round(100 * (_hld_size[_Y1] / _hld_size[_Y0] - 1), 1),
                "stockMm2": round(_sur_tot[_Y0], 0),
                "areaDrivenTwh": round(_area_driven_twh[_Y0], 1),
                "areaDrivenSharePct": _pct(_area_driven_twh[_Y0], _tes_tot[_Y0]),
                "thermalTwh": round(_tes_tot[_Y0], 1)},
         spoilers=["changePct", "m2PerHouseholdTarget", "householdChangePct"],
         notebook=_NB + "#section_1", reference="nW-BE §1.2.1")

    _add("renovation-rate", _T, "Renovation rate of the housing stock",
         "% of the housing stock per year",
         ref_RS_reno_rate, _reno_rate_target, ref_year=_Y0, target_year=_Y1,
         slider={"min": 1, "max": 9, "step": 0.25},
         impact=_impact("linear-shift", _reno_rate_target, slope=_reno_slope),
         model={"var": "acc_RS_tes_sht_ren", "section": "2.1.1",
                "prose": "the rate itself is quoted only in the markdown of section "
                         "2.1.1; the model input is the multiplier on the observed "
                         "-0.458 kWh/m²/year improvement",
                "note": "renovation and new construction together, measured on floor "
                        "area, not on the number of dwellings"},
         history="res_renovation_rate",
         facts={"accTarget": acc_RS_tes_sht_ren,
                "improvementHist": cur_RS_tes_sht_ren,
                "improvementTarget": round(acc_RS_tes_sht_ren * cur_RS_tes_sht_ren, 3),
                "intensity2019": round(ref_RS_tes_sht, 1),
                "intensityEffTarget": round(trg_RS_tes_sht, 1),
                "intensityTarget": round(trg_RS_tes_sht * suf_RS_tes_sht, 1),
                "heatTwh": round(_tes[_Y0]["space heating"], 1),
                "heatSharePct": _pct(_tes[_Y0]["space heating"], _tes_tot[_Y0]),
                "thermalTwh": round(_tes_tot[_Y0], 1),
                "rateHist": ref_RS_reno_rate},
         spoilers=["accTarget", "improvementTarget", "intensityEffTarget",
                   "intensityTarget"],
         notebook=_NB + "#section_2", reference="nW-BE §2.1.1")

    _add("thermostat", _T, "Degrees off the heating setpoint",
         "°C less on the home thermostat",
         0.0, _setpoint_drop, ref_year=_Y0, target_year=_Y1,
         slider={"min": -1, "max": 6, "step": 0.5},
         impact=_impact("linear-shift", _setpoint_drop, slope=_setpoint_slope),
         model={"var": "suf_RS_tes_sht", "section": "2.1.2",
                "note": "d_temp cannot be read from the notebook's globals: cell 65 "
                        "reassigns it to the tertiary value, so the residential drop is "
                        "recovered from suf_RS_tes_sht"},
         facts={"pctPerDegC": ref_pct_per_degc,
                "demandCutPct": round((1 - suf_RS_tes_sht) * 100, 1),
                "slowheatOk": ref_slowheat_ok_degc,
                "slowheatMin": ref_slowheat_min_degc,
                "setpoint2022": ref_setpoint_2022_degc,
                "degCTwh2019": round(_degc_twh_2019, 2),
                "degCKwhPerPerson2019": round(_degc_twh_2019 * 1e9 / _pop[_Y0], 0),
                "heatTwh": round(_tes[_Y0]["space heating"], 1),
                "heatSharePct": _pct(_tes[_Y0]["space heating"], _tes_tot[_Y0])},
         spoilers=["demandCutPct"],
         notebook=_NB + "#section_2", reference="nW-BE §2.1.2")

    _add("hot-water", _T, "Domestic hot water",
         "kWh/person/year of home hot water",
         ref_RS_tes_shw, trg_RS_tes_shw, ref_year=_Y0, target_year=_Y1,
         slider={"min": 300, "max": 900, "step": 10},
         impact=_impact("proportional", trg_RS_tes_shw,
                        scaled=_tes[_Y1]["sanitary hot water"]),
         model={"var": "trg_RS_tes_shw", "section": "2.1.2"},
         history="res_hot_water_per_person",
         facts={"changePct": round(pro_RS_tes_shw * 100, 1),
                "litres2019": round(_litres_40c[_Y0], 1),
                "litresTarget": round(_litres_40c[_Y1], 1),
                "kwhPer100Litres": round(_kwh_per_100l_40c, 2),
                "showerMinutes": shower_duration,
                "showerFlow": shower_flow_rate,
                "showerTemp": shower_temperature,
                "showerKwh": round(_shower_kwh, 2),
                "othersLitres": others_volume,
                "othersTemp": others_temperature,
                "othersKwh": round(_others_kwh, 2),
                "hotWaterTwh": round(_tes[_Y0]["sanitary hot water"], 1),
                "hotWaterSharePct": _pct(_tes[_Y0]["sanitary hot water"], _tes_tot[_Y0]),
                "thermalTwh": round(_tes_tot[_Y0], 1)},
         spoilers=["changePct", "litresTarget", "showerMinutes", "showerFlow",
                   "showerTemp", "showerKwh", "othersLitres", "othersTemp",
                   "othersKwh"],
         notebook=_NB + "#section_2", reference="nW-BE §2.1.2")

    _add("cooling", _T, "Home cooling", "kWh/m²/year of home cooling",
         ref_RS_tes_scl, trg_RS_tes_scl, ref_year=_Y0, target_year=_Y1,
         slider={"min": 0, "max": 6, "step": 0.05},
         impact=_impact("proportional", trg_RS_tes_scl,
                        scaled=_tes[_Y1]["space cooling"]),
         model={"var": "acc_RS_tes_scl_ren", "section": "2.1.2",
                "note": "a multiplier on the observed +0.035 kWh/m²/year deployment "
                        "rate; see the report on how that rate compares with the "
                        "observed series"},
         history="res_cooling_per_m2",
         facts={"accTarget": acc_RS_tes_scl_ren,
                "rateAssumed": cur_RS_tes_scl_ren,
                "growthPct": round(100 * (trg_RS_tes_scl / ref_RS_tes_scl - 1), 0),
                "coolingTwh": round(_tes[_Y0]["space cooling"], 2),
                "coolingSharePct": _pct(_tes[_Y0]["space cooling"], _tes_tot[_Y0]),
                "coolingTwhTarget": round(_tes[_Y1]["space cooling"], 2),
                "heatTwh": round(_tes[_Y0]["space heating"], 1),
                "thermalTwh": round(_tes_tot[_Y0], 1)},
         spoilers=["accTarget", "rateAssumed", "growthPct", "coolingTwhTarget"],
         notebook=_NB + "#section_2", reference="nW-BE §2.1.2")

    _add("cooking", _T, "Home cooking", "kWh/household/year of home cooking",
         ref_RS_tes_cok, trg_RS_tes_cok, ref_year=_Y0, target_year=_Y1,
         slider={"min": 150, "max": 420, "step": 10},
         impact=_impact("proportional", trg_RS_tes_cok, scaled=_tes[_Y1]["cooking"]),
         model={"var": "pro_RS_tes_cok", "section": "2.1.2"},
         facts={"changePct": round(pro_RS_tes_cok * 100, 1),
                "trendHist": ref_RS_cook_trend,
                "perDay2019": round(ref_RS_tes_cok / 365.0, 2),
                "gasShare2019": round(ref_RS_tes_cok_gas * 100, 1),
                "gasShareTarget": round(trg_RS_tes_cok_gas * 100, 1),
                "cookingTwh": round(_tes[_Y0]["cooking"], 2),
                "cookingSharePct": _pct(_tes[_Y0]["cooking"], _tes_tot[_Y0]),
                "householdSize": round(_hld_size[_Y0], 2),
                "thermalTwh": round(_tes_tot[_Y0], 1)},
         spoilers=["changePct", "gasShareTarget"],
         notebook=_NB + "#section_2", reference="nW-BE §2.1.2")

    _add("district-heat", _T, "District heating share of home heat", "% of home heat",
         ref_RS_tes_dhn * 100, trg_RS_tes_dhn * 100, ref_year=_Y0, target_year=_Y1,
         slider={"min": 0, "max": 45, "step": 0.5},
         # The carrier split leaves the end-use demand untouched — share_heat_dhn
         # and share_heat_ihs add to 1 in cell 57 — so this lever has, by
         # construction, no leverage on the demand this topic measures. The gain
         # sits upstream, in how the heat is produced, which the demand model does
         # not represent. That is the point of the lever, not a defect in it.
         impact=_impact("negligible", trg_RS_tes_dhn * 100),
         model={"var": "trg_RS_tes_dhn", "section": "2.1.3",
                "note": "splits space heating + hot water between a network and "
                        "individual boilers; the end-use total is unchanged"},
         facts={"potentialPct": ref_dhn_potential_pct,
                "paths2050Pct": ref_dhn_paths2050_pct,
                "networkableTwh": round(_networkable_twh[_Y0], 1),
                "networkableSharePct": _pct(_networkable_twh[_Y0], _tes_tot[_Y0]),
                "networkableTwhTarget": round(_networkable_twh[_Y1], 1),
                "servedTwh2019": round(_networkable_twh[_Y0] * ref_RS_tes_dhn, 2),
                "thermalTwh": round(_tes_tot[_Y0], 1)},
         spoilers=["networkableTwhTarget"],
         notebook=_NB + "#section_2", reference="nW-BE §2.1.3")

    # --- Shared model quantities (context + the leverage arithmetic) -------------
    _ws_model = {
        "scope": "residential thermal demand: space heating, space cooling, sanitary "
                 "hot water and cooking (household appliances excluded)",
        "refYear": _Y0, "targetYear": _Y1,
        "population": {str(y): round(_pop[y]) for y in (_Y0, _Y1)},
        "households": {str(y): round(_hld[y]) for y in (_Y0, _Y1)},
        "floorAreaMm2": {str(y): round(_sur_tot[y], 1) for y in (_Y0, _Y1)},
        "thermalTwh": {str(y): round(_tes_tot[y], 3) for y in (_Y0, _Y1)},
        "applianceTwh": {str(y): round(_ees_tot[y], 3) for y in (_Y0, _Y1)},
        "residentialTwh": {str(y): round(_res_tot[y], 3) for y in (_Y0, _Y1)},
        "serviceTwh": {str(y): {c: round(_tes[y][c], 3) for c in _TES}
                       for y in (_Y0, _Y1)},
        "areaDrivenTwh": {str(y): round(_area_driven_twh[y], 3) for y in (_Y0, _Y1)},
        "networkableTwh": {str(y): round(_networkable_twh[y], 3) for y in (_Y0, _Y1)},
        "intensityKwhM2": {str(_Y0): round(ref_RS_tes_sht, 3),
                           str(_Y1): round(trg_RS_tes_sht * suf_RS_tes_sht, 3)},
        "districtHeatShare": {str(_Y0): round(ref_RS_tes_dhn * 100, 3),
                              str(_Y1): round(trg_RS_tes_dhn * 100, 3)},
    }

    return {"levers": _L, "model": _ws_model}
