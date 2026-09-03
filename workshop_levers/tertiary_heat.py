"""Tertiary heat: heating, cooling, hot water and catering in Belgian service buildings.

Offices, shops, schools, hospitals, hotels, restaurants and the rest of the
service sector. Seven levers, all of them genuine degrees of freedom of the
buildings notebook's tertiary block, so the reveal is exact and no group can
enter a self-inconsistent scenario:

    ter-floor-area     m² of service building per person  (pro_TS_sur_spe)
    ter-insulation     kWh/m² of heating need per year    (acc_TS_tes_sht_ren)
    ter-thermostat     °C off the heating setpoint        (suf_TS_tes_sht)
    ter-cooling        kWh/m²/year of cooling             (acc_TS_tes_scl_ren)
    ter-hot-water      kWh/person/year                    (pro_TS_tes_shw)
    ter-catering       kWh/person/year                    (pro_TS_tes_cat)
    ter-district-heat  % of service-building heat          (trg_TS_tes_dhn)

The residential sector is a separate topic (`residential_heat.py`) out of the
same notebook, so nothing here reads a `*_RS_*` variable. Lever ids must be
unique within a sector, hence the `ter-` prefix.

This sector's story is the divergence of its two space-conditioning services:
between 2000 and 2019 cooling multiplied by about ten (1,8 → 17,2 kWh/m²) while
heating stayed flat and then fell. Groups reliably expect the opposite.

Everything is read from quantities the notebook has already computed — this
module adds no assumptions of its own. It also cross-checks the model against
the figures written in the notebook's prose, so that editing either the text or
the code without the other fails loudly here rather than feeding a wrong number
into a workshop.

Three traps in the notebook's namespace, all handled below:

  * `d_temp`, `d_cons_temp`, `share_heat_dhn` and `share_heat_ihs` are assigned
    twice, once per sector. By the time this module runs they happen to hold the
    *tertiary* values, but relying on assignment order is fragile: the setpoint
    drop is recovered from `suf_TS_tes_sht` and the carrier split from
    `ref_TS_tes_dhn` / `trg_TS_tes_dhn`, exactly as the residential module does.
  * `linear_growth()` rounds every series it returns to three decimals, so a
    quantity read back out of `df_SUF` or `df_tes_TS_tot` carries up to 5e-4 of
    rounding in its own unit. The tolerances below are that rounding carried
    through the arithmetic.
  * the notebook quotes a 1,7 %/year "renewal rate" of tertiary floor area in
    §3.1.1. As in the residential sector that is renovation at any depth plus new
    construction, and its renovation component is a constant of the JRC-IDEES
    building-stock model — so it is not an observation of renovation activity and
    is not comparable with the ~1 %/year *energy* renovation rate of European
    policy. It is used on the fact cards only; the `ter-insulation` lever is
    defined on the kWh/m² the model actually moves.

See docs/workshop_module.md for the design, and
website/workshop/content/tertiary-heat.yaml for the wording.
"""
from nW_BE_demand_model_sub_functions import make_lever

from . import need

TOPIC = "tertiary-heat"
SECTOR = "buildings"
ORDER = 40

NOTEBOOK = "../notebooks/nW_BE_demand_model_buildings.html"


def build(ctx):
    (years, population_dict, df_SUF,
     df_tes_TS_tot, df_ees_TS_tot,
     ref_TS_sur_tot, ref_TS_sur_spe, pro_TS_sur_spe,
     ref_TS_tes_sht, trg_TS_tes_sht, suf_TS_tes_sht,
     acc_TS_tes_sht_ren, cur_TS_tes_sht_ren,
     ref_TS_tes_scl, trg_TS_tes_scl, acc_TS_tes_scl_ren, cur_TS_tes_scl_ren,
     ref_TS_tes_shw, trg_TS_tes_shw, pro_TS_tes_shw,
     ref_TS_tes_cat, trg_TS_tes_cat, pro_TS_tes_cat,
     ref_TS_tes_cat_gas, trg_TS_tes_cat_gas,
     ref_TS_tes_cat_bio, trg_TS_tes_cat_bio,
     ref_TS_tes_dhn, trg_TS_tes_dhn,
     cp_h2o, rho_h2o) = need(
        ctx, 'years',
        'population_dict',
        'df_SUF',
        'df_tes_TS_tot',
        'df_ees_TS_tot',
        'ref_TS_sur_tot',
        'ref_TS_sur_spe',
        'pro_TS_sur_spe',
        'ref_TS_tes_sht',
        'trg_TS_tes_sht',
        'suf_TS_tes_sht',
        'acc_TS_tes_sht_ren',
        'cur_TS_tes_sht_ren',
        'ref_TS_tes_scl',
        'trg_TS_tes_scl',
        'acc_TS_tes_scl_ren',
        'cur_TS_tes_scl_ren',
        'ref_TS_tes_shw',
        'trg_TS_tes_shw',
        'pro_TS_tes_shw',
        'ref_TS_tes_cat',
        'trg_TS_tes_cat',
        'pro_TS_tes_cat',
        'ref_TS_tes_cat_gas',
        'trg_TS_tes_cat_gas',
        'ref_TS_tes_cat_bio',
        'trg_TS_tes_cat_bio',
        'ref_TS_tes_dhn',
        'trg_TS_tes_dhn',
        'cp_h2o',
        'rho_h2o')

    _NB = NOTEBOOK
    _Y0, _Y1 = years[0], years[-1]
    _pop = {y: float(population_dict[y]) for y in (_Y0, _Y1)}

    # --- Reference values quoted only in this notebook's prose -------------------
    # Named here so the workshop can export them, and so that editing either the
    # text or the model trips the assertions below instead of drifting silently.
    # Section numbers refer to the markdown headings of the buildings notebook.
    ref_pct_per_degc        = 7.0    # % of heat demand saved per -1 °C [3]     -- §2.1.2
    ref_setpoint_drop_degc  = 1.0    # the tertiary setpoint reduction for 2050  -- §3.1.2
    ref_TS_renewal_rate     = 1.7    # % of floor area renewed per year          -- §3.1.1
    ref_TS_reno_rate_jrc    = 0.51   # of which renovation, a JRC-IDEES constant -- §3.1.1
    ref_TS_new_build_rate   = 1.22   # of which net new construction, observed   -- §3.1.1
    ref_TS_sht_2021         = 102.6  # heat intensity in 2021, kWh/m² [1,2]      -- §3.1.2
    ref_TS_sht_2023         = 78.6   # heat intensity in 2023, kWh/m² [1,2]      -- §3.1.2
    ref_TS_scl_growth_x     = 9.7    # cooling multiplier, 2000 to 2019 [1,2]    -- §3.1.2
    ref_TS_scl_2000         = 1.8    # cooling in 2000, kWh/m² [1,2]             -- §3.1.2
    ref_TS_scl_2023         = 16.1   # cooling in 2023, kWh/m² [1,2]             -- §3.1.2
    ref_TS_cat_trend        = 4.297  # catering, kWh/person/year, 2000-2023      -- §3.1.2
    ref_RS_cok_trend        = -0.523 # home cooking, kWh/person/year, 2000-2023  -- §3.1.2
    ref_TS_shw_2023         = 323.9  # tertiary hot water in 2023, kWh/person    -- §3.1.2
    ref_dhn_potential_pct   = 45.0   # techno-economic potential for 2050 [4]    -- §2.1.3
    ref_dhn_paths2050_pct   = 13.0   # EnergyVille PATHS2050, buildings [5]      -- §2.1.3

    # --- Scope ------------------------------------------------------------------
    # The four thermal services of the tertiary sector: everything the seven levers
    # below act on. Ventilation, lighting, commercial refrigeration and ICT are
    # tertiary too but are specific electrical uses, not part of this topic, so
    # they stay out of the total against which the leverage readout is calibrated.
    _TES = ["space heating", "space cooling", "sanitary hot water", "catering"]
    _EES = list(df_ees_TS_tot.columns)

    def _twh(df, column, year):
        return float(df.loc[year, column])

    _tes = {y: {c: _twh(df_tes_TS_tot, c, y) for c in _TES} for y in (_Y0, _Y1)}
    _tes_tot = {y: sum(_tes[y].values()) for y in (_Y0, _Y1)}
    _ees_tot = {y: sum(_twh(df_ees_TS_tot, c, y) for c in _EES) for y in (_Y0, _Y1)}
    _ter_tot = {y: _tes_tot[y] + _ees_tot[y] for y in (_Y0, _Y1)}
    _TOT = _tes_tot[_Y1]

    _sur_spe = {y: float(df_SUF["TS specific surface [m²/person]"][y]) for y in (_Y0, _Y1)}
    _sur_tot = {y: float(df_SUF["TS total surface [Mm²]"][y]) for y in (_Y0, _Y1)}   # Mm²

    # --- Consistency checks: the model vs. the figures written in the prose -----
    assert abs(_sur_spe[_Y0] - ref_TS_sur_spe) < 1e-3, (
        f"df_SUF's specific tertiary surface starts at {_sur_spe[_Y0]:.4f} m2/person "
        f"but ref_TS_sur_spe is {ref_TS_sur_spe:.4f} -- section 1.2.2")
    assert abs(_sur_tot[_Y0] - ref_TS_sur_tot * 1e-6) < 1e-2, (
        f"df_SUF's total tertiary surface starts at {_sur_tot[_Y0]:.3f} Mm2 but "
        f"ref_TS_sur_tot is {ref_TS_sur_tot * 1e-6:.3f} Mm2 -- section 1.2.2")

    # -1 °C at -7% per degree = the 0.93 factor of section 3.1.2. `d_temp` itself
    # is shared with the residential block and must not be read here.
    _setpoint_drop = (1.0 - suf_TS_tes_sht) / (ref_pct_per_degc / 100.0)
    assert abs(_setpoint_drop - ref_setpoint_drop_degc) < 1e-9, (
        f"the tertiary setpoint reduction implied by suf_TS_tes_sht is now "
        f"{_setpoint_drop:.3f} °C at {ref_pct_per_degc} %/°C, but section 3.1.2 "
        f"quotes {ref_setpoint_drop_degc} °C -- update one or the other")

    # Space heating is the only service that carries both the renovation and the
    # setpoint assumption, so its 2050 value must be reproducible from the two.
    _sht_twh_eff_only = trg_TS_tes_sht * _sur_tot[_Y1] * 1e-3
    assert abs(_sht_twh_eff_only * suf_TS_tes_sht
               - _tes[_Y1]["space heating"]) < 1e-3, (
        f"space heating in 2050 is {_tes[_Y1]['space heating']:.4f} TWh but "
        f"trg_TS_tes_sht x suf_TS_tes_sht x floor area gives "
        f"{_sht_twh_eff_only * suf_TS_tes_sht:.4f} TWh; the levers below would "
        f"misreport their leverage")
    assert abs(trg_TS_tes_sht - (ref_TS_tes_sht
                                 + acc_TS_tes_sht_ren * cur_TS_tes_sht_ren
                                 * (_Y1 - _Y0))) < 1e-9, (
        f"trg_TS_tes_sht is {trg_TS_tes_sht:.4f} kWh/m2, no longer the linear "
        f"improvement of section 3.1.1 from {ref_TS_tes_sht:.4f}")
    assert abs(trg_TS_tes_scl - (ref_TS_tes_scl
                                 + acc_TS_tes_scl_ren * cur_TS_tes_scl_ren
                                 * (_Y1 - _Y0))) < 1e-9, (
        f"trg_TS_tes_scl is {trg_TS_tes_scl:.4f} kWh/m2, no longer the linear "
        f"deployment of section 3.1.2 from {ref_TS_tes_scl:.4f}")
    # Hot water is *contained* at its 2019 level, which is the whole point of that
    # lever: the observed series has risen for twenty-three years.
    assert abs(pro_TS_tes_shw) < 1e-12 and abs(trg_TS_tes_shw - ref_TS_tes_shw) < 1e-9, (
        f"section 3.1.2 contains tertiary hot water at its 2019 level, but "
        f"pro_TS_tes_shw is now {pro_TS_tes_shw} -- the ter-hot-water lever's "
        f"wording assumes containment")
    # The renewal rate quoted in the prose is the sum of its two components.
    assert abs(ref_TS_reno_rate_jrc + ref_TS_new_build_rate
               - ref_TS_renewal_rate) < 0.05, (
        f"the renovation ({ref_TS_reno_rate_jrc}) and new-build "
        f"({ref_TS_new_build_rate}) components no longer add up to the renewal "
        f"rate of {ref_TS_renewal_rate} % quoted in section 3.1.1")

    # --- Derived lever quantities ----------------------------------------------
    # Space heating and cooling both scale with the floor area; hot water and
    # catering scale with population, so neither moves with m²/person.
    _area_driven_twh = {y: _tes[y]["space heating"] + _tes[y]["space cooling"]
                        for y in (_Y0, _Y1)}
    # Heat a network could carry: space heating + sanitary hot water (cell 75).
    _networkable_twh = {y: _tes[y]["space heating"] + _tes[y]["sanitary hot water"]
                        for y in (_Y0, _Y1)}
    # TWh per degree off the thermostat. Exact for a single lever moved alone: the
    # 2050 intensity is affine in the setpoint drop, and the floor area does not
    # depend on it.
    _setpoint_slope = -(ref_pct_per_degc / 100.0) * _sht_twh_eff_only
    # What one degree is worth on today's stock, for the tangible card.
    _degc_twh_2019 = (ref_pct_per_degc / 100.0) * _tes[_Y0]["space heating"]
    # Useful heat the average m² needed in 2019, per service building "unit" of
    # 1 000 m² — a school, a supermarket, a small office block.
    _heat_kwh_per_1000m2 = ref_TS_tes_sht * 1000.0
    # Equivalent litres of 40 °C water per person per day, the same yardstick the
    # residential topic uses, so the two are directly comparable.
    _litres_40c = ref_TS_tes_shw / (rho_h2o * cp_h2o * (40.0 - 15.0)) / 365.0
    # Food preparation, both sides of the scenario, for the catering caution card.
    _cat_change_twh = _tes[_Y1]["catering"] - _tes[_Y0]["catering"]

    def _impact(kind, v_target, scaled=0.0, slope=None):
        """Leverage record read by website/assets/js/workshop/impact.js.

        TWh(vTarget) always equals `total`, the négaWatt 2050 demand for the four
        tertiary thermal services, so every lever's readout is on the same
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

    _T = "tertiary-heat"

    _add("ter-floor-area", _T, "Service floor area per person",
         "m² of service building per person",
         _sur_spe[_Y0], _sur_spe[_Y1], ref_year=_Y0, target_year=_Y1,
         slider={"min": 12, "max": 30, "step": 0.25},
         impact=_impact("proportional", _sur_spe[_Y1], scaled=_area_driven_twh[_Y1]),
         model={"var": "pro_TS_sur_spe", "section": "1.2.2",
                "note": "heating and cooling scale with the floor area; hot water and "
                        "catering scale with population, so they do not move with "
                        "this lever"},
         history="ter_m2_per_person",
         facts={"changePct": round(pro_TS_sur_spe * 100, 1),
                "stockMm2": round(_sur_tot[_Y0], 0),
                "stockMm2Target": round(_sur_tot[_Y1], 0),
                "areaDrivenTwh": round(_area_driven_twh[_Y0], 1),
                "areaDrivenSharePct": _pct(_area_driven_twh[_Y0], _tes_tot[_Y0]),
                "thermalTwh": round(_tes_tot[_Y0], 1),
                "electricalTwh": round(_ees_tot[_Y0], 1)},
         spoilers=["changePct", "stockMm2Target"],
         notebook=_NB + "#section_3", reference="nW-BE §1.2.2")

    # Same reasoning as residential-heat's `insulation` lever: the model's degree
    # of freedom is acc_TS_tes_sht_ren, a multiplier on the observed
    # -0.154 kWh/m²/year improvement, and this lever is the 2050 intensity it
    # produces. Efficiency only, before the suf_TS_tes_sht thermostat multiplier,
    # so it sits on the same basis as the observed kWh/m² series.
    _add("ter-insulation", _T, "Heating need of the average service building",
         "kWh/m²/year of service heating need",
         ref_TS_tes_sht, trg_TS_tes_sht, ref_year=_Y0, target_year=_Y1,
         slider={"min": 45, "max": 110, "step": 0.5},
         impact=_impact("proportional", trg_TS_tes_sht,
                        scaled=_tes[_Y1]["space heating"]),
         model={"var": "acc_TS_tes_sht_ren", "section": "3.1.1",
                "prose": "the model input is the multiplier on the observed "
                         "-0.154 kWh/m²/year improvement quoted in section 3.1.1; "
                         "this lever is the 2050 intensity it produces",
                "note": "efficiency only, before the suf_TS_tes_sht thermostat "
                        "multiplier"},
         history="ter_heat_per_m2",
         facts={"accTarget": acc_TS_tes_sht_ren,
                "improvementHist": cur_TS_tes_sht_ren,
                "improvementTarget": round(acc_TS_tes_sht_ren * cur_TS_tes_sht_ren, 3),
                "intensityEffTarget": round(trg_TS_tes_sht, 1),
                "intensityTarget": round(trg_TS_tes_sht * suf_TS_tes_sht, 1),
                "renewalRateJrc": ref_TS_renewal_rate,
                "renoRateJrc": ref_TS_reno_rate_jrc,
                "newBuildRateJrc": ref_TS_new_build_rate,
                "sht2021": ref_TS_sht_2021, "sht2023": ref_TS_sht_2023,
                "crisisDropPerYear": round((ref_TS_sht_2021 - ref_TS_sht_2023) / 2.0, 1),
                "heatKwhPer1000m2": round(_heat_kwh_per_1000m2),
                "heatTwh": round(_tes[_Y0]["space heating"], 1),
                "heatSharePct": _pct(_tes[_Y0]["space heating"], _tes_tot[_Y0]),
                "thermalTwh": round(_tes_tot[_Y0], 1)},
         spoilers=["accTarget", "improvementTarget", "intensityEffTarget",
                   "intensityTarget"],
         notebook=_NB + "#section_3", reference="nW-BE §3.1.1")

    _add("ter-thermostat", _T, "Degrees off the service-building setpoint",
         "°C less in service buildings",
         0.0, _setpoint_drop, ref_year=_Y0, target_year=_Y1,
         slider={"min": -1, "max": 5, "step": 0.5},
         impact=_impact("linear-shift", _setpoint_drop, slope=_setpoint_slope),
         model={"var": "suf_TS_tes_sht", "section": "3.1.2",
                "note": "d_temp is shared with the residential block, so the "
                        "tertiary drop is recovered from suf_TS_tes_sht"},
         facts={"pctPerDegC": ref_pct_per_degc,
                "demandCutPct": round((1 - suf_TS_tes_sht) * 100, 1),
                "degCTwh2019": round(_degc_twh_2019, 2),
                "heatTwh": round(_tes[_Y0]["space heating"], 1),
                "heatSharePct": _pct(_tes[_Y0]["space heating"], _tes_tot[_Y0]),
                "thermalTwh": round(_tes_tot[_Y0], 1)},
         spoilers=["demandCutPct"],
         notebook=_NB + "#section_3", reference="nW-BE §3.1.2")

    _add("ter-cooling", _T, "Cooling of service buildings",
         "kWh/m²/year of service cooling",
         ref_TS_tes_scl, trg_TS_tes_scl, ref_year=_Y0, target_year=_Y1,
         slider={"min": 8, "max": 48, "step": 0.5},
         impact=_impact("proportional", trg_TS_tes_scl,
                        scaled=_tes[_Y1]["space cooling"]),
         model={"var": "acc_TS_tes_scl_ren", "section": "3.1.2",
                "note": "a multiplier on the observed +0.743 kWh/m²/year deployment "
                        "rate; the scenario keeps one third of it"},
         history="ter_cooling_per_m2",
         facts={"accTarget": round(acc_TS_tes_scl_ren, 3),
                "rateHist": cur_TS_tes_scl_ren,
                "rateAssumed": round(acc_TS_tes_scl_ren * cur_TS_tes_scl_ren, 3),
                "growthPct": round(100 * (trg_TS_tes_scl / ref_TS_tes_scl - 1), 0),
                "growthX2019": ref_TS_scl_growth_x,
                "scl2000": ref_TS_scl_2000, "scl2023": ref_TS_scl_2023,
                "coolingTwh": round(_tes[_Y0]["space cooling"], 2),
                "coolingSharePct": _pct(_tes[_Y0]["space cooling"], _tes_tot[_Y0]),
                "coolingTwhTarget": round(_tes[_Y1]["space cooling"], 2),
                "heatTwh": round(_tes[_Y0]["space heating"], 1),
                "thermalTwh": round(_tes_tot[_Y0], 1)},
         spoilers=["accTarget", "rateAssumed", "growthPct", "coolingTwhTarget"],
         notebook=_NB + "#section_3", reference="nW-BE §3.1.2")

    # Contained at the 2019 level, against a series that has risen for 23 years,
    # so *down* is the ambitious direction even though the target equals the
    # reference. Left to its default, `better` would read "up".
    _add("ter-hot-water", _T, "Hot water in service buildings",
         "kWh/person/year of service hot water",
         ref_TS_tes_shw, trg_TS_tes_shw, ref_year=_Y0, target_year=_Y1,
         slider={"min": 200, "max": 460, "step": 5}, better="down",
         impact=_impact("proportional", trg_TS_tes_shw,
                        scaled=_tes[_Y1]["sanitary hot water"]),
         model={"var": "pro_TS_tes_shw", "section": "3.1.2"},
         history="ter_hot_water_per_person",
         facts={"changePct": round(pro_TS_tes_shw * 100, 1),
                "shw2023": ref_TS_shw_2023,
                "litres2019": round(_litres_40c, 1),
                "gap2023Pct": round(100 * (ref_TS_shw_2023 / ref_TS_tes_shw - 1), 1),
                "hotWaterTwh": round(_tes[_Y0]["sanitary hot water"], 2),
                "hotWaterSharePct": _pct(_tes[_Y0]["sanitary hot water"], _tes_tot[_Y0]),
                "hotWaterTwhTarget": round(_tes[_Y1]["sanitary hot water"], 2),
                "thermalTwh": round(_tes_tot[_Y0], 1)},
         # gap2023Pct is the distance from the observed 2023 level to the target,
         # so printing it before a group answers hands over the answer.
         spoilers=["changePct", "gap2023Pct", "hotWaterTwhTarget"],
         notebook=_NB + "#section_3", reference="nW-BE §3.1.2")

    _add("ter-catering", _T, "Catering", "kWh/person/year of catering",
         ref_TS_tes_cat, trg_TS_tes_cat, ref_year=_Y0, target_year=_Y1,
         slider={"min": 150, "max": 450, "step": 5},
         impact=_impact("proportional", trg_TS_tes_cat, scaled=_tes[_Y1]["catering"]),
         model={"var": "pro_TS_tes_cat", "section": "3.1.2"},
         facts={"changePct": round(pro_TS_tes_cat * 100, 1),
                "trendHist": ref_TS_cat_trend,
                "homeCookingTrend": ref_RS_cok_trend,
                "gasShare2019": round(ref_TS_tes_cat_gas * 100, 1),
                "gasShareTarget": round(trg_TS_tes_cat_gas * 100, 1),
                "bioShare2019": round(ref_TS_tes_cat_bio * 100, 1),
                "cateringTwh": round(_tes[_Y0]["catering"], 2),
                "cateringSharePct": _pct(_tes[_Y0]["catering"], _tes_tot[_Y0]),
                "cateringTwhTarget": round(_tes[_Y1]["catering"], 2),
                "cateringChangeTwh": round(_cat_change_twh, 2),
                "thermalTwh": round(_tes_tot[_Y0], 1)},
         spoilers=["changePct", "gasShareTarget", "cateringTwhTarget",
                   "cateringChangeTwh"],
         notebook=_NB + "#section_3", reference="nW-BE §3.1.2")

    _add("ter-district-heat", _T, "District heating share of service-building heat",
         "% of service-building heat",
         ref_TS_tes_dhn * 100, trg_TS_tes_dhn * 100, ref_year=_Y0, target_year=_Y1,
         slider={"min": 0, "max": 45, "step": 0.5},
         # The carrier split leaves the end-use demand untouched — share_heat_dhn
         # and share_heat_ihs add to 1 in cell 75 — so this lever has, by
         # construction, no leverage on the demand this topic measures. The gain
         # sits upstream, in how the heat is produced, which the demand model does
         # not represent. That is the point of the lever, not a defect in it.
         impact=_impact("negligible", trg_TS_tes_dhn * 100),
         model={"var": "trg_TS_tes_dhn", "section": "3.1.3",
                "note": "splits space heating + hot water between a network and "
                        "individual boilers; the end-use total is unchanged"},
         facts={"potentialPct": ref_dhn_potential_pct,
                "paths2050Pct": ref_dhn_paths2050_pct,
                "networkableTwh": round(_networkable_twh[_Y0], 1),
                "networkableSharePct": _pct(_networkable_twh[_Y0], _tes_tot[_Y0]),
                "networkableTwhTarget": round(_networkable_twh[_Y1], 1),
                "servedTwh2019": round(_networkable_twh[_Y0] * ref_TS_tes_dhn, 2),
                "thermalTwh": round(_tes_tot[_Y0], 1)},
         spoilers=["networkableTwhTarget"],
         notebook=_NB + "#section_3", reference="nW-BE §3.1.3")

    # --- Shared model quantities (context + the leverage arithmetic) -------------
    _ws_model = {
        "scope": "tertiary thermal demand: space heating, space cooling, sanitary "
                 "hot water and catering (ventilation, lighting, commercial "
                 "refrigeration and ICT excluded)",
        "refYear": _Y0, "targetYear": _Y1,
        "population": {str(y): round(_pop[y]) for y in (_Y0, _Y1)},
        "floorAreaMm2": {str(y): round(_sur_tot[y], 1) for y in (_Y0, _Y1)},
        "floorAreaPerPerson": {str(y): round(_sur_spe[y], 3) for y in (_Y0, _Y1)},
        "thermalTwh": {str(y): round(_tes_tot[y], 3) for y in (_Y0, _Y1)},
        "electricalTwh": {str(y): round(_ees_tot[y], 3) for y in (_Y0, _Y1)},
        "tertiaryTwh": {str(y): round(_ter_tot[y], 3) for y in (_Y0, _Y1)},
        "serviceTwh": {str(y): {c: round(_tes[y][c], 3) for c in _TES}
                       for y in (_Y0, _Y1)},
        "areaDrivenTwh": {str(y): round(_area_driven_twh[y], 3) for y in (_Y0, _Y1)},
        "networkableTwh": {str(y): round(_networkable_twh[y], 3) for y in (_Y0, _Y1)},
        "intensityKwhM2": {str(_Y0): round(ref_TS_tes_sht, 3),
                           str(_Y1): round(trg_TS_tes_sht * suf_TS_tes_sht, 3)},
        "coolingKwhM2": {str(_Y0): round(ref_TS_tes_scl, 3),
                         str(_Y1): round(trg_TS_tes_scl, 3)},
        "districtHeatShare": {str(_Y0): round(ref_TS_tes_dhn * 100, 3),
                              str(_Y1): round(trg_TS_tes_dhn * 100, 3)},
    }

    return {"levers": _L, "model": _ws_model}
