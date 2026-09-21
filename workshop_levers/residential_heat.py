"""Residential heat: heating, cooling, hot water and cooking in Belgian homes.

Eight levers over seven degrees of freedom of the buildings notebook's residential
block, so the reveal is exact and no group can enter a self-inconsistent scenario.
The one pair sharing a degree of freedom is `renovation-rate` x `renovation-depth`:
section 2.1.1 pins their product, never either one alone (D51).

    floor-area        m² of dwelling per person       (pro_RS_sur_spe)
    renovation-rate   % of the stock renovated a year (rat_RS_tes_sht_ren)
    renovation-depth  % of the heating need one cuts  (dep_RS_tes_sht_ren)
    thermostat        °C off the heating setpoint     (suf_RS_tes_sht)
    hot-water         kWh/person/year                 (trg_RS_tes_shw)
    cooling           kWh/m²/year                     (acc_RS_tes_scl_ren)
    cooking           kWh/household/year              (pro_RS_tes_cok)
    district-heat     % of home heat from a network   (trg_RS_tes_dhn)

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
  * "renovation rate" means three different things in the same paragraph. The
    3 %/year of section 2.1.1 is a *renewal* rate -- renovation at any depth plus
    new construction; its renovation component (2.261 %/year) is a constant booked
    by JRC-IDEES rather than an observation; and the ~1 %/year the Commission
    measures counts *energy* renovation only. Section 2.1.1 now decomposes the
    first two in code, and the two levers below are defined on the JRC renovation
    component together with the depth that goes with it, so that their product
    reproduces the observed -0.458 kWh/m²/year exactly.

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
     dep_RS_tes_sht_ren, shr_RS_tes_sht_ren, rat_RS_tes_sht_ren,
     obs_RS_tes_sht_2000, obs_RS_tes_sht_dep,
     obs_RS_sht_rat_ren, obs_RS_sht_rat_new, obs_RS_sht_rat_all,
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
        'dep_RS_tes_sht_ren',
        'shr_RS_tes_sht_ren',
        'rat_RS_tes_sht_ren',
        'obs_RS_tes_sht_2000',
        'obs_RS_tes_sht_dep',
        'obs_RS_sht_rat_ren',
        'obs_RS_sht_rat_new',
        'obs_RS_sht_rat_all',
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
    # The renewal / renovation / new-build decomposition now lives in the notebook
    # (section 2.1.1) together with the renovation reading of the efficiency
    # assumption, because the two renovation levers are defined on it. Read from
    # there rather than restated here: obs_RS_sht_rat_all / _ren / _new.
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
    # `linear_growth()` rounds every series it returns to three decimals, so a
    # quantity read back out of df_SUF or df_tes_RS_tot carries up to 5e-4 of
    # rounding in its own unit. The tolerances below are that rounding, carried
    # through the arithmetic -- not a licence for the model to drift.
    assert abs(_sur_spe[_Y0] - ref_RS_sur_spe) < 1e-3, (
        f"df_SUF's specific residential surface starts at {_sur_spe[_Y0]:.4f} m2/person "
        f"but ref_RS_sur_spe is {ref_RS_sur_spe:.4f} -- section 1.2.1")
    # the household figure is the same series divided by the household size, so
    # the 5e-4 above is magnified by ~2.3 persons per household
    assert abs(_sur_hld[_Y0] - ref_RS_sur_hld) < 5e-3, (
        f"df_SUF's household surface starts at {_sur_hld[_Y0]:.4f} m2/household "
        f"but ref_RS_sur_hld is {ref_RS_sur_hld:.4f} -- section 1.2.1")

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
    assert abs(_sht_twh_eff_only * suf_RS_tes_sht
               - _tes[_Y1]["space heating"]) < 1e-3, (
        f"space heating in 2050 is {_tes[_Y1]['space heating']:.4f} TWh but "
        f"trg_RS_tes_sht x suf_RS_tes_sht x floor area gives "
        f"{_sht_twh_eff_only * suf_RS_tes_sht:.4f} TWh; the levers below would "
        f"misreport their leverage")
    assert abs(trg_RS_tes_sht - (ref_RS_tes_sht
                                 + acc_RS_tes_sht_ren * cur_RS_tes_sht_ren
                                 * (_Y1 - _Y0))) < 1e-9, (
        f"trg_RS_tes_sht is {trg_RS_tes_sht:.4f} kWh/m2, no longer the linear "
        f"improvement of section 2.1.1 from {ref_RS_tes_sht:.4f}")
    assert abs(trg_RS_tes_scl - (ref_RS_tes_scl
                                 + acc_RS_tes_scl_ren * cur_RS_tes_scl_ren
                                 * (_Y1 - _Y0))) < 1e-9, (
        f"trg_RS_tes_scl is {trg_RS_tes_scl:.4f} kWh/m2, no longer the linear "
        f"deployment of section 2.1.2 from {ref_RS_tes_scl:.4f}")

    # The renewal rate quoted in the prose is the sum of its two components.
    assert abs(obs_RS_sht_rat_ren + obs_RS_sht_rat_new
               - obs_RS_sht_rat_all) < 0.02, (
        f"the renovation ({obs_RS_sht_rat_ren}) and new-build "
        f"({obs_RS_sht_rat_new}) components no longer add up to the renewal "
        f"rate of {obs_RS_sht_rat_all} % quoted in section 2.1.1")

    # The two renovation levers are exact only if their product reproduces the
    # intensity trajectory they are read from, and if the observed pair
    # reproduces the observed improvement. Both are arithmetic, so both must hold
    # to the last digit.
    assert abs(ref_RS_tes_sht * (1 - dep_RS_tes_sht_ren * rat_RS_tes_sht_ren
                                 * (_Y1 - _Y0)) - trg_RS_tes_sht) < 1e-9, (
        f"the renovation rate x depth pair of section 2.1.1 no longer lands on "
        f"trg_RS_tes_sht = {trg_RS_tes_sht:.4f} kWh/m2")
    assert abs(obs_RS_tes_sht_dep * obs_RS_sht_rat_ren / 100 * obs_RS_tes_sht_2000
               + cur_RS_tes_sht_ren) < 1e-9, (
        f"the observed renovation pair no longer reproduces the observed "
        f"{cur_RS_tes_sht_ren} kWh/m2/year improvement")
    assert 0 < shr_RS_tes_sht_ren <= 1, (
        f"section 2.1.1 renovates {shr_RS_tes_sht_ren:.0%} of the stock by 2050")

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

    # TWh per degree off the thermostat. Exact for a single lever moved alone: the
    # 2050 intensity is affine in the setpoint drop, and the floor area does not
    # depend on it.
    _setpoint_slope = -(ref_pct_per_degc / 100.0) * _sht_twh_eff_only
    # Useful heat the average 2019 dwelling needs, for the tangible card.
    _heat_kwh_per_household = ref_RS_tes_sht * _sur_hld[_Y0]
    # What one degree is worth on today's housing stock, for the tangible card.
    _degc_twh_2019 = (ref_pct_per_degc / 100.0) * _tes[_Y0]["space heating"]

    # --- Renovation levers: volumes, in dwellings a year -------------------------
    # The model counts floor area, not dwellings, so a rate in % of the stock is
    # turned into homes a year through the average dwelling size -- which gives
    # 4.9 million dwellings against 5.0 million households, close enough for the
    # order of magnitude a card is quoting and stated as such on the card.
    _dwellings = _sur_tot[_Y0] * 1e6 / _sur_hld[_Y0]
    _homes_per_year = {"obs": obs_RS_sht_rat_ren / 100.0 * _dwellings,
                       "trg": rat_RS_tes_sht_ren * _dwellings}
    # How far the observed trend took the stock over the whole measured period,
    # on the fitted line rather than on two weather-dependent endpoints.
    _intensity_drop_pct = -cur_RS_tes_sht_ren * 23.0 / obs_RS_tes_sht_2000 * 100.0

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

    def _impact_ren(axis, v_target, other, scaled):
        """Leverage record for the two renovation levers (impact.js, kind
        "renovation").

        Renovate a constant share `rate` of the stock every year, each renovation
        cutting a fraction `depth` off that dwelling's heating need, and the stock
        average falls linearly -- the shape section 2.1.1 assumes:

            I(N) = I(0) * [1 - depth * min(rate*N, 1)]

        Moving one of the two on its own, with the other held at négaWatt's value,
        scales space heating by the ratio of that factor to its value at
        négaWatt's pair. `other` carries the companion lever's value in the same
        percent units as the slider, `axis` says which of the two this lever is.
        Exact for a single lever, like every other kind; the cap at 1 is the point
        where every dwelling has been renovated once and more speed buys nothing.
        """
        return {"kind": "renovation", "axis": axis,
                "vTarget": round(float(v_target), 4),
                "other": round(float(other), 4),
                "years": _Y1 - _Y0,
                "total": round(_TOT, 4), "scaled": round(float(scaled), 4)}

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

    # Section 2.1.1 has one degree of freedom, acc_RS_tes_sht_ren, a multiplier on
    # the observed -0.458 kWh/m²/year improvement of the stock average. Two ways to
    # put that to a group, and the first round of this workshop took the wrong one:
    #   * as the 2050 kWh/m² the multiplier produces -- exact, with a real observed
    #     curve, but a stock average in kWh/m² is not a quantity anyone at the table
    #     has a feel for, and it is not the form any renovation policy takes.
    #   * as renovation activity: how many dwellings a year, and how deep. That is
    #     the form the question is asked in everywhere, and it is the same
    #     arithmetic -- a constant share of the stock renovated each year at a
    #     constant depth gives exactly the linear fall in the stock average that
    #     section 2.1.1 assumes, so depth x rate is pinned by the trajectory.
    # The second is taken here, as two levers. Only the *product* of the pair is
    # observed, so the split needs one assumption; it is made in the notebook (the
    # depth, set at the European Commission's threshold for a "deep" renovation)
    # and the rate follows, leaving trg_RS_tes_sht and everything downstream of it
    # numerically unchanged. See docs/workshop_module.md, D51.
    #
    # Neither lever gets an observed curve, and for opposite reasons: the JRC
    # renovation series is flat by construction (a booking constant, 2.261 %/year
    # with a standard deviation of 0.008 points), and nobody measures the depth of
    # the average Belgian renovation at all. The renewal series that *does* move is
    # plotted on a fact card instead, where the caveat can be written next to it.
    _add("renovation-rate", _T, "Energy renovation rate of the dwelling stock",
         "% of homes renovated per year",
         obs_RS_sht_rat_ren, rat_RS_tes_sht_ren * 100,
         ref_year=_Y0, target_year=_Y1,
         # 0-6 rather than 0-4: the Walloon draft renovation plan asks for 3 %/year
         # now and 5 %/year by 2050, and a slider a group cannot push that far
         # would contradict the card quoting it.
         slider={"min": 0, "max": 6, "step": 0.1}, better="up",
         impact=_impact_ren("rate", rat_RS_tes_sht_ren * 100,
                            dep_RS_tes_sht_ren * 100,
                            scaled=_tes[_Y1]["space heating"]),
         model={"var": "rat_RS_tes_sht_ren", "section": "2.1.1",
                "prose": "derived in section 2.1.1 from the -0.916 kWh/m²/year "
                         "trajectory and the assumed renovation depth; the model's "
                         "own input remains acc_RS_tes_sht_ren",
                "note": "the reference value is the renovation component of the "
                        "JRC-IDEES floor-area series, renewal minus net new build, "
                        "and the depth lever is defined on the same basis so that "
                        "the pair reproduces the observed improvement"},
         facts={"renoRateObs": round(obs_RS_sht_rat_ren, 3),
                "newBuildRateObs": round(obs_RS_sht_rat_new, 3),
                "renewalRateObs": round(obs_RS_sht_rat_all, 3),
                "improvementHist": cur_RS_tes_sht_ren,
                "intensity2000": round(obs_RS_tes_sht_2000, 1),
                "intensity2019": round(ref_RS_tes_sht, 1),
                "intensityDropPct": round(_intensity_drop_pct, 1),
                "depthObs": round(obs_RS_tes_sht_dep * 100, 1),
                "dwellingsM": round(_dwellings / 1e6, 2),
                "homesPerPoint": round(_dwellings / 100.0, -2),
                "homesPerYearObs": round(_homes_per_year["obs"], -3),
                "homesPerYearTarget": round(_homes_per_year["trg"], -3),
                "rateTarget": round(rat_RS_tes_sht_ren * 100, 2),
                "shareTarget": round(shr_RS_tes_sht_ren * 100, 1),
                "depthTarget": round(dep_RS_tes_sht_ren * 100, 1),
                "m2PerHousehold": round(_sur_hld[_Y0], 1),
                "heatKwhPerHousehold": round(_heat_kwh_per_household),
                "heatTwh": round(_tes[_Y0]["space heating"], 1),
                "heatSharePct": _pct(_tes[_Y0]["space heating"], _tes_tot[_Y0]),
                "thermalTwh": round(_tes_tot[_Y0], 1)},
         spoilers=["rateTarget", "shareTarget", "depthTarget",
                   "homesPerYearTarget"],
         notebook=_NB + "#section_2", reference="nW-BE §2.1.1")

    # The depth is the half nobody measures. What the observed period fixes is the
    # *product*: 2.261 %/year of floor area renovated at 25.5 % each reproduces the
    # -0.458 kWh/m²/year exactly, and so would 1 %/year at 58 %. The reference value
    # below is therefore "the depth that goes with the JRC rate", and the cards say
    # so -- it is the honest form of a quantity that is inferred, not measured.
    _add("renovation-depth", _T, "Depth of one energy renovation",
         "% of the heating need cut",
         obs_RS_tes_sht_dep * 100, dep_RS_tes_sht_ren * 100,
         ref_year=_Y0, target_year=_Y1,
         slider={"min": 0, "max": 100, "step": 5}, better="up",
         impact=_impact_ren("depth", dep_RS_tes_sht_ren * 100,
                            rat_RS_tes_sht_ren * 100,
                            scaled=_tes[_Y1]["space heating"]),
         model={"var": "dep_RS_tes_sht_ren", "section": "2.1.1",
                "prose": "the one assumption section 2.1.1's renovation reading "
                         "adds; the rate follows from it and the trajectory",
                "note": "inferred, not measured: only the product of rate and "
                        "depth is observed, so the reference value is the depth "
                        "implied by the JRC renovation rate"},
         facts={"depthObs": round(obs_RS_tes_sht_dep * 100, 1),
                "renoRateObs": round(obs_RS_sht_rat_ren, 3),
                "improvementHist": cur_RS_tes_sht_ren,
                "intensity2000": round(obs_RS_tes_sht_2000, 1),
                "intensity2019": round(ref_RS_tes_sht, 1),
                "intensityDropPct": round(_intensity_drop_pct, 1),
                # the depth implied by the ~1 %/year of *energy* renovation the
                # Commission measures, on the same observed improvement: the same
                # arithmetic, a different count of what a renovation is.
                "depthIfOnePct": round(-cur_RS_tes_sht_ren
                                       / (1.0 / 100 * obs_RS_tes_sht_2000) * 100, 1),
                "depthHalveAll": 50.0,
                "rateTarget": round(rat_RS_tes_sht_ren * 100, 2),
                "shareTarget": round(shr_RS_tes_sht_ren * 100, 1),
                "depthTarget": round(dep_RS_tes_sht_ren * 100, 1),
                "heatKwhPerHousehold": round(_heat_kwh_per_household),
                "m2PerHousehold": round(_sur_hld[_Y0], 1),
                "heatTwh": round(_tes[_Y0]["space heating"], 1),
                "heatSharePct": _pct(_tes[_Y0]["space heating"], _tes_tot[_Y0]),
                "thermalTwh": round(_tes_tot[_Y0], 1)},
         spoilers=["rateTarget", "shareTarget", "depthTarget"],
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
