import json
import os
from datetime import date

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import fsolve

#%% PROJECTIONS

# If we want change the initial years:
def generate_target_years(start_year, end_year=2050, step=5, min_gap=2):
    """
    Returns a list of years from start_year to end_year (included), spaced by 
    'step', ensuring at least 'min_gap' years between start_year and the next 
    value.
    """
    years = [start_year]
    next_year = ((start_year // step) + 1) * step
    while next_year <= end_year:
        if next_year - start_year >= min_gap:
            years.append(next_year)
        next_year += step
    return years

#%% POST-PROCESSING

def highlight_lines(row):
    if row['Unit'] == '% of total':
        return ['border-top:2px solid black'] * len(row)
    if row['Mode'] == 'TOTAL':
        return ['border-top:4px double black'] * len(row)
    return [''] * len(row)

def highlight_mode_separator(row):
    if 'Carrier' in row:
        if row['Carrier'] != '':
            return ['border-top:2px solid black'] * len(row)
    if 'Mode' in row:
        if row['Mode'] != '':
            return ['border-top:2px solid black'] * len(row)
    if 'Sector' in row:
        if row['Sector'] != '':
            return ['border-top:2px solid black'] * len(row)
    return [''] * len(row)

def bold_mode(cell, mode_cell, col):
    if mode_cell != '' and col == 'Carrier':
        return 'font-weight:bold'
    elif mode_cell != '' and col == 'Mode':
        return 'font-weight:bold'
    elif mode_cell != '' and col == 'Sector':
        return 'font-weight:bold'
    return ''

#%% MISC

# LINEAR: how to use, exemple :
def linear_growth(start_year, start_value, end_year, end_value, target_years):
    """
    Computes linear interpolation between start_value and end_value over time.

    Returns a list of values for each year in target_years.

    Example: linear_growth(2020, 10, 2030, 20, [2020, 2025, 2030]) → [10.0, 15.0, 20.0]
    """
    return [round(start_value + (end_value - start_value) * (y - start_year) / (end_year - start_year), 3)
            for y in target_years]

# General Constant Function with CONTROL POINT:
def constant_with_control_point(start_year, start_value, control_year, control_value, target_years):
    """
    Returns values with piecewise constant growth: 
    from start to control point, then from control to end.

    Useful to model changes with an intermediate value at a specific year.

    Example: control point at 2030 with a plateau.
    """
    values = []
    for y in target_years:
        if y < control_year:
            val = start_value 
        else:
            val = control_value 
        values.append(round(val, 3))
    return values

# General LINEAR Function with CONTROL POINT:
def linear_with_middle_point(start_year, start_value, control_year, control_value, end_year, end_value, target_years):
    """
    Piecewise linear interpolation with one control point:
    - start_year → control_year
    - control_year → end_year

    Useful to model changes with an intermediate value at a specific year.

    Notes / edge cases:
    - If control_year == start_year and control_value != start_value, the first segment is undefined (division by zero).
      In that case, the function keeps start_value up to control_year.
    - If end_year == control_year and end_value != control_value, the second segment is undefined.
      In that case, the function keeps control_value from control_year onward.
    - For flat segments (same values), the function returns the constant value.
    """
    values = []
    for y in target_years:
        if y <= control_year:
            if control_value == start_value:
                val = start_value
            else:
                val = start_value + (control_value - start_value) * (y - start_year) / (control_year - start_year)
        else:
            if control_value == end_value:
                val = end_value
            else:
                val = control_value + (end_value - control_value) * (y - control_year) / (end_year - control_year)
        values.append(round(val, 3))
    return values

def curved_with_middle(start_year, start_value, control_year, control_value, end_year, end_value, target_years, shape_start=1.0, shape_end=1.0, smooth_power=5):
    """
    Returns a smooth curve between three points: start, control, and end.

    The curve is made of two segments (start→control and control→end), each shaped 
    by a curvature factor:
    - shape_start and shape_end ∈ [0, 1], where 0 = linear and 1 = fully curved.

    The 'smooth_power' parameter controls the strength of curvature when shape = 1.
    Higher values produce more pronounced acceleration or deceleration near the control point.

    Useful for modeling soft transitions that are not purely linear.
    """
    def normalize(x, a, b):
        return (x - a) / (b - a)

    p1 = 1 + shape_start * (smooth_power - 1)
    p2 = 1 + shape_end   * (smooth_power - 1)
    
    result = []
    for y in target_years:
        if y <= control_year:
            t = normalize(y, start_year, control_year)
            v = start_value + (control_value - start_value) * (t ** p1)
        else:
            u = normalize(y, control_year, end_year)
            v = control_value + (end_value - control_value) * (1 - (1 - u) ** p2)
        result.append(round(v, 3))
    return result
    

# S-CURVE without a CONTROL POINT:
def s_curve_growth(start_year, start_value, end_year, end_value, target_years, slope_factor):
    """
    Returns a smooth S-shaped curve if we give the start and the end value.

    The curve uses a logistic function centered between start and end years,
    scaled to match the given start and end values.

    - 'slope_factor' controls how steep the S-curve is (The transition is faster the higher the number.).

    Useful for modeling progressive adoption, saturation, or demand evolution.
    """
    # Normalization of the abscissa
    def normalize(y):
        return (y - start_year) / (end_year - start_year)
    y0, y1 = start_value, end_value
    c = 0.5  # always centered in the temporal middle
    
    # Raw logistic
    def f_raw(x, k):
        return 1 / (1 + np.exp(-k * (x - c)))

    # Normalized logistic to exactly match endpoints y0 and y1
    def f_norm(x, k):
        f0 = f_raw(0, k)
        f1 = f_raw(1, k)
        return y0 + (y1 - y0) * (f_raw(x, k) - f0) / (f1 - f0)
    
    # Determine k
    k0 = 10.0 * slope_factor

    # Generates the curve for each target year
    x_targets = [normalize(y) for y in target_years]
    return [round(float(f_norm(x, k0)), 3) for x in x_targets]

# S-CURVE that must pass through the CONTROL POINT:
def s_curve_with_control_value(start_year, start_value, control_year, control_value, end_year, end_value, target_years, slope_factor):
    """
    Returns a smooth S-shaped curve passing through a fixed control point.

    The curve uses a logistic function centered between start and end years,
    scaled to match the given start and end values.

    If the control point is not centered, the function automatically adjusts the slope 
    (steepness) so that the curve still passes through it.

    - 'slope_factor' controls how steep the S-curve is (The transition is faster the higher the number).

    Useful for modeling progressive adoption, saturation, or demand evolution.
    """
    # Normalization of the abscissa
    def normalize(y):
        return (y - start_year) / (end_year - start_year)
    x_ctrl = normalize(control_year)
    y0, y1, y2 = start_value, control_value, end_value
    c = 0.5  # always centered in the temporal middle
    
    # Raw logistic
    def f_raw(x, k):
        return 1 / (1 + np.exp(-k * (x - c)))

    # Normalized logistic to exactly match endpoints y0 and y2
    def f_norm(x, k):
        f0 = f_raw(0, k)
        f1 = f_raw(1, k)
        return y0 + (y2 - y0) * (f_raw(x, k) - f0) / (f1 - f0)
    
    # Determine k
    mid_time = (start_year + end_year) / 2
    if control_year == mid_time:
        # Case 1: the control point is exactly in the middle
        k0 = 10.0 * slope_factor
    else:
        # Case 2: find k to pass through the control point.
        def objective(k):
            return f_norm(x_ctrl, k) - y1
        k_guess = 10.0 * slope_factor * (1 if y1 > (y0 + y2)/2 else -1)
        k0 = fsolve(objective, k_guess)[0]
    
    # Generates the curve for each target year
    x_targets = [normalize(y) for y in target_years]
    return [round(float(f_norm(x, k0)), 3) for x in x_targets]

# BELL-CURVE that must pass through the CONTROL POINT:
def b_curve_with_control_value(start_year, start_value, control_year, control_value, end_year, end_value, target_years, slope_factor):
    """
    Returns a bell-shaped curve that EXACTLY passes through:
    (start_year, start_value), (control_year, control_value), and (end_year, end_value).
    
    - 'slope_factor': Controls the width. 1.0 is standard, >1.0 is narrow, <1.0 is wide.
    """
    x_s, y_s = float(start_year), float(start_value)
    x_c, y_c = float(control_year), float(control_value)
    x_e, y_e = float(end_year), float(end_value)
    
    total_range = x_e - x_s
    # k controls the "tightness" of the bell.
    # We normalize by total_range so slope_factor feels consistent regardless of the year span.
    k = (slope_factor ** 2) * 8.0 / (total_range ** 2)

    def f_raw_gaussian(x):
        return np.exp(-k * (x - x_c)**2)

    # 1. Create a linear baseline between start and end values
    def L(x):
        return y_s + (y_e - y_s) * (x - x_s) / total_range

    # 2. Create a correction for the Gaussian so it equals 0 at x_s and x_e
    def S(x):
        g_x = f_raw_gaussian(x)
        g_s = f_raw_gaussian(x_s)
        g_e = f_raw_gaussian(x_e)
        # Subtract the line connecting the Gaussian's own start/end values
        gaussian_trend = g_s + (g_e - g_s) * (x - x_s) / total_range
        return g_x - gaussian_trend

    # 3. Calculate scaling factor to ensure the peak hits control_value
    s_c = S(x_c)
    l_c = L(x_c)
    
    # If control_year is at the very edge, avoid division by zero
    if abs(s_c) < 1e-10:
        return [round(float(L(y)), 3) for y in target_years]

    def get_val(x):
        # Result = Linear Trend + (Distance needed to hit Peak * Normalized Shape)
        return L(x) + (y_c - l_c) * (S(x) / s_c)

    return [round(float(get_val(y)), 3) for y in target_years]

# ACCELERATED GROWTH 
def accelerated_growth(start_year, start_value, end_year, end_value, target_years, factor, mode="ease_in"):
    """
    Returns a curved growth or decay between start and end values with an accelerating shape.

    The curve is based on a power easing function:
      - 'factor' controls how strong the acceleration is (The transition is faster the higher the number).
      - 'mode' defines the shape:
          * "ease_in"     → slow start, faster finish
          * "ease_out"    → fast start, slower finish
          * "ease_in_out" → slow at start and end, faster in the middle

    Useful for scenarios such as technology uptake, emissions reduction,
    or gradual policy impacts.
    """
    import numpy as np

    t = (np.array(target_years, dtype=float) - float(start_year)) / (float(end_year) - float(start_year))
    a = max(1.0, float(factor))  # sécurité minimale

    if mode == "ease_in":
        s = t**a
    elif mode == "ease_out":
        s = 1.0 - (1.0 - t)**a
    elif mode == "ease_in_out":
        # in/out symétrique (généralisation lisse)
        left  = 0.5 * (2.0 * t)**a
        right = 1.0 - 0.5 * (2.0 * (1.0 - t))**a
        s = np.where(t < 0.5, left, right)
    else:
        raise ValueError('mode must be "ease_in", "ease_out", or "ease_in_out"')

    vals = float(start_value) + s * (float(end_value) - float(start_value))
    return [round(float(v), 3) for v in vals]

# LINEAR AND THEN ACCELERATED GROWTH 
def strong_acceleration_growth(start_year, start_value, end_year, end_value, target_years, power=3):
    """
    Returns a growth or decay curve with very little change at the start, and most of the change happening near the end.

    - 'power' >= 1 controls how late the acceleration happens (The transition is faster the higher the number) 

    Useful when we want a trajectory almost flat earlu on, 
    followed by a strong acceleration close to the horizon year.
    """
    u = np.array([(y - start_year) / (end_year - start_year) for y in target_years])

    vals = float(start_value) + (u**power) * (float(end_value) - float(start_value))
    return [round(float(v), 3) for v in vals]


#%% WEBSITE EXPORT
#
# These helpers connect the notebooks to the public website in ``website/``.
# Each notebook builds a list of "elementary hypotheses" (current value vs.
# 2050 target) and calls ``write_hypotheses_js`` to (re)write the matching
# ``website/data/<sector>.js`` file. The static HTML pages read those files, so
# re-running a notebook automatically refreshes the figures shown online.
#
# See the "Public sufficiency website" section of README.md for the full data
# contract.

def _json_safe(value):
    """Convert numpy / pandas scalars to plain Python types for json.dumps."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return round(float(value), 4)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def make_hypothesis(id, name, category, ref_value, target_value,
                    unit="", ref_year=2019, target_year=2050,
                    pct_change=None, change_label=None, direction=None,
                    display_ref=None, display_target=None,
                    notebook=None, reference=None):
    """Build one website hypothesis record (a plain dict).

    Only ``id``, ``name``, ``category``, ``ref_value`` and ``target_value`` are
    required. ``pct_change`` is computed from the two values when both are
    numeric and it is not supplied (relative change vs. the reference value).

    The optional fields mirror the keys consumed by ``website/assets/js/render.js``:
    ``change_label`` / ``direction`` override the auto-generated badge,
    ``display_ref`` / ``display_target`` override number formatting, ``notebook``
    is the deep-link to the flattened notebook, ``reference`` is the source shown
    in the card footer.
    """
    rec = {
        "name": name,
        "category": category,
        "refYear": ref_year,
        "refValue": ref_value,
        "targetYear": target_year,
        "targetValue": target_value,
        "unit": unit,
    }
    if pct_change is None and change_label is None:
        try:
            rv, tv = float(ref_value), float(target_value)
            if rv != 0:
                pct_change = (tv - rv) / abs(rv) * 100.0
        except (TypeError, ValueError):
            pct_change = None
    if pct_change is not None:
        rec["pctChange"] = round(float(pct_change), 1)
    if change_label is not None:
        rec["changeLabel"] = change_label
    if direction is not None:
        rec["direction"] = direction
    if display_ref is not None:
        rec["displayRef"] = display_ref
    if display_target is not None:
        rec["displayTarget"] = display_target
    if notebook is not None:
        rec["notebook"] = notebook
    if reference is not None:
        rec["reference"] = reference
    return {"id": id, **rec}


def write_hypotheses_js(sector_key, hypotheses, plots=None, title=None,
                        out_path=None):
    """Write ``website/data/<sector_key>.js`` for the public website.

    Parameters
    ----------
    sector_key : str
        Key under ``window.NW_DATA`` (e.g. ``"transport"``, ``"buildings"``).
    hypotheses : list[dict]
        Records, ideally produced by :func:`make_hypothesis`. Each must carry an
        ``id``; the remaining keys populate the card.
    plots : dict, optional
        Named plot specs consumed by ``render.js`` (``data-plot`` figures), e.g.
        ``{"modalShare": {"type": "stackedBar", "x": [...], "series": [...]}}``.
    title : str, optional
        Human-readable sector title stored alongside the data.
    out_path : str, optional
        Override the destination file. Defaults to ``website/data/<sector_key>.js``
        next to this module.

    Returns
    -------
    str
        The path that was written.
    """
    hyp_map = {}
    for h in hypotheses:
        h = dict(h)
        hid = h.pop("id")
        hyp_map[hid] = h

    payload = {
        "title": title or sector_key.capitalize(),
        "generated": date.today().isoformat(),
        "hypotheses": hyp_map,
    }
    if plots:
        payload["plots"] = plots
    payload = _json_safe(payload)

    if out_path is None:
        here = os.path.dirname(os.path.abspath(__file__))
        out_path = os.path.join(here, "website", "data", sector_key + ".js")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    body = json.dumps(payload, indent=2, ensure_ascii=False)
    content = (
        "/* AUTO-GENERATED by the notebook 'Website export' cell. Do not edit by hand. */\n"
        "window.NW_DATA = window.NW_DATA || {};\n"
        "window.NW_DATA[\"" + sector_key + "\"] = " + body + ";\n"
    )
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(content)
    print(f"[website] wrote {len(hyp_map)} hypotheses to {out_path}")
    return out_path


# # S-CURVE (logistic function centered on midpoint)
# def s_curve_negaWatt_france(start_year, start_value, end_year, end_value, target_years, midpoint, sigma=0.15):
#     """
#     Returns an S-shaped growth curve based on a normal CDF, scaled between start and end values.

#     The curve is centered on 'midpoint' and controlled by 'sigma':
#     - A small sigma → steep transition (quick change)
#     - A large sigma → smoother, slower transition

#     The result is normalized to ensure it starts at start_value and ends at end_value.

#     Useful for modeling gradual adoption, saturation effects, or phased rollouts.
#     """
#     if midpoint is None:
#         midpoint = (start_year + end_year) / 2
#     t_norm = [(y - start_year) / (end_year - start_year) for y in target_years]
#     mid_t = (midpoint - start_year) / (end_year - start_year)
    
#     # Solve mu such that CDF(mu+1) = 0.9
#     PROB_UMAX = 0.9
#     def eq(mu, prob): return norm.cdf(1, mu, sigma) - prob
#     mu = fsolve(eq, 0, args=PROB_UMAX)[0]
    
#     # Apply scaled normal CDF
#     s_vals = [1 - norm.cdf((t - mid_t) * 2, mu, sigma) for t in t_norm]
#     s_vals = (np.array(s_vals) - min(s_vals)) / (max(s_vals) - min(s_vals))  # normalize to [0, 1]
#     vals = start_value + s_vals * (end_value - start_value)
#     return [round(v, 3) for v in vals]

#%% ENERGY TOTALS OVERRIDES (notebook <-> nW_BE.py consistency)

ENERGY_TOTALS_OVERRIDE_YEARS = (2019, 2030, 2040, 2050)
ENERGY_TOTALS_OVERRIDE_COLUMNS = ["source", "sector", "key", "year", "value_twh"]

ENERGY_TOTALS_MODE_MAPPING = {
    "bicycle": "total road",
    "bus&coach": "total road",
    "car": "total road",
    "truck-heavy duty": "total road",
    "truck-light commercial": "total road",
    "two-wheeler": "total road",
    "navigation-coastal": "navigation",
    "navigation-inland": "navigation",
    "plane-extra EU": "aviation",
    "plane-intra EU": "aviation",
    "train": "total train",
    "train-conventional": "total train",
    "train-high speed": "total train",
    "tram&metro": "total train",
}

# (mapped mode, sector, total energy_totals key, electricity energy_totals key or None)
_TRANSPORT_EMIT = (
    ("total road", "road", "total road", "electricity road"),
    ("total train", "rail", "total rail", "electricity rail"),
    ("aviation", "aviation", "total international aviation", None),
    ("navigation", "navigation", "total domestic navigation", None),
)


def _label_for_year(labels, year):
    year = int(year)
    for lab in labels:
        try:
            if int(lab) == year:
                return lab
        except (TypeError, ValueError):
            continue
    raise KeyError("year %s not found" % year)


def _idx_at(df, year, col=None):
    lab = _label_for_year(df.index, year)
    row = df.loc[lab]
    if col is None:
        if isinstance(row, pd.Series):
            return float(row.sum())
        return float(row)
    return float(row[col])


def _col_at(df, year, row=None):
    if df is None or len(df) == 0:
        return 0.0
    lab = _label_for_year(df.columns, year)
    if row is None:
        return float(df[lab].sum())
    if row not in df.index:
        return 0.0
    val = df.loc[row, lab]
    if isinstance(val, pd.Series):
        return float(val.sum())
    return float(val)


def _override_row(source, sector, key, year, value):
    return {
        "source": source,
        "sector": sector,
        "key": key,
        "year": int(year),
        "value_twh": round(float(value), 6),
    }


def _split_powertrains(df):
    year_cols = [c for c in df.columns if c not in ("Mode", "Powertrain")]

    def prep(mask):
        out = df.loc[mask, ["Mode"] + list(year_cols)].copy()
        if out.empty:
            return pd.DataFrame(columns=year_cols)
        return out.groupby("Mode", as_index=True).sum(numeric_only=True)

    return {
        "total": prep(df["Powertrain"] == "TOTAL"),
        "elec": prep(df["Powertrain"] == "electrical"),
    }


def _map_transport_modes(df):
    if df is None or len(df) == 0:
        return df
    mapped = df.copy()
    mapped.index = mapped.index.to_series().replace(ENERGY_TOTALS_MODE_MAPPING)
    return mapped.groupby(level=0).sum(numeric_only=True)


def energy_totals_overrides_from_transport(mobility, freight, years=ENERGY_TOTALS_OVERRIDE_YEARS):
    """Tidy energy_totals rows from passenger + freight TWh tables (Mode/Powertrain)."""
    mobility_pt = _split_powertrains(mobility)
    freight_pt = _split_powertrains(freight)
    total = _map_transport_modes(
        pd.concat([mobility_pt["total"], freight_pt["total"]]).groupby(level=0).sum()
    )
    elec = _map_transport_modes(
        pd.concat([mobility_pt["elec"], freight_pt["elec"]]).groupby(level=0).sum()
    )
    rows = []
    for mapped, sector, total_key, elec_key in _TRANSPORT_EMIT:
        for year in years:
            rows.append(_override_row("transport", sector, total_key, year, _col_at(total, year, mapped)))
            if elec_key is not None:
                rows.append(_override_row("transport", sector, elec_key, year, _col_at(elec, year, mapped)))
    return pd.DataFrame(rows, columns=ENERGY_TOTALS_OVERRIDE_COLUMNS)


def energy_totals_overrides_from_buildings(
    residential_thermal,
    residential_elec,
    tertiary_thermal,
    tertiary_elec,
    years=ENERGY_TOTALS_OVERRIDE_YEARS,
):
    """Tidy energy_totals rows from residential/tertiary service DataFrames (index=year)."""
    rows = []
    for year in years:
        rs_space = (
            _idx_at(residential_thermal, year, "space heating")
            + _idx_at(residential_thermal, year, "cooking")
            + _idx_at(residential_thermal, year, "space cooling")
        )
        ts_space = (
            _idx_at(tertiary_thermal, year, "space heating")
            + _idx_at(tertiary_thermal, year, "catering")
            + _idx_at(tertiary_thermal, year, "space cooling")
        )
        rows.extend(
            [
                _override_row("buildings", "residential", "total residential space", year, rs_space),
                _override_row(
                    "buildings",
                    "residential",
                    "total residential water",
                    year,
                    _idx_at(residential_thermal, year, "sanitary hot water"),
                ),
                _override_row(
                    "buildings",
                    "residential",
                    "distributed heat residential",
                    year,
                    _idx_at(residential_thermal, year, "heat_dhn"),
                ),
                _override_row(
                    "buildings",
                    "residential",
                    "electricity residential",
                    year,
                    _idx_at(residential_elec, year),
                ),
                _override_row("buildings", "services", "total services space", year, ts_space),
                _override_row(
                    "buildings",
                    "services",
                    "total services water",
                    year,
                    _idx_at(tertiary_thermal, year, "sanitary hot water"),
                ),
                _override_row(
                    "buildings",
                    "services",
                    "distributed heat services",
                    year,
                    _idx_at(tertiary_thermal, year, "heat_dhn"),
                ),
                _override_row(
                    "buildings",
                    "services",
                    "electricity services",
                    year,
                    _idx_at(tertiary_elec, year),
                ),
            ]
        )
    return pd.DataFrame(rows, columns=ENERGY_TOTALS_OVERRIDE_COLUMNS)


def write_energy_totals_overrides(df, path, comment=None, replace_sources=None):
    """Write (or merge-by-source) a tidy energy_totals overrides CSV."""
    path = os.path.abspath(path)
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    out = df[list(ENERGY_TOTALS_OVERRIDE_COLUMNS)].copy()
    out["year"] = out["year"].astype(int)
    out["value_twh"] = out["value_twh"].astype(float)
    if replace_sources is not None and os.path.isfile(path):
        existing = pd.read_csv(path, comment="#")
        keep = existing[~existing["source"].isin(list(replace_sources))]
        out = pd.concat([keep, out], ignore_index=True)
    out = out.sort_values(["source", "sector", "key", "year"]).reset_index(drop=True)
    if comment is None:
        comment = "generated from nW_BE.py; refresh from notebooks to detect transcription drift"
    with open(path, "w", encoding="utf-8") as fh:
        for line in str(comment).splitlines():
            fh.write("# " + line.lstrip("# ").rstrip() + "\n")
        out.to_csv(fh, index=False)
    return path


def make_energy_totals_overrides(df, path=None, comment=None, replace_sources=None):
    """Write the overrides CSV. Default path is ``data/energy_totals_overrides.csv`` next to this file."""
    if path is None:
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(here, "data", "energy_totals_overrides.csv")
    return write_energy_totals_overrides(
        df, path, comment=comment, replace_sources=replace_sources
    )
