#!/usr/bin/env python
"""Refresh the international comparison figures quoted in the workshop content.

The workshop fact cards quote a handful of Eurostat modal-split numbers for
Belgium and its neighbours. They are written by hand into
website/workshop/content/<topic>.yaml (prose belongs there, not in generated
files), so this script exists to make them *auditable*: run it to see the current
values straight from the Eurostat API and check the YAML still matches.

    python scripts/fetch_eurostat_benchmarks.py            # print the tables
    python scripts/fetch_eurostat_benchmarks.py --json out.json

Datasets
    tran_hv_frmod   modal split of inland freight transport   (% of tonne-km)
    tran_hv_psmod   modal split of inland passenger transport (% of passenger-km)
    road_go_ta_tott road freight by type of operation: the empty-running share and
                    the loads behind `truck-fill` (the notebook's ref_*_FT_trk_* values,
                    section 3.3.3, 2024 and the EU 2008 start of the trend)
    road_go_ta_lc   laden vehicle-km by load-capacity class: the payload capacity the
                    filling rate divides by (ref_vkm_class_FT_trk_*, cap_FT_trk_hvy)

Note the bases, which differ from the model's:
  * frmod covers road + rail + inland waterways only (no air, no sea);
  * psmod covers cars + buses & coaches + trains only (no cycling, no walking,
    no tram/metro), which is why the Netherlands shows the *highest* car share
    despite cycling the most.
"""
import argparse
import json
import sys
import urllib.request

API = ("https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/"
       "{ds}?format=JSON&lang=EN{geo}{time}")
COUNTRIES = ["BE", "NL", "DE", "CH", "AT", "DK", "EU27_2020"]
YEARS = ["2019", "2023"]
DATASETS = {
    "tran_hv_frmod": ("Inland freight modal split", "% of tonne-km", "tra_mode"),
    "tran_hv_psmod": ("Inland passenger modal split", "% of passenger-km", "vehicle"),
}


def fetch(dataset, timeout=45):
    url = API.format(ds=dataset,
                     geo="".join("&geo=" + g for g in COUNTRIES),
                     time="".join("&time=" + y for y in YEARS))
    with urllib.request.urlopen(url, timeout=timeout) as fh:
        return json.load(fh)


def tabulate(doc, mode_dim):
    order, sizes = doc["id"], doc["size"]
    index = {k: doc["dimension"][k]["category"]["index"] for k in order}
    back = {k: {v: name for name, v in index[k].items()} for k in order}

    def decode(flat):
        rem, out = int(flat), {}
        for key, n in zip(reversed(order), reversed(sizes)):
            out[key] = back[key][rem % n]
            rem //= n
        return out

    rows = {}
    for flat, value in doc["value"].items():
        cell = decode(flat)
        rows.setdefault((cell["geo"], cell["time"]), {})[cell[mode_dim]] = value
    return rows


FREIGHT_COUNTRIES = ["BE", "FR", "DE", "NL", "AT", "EU27_2020"]
FREIGHT_YEARS = ["2008", "2019", "2024"]


def freight_loads(timeout=60):
    """Empty running and loads of heavy goods vehicles, by country of registration.

    Goods per truck-km = (1 - empty share) x load when laden; the notebook quotes
    the 2024 Belgian and EU values and the EU 2008 ones.
    """
    url = ("https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/"
           "road_go_ta_tott?format=JSON&lang=EN&tra_type=TOTAL"
           "&unit=MIO_VKM&unit=MIO_TKM&tra_oper=TOTAL&tra_oper=EMPTY&tra_oper=LOADED"
           + "".join("&geo=" + g for g in FREIGHT_COUNTRIES)
           + "".join("&time=" + y for y in FREIGHT_YEARS))
    with urllib.request.urlopen(url, timeout=timeout) as fh:
        doc = json.load(fh)
    order, sizes = doc["id"], doc["size"]
    index = {k: doc["dimension"][k]["category"]["index"] for k in order}
    back = {k: {v: name for name, v in index[k].items()} for k in order}
    cells = {}
    for flat, value in doc["value"].items():
        rem, cell = int(flat), {}
        for key, n in zip(reversed(order), reversed(sizes)):
            cell[key] = back[key][rem % n]
            rem //= n
        cells[(cell["geo"], cell["time"], cell["tra_oper"], cell["unit"])] = value
    rows = {}
    for geo in FREIGHT_COUNTRIES:
        for year in FREIGHT_YEARS:
            vkm = cells.get((geo, year, "TOTAL", "MIO_VKM"))
            empty = cells.get((geo, year, "EMPTY", "MIO_VKM"))
            laden = cells.get((geo, year, "LOADED", "MIO_VKM"))
            tkm = cells.get((geo, year, "TOTAL", "MIO_TKM"))
            if vkm and empty is not None and laden and tkm:
                rows[(geo, year)] = {"empty %": 100 * empty / vkm,
                                     "t per laden vkm": tkm / laden,
                                     "t per vkm": tkm / vkm}
    return rows


# Same class capacities as cap_class_FT_trk in the notebook (section 3.3.3).
LC_CLASSES = ["T_LE9P5", "T9P6-15P5", "T15P6-20P5", "T20P6-25P5", "T25P6-30P5", "T_GT30P5"]
LC_CAPACITY = [6.0, 12.5, 18.0, 23.0, 28.0, 32.0]
LC_YEARS = {"BE": ["2010", "2011", "2012", "2013", "2014", "2015", "2019"],
            "EU27_2020": ["2008", "2024"]}


def freight_capacity(timeout=90):
    """Payload capacity per laden vehicle-km, and the filling rate when laden.

    A filling rate above 100 % means the classification no longer holds, which is
    why the notebook takes the Belgian capacity from 2010-2014 only.
    """
    years = sorted({y for ys in LC_YEARS.values() for y in ys})
    url = ("https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/"
           "road_go_ta_lc?format=JSON&lang=EN&tra_type=TOTAL&unit=MIO_VKM&unit=MIO_TKM"
           + "".join("&geo=" + g for g in LC_YEARS)
           + "".join("&time=" + y for y in years))
    with urllib.request.urlopen(url, timeout=timeout) as fh:
        doc = json.load(fh)
    order, sizes = doc["id"], doc["size"]
    index = {k: doc["dimension"][k]["category"]["index"] for k in order}
    back = {k: {v: name for name, v in index[k].items()} for k in order}
    cells = {}
    for flat, value in doc["value"].items():
        rem, cell = int(flat), {}
        for key, n in zip(reversed(order), reversed(sizes)):
            cell[key] = back[key][rem % n]
            rem //= n
        cells[(cell["geo"], cell["time"], cell["weight"], cell["unit"])] = value
    rows = {}
    for geo, ys in LC_YEARS.items():
        for year in ys:
            vkm = [cells.get((geo, year, c, "MIO_VKM")) or 0 for c in LC_CLASSES]
            tkm = cells.get((geo, year, "TOTAL", "MIO_TKM"))
            if sum(vkm) and tkm:
                cap = sum(v * c for v, c in zip(vkm, LC_CAPACITY)) / sum(vkm)
                rows[(geo, year)] = {"laden vkm by class": vkm, "capacity t": cap,
                                     "filled when laden %": 100 * tkm / sum(vkm) / cap}
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", metavar="PATH", help="also write the raw tables as JSON")
    args = ap.parse_args()

    collected = {}
    for dataset, (title, unit, mode_dim) in DATASETS.items():
        try:
            doc = fetch(dataset)
        except Exception as exc:                      # offline is not a failure here
            print(f"{dataset}: could not fetch ({exc})", file=sys.stderr)
            continue
        rows = tabulate(doc, mode_dim)
        modes = sorted({m for r in rows.values() for m in r})
        print(f"\n{title} — {dataset} ({unit})")
        print(f"  {'geo':<10}{'year':<7}" + "".join(f"{m:>12}" for m in modes))
        for (geo, year), values in sorted(rows.items()):
            cells = "".join(f"{values.get(m, float('nan')):>12.1f}" for m in modes)
            print(f"  {geo:<10}{year:<7}{cells}")
        collected[dataset] = {f"{g}:{y}": v for (g, y), v in rows.items()}

    try:
        rows = freight_loads()
    except Exception as exc:                          # offline is not a failure here
        print(f"road_go_ta_tott: could not fetch ({exc})", file=sys.stderr)
        rows = {}
    if rows:
        cols = ["empty %", "t per laden vkm", "t per vkm"]
        print("\nRoad freight, heavy goods vehicles by country of registration — road_go_ta_tott")
        print(f"  {'geo':<10}{'year':<7}" + "".join(f"{c:>17}" for c in cols))
        for (geo, year), values in sorted(rows.items()):
            print(f"  {geo:<10}{year:<7}" + "".join(f"{values[c]:>17.2f}" for c in cols))
        collected["road_go_ta_tott"] = {f"{g}:{y}": v for (g, y), v in rows.items()}

    try:
        caps = freight_capacity()
    except Exception as exc:                          # offline is not a failure here
        print(f"road_go_ta_lc: could not fetch ({exc})", file=sys.stderr)
        caps = {}
    if caps:
        print("\nPayload capacity by load-capacity class — road_go_ta_lc")
        print(f"  {'geo':<10}{'year':<7}{'capacity t':>12}{'filled laden %':>16}  laden vkm by class")
        for (geo, year), r in sorted(caps.items()):
            print(f"  {geo:<10}{year:<7}{r['capacity t']:>12.2f}{r['filled when laden %']:>16.1f}"
                  f"  {[round(v) for v in r['laden vkm by class']]}")
        collected["road_go_ta_lc"] = {f"{g}:{y}": v for (g, y), v in caps.items()}

    if args.json and collected:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(collected, fh, indent=2, sort_keys=True)
        print(f"\nwrote {args.json}")
    return 0 if collected else 1


if __name__ == "__main__":
    sys.exit(main())
