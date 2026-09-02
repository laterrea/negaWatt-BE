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

    if args.json and collected:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(collected, fh, indent=2, sort_keys=True)
        print(f"\nwrote {args.json}")
    return 0 if collected else 1


if __name__ == "__main__":
    sys.exit(main())
