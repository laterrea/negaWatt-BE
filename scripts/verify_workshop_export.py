#!/usr/bin/env python
"""Acceptance test for the workshop data exported by the notebooks (milestone M1).

Checks the generated files, not the notebooks, so it is fast and can run in CI:

  python scripts/verify_workshop_export.py

Re-run the "Workshop export" cell of nW_BE_demand_model_transports.ipynb and the
"Workshop history export" cell of nW_BE_demand_data_aux.ipynb first if the model
has changed.
"""
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "website", "data")

# The eight levers played in the proof of concept, with the values documented in
# docs/workshop_module.md. Tolerances are loose enough to survive a rounding
# change and tight enough to catch a real shift in an assumption.
EXPECTED = {
    # id                 unit                 2019      2050    tol
    "ground-km-day":  ("km/person/day",      32.68,    30.48,   0.05),
    "car-share":      ("% of motorised km",  78.80,    55.49,   0.05),
    "car-occupancy":  ("persons/car",         1.22,     2.00,   0.01),
    "car-energy":     ("% of 2019",         100.00,    75.00,   0.01),
    "bike-km-day":    ("km/person/day",       1.68,     3.27,   0.01),
    "freight-tkm":    ("tkm/person/year",  7013.63,  6312.27,   1.00),
    "truck-share":    ("% of tonne-km",      66.77,    50.42,   0.05),
    "truck-load":     ("tonnes",             12.65,    13.29,   0.01),
}
SPARE = {"bus-occupancy", "train-occupancy"}
IMPACT_KINDS = {"proportional", "inverse", "linear-shift", "negligible"}
EDGE_MARGIN = 0.12


def payload(name):
    path = os.path.join(DATA, name)
    if not os.path.isfile(path):
        raise SystemExit(f"MISSING {path}\n  -> run the notebook export cell first")
    txt = open(path, encoding="utf-8").read()
    return json.loads(re.search(r"= (\{.*\});\n\Z", txt, re.S).group(1))


def main():
    failures, checks = [], 0

    def check(cond, msg):
        nonlocal checks
        checks += 1
        if not cond:
            failures.append(msg)

    levers_doc = payload("levers_transport.js")
    history_doc = payload("history_transport.js")
    levers = levers_doc["levers"]
    series = history_doc["series"]

    # --- the lever set ------------------------------------------------------
    shown = {k for k, v in levers.items() if v.get("shown")}
    check(shown == set(EXPECTED), f"shown levers are {sorted(shown)}, expected {sorted(EXPECTED)}")
    check(SPARE <= set(levers), f"spare levers missing: {sorted(SPARE - set(levers))}")

    for lid, (unit, ref, target, tol) in EXPECTED.items():
        lv = levers.get(lid)
        if lv is None:
            failures.append(f"{lid}: absent")
            checks += 1
            continue
        check(lv["unit"] == unit, f"{lid}: unit is {lv['unit']!r}, expected {unit!r}")
        check(abs(lv["refValue"] - ref) <= tol,
              f"{lid}: 2019 value {lv['refValue']} != {ref} (+-{tol})")
        check(abs(lv["targetValue"] - target) <= tol,
              f"{lid}: 2050 value {lv['targetValue']} != {target} (+-{tol})")
        check(lv["refYear"] == 2019 and lv["targetYear"] == 2050,
              f"{lid}: unexpected years {lv['refYear']}/{lv['targetYear']}")

    # --- every lever is well formed ----------------------------------------
    for lid, lv in levers.items():
        for field in ("topic", "name", "unit", "slider", "better", "decimals"):
            check(field in lv, f"{lid}: missing field {field!r}")
        s = lv["slider"]
        span = s["max"] - s["min"]
        check(span > 0, f"{lid}: empty slider range")
        # The anti-anchoring rule (docs/workshop_module.md D4): a range that ends
        # on the negaWatt value hands the answer to the participants.
        edge = min(lv["targetValue"] - s["min"], s["max"] - lv["targetValue"]) / span
        check(edge >= EDGE_MARGIN,
              f"{lid}: negaWatt target sits {edge:.0%} from a slider end (need >={EDGE_MARGIN:.0%})")
        check(s["min"] <= lv["refValue"] <= s["max"],
              f"{lid}: 2019 value {lv['refValue']} outside the slider range")
        imp = lv.get("impact")
        check(imp is not None, f"{lid}: no impact record")
        if imp:
            check(imp["kind"] in IMPACT_KINDS, f"{lid}: bad impact kind {imp['kind']!r}")
            topic_model = levers_doc.get("model", {}).get(lv.get("topic"), {})
            expected = (topic_model.get("inlandTwh") or {}).get("2050")
            if expected is not None:
                check(abs(imp["total"] - expected) < 0.01,
                      f"{lid}: impact total {imp['total']} != the topic's 2050 total")
            if imp["kind"] == "linear-shift":
                check("slope" in imp, f"{lid}: linear-shift without a slope")
            else:
                check(imp["scaled"] > 0, f"{lid}: {imp['kind']} with scaled=0 "
                                         f"(the mode lookup probably failed)")

    # --- the energy model --------------------------------------------------
    # The model block is keyed by topic, so two topics of one sector cannot
    # overwrite each other's context quantities.
    check("inland-mobility" in levers_doc.get("model", {}),
          "the model block has no inland-mobility entry")
    m = levers_doc["model"]["inland-mobility"]
    tot19, tot50 = m["inlandTwh"]["2019"], m["inlandTwh"]["2050"]
    check(90 < tot19 < 115, f"2019 inland demand {tot19} TWh is outside a sane range")
    check(15 < tot50 < 35, f"2050 inland demand {tot50} TWh is outside a sane range")
    check(abs(m["inlandPassengerTwh"]["2050"] + m["inlandFreightTwh"]["2050"] - tot50) < 0.01,
          "passenger + freight do not add up to the inland total")
    for name in ("modeTwhTarget", "freightModeTwhTarget"):
        check(sum(m[name].values()) > 0, f"{name} is all zeros (mode lookup failed)")
    check(m["modeTwhTarget"]["car"] > 1, "car 2050 demand looks wrong")

    # --- history series ----------------------------------------------------
    for key, s in series.items():
        check(len(s["x"]) == len(s["y"]), f"series {key}: x/y length mismatch")
        check(s["x"] == sorted(s["x"]), f"series {key}: years not sorted")
        check(any(v is not None for v in s["y"]), f"series {key}: no data at all")
        check("source" in s, f"series {key}: no source recorded")

    # A lever may point at a series exported here or at a hand-curated one in the
    # topic YAML; only the former can be checked at this stage.
    for lid, lv in levers.items():
        key = lv.get("history")
        if key and key in series:
            s = series[key]
            hist = next((v for x, v in zip(s["x"], s["y"]) if x == lv["refYear"]), None)
            check(hist is not None, f"{lid}: series {key} has no {lv['refYear']} point")
            if hist is not None:
                tol = max(0.05, abs(lv["refValue"]) * 0.003)
                check(abs(hist - lv["refValue"]) <= tol,
                      f"{lid}: lever {lv['refValue']} and history {hist} disagree in "
                      f"{lv['refYear']} — the two are on different bases")

    print(f"levers_transport.js   {len(levers)} levers ({len(shown)} shown), "
          f"generated {levers_doc['generated']}")
    print(f"history_transport.js  {len(series)} series, generated {history_doc['generated']}")
    print(f"inland mobility       {tot19:.1f} TWh (2019) -> {tot50:.1f} TWh (2050)")
    print(f"\n{checks} checks run")
    if failures:
        print(f"{len(failures)} FAILED:")
        for f in failures:
            print("  -", f)
        return 1
    print("all passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
