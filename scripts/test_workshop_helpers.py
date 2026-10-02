#!/usr/bin/env python
"""Unit tests for the workshop export helpers in nW_BE_demand_model_sub_functions.py.

Run from the repository root:  python scripts/test_workshop_helpers.py
"""
import importlib.util
import json
import os
import re
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_sub_functions():
    path = os.path.join(ROOT, "nW_BE_demand_model_sub_functions.py")
    spec = importlib.util.spec_from_file_location("sf", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def payload_of(js_path):
    """Parse the JSON payload back out of a generated window.* .js file."""
    txt = open(js_path, encoding="utf-8").read()
    assert txt.startswith("/* AUTO-GENERATED"), "missing generated-file banner"
    return json.loads(re.search(r"= (\{.*\});\n\Z", txt, re.S).group(1))


def main():
    sf = load_sub_functions()
    passed = []

    def ok(msg):
        passed.append(msg)
        print("  PASS", msg)

    lever = sf.make_lever(
        "car-occupancy", "inland-mobility", "Car occupancy", "persons/car",
        1.22, 2.00, slider={"min": 1.0, "max": 2.5, "step": 0.05},
        impact={"kind": "inverse", "mode": "car", "twh2050": 8.9},
        model={"var": "occu_trgt_PM_car", "cell": 77}, history="car_occupancy")
    assert lever["better"] == "up", lever["better"]
    assert lever["decimals"] == 2, lever["decimals"]
    assert lever["shown"] is True
    ok("make_lever basic record, derived better/decimals")

    # The anti-anchoring guard (docs/workshop_module.md D4) is the reason this
    # export can fail: a range that ends on the negaWatt value gives it away.
    for bad in ({"min": 1.2, "max": 2.0, "step": 0.05},
                {"min": 2.0, "max": 3.0, "step": 0.05}):
        try:
            sf.make_lever("x", "t", "x", "u", 1.22, 2.00, slider=bad)
        except ValueError as exc:
            assert "slider end" in str(exc), exc
        else:
            raise AssertionError(f"edge guard did not fire for {bad}")
    ok("edge-margin guard rejects a target at an end stop")

    for ref, trg in [(1.22, 2.0), (35.1, 34.7), (0, 30), (7014, 6312),
                     (100, 75), (12.65, 13.29), (1.68, 3.27), (0, 25)]:
        lv = sf.make_lever("a", "t", "a", "u", ref, trg)
        s = lv["slider"]
        span = s["max"] - s["min"]
        edge = min(trg - s["min"], s["max"] - trg) / span
        assert edge >= sf.LEVER_MIN_EDGE_MARGIN, (ref, trg, s, edge)
    ok("auto-derived sliders always satisfy the edge guard")

    try:
        sf.make_lever("a", "t", "a", "u", 1, 2, impact={"kind": "magic"})
    except ValueError as exc:
        assert "impact kind" in str(exc), exc
    else:
        raise AssertionError("unknown impact kind accepted")
    ok("unknown impact kind rejected")

    # --- a lever answered in parts (D61) -------------------------------------
    def cut(pid, target=None, lo=0, hi=20):
        return sf.make_lever_part(pid, pid, "% less energy per km", 0,
                                  {"min": lo, "max": hi, "step": 1}, target_value=target)

    assert abs(sf.combine_parts("cuts", 100, [10, 10]) - 81.0) < 1e-12
    assert sf.combine_parts("sum", 0, [1.5, 2.5]) == 4.0
    ok("combine_parts: cuts compound, sums add")

    # a part may speak its own unit (D64): km/h less, or an average mass in kg
    kmh = sf.make_lever_part("speed", "Speed", "km/h", 0, {"min": 0, "max": 20, "step": 1},
                             cut_per_unit=0.6)
    kg = sf.make_lever_part("mass", "Mass", "kg", 1400, {"min": 900, "max": 1800, "step": 10},
                            cut_per_unit=-0.04)
    assert abs(sf.part_cut(kmh, 10) - 6.0) < 1e-12
    assert abs(sf.part_cut(kg, 1300) - 4.0) < 1e-12           # 100 kg lighter: -4 %
    assert abs(sf.part_cut(kg, 1500) + 4.0) < 1e-12           # 100 kg heavier: +4 %
    assert abs(sf.combine_parts("cuts", 100, [10, 1300], [kmh, kg]) - 100 * 0.94 * 0.96) < 1e-9
    assert abs(sf.combine_parts("cuts", 100, [0, 1400], [kmh, kg]) - 100) < 1e-9
    unit_lever = sf.make_lever("u", "t", "u", "% of 2019", 100, 100 * 0.94 * 0.96,
                               slider={"min": 60, "max": 125, "step": 1},
                               parts=[sf.make_lever_part("speed", "Speed", "km/h", 0,
                                                         {"min": 0, "max": 20, "step": 1},
                                                         target_value=10, cut_per_unit=0.6),
                                      sf.make_lever_part("mass", "Mass", "kg", 1400,
                                                         {"min": 900, "max": 1800, "step": 10},
                                                         target_value=1300, cut_per_unit=-0.04)],
                               combine="cuts")
    assert unit_lever["parts"][1]["cutPerUnit"] == -0.04
    ok("parts in km/h and kg convert to cuts, both ways round")

    parted = sf.make_lever(
        "car-energy", "inland-mobility", "Car energy", "% of 2019", 100, 75,
        slider={"min": 45, "max": 115, "step": 1},
        parts=[cut("speed", 8), cut("eco", 7, hi=15),
               cut("size", 100 * (1 - 0.75 / (0.92 * 0.93)), lo=-15, hi=30)],
        combine="cuts")
    assert parted["combine"] == "cuts"
    assert [p["answerId"] for p in parted["parts"]] == [
        "car-energy__speed", "car-energy__eco", "car-energy__size"]
    ok("make_lever exports parts with their API answer ids")

    def refused(fragment, **kw):
        args = dict(slider={"min": 45, "max": 115, "step": 1},
                    parts=[cut("a"), cut("b")], combine="cuts")
        args.update(kw)
        try:
            sf.make_lever("x", "t", "x", "% of 2019", 100, 75, **args)
        except ValueError as exc:
            assert fragment in str(exc), (fragment, exc)
        else:
            raise AssertionError(f"accepted although it should fail with {fragment!r}")

    refused("combine must be", combine="product")
    refused("at least two", parts=[cut("a")])
    refused("duplicate part ids", parts=[cut("a"), cut("a")])
    refused("go together", combine=None)
    refused("every part carries a target", parts=[cut("a", 10), cut("b")])
    refused("not the lever's target", parts=[cut("a", 10), cut("b", 10)])
    refused("outside the lever's range", parts=[cut("a", hi=40), cut("b", hi=40)])
    refused("not accepted by the API", parts=[cut("a" * 70), cut("b")])
    ok("parts that cannot produce the lever's own values are refused")

    try:
        cut("speed", 1)
    except ValueError as exc:
        assert "slider end" in str(exc), exc
    else:
        raise AssertionError("a part target at an end stop was accepted")
    ok("edge-margin guard applies to every part")

    with tempfile.TemporaryDirectory() as tmp:
        levers_js = os.path.join(tmp, "levers_transport.js")
        clash = sf.make_lever("car-energy__speed", "t", "x", "u", 1.22, 2.00,
                              slider={"min": 1.0, "max": 2.5, "step": 0.05})
        try:
            sf.write_levers_js("transport", [parted, clash], out_path=levers_js)
        except ValueError as exc:
            assert "used twice" in str(exc), exc
        else:
            raise AssertionError("a lever id equal to a part answer id was accepted")
        ok("part answer ids cannot collide with a lever id")

    with tempfile.TemporaryDirectory() as tmp:
        levers_js = os.path.join(tmp, "levers_transport.js")
        try:
            sf.write_levers_js("transport", [lever, dict(lever)], out_path=levers_js)
        except ValueError as exc:
            assert "duplicate" in str(exc), exc
        else:
            raise AssertionError("duplicate lever id accepted")
        ok("duplicate lever id rejected")

        sf.write_levers_js("transport", [lever], model={"carTwh2050": 8.9},
                           title="Mobility & transport", out_path=levers_js)
        data = payload_of(levers_js)
        assert data["levers"]["car-occupancy"]["targetValue"] == 2.0
        assert data["model"]["carTwh2050"] == 8.9
        assert 'window.NW_LEVERS["transport"]' in open(levers_js, encoding="utf-8").read()
        ok("write_levers_js emits a parseable window.NW_LEVERS payload")

        series = sf.make_history_series("pkm", "Ground mobility", "km/day",
                                        [2000, 2001], [33.1, float("nan")],
                                        source="JRC-IDEES-2023")
        assert series["y"] == [33.1, None]
        history_js = os.path.join(tmp, "history_transport.js")
        sf.write_history_js("transport", [series], out_path=history_js)
        hist = payload_of(history_js)
        assert hist["series"]["pkm"]["y"] == [33.1, None]
        ok("history export converts NaN to null")

        try:
            sf.make_history_series("bad", "x", "u", [2000, 2001], [1.0])
        except ValueError as exc:
            assert "years vs" in str(exc), exc
        else:
            raise AssertionError("mismatched series lengths accepted")
        ok("mismatched years/values length rejected")

    print(f"\n{len(passed)} helper tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
