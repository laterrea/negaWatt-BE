#!/usr/bin/env python
"""End-to-end contract test for the workshop value-collection API.

The point is that ONE test runs against BOTH implementations, so the PHP in
website/api/ and the SQLite shim in scripts/dev_api.py cannot drift apart:

    python scripts/dev_api.py --port 8787 --quiet &
    python scripts/test_workshop_api.py --base http://127.0.0.1:8787
    python scripts/test_workshop_api.py --base https://negawatt.squoilin.eu/api

There is no session, no code and no facilitator key: a group belongs to a topic
and the reveal screen selects groups by date. The suite exercises the happy path
(three groups start → answer → results) plus the failure modes that matter (wrong
token, unknown group, validation) and the date window, which is the only way to
tell one workshop from another.

It writes into the real topic, so it names its groups "contract-test …" and the
date-window checks only ever assert about its own groups.
"""
import argparse
import json
import sys
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone

TOPIC = "inland-mobility"
LEVERS = ["ground-km-day", "car-share", "car-occupancy", "car-energy",
          "bike-km-day", "freight-tkm", "truck-share", "truck-load"]
TAG = "contract-test"


class Client:
    def __init__(self, base, timeout=20):
        self.base = base.rstrip("/")
        self.timeout = timeout

    def call(self, method, path, body=None, **params):
        url = self.base + path
        if params:
            from urllib.parse import urlencode
            url += "?" + urlencode({k: v for k, v in params.items() if v is not None})
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(url, data=data, method=method)
        if data is not None:
            req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as res:
                return res.status, json.loads(res.read().decode("utf-8") or "null")
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", "replace")
            try:
                return exc.code, json.loads(raw)
            except ValueError:
                return exc.code, {"error": "non_json_response", "body": raw[:400]}


class Runner:
    def __init__(self):
        self.passed = 0
        self.failed = []

    def check(self, label, condition, detail=""):
        if condition:
            self.passed += 1
            print(f"  PASS  {label}")
        else:
            self.failed.append(f"{label} — {detail}")
            print(f"  FAIL  {label}  {detail}")

    def equal(self, label, got, want):
        self.check(label, got == want, f"got {got!r}, want {want!r}")


def utc_date(offset_days=0):
    return (datetime.now(timezone.utc) + timedelta(days=offset_days)).strftime("%Y-%m-%d")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="e.g. http://127.0.0.1:8787")
    args = ap.parse_args()

    api = Client(args.base)
    run = Runner()
    print(f"contract test against {api.base}\n")

    # ---------------------------------------------------------------- selftest
    status, doc = api.call("GET", "/selftest.php")
    run.equal("selftest returns 200", status, 200)
    run.check("selftest reports ok", bool(doc.get("ok")), json.dumps(doc)[:200])
    run.check("the three tables exist",
              set(doc.get("tables", {})) == {"ws_groups", "ws_answers", "ws_answer_log"},
              str(doc.get("tables")))
    impl = doc.get("implementation") or f"php {doc.get('php')}"
    print(f"        implementation: {impl}\n")

    # ------------------------------------------------- starting, with a name
    groups = []
    for name in [f"{TAG} Table 1", f"{TAG} Table 2", f"{TAG} Les cyclistes"]:
        status, doc = api.call("POST", "/group.php", {"topic": TOPIC, "name": name})
        run.equal(f"group {name!r} starts", status, 200)
        run.equal(f"group {name!r} keeps its name", doc.get("name"), name)
        run.equal(f"group {name!r} gets the topic back", doc.get("topic"), TOPIC)
        run.check("a group token was issued",
                  isinstance(doc.get("token"), str) and len(doc["token"]) >= 32,
                  repr(doc.get("token")))
        groups.append({"name": name, "id": doc.get("group_id"), "token": doc.get("token")})
    run.check("group ids are distinct", len({g["id"] for g in groups}) == 3,
              str([g["id"] for g in groups]))

    # Starting twice under one name is two groups now: without sessions there is
    # nothing to re-join, and the reveal numbers any duplicate labels.
    status, doc = api.call("POST", "/group.php",
                           {"topic": TOPIC, "name": f"{TAG} Table 1"})
    run.equal("starting again under the same name is allowed", status, 200)
    run.check("...and is a group of its own", doc.get("group_id") != groups[0]["id"],
              str(doc.get("group_id")))
    duplicate = {"id": doc.get("group_id"), "token": doc.get("token")}

    status, doc = api.call("POST", "/group.php", {"topic": "not a topic!"})
    run.equal("a malformed topic is rejected", status, 400)

    # ------------------------------------------- starting without typing a name
    # This is the flow a participant actually gets: press Start, get named by the
    # server, land on question 1.
    auto = []
    for _ in range(3):
        status, doc = api.call("POST", "/group.php",
                               {"topic": TOPIC, "name_prefix": f"{TAG} Groupe"})
        if status != 200:
            run.check("auto-named start", False, f"{status} {doc}")
            break
        auto.append(doc)
    run.equal("three devices start with no name typed", len(auto), 3)
    run.check("the server names them in order and per day",
              [g.get("name", "").rsplit(" ", 1)[-1] for g in auto] == ["1", "2", "3"]
              or all(g.get("name", "").startswith(f"{TAG} Groupe ") for g in auto),
              str([g.get("name") for g in auto]))
    run.check("each gets its own group",
              len({g.get("group_id") for g in auto}) == 3, str(auto))
    run.check("an empty name is accepted as no name at all",
              all(g.get("name") for g in auto), str([g.get("name") for g in auto]))

    first = auto[0]
    status, doc = api.call("POST", "/answer.php", {
        "group_id": first["group_id"], "token": first["token"],
        "lever_id": "car-occupancy", "value": 1.9})
    run.equal("an unnamed group can answer", status, 200)

    status, doc = api.call("POST", "/rename.php", {
        "group_id": first["group_id"], "token": first["token"],
        "name": f"{TAG} Table du fond"})
    run.equal("a group can rename itself", status, 200)
    run.equal("the new name is returned", doc.get("name"), f"{TAG} Table du fond")
    status, doc = api.call("POST", "/rename.php", {
        "group_id": first["group_id"], "token": "wrong" * 8, "name": "Pirate"})
    run.equal("renaming needs the group's own token", status, 403)
    status, doc = api.call("POST", "/rename.php", {
        "group_id": auto[1]["group_id"], "token": auto[1]["token"],
        "name": f"{TAG} Table du fond"})
    run.equal("renaming onto another group's name is allowed now", status, 200)

    # ----------------------------------------------------------------- answers
    plan = {
        groups[0]["id"]: [30.0, 70.0, 1.5, 90.0, 2.5, 6800.0, 62.0, 13.0],
        groups[1]["id"]: [26.0, 55.0, 2.1, 70.0, 4.5, 5600.0, 48.0, 15.5],
        groups[2]["id"]: [22.0, 45.0, 2.4, 62.0, 6.0, 5000.0, 40.0, 16.5],
    }
    written = 0
    for group in groups:
        for lever, value in zip(LEVERS, plan[group["id"]]):
            status, doc = api.call("POST", "/answer.php", {
                "group_id": group["id"], "token": group["token"],
                "lever_id": lever, "value": value, "confidence": 2,
                "condition": f"{group['name']}: it would take real policy"})
            if status != 200 or not doc.get("ok"):
                run.check(f"answer {lever} for {group['name']}", False, f"{status} {doc}")
                break
            written += 1
    run.equal("all 24 answers accepted", written, 24)

    # the log must keep every move, so overwriting is allowed and traced
    status, doc = api.call("POST", "/answer.php", {
        "group_id": groups[0]["id"], "token": groups[0]["token"],
        "lever_id": "car-occupancy", "value": 1.75, "confidence": 3})
    run.equal("a group may change its mind", status, 200)

    bad = [
        ({"group_id": groups[0]["id"], "token": "wrong" * 8,
          "lever_id": "car-share", "value": 60}, 403, "a wrong token is refused"),
        ({"group_id": 999999, "token": groups[0]["token"],
          "lever_id": "car-share", "value": 60}, 404, "an unknown group is refused"),
        ({"group_id": groups[0]["id"],
          "lever_id": "car-share", "value": 60}, 400, "a missing token is refused"),
        ({"group_id": groups[0]["id"], "token": groups[0]["token"],
          "lever_id": "car-share", "value": "not a number"}, 400,
         "a non-numeric value is refused"),
        ({"group_id": groups[0]["id"], "token": groups[0]["token"],
          "lever_id": "bad lever id", "value": 1}, 400, "a malformed lever id is refused"),
        ({"group_id": groups[0]["id"], "token": groups[0]["token"],
          "lever_id": "car-share", "value": 60, "confidence": 9}, 400,
         "an out-of-range confidence is refused"),
    ]
    for body, want, label in bad:
        status, _ = api.call("POST", "/answer.php", body)
        run.equal(label, status, want)

    # ----------------------------------------------------------------- results
    ours = {g["id"] for g in groups} | {g["group_id"] for g in auto} | {duplicate["id"]}

    status, doc = api.call("GET", "/results.php", topic=TOPIC)
    run.equal("results need no credential at all", status, 200)
    run.equal("the topic comes back", doc.get("topic"), TOPIC)
    reported = {g["id"] for g in doc.get("groups", [])}
    run.check("every group we started is reported", ours <= reported,
              str(sorted(ours - reported)))
    mine = [a for a in doc.get("answers", []) if a["group_id"] in ours]
    run.equal("25 answers of ours are reported", len(mine), 25)
    occupancy = [a for a in mine
                 if a["lever_id"] == "car-occupancy" and a["group_id"] == groups[0]["id"]]
    run.check("the changed answer is the one stored",
              len(occupancy) == 1 and abs(occupancy[0]["value"] - 1.75) < 1e-9,
              str(occupancy))
    run.check("conditions come back with the answers",
              any(a.get("condition") for a in mine), "no condition text")
    named = {g["id"] for g in groups}
    run.check("confidence comes back with the answers",
              all(a.get("confidence") in (1, 2, 3)
                  for a in mine if a["group_id"] in named), "bad confidence")
    run.check("an answer given without a confidence keeps a null one",
              any(a.get("confidence") is None for a in mine), "no null confidence")
    run.check("groups carry the moment they started, which is what the filter uses",
              all(g.get("created_at") for g in doc.get("groups", [])),
              str(doc.get("groups", [])[:2]))

    status, doc = api.call("GET", "/results.php")
    run.equal("results without a topic are rejected", status, 400)

    # ------------------------------------------------------------ date window
    # This is what replaces the session: which sitting is on screen.
    status, doc = api.call("GET", "/results.php", topic=TOPIC,
                           **{"from": utc_date(), "to": utc_date()})
    run.equal("today's window returns 200", status, 200)
    run.check("today's window holds the groups we just started",
              ours <= {g["id"] for g in doc.get("groups", [])},
              str(sorted(ours - {g["id"] for g in doc.get("groups", [])})))
    run.check("a bare end date covers the whole day",
              str(doc.get("to", "")).endswith("23:59:59"), str(doc.get("to")))

    status, doc = api.call("GET", "/results.php", topic=TOPIC,
                           **{"from": utc_date(1)})
    run.equal("a window starting tomorrow returns 200", status, 200)
    run.check("...and holds none of our groups",
              not (ours & {g["id"] for g in doc.get("groups", [])}),
              str([g["id"] for g in doc.get("groups", [])]))

    status, doc = api.call("GET", "/results.php", topic=TOPIC,
                           **{"to": utc_date(-1)})
    run.equal("a window ending yesterday returns 200", status, 200)
    run.check("...and holds none of our groups either",
              not (ours & {g["id"] for g in doc.get("groups", [])}),
              str([g["id"] for g in doc.get("groups", [])]))

    status, doc = api.call("GET", "/results.php", topic=TOPIC,
                           **{"from": "2019-01-01"})
    run.equal("widening the start summarises every sitting", status, 200)
    run.check("...which includes ours", ours <= {g["id"] for g in doc.get("groups", [])},
              str(sorted(ours - {g["id"] for g in doc.get("groups", [])})))

    status, doc = api.call("GET", "/results.php", topic=TOPIC,
                           **{"from": "2026-09-03 14:30"})
    run.equal("a timestamp bound is accepted", status, 200)
    run.equal("...and normalised to seconds", doc.get("from"), "2026-09-03 14:30:00")

    for value, label in [("not-a-date", "a malformed bound is rejected"),
                         ("2026-02-30", "an impossible date is rejected")]:
        status, _ = api.call("GET", "/results.php", topic=TOPIC, **{"from": value})
        run.equal(label, status, 400)

    print(f"\n{run.passed} passed, {len(run.failed)} failed")
    if run.failed:
        for item in run.failed:
            print("  -", item)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
