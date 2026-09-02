#!/usr/bin/env python
"""End-to-end contract test for the workshop value-collection API.

The point is that ONE test runs against BOTH implementations, so the PHP in
website/api/ and the SQLite shim in scripts/dev_api.py cannot drift apart:

    python scripts/dev_api.py --port 8787 --quiet &
    python scripts/test_workshop_api.py --base http://127.0.0.1:8787
    python scripts/test_workshop_api.py --base https://negawatt.squoilin.eu/api \\
                                        --admin-key <key from config.php>

It exercises the happy path (create → three groups join → answers → results →
reveal) and the failure modes that matter (bad code, wrong token, gating,
validation, closed session, solo exclusion). It writes only into sessions it
creates, and closes them at the end.
"""
import argparse
import json
import sys
import urllib.error
import urllib.request

TOPIC = "inland-mobility"
LEVERS = ["ground-km-day", "car-share", "car-occupancy", "car-energy",
          "bike-km-day", "freight-tkm", "truck-share", "truck-load"]


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="e.g. http://127.0.0.1:8787")
    ap.add_argument("--admin-key", default="dev-admin-key",
                    help="value of admin_key in config.php (topic-scope results)")
    ap.add_argument("--keep", action="store_true", help="do not close the test sessions")
    args = ap.parse_args()

    api = Client(args.base)
    run = Runner()
    print(f"contract test against {api.base}\n")

    # ---------------------------------------------------------------- selftest
    status, doc = api.call("GET", "/selftest.php")
    run.equal("selftest returns 200", status, 200)
    run.check("selftest reports ok", bool(doc.get("ok")), json.dumps(doc)[:200])
    run.check("all four tables exist",
              set(doc.get("tables", {})) == {"ws_sessions", "ws_groups", "ws_answers",
                                             "ws_answer_log"},
              str(doc.get("tables")))
    impl = doc.get("implementation") or f"php {doc.get('php')}"
    print(f"        implementation: {impl}\n")

    # ------------------------------------------------------ create the session
    status, sess = api.call("POST", "/session.php",
                            {"topic": TOPIC, "label": "contract test"})
    run.equal("create session returns 201", status, 201)
    code, admin = sess.get("code"), sess.get("admin_token")
    run.check("session code has the right shape",
              isinstance(code, str) and len(code) == 4
              and all(c in "ABCDEFGHJKMNPQRSTUVWXYZ23456789" for c in code), repr(code))
    run.check("an admin token was issued", isinstance(admin, str) and len(admin) >= 32,
              repr(admin))
    run.equal("reveal starts before the first lever", sess.get("reveal_step"), -1)

    status, doc = api.call("POST", "/session.php", {"topic": "not a topic!"})
    run.equal("a malformed topic is rejected", status, 400)

    # ------------------------------------------------------------- groups join
    groups = []
    for name in ["Table 1", "Table 2", "Les cyclistes"]:
        status, doc = api.call("POST", "/group.php", {"code": code, "name": name})
        run.equal(f"group {name!r} joins", status, 200)
        run.equal(f"group {name!r} gets the session topic", doc.get("topic"), TOPIC)
        groups.append({"name": name, "id": doc.get("group_id"), "token": doc.get("token")})
    run.check("group ids are distinct", len({g["id"] for g in groups}) == 3,
              str([g["id"] for g in groups]))

    status, doc = api.call("POST", "/group.php", {"code": code, "name": "Table 1"})
    run.equal("re-joining an existing name is allowed", status, 200)
    run.equal("re-joining returns the same group", doc.get("group_id"), groups[0]["id"])
    run.check("re-joining is flagged as such", doc.get("rejoined") is True, str(doc))
    groups[0]["token"] = doc.get("token")

    status, doc = api.call("POST", "/group.php", {"code": "ZZZZ", "name": "nobody"})
    run.equal("joining an unknown session is a 404", status, 404)
    status, doc = api.call("POST", "/group.php", {"code": code, "name": ""})
    run.equal("an empty group name is rejected", status, 400)

    # ------------------------------------------- joining from a link, no typing
    # This is the flow a participant actually gets: a link carrying the slug, a
    # group named by the server, and straight into question 1.
    slug = "contract-test-" + code.lower()
    status, doc = api.call("POST", "/session.php",
                           {"topic": TOPIC, "label": "slug test", "slug": slug})
    run.equal("a session can be created with a slug", status, 201)
    run.equal("the slug comes back", doc.get("slug"), slug)
    slug_code, slug_admin = doc.get("code"), doc.get("admin_token")

    status, doc = api.call("POST", "/session.php",
                           {"topic": TOPIC, "label": "dup", "slug": slug})
    run.equal("a duplicate slug is refused", status, 409)

    status, doc = api.call("POST", "/session.php",
                           {"topic": TOPIC, "label": "bad", "slug": "Not A Slug"})
    run.equal("a malformed slug is refused", status, 400)

    status, doc = api.call("GET", "/session.php", slug=slug)
    run.equal("a session resolves by slug", status, 200)
    run.equal("...to the right session", doc.get("code"), slug_code)
    status, doc = api.call("GET", "/session.php", slug="no-such-workshop")
    run.equal("an unknown slug is a 404", status, 404)

    auto = []
    for _ in range(3):
        status, doc = api.call("POST", "/group.php",
                               {"slug": slug, "auto_name": True, "name_prefix": "Groupe"})
        if status != 200:
            run.check("auto-named join", False, f"{status} {doc}")
            break
        auto.append(doc)
    run.equal("three devices join from the link", len(auto), 3)
    run.equal("the server names them in order",
              [g.get("name") for g in auto], ["Groupe 1", "Groupe 2", "Groupe 3"])
    run.check("each gets its own group",
              len({g.get("group_id") for g in auto}) == 3, str(auto))
    run.check("the join returns the topic, so the page knows what to show",
              all(g.get("topic") == TOPIC for g in auto), str([g.get("topic") for g in auto]))

    first = auto[0]
    status, doc = api.call("POST", "/answer.php", {
        "code": slug_code, "group_id": first["group_id"], "token": first["token"],
        "lever_id": "car-occupancy", "value": 1.9})
    run.equal("a link-joined group can answer", status, 200)

    status, doc = api.call("POST", "/rename.php", {
        "slug": slug, "group_id": first["group_id"], "token": first["token"],
        "name": "Table du fond"})
    run.equal("a group can rename itself", status, 200)
    run.equal("the new name is returned", doc.get("name"), "Table du fond")
    status, doc = api.call("POST", "/rename.php", {
        "slug": slug, "group_id": first["group_id"], "token": "wrong" * 8,
        "name": "Pirate"})
    run.equal("renaming needs the group's own token", status, 403)
    status, doc = api.call("POST", "/rename.php", {
        "slug": slug, "group_id": auto[1]["group_id"], "token": auto[1]["token"],
        "name": "Table du fond"})
    run.equal("renaming onto another group's name is refused", status, 409)

    status, doc = api.call("GET", "/results.php", code=slug_code, admin_token=slug_admin)
    names = sorted(g["name"] for g in doc.get("groups", []))
    run.equal("the rename is reflected in the results",
              names, ["Groupe 2", "Groupe 3", "Table du fond"])
    api.call("POST", "/reveal.php",
             {"code": slug_code, "admin_token": slug_admin, "step": -1, "close": True})

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
                "code": code, "group_id": group["id"], "token": group["token"],
                "lever_id": lever, "value": value, "confidence": 2,
                "condition": f"{group['name']}: it would take real policy"})
            if status != 200 or not doc.get("ok"):
                run.check(f"answer {lever} for {group['name']}", False, f"{status} {doc}")
                break
            written += 1
    run.equal("all 24 answers accepted", written, 24)

    # the log must keep every move, so overwriting is allowed and traced
    status, doc = api.call("POST", "/answer.php", {
        "code": code, "group_id": groups[0]["id"], "token": groups[0]["token"],
        "lever_id": "car-occupancy", "value": 1.75, "confidence": 3})
    run.equal("a group may change its mind", status, 200)

    bad = [
        ({"code": code, "group_id": groups[0]["id"], "token": "wrong" * 8,
          "lever_id": "car-share", "value": 60}, 403, "a wrong token is refused"),
        ({"code": code, "group_id": 999999, "token": groups[0]["token"],
          "lever_id": "car-share", "value": 60}, 404, "an unknown group is refused"),
        ({"code": code, "group_id": groups[0]["id"], "token": groups[0]["token"],
          "lever_id": "car-share", "value": "not a number"}, 400,
         "a non-numeric value is refused"),
        ({"code": code, "group_id": groups[0]["id"], "token": groups[0]["token"],
          "lever_id": "bad lever id", "value": 1}, 400, "a malformed lever id is refused"),
        ({"code": code, "group_id": groups[0]["id"], "token": groups[0]["token"],
          "lever_id": "car-share", "value": 60, "confidence": 9}, 400,
         "an out-of-range confidence is refused"),
    ]
    for body, want, label in bad:
        status, _ = api.call("POST", "/answer.php", body)
        run.equal(label, status, want)

    # ----------------------------------------------------------------- results
    status, doc = api.call("GET", "/results.php", code=code)
    run.equal("session results are gated without the admin token", status, 403)

    status, doc = api.call("GET", "/results.php", code=code, admin_token=admin)
    run.equal("session results with the admin token", status, 200)
    run.equal("scope is the session", doc.get("scope"), "session")
    run.equal("three groups are reported", len(doc.get("groups", [])), 3)
    run.equal("24 answers are reported", len(doc.get("answers", [])), 24)
    occupancy = [a for a in doc["answers"]
                 if a["lever_id"] == "car-occupancy" and a["group_id"] == groups[0]["id"]]
    run.check("the changed answer is the one stored",
              len(occupancy) == 1 and abs(occupancy[0]["value"] - 1.75) < 1e-9,
              str(occupancy))
    run.check("conditions come back with the answers",
              any(a.get("condition") for a in doc["answers"]), "no condition text")
    run.check("confidence comes back with the answers",
              all(a.get("confidence") in (1, 2, 3) for a in doc["answers"]), "bad confidence")

    status, doc = api.call("GET", "/results.php", topic=TOPIC)
    run.equal("topic results need the facilitator key", status, 403)
    status, doc = api.call("GET", "/results.php", topic=TOPIC, admin_key=args.admin_key)
    if status == 403:
        print("  SKIP  topic-scope aggregation (admin key not supplied/matching)")
    else:
        run.equal("topic results with the facilitator key", status, 200)
        run.equal("scope is the topic", doc.get("scope"), "topic")
        run.check("our session is in the topic aggregation",
                  code in [s["code"] for s in doc.get("sessions", [])],
                  str([s["code"] for s in doc.get("sessions", [])]))

        # a solo session must stay out of the collective totals
        _, solo = api.call("POST", "/session.php",
                           {"topic": TOPIC, "label": "solo test", "mode": "solo"})
        status, doc = api.call("GET", "/results.php", topic=TOPIC, admin_key=args.admin_key)
        run.check("a solo session is excluded from the topic aggregation",
                  solo["code"] not in [s["code"] for s in doc.get("sessions", [])],
                  str([s["code"] for s in doc.get("sessions", [])]))
        api.call("POST", "/reveal.php", {"code": solo["code"],
                                         "admin_token": solo["admin_token"],
                                         "step": -1, "close": True})

    # ------------------------------------------------------------------ reveal
    status, doc = api.call("POST", "/reveal.php", {"code": code, "step": 0})
    run.equal("advancing the reveal needs the admin token", status, 403)
    status, doc = api.call("POST", "/reveal.php",
                           {"code": code, "admin_token": admin, "step": 3})
    run.equal("the facilitator advances the reveal", status, 200)
    run.equal("the step is stored", doc.get("reveal_step"), 3)
    status, doc = api.call("GET", "/session.php", code=code)
    run.equal("the session reports the new step", doc.get("reveal_step"), 3)
    run.equal("group progress is visible to participants",
              sorted(g["answered"] for g in doc.get("groups", [])), [8, 8, 8])

    status, doc = api.call("POST", "/reveal.php",
                           {"code": code, "admin_token": admin, "step": 5000})
    run.equal("an absurd step is rejected", status, 400)

    # ------------------------------------------------------------------- close
    if not args.keep:
        status, doc = api.call("POST", "/reveal.php",
                               {"code": code, "admin_token": admin, "step": 7, "close": True})
        run.equal("the facilitator closes the session", status, 200)
        status, doc = api.call("POST", "/answer.php", {
            "code": code, "group_id": groups[0]["id"], "token": groups[0]["token"],
            "lever_id": "car-share", "value": 60})
        run.equal("a closed session refuses new answers", status, 409)
        status, doc = api.call("GET", "/results.php", code=code, admin_token=admin)
        run.equal("a closed session still returns its results", status, 200)

    print(f"\n{run.passed} passed, {len(run.failed)} failed")
    if run.failed:
        for item in run.failed:
            print("  -", item)
        return 1
    print(f"session {code} " + ("kept open" if args.keep else "closed"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
