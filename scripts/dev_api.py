#!/usr/bin/env python
"""negaWatt-BE workshop API — SQLite implementation of the same contract as website/api/*.php.

Two jobs:

  1. develop and test the workshop UI without PHP or MySQL;
  2. run a workshop from a laptop with no internet at all — start this, point the
     participant pages at it, and everything works on the local network.

    python scripts/dev_api.py --port 8787
    python scripts/dev_api.py --port 8787 --db /tmp/ws.sqlite --host 0.0.0.0

Then open the pages with ?api=http://<this machine>:8787 once; the base URL is
remembered in the browser.

The contract lives in scripts/test_workshop_api.py, which is run against this
server *and* against the deployed PHP so the two cannot drift apart.
"""
import argparse
import hashlib
import http.server
import json
import os
import re
import secrets
import socketserver
import sqlite3
import sys
import threading
from datetime import datetime, timezone
from urllib.parse import urlparse, parse_qs

CODE_ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"      # no O/0/I/1/L
CODE_LENGTH = 4
MAX_BODY = 8192
DEFAULT_ADMIN_KEY = "dev-admin-key"

SCHEMA = """
CREATE TABLE IF NOT EXISTS ws_sessions (
  code TEXT PRIMARY KEY,
  slug TEXT UNIQUE,
  topic TEXT NOT NULL,
  label TEXT NOT NULL DEFAULT '',
  mode TEXT NOT NULL DEFAULT 'group',
  results_public INTEGER NOT NULL DEFAULT 0,
  reveal_step INTEGER NOT NULL DEFAULT -1,
  admin_token_hash TEXT NOT NULL,
  created_at TEXT NOT NULL,
  closed_at TEXT
);
CREATE TABLE IF NOT EXISTS ws_groups (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_code TEXT NOT NULL REFERENCES ws_sessions(code) ON DELETE CASCADE,
  name TEXT NOT NULL,
  token_hash TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  UNIQUE (session_code, name)
);
CREATE TABLE IF NOT EXISTS ws_answers (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  group_id INTEGER NOT NULL REFERENCES ws_groups(id) ON DELETE CASCADE,
  lever_id TEXT NOT NULL,
  value REAL NOT NULL,
  confidence INTEGER,
  condition_text TEXT,
  updated_at TEXT NOT NULL,
  UNIQUE (group_id, lever_id)
);
CREATE TABLE IF NOT EXISTS ws_answer_log (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  group_id INTEGER NOT NULL,
  lever_id TEXT NOT NULL,
  value REAL NOT NULL,
  confidence INTEGER,
  created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_topic_open ON ws_sessions(topic, closed_at);
CREATE INDEX IF NOT EXISTS idx_group_lever ON ws_answer_log(group_id, lever_id);
"""


class ApiError(Exception):
    def __init__(self, error, status=400, **extra):
        super().__init__(error)
        self.payload = dict(extra)
        self.payload["error"] = error
        self.status = status


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def check_str(value, maximum, field, required=True):
    if value is None or value == "":
        if required:
            raise ApiError("missing_field", 400, field=field)
        return None
    if not isinstance(value, str):
        raise ApiError("invalid_field", 400, field=field)
    value = value.strip()[:maximum]
    if not value:
        if required:
            raise ApiError("missing_field", 400, field=field)
        return None
    return value


def check_id(value, field):
    value = "" if value is None else str(value)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", value):
        raise ApiError("invalid_field", 400, field=field)
    return value


def check_code(value):
    value = str(value or "").strip().upper()
    if not re.fullmatch(f"[{CODE_ALPHABET}]{{{CODE_LENGTH},8}}", value):
        raise ApiError("invalid_code", 400)
    return value


def check_slug(value, required=True):
    value = str(value or "").strip().lower()
    if not value:
        if required:
            raise ApiError("missing_field", 400, field="slug")
        return None
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{1,63}", value):
        raise ApiError("invalid_slug", 400)
    return value


def check_number(value, field):
    if isinstance(value, bool) or value is None:
        raise ApiError("invalid_field", 400, field=field)
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ApiError("invalid_field", 400, field=field)
    if number != number or number in (float("inf"), float("-inf")):
        raise ApiError("invalid_field", 400, field=field)
    return number


class Store:
    def __init__(self, path):
        self.path = path
        self.lock = threading.Lock()
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def query(self, sql, args=()):
        with self.lock:
            return [dict(r) for r in self.conn.execute(sql, args).fetchall()]

    def one(self, sql, args=()):
        rows = self.query(sql, args)
        return rows[0] if rows else None

    def run(self, sql, args=()):
        with self.lock:
            cur = self.conn.execute(sql, args)
            self.conn.commit()
            return cur

    def new_code(self):
        for _ in range(40):
            code = "".join(secrets.choice(CODE_ALPHABET) for _ in range(CODE_LENGTH))
            if not self.one("SELECT 1 FROM ws_sessions WHERE code = ?", (code,)):
                return code
        raise ApiError("code_space_exhausted", 503)

    def session(self, code):
        row = self.one("SELECT * FROM ws_sessions WHERE code = ?", (code,))
        if row is None:
            raise ApiError("unknown_session", 404)
        return row

    def session_by(self, code, slug):
        """Participants arrive with a slug from a link; a projector gives a code."""
        if slug:
            row = self.one("SELECT * FROM ws_sessions WHERE slug = ?", (slug,))
            if row is None:
                raise ApiError("unknown_session", 404)
            return row
        if not code:
            raise ApiError("missing_field", 400, field="code or slug")
        return self.session(code)


def is_admin(session, token):
    return bool(token) and secrets.compare_digest(session["admin_token_hash"], sha256(token))


# --------------------------------------------------------------------- handlers

def handle_session_post(store, body, _query, admin_key):
    topic = check_id(body.get("topic"), "topic")
    label = check_str(body.get("label", ""), 160, "label", required=False) or ""
    mode = body.get("mode", "group")
    if mode not in ("group", "solo"):
        mode = "group"
    results_public = 1 if body.get("results_public") else 0
    slug = check_slug(body.get("slug"), required=False)
    if slug and store.one("SELECT code FROM ws_sessions WHERE slug = ?", (slug,)):
        raise ApiError("slug_taken", 409, slug=slug)
    code = store.new_code()
    token = secrets.token_hex(24)
    store.run(
        "INSERT INTO ws_sessions (code, slug, topic, label, mode, results_public,"
        " reveal_step, admin_token_hash, created_at) VALUES (?, ?, ?, ?, ?, ?, -1, ?, ?)",
        (code, slug, topic, label, mode, results_public, sha256(token), now()))
    return 201, {"code": code, "slug": slug, "admin_token": token, "topic": topic,
                 "label": label, "mode": mode, "reveal_step": -1}


def handle_session_get(store, _body, query, admin_key):
    raw_code = query.get("code", [None])[0]
    raw_slug = query.get("slug", [None])[0]
    session = store.session_by(check_code(raw_code) if raw_code else None,
                               check_slug(raw_slug) if raw_slug else None)
    code = session["code"]
    rows = store.query(
        "SELECT g.id, g.name, g.updated_at, COUNT(a.id) AS answered FROM ws_groups g"
        " LEFT JOIN ws_answers a ON a.group_id = g.id WHERE g.session_code = ?"
        " GROUP BY g.id, g.name, g.updated_at ORDER BY g.id", (code,))
    groups = [{"id": r["id"], "name": r["name"], "answered": r["answered"],
               "updated_at": r["updated_at"]} for r in rows]
    return 200, {"code": session["code"], "slug": session["slug"],
                 "topic": session["topic"],
                 "label": session["label"], "mode": session["mode"],
                 "reveal_step": session["reveal_step"],
                 "results_public": bool(session["results_public"]),
                 "closed": session["closed_at"] is not None,
                 "created_at": session["created_at"], "groups": groups}


def handle_group(store, body, _query, admin_key):
    session = store.session_by(
        check_code(body["code"]) if body.get("code") else None,
        check_slug(body["slug"]) if body.get("slug") else None)
    code = session["code"]
    auto = bool(body.get("auto_name"))
    name = None if auto else check_str(body.get("name"), 80, "name")
    if session["closed_at"] is not None:
        raise ApiError("session_closed", 409)
    token = secrets.token_hex(24)
    stamp = now()

    if auto:
        # Next free ordinal; the unique index settles any race between devices.
        prefix = check_str(body.get("name_prefix"), 40, "name_prefix", required=False) or "Group"
        start = store.one("SELECT COUNT(*) AS n FROM ws_groups WHERE session_code = ?",
                          (code,))["n"] + 1
        for n in range(start, start + 200):
            candidate = (prefix + " " + str(n))[:80]
            try:
                cur = store.run(
                    "INSERT INTO ws_groups (session_code, name, token_hash, created_at,"
                    " updated_at) VALUES (?, ?, ?, ?, ?)",
                    (code, candidate, sha256(token), stamp, stamp))
            except sqlite3.IntegrityError:
                continue
            return 200, {"group_id": cur.lastrowid, "token": token, "name": candidate,
                         "ordinal": n, "topic": session["topic"], "code": code,
                         "slug": session["slug"], "rejoined": False}
        raise ApiError("too_many_groups", 503)

    existing = store.one("SELECT id FROM ws_groups WHERE session_code = ? AND name = ?",
                         (code, name))
    if existing:
        store.run("UPDATE ws_groups SET token_hash = ?, updated_at = ? WHERE id = ?",
                  (sha256(token), stamp, existing["id"]))
        group_id, rejoined = existing["id"], True
    else:
        cur = store.run(
            "INSERT INTO ws_groups (session_code, name, token_hash, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?)", (code, name, sha256(token), stamp, stamp))
        group_id, rejoined = cur.lastrowid, False
    return 200, {"group_id": group_id, "token": token, "name": name,
                 "topic": session["topic"], "code": code, "slug": session["slug"],
                 "rejoined": rejoined}


def handle_answer(store, body, _query, admin_key):
    code = check_code(body.get("code"))
    group_id = int(body.get("group_id") or 0)
    token = check_str(body.get("token"), 96, "token")
    lever_id = check_id(body.get("lever_id"), "lever_id")
    value = check_number(body.get("value"), "value")
    confidence = body.get("confidence")
    if confidence is not None:
        confidence = int(confidence)
        if confidence < 1 or confidence > 3:
            raise ApiError("invalid_field", 400, field="confidence")
    condition = check_str(body.get("condition"), 280, "condition", required=False)

    session = store.session(code)
    if session["closed_at"] is not None:
        raise ApiError("session_closed", 409)
    group = store.one("SELECT token_hash FROM ws_groups WHERE id = ? AND session_code = ?",
                      (group_id, code))
    if group is None:
        raise ApiError("unknown_group", 404)
    if not secrets.compare_digest(group["token_hash"], sha256(token)):
        raise ApiError("bad_token", 403)

    stamp = now()
    store.run(
        "INSERT INTO ws_answers (group_id, lever_id, value, confidence, condition_text,"
        " updated_at) VALUES (?, ?, ?, ?, ?, ?)"
        " ON CONFLICT(group_id, lever_id) DO UPDATE SET value=excluded.value,"
        " confidence=excluded.confidence, condition_text=excluded.condition_text,"
        " updated_at=excluded.updated_at",
        (group_id, lever_id, value, confidence, condition, stamp))
    store.run("INSERT INTO ws_answer_log (group_id, lever_id, value, confidence, created_at)"
              " VALUES (?, ?, ?, ?, ?)", (group_id, lever_id, value, confidence, stamp))
    store.run("UPDATE ws_groups SET updated_at = ? WHERE id = ?", (stamp, group_id))
    return 200, {"ok": True, "lever_id": lever_id, "updated_at": stamp}


def handle_results(store, _body, query, admin_key):
    topic = query.get("topic", [None])[0]
    code = query.get("code", [None])[0]
    if not topic and not code:
        raise ApiError("missing_field", 400, field="code or topic")

    sessions, codes, scope = [], [], "session"
    if code:
        code = check_code(code)
        session = store.session(code)
        token = query.get("admin_token", [None])[0]
        if not session["results_public"] and not is_admin(session, token):
            raise ApiError("forbidden", 403, detail="admin_token required for this session")
        codes = [code]
        topic = session["topic"]
        sessions.append({"code": code, "label": session["label"],
                         "reveal_step": session["reveal_step"],
                         "closed": session["closed_at"] is not None})
    else:
        scope = "topic"
        topic = check_id(topic, "topic")
        key = query.get("admin_key", [""])[0]
        if not admin_key or not secrets.compare_digest(admin_key, key):
            raise ApiError("forbidden", 403, detail="admin_key required for the topic scope")
        for row in store.query(
                "SELECT code, label, reveal_step, closed_at FROM ws_sessions"
                " WHERE topic = ? AND closed_at IS NULL AND mode <> 'solo'"
                " ORDER BY created_at", (topic,)):
            codes.append(row["code"])
            sessions.append({"code": row["code"], "label": row["label"],
                             "reveal_step": row["reveal_step"],
                             "closed": row["closed_at"] is not None})

    groups, answers = [], []
    if codes:
        marks = ",".join("?" * len(codes))
        rows = store.query(
            f"SELECT id, session_code, name FROM ws_groups WHERE session_code IN ({marks})"
            " ORDER BY session_code, id", tuple(codes))
        groups = [{"id": r["id"], "session": r["session_code"], "name": r["name"]}
                  for r in rows]
        ids = [r["id"] for r in rows]
        if ids:
            marks = ",".join("?" * len(ids))
            for r in store.query(
                    "SELECT group_id, lever_id, value, confidence, condition_text, updated_at"
                    f" FROM ws_answers WHERE group_id IN ({marks})"
                    " ORDER BY lever_id, group_id", tuple(ids)):
                answers.append({"group_id": r["group_id"], "lever_id": r["lever_id"],
                                "value": float(r["value"]),
                                "confidence": r["confidence"],
                                "condition": r["condition_text"],
                                "updated_at": r["updated_at"]})

    return 200, {"scope": scope, "topic": topic, "sessions": sessions,
                 "groups": groups, "answers": answers, "served_at": now()}


def handle_reveal(store, body, _query, admin_key):
    code = check_code(body.get("code"))
    session = store.session(code)
    if not is_admin(session, body.get("admin_token")):
        raise ApiError("forbidden", 403, detail="admin_token required")
    step = int(body.get("step", -1))
    if step < -1 or step > 999:
        raise ApiError("invalid_field", 400, field="step")
    if body.get("close"):
        store.run("UPDATE ws_sessions SET reveal_step = ?, closed_at = ? WHERE code = ?",
                  (step, now(), code))
    else:
        store.run("UPDATE ws_sessions SET reveal_step = ? WHERE code = ?", (step, code))
    return 200, {"ok": True, "code": code, "reveal_step": step,
                 "closed": bool(body.get("close"))}


def handle_selftest(store, _body, _query, admin_key):
    tables = {}
    for table in ("ws_sessions", "ws_groups", "ws_answers", "ws_answer_log"):
        tables[table] = store.one(f"SELECT COUNT(*) AS n FROM {table}")["n"]
    return 200, {"ok": True, "php": None, "implementation": "dev_api.py (sqlite)",
                 "configured": True, "pdo_mysql": False,
                 "database": "reachable", "tables": tables, "db_path": store.path}


def handle_rename(store, body, _query, admin_key):
    session = store.session_by(
        check_code(body["code"]) if body.get("code") else None,
        check_slug(body["slug"]) if body.get("slug") else None)
    code = session["code"]
    group_id = int(body.get("group_id") or 0)
    token = check_str(body.get("token"), 96, "token")
    name = check_str(body.get("name"), 80, "name")

    group = store.one("SELECT token_hash FROM ws_groups WHERE id = ? AND session_code = ?",
                      (group_id, code))
    if group is None:
        raise ApiError("unknown_group", 404)
    if not secrets.compare_digest(group["token_hash"], sha256(token)):
        raise ApiError("bad_token", 403)
    if store.one("SELECT id FROM ws_groups WHERE session_code = ? AND name = ? AND id <> ?",
                 (code, name, group_id)):
        raise ApiError("name_taken", 409)
    store.run("UPDATE ws_groups SET name = ?, updated_at = ? WHERE id = ?",
              (name, now(), group_id))
    return 200, {"ok": True, "group_id": group_id, "name": name}


ROUTES = {
    ("POST", "/session.php"): handle_session_post,
    ("GET", "/session.php"): handle_session_get,
    ("POST", "/group.php"): handle_group,
    ("POST", "/rename.php"): handle_rename,
    ("POST", "/answer.php"): handle_answer,
    ("GET", "/results.php"): handle_results,
    ("POST", "/reveal.php"): handle_reveal,
    ("GET", "/selftest.php"): handle_selftest,
}


def make_handler(store, admin_key, quiet):
    class Handler(http.server.BaseHTTPRequestHandler):
        server_version = "nwWorkshopDevAPI/1.0"

        def log_message(self, fmt, *args):
            if not quiet:
                sys.stderr.write("  %s %s\n" % (self.address_string(), fmt % args))

        def _send(self, status, payload):
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self):
            self._send(200, {"ok": True})

        def _dispatch(self, method):
            parsed = urlparse(self.path)
            path = parsed.path
            if not path.endswith(".php"):            # tolerate /session as well
                path = path.rstrip("/") + ".php"
            handler = ROUTES.get((method, path))
            if handler is None:
                allowed = [m for (m, p) in ROUTES if p == path]
                if allowed:
                    return self._send(405, {"error": "method_not_allowed", "allowed": allowed})
                return self._send(404, {"error": "unknown_endpoint", "path": parsed.path})

            body = {}
            if method == "POST":
                length = int(self.headers.get("Content-Length") or 0)
                if length > MAX_BODY:
                    return self._send(413, {"error": "body_too_large"})
                raw = self.rfile.read(length) if length else b""
                if raw:
                    try:
                        body = json.loads(raw.decode("utf-8"))
                    except Exception:
                        return self._send(400, {"error": "invalid_json"})
                    if not isinstance(body, dict):
                        return self._send(400, {"error": "invalid_json"})
            try:
                status, payload = handler(store, body, parse_qs(parsed.query), admin_key)
            except ApiError as exc:
                return self._send(exc.status, exc.payload)
            except Exception as exc:                 # never leak a stack trace
                sys.stderr.write(f"  ERROR {exc!r}\n")
                return self._send(500, {"error": "write_failed"})
            self._send(status, payload)

        def do_GET(self):
            self._dispatch("GET")

        def do_POST(self):
            self._dispatch("POST")

    return Handler


class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--host", default="127.0.0.1",
                    help="0.0.0.0 to serve a workshop over the local network")
    ap.add_argument("--db", default=None, help="SQLite file (default: alongside this script)")
    ap.add_argument("--admin-key", default=os.environ.get("NW_WS_ADMIN_KEY", DEFAULT_ADMIN_KEY),
                    help="key that unlocks results.php?topic=…")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    db = args.db or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "workshop_dev.sqlite")
    store = Store(db)
    handler = make_handler(store, args.admin_key, args.quiet)
    server = Server((args.host, args.port), handler)
    print(f"negaWatt workshop dev API on http://{args.host}:{args.port}")
    print(f"  database  {db}")
    print(f"  admin key {args.admin_key}")
    print(f"  point the pages at it once with ?api=http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
