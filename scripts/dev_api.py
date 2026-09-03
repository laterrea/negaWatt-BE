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

MAX_BODY = 8192

SCHEMA = """
CREATE TABLE IF NOT EXISTS ws_groups (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  topic TEXT NOT NULL,
  name TEXT NOT NULL,
  token_hash TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
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
CREATE INDEX IF NOT EXISTS idx_topic_created ON ws_groups(topic, created_at);
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


def today():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d 00:00:00")


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


def check_stamp(value, field, end_of_day=False):
    """A results-window bound in UTC; see ws_stamp() in website/api/db.php."""
    value = str(value or "").strip().replace("T", " ").replace("Z", "")
    if not value:
        return None
    match = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", value)
    if match:
        time = "23:59:59" if end_of_day else "00:00:00"
    else:
        match = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2})(?::(\d{2}))?", value)
        if not match:
            raise ApiError("invalid_field", 400, field=field)
        time = f"{match.group(4)}:{match.group(5)}:{match.group(6) or '00'}"
    try:
        datetime(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    except ValueError:
        raise ApiError("invalid_field", 400, field=field)
    return f"{match.group(1)}-{match.group(2)}-{match.group(3)} {time}"


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

    def group(self, group_id, token):
        """The group token is the only credential: it stops one device
        overwriting another group's answers. It does not gate the results."""
        token = check_str(token, 96, "token")
        row = self.one("SELECT * FROM ws_groups WHERE id = ?", (int(group_id or 0),))
        if row is None:
            raise ApiError("unknown_group", 404)
        if not secrets.compare_digest(row["token_hash"], sha256(token)):
            raise ApiError("bad_token", 403)
        return row


# --------------------------------------------------------------------- handlers

def handle_group(store, body, _query):
    topic = check_id(body.get("topic"), "topic")
    name = check_str(body.get("name"), 80, "name", required=False)
    prefix = check_str(body.get("name_prefix"), 40, "name_prefix", required=False) or "Group"
    token = secrets.token_hex(24)
    stamp = now()
    cur = store.run(
        "INSERT INTO ws_groups (topic, name, token_hash, created_at, updated_at)"
        " VALUES (?, ?, ?, ?, ?)", (topic, name or "", sha256(token), stamp, stamp))
    group_id = cur.lastrowid
    if name is None:
        rank = store.one(
            "SELECT COUNT(*) AS n FROM ws_groups WHERE topic = ? AND created_at >= ?"
            " AND id <= ?", (topic, today(), group_id))["n"]
        name = f"{prefix} {rank}"[:80]
        store.run("UPDATE ws_groups SET name = ? WHERE id = ?", (name, group_id))
    return 200, {"group_id": group_id, "token": token, "name": name,
                 "topic": topic, "created_at": stamp}


def handle_rename(store, body, _query):
    group = store.group(body.get("group_id"), body.get("token"))
    name = check_str(body.get("name"), 80, "name")
    store.run("UPDATE ws_groups SET name = ?, updated_at = ? WHERE id = ?",
              (name, now(), group["id"]))
    return 200, {"ok": True, "group_id": group["id"], "name": name}


def handle_answer(store, body, _query):
    group = store.group(body.get("group_id"), body.get("token"))
    group_id = group["id"]
    lever_id = check_id(body.get("lever_id"), "lever_id")
    value = check_number(body.get("value"), "value")
    confidence = body.get("confidence")
    if confidence is not None:
        confidence = int(confidence)
        if confidence < 1 or confidence > 3:
            raise ApiError("invalid_field", 400, field="confidence")
    condition = check_str(body.get("condition"), 280, "condition", required=False)

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


def handle_results(store, _body, query):
    topic = check_id(query.get("topic", [None])[0], "topic")
    start = check_stamp(query.get("from", [None])[0], "from")
    end = check_stamp(query.get("to", [None])[0], "to", end_of_day=True)

    sql = "SELECT id, name, created_at, updated_at FROM ws_groups WHERE topic = ?"
    args = [topic]
    if start is not None:
        sql += " AND created_at >= ?"
        args.append(start)
    if end is not None:
        sql += " AND created_at <= ?"
        args.append(end)
    rows = store.query(sql + " ORDER BY id", tuple(args))
    groups = [{"id": r["id"], "name": r["name"], "created_at": r["created_at"],
               "updated_at": r["updated_at"]} for r in rows]

    answers = []
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

    return 200, {"topic": topic, "from": start, "to": end,
                 "groups": groups, "answers": answers, "served_at": now()}


def handle_selftest(store, _body, _query):
    tables = {}
    for table in ("ws_groups", "ws_answers", "ws_answer_log"):
        tables[table] = store.one(f"SELECT COUNT(*) AS n FROM {table}")["n"]
    return 200, {"ok": True, "php": None, "implementation": "dev_api.py (sqlite)",
                 "configured": True, "pdo_mysql": False,
                 "database": "reachable", "tables": tables, "db_path": store.path}


ROUTES = {
    ("POST", "/group.php"): handle_group,
    ("POST", "/rename.php"): handle_rename,
    ("POST", "/answer.php"): handle_answer,
    ("GET", "/results.php"): handle_results,
    ("GET", "/selftest.php"): handle_selftest,
}


def make_handler(store, quiet):
    class Handler(http.server.BaseHTTPRequestHandler):
        server_version = "nwWorkshopDevAPI/2.0"

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
            if not path.endswith(".php"):            # tolerate /group as well
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
                status, payload = handler(store, body, parse_qs(parsed.query))
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
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    db = args.db or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "workshop_dev.sqlite")
    store = Store(db)
    handler = make_handler(store, args.quiet)
    server = Server((args.host, args.port), handler)
    print(f"negaWatt workshop dev API on http://{args.host}:{args.port}")
    print(f"  database  {db}")
    print(f"  point the pages at it once with ?api=http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
