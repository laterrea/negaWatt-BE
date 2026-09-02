#!/usr/bin/env python
"""Serve website/ for local development, with caching switched off.

`python -m http.server` lets the browser cache aggressively, which is a real
nuisance here: the whole point of this module is that re-running a notebook or the
content build changes the .js files under website/data/, and a cached copy makes
it look as though nothing happened.

    python scripts/dev_static.py --port 8080
    #  -> http://127.0.0.1:8080/workshop/index.html

In production the same behaviour comes from website/.htaccess, which sets
Cache-Control: no-cache on .js/.css/.html and caches only the vendored assets.
"""
import argparse
import functools
import http.server
import os
import socketserver
import sys

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "website")


class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def log_message(self, fmt, *args):
        if not getattr(self.server, "quiet", False):
            sys.stderr.write("  %s\n" % (fmt % args))


class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="127.0.0.1",
                    help="0.0.0.0 to serve a workshop over the local network")
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    handler = functools.partial(Handler, directory=args.root)
    server = Server((args.host, args.port), handler)
    server.quiet = args.quiet
    print(f"serving {args.root} on http://{args.host}:{args.port} (no caching)")
    print(f"  workshop: http://{args.host}:{args.port}/workshop/index.html")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
