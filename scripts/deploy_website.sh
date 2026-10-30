#!/usr/bin/env bash
# Publish the workshop changes. Safe to re-run; no --delete, so intervec/ and
# RFNBO_final_results/ are untouched. Markdown files are working notes (e.g.
# workshop/participant_feedback.md) and are never published.
#
# The docroot is negawatt@negawatt.squoilin.eu:public_html/. Since October 2026 the
# server also answers on two names of its own (Apache configuration, not in this
# repository): https://workshop.negawatt.be/ serves public_html/workshop/ with
# /assets, /data and /api alongside, and https://hypotheses.negawatt.be/ the card
# site; the old negawatt.squoilin.eu URLs redirect there.
#
# To go back to an earlier state, check out that commit (e.g. the tag
# pre-workshop-2026-11) and run this script again; a tarball of the site as it was
# before the November 2026 changes is in ~negawatt/backups/ on the server.
set -euo pipefail
cd "$(dirname "$0")/.."                      # the repository root, wherever it is

# The key used on the original workstation, if present; otherwise ~/.ssh/config
# decides, as for any other ssh call.
SSH="ssh"
if [ -f ~/.ssh/rsa_labothap ]; then SSH="ssh -i $HOME/.ssh/rsa_labothap"; fi

rsync -avz --no-perms --omit-dir-times \
      --exclude '.DS_Store' --exclude '__pycache__' \
      --exclude '*.md' \
      -e "$SSH" \
      website/ negawatt@negawatt.squoilin.eu:public_html/
echo
echo "--- verifying ---"
BASE=https://workshop.negawatt.be
for f in data/workshop_content.js data/levers_transport.js data/levers_buildings.js \
         data/history_transport.js \
         assets/js/workshop/impact.js assets/js/workshop/play.js \
         assets/js/workshop/reveal.js assets/js/workshop/cards.js \
         assets/js/workshop/join.js workshop/play.html workshop/reveal.html \
         assets/css/workshop.css; do
  url="$BASE/${f#workshop/}"                 # workshop/x.html is served at the root
  if diff -q <(curl -sL "$url") "website/$f" >/dev/null; then
    printf "  OK   %s\n" "$url"
  else
    printf "  DIFF %s\n" "$url"
  fi
done
