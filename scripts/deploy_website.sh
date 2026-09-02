#!/usr/bin/env bash
# Publish the workshop changes. Safe to re-run; no --delete, so intervec/ and
# RFNBO_final_results/ are untouched.
set -euo pipefail
cd /home/sylvain/svn/negaWatt-BE
rsync -avz --no-perms --omit-dir-times \
      --exclude '.DS_Store' --exclude '__pycache__' \
      -e "ssh -i ~/.ssh/rsa_nopasswd" \
      website/ negawatt@negawatt.squoilin.eu:public_html/
echo
echo "--- verifying ---"
for f in data/workshop_content.js data/levers_transport.js \
         assets/js/workshop/impact.js assets/js/workshop/play.js \
         assets/css/workshop.css; do
  if diff -q <(curl -s "https://negawatt.squoilin.eu/$f") "website/$f" >/dev/null; then
    printf "  OK   %s\n" "$f"
  else
    printf "  DIFF %s\n" "$f"
  fi
done
