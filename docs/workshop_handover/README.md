# Handover — the three remaining workshop topics

Three agents were dispatched on 2026-09-02 to add `international-mobility`,
`residential-heat` and `tertiary-heat`. All three, and the five web-research
agents they had spawned between them, were stopped by an API session limit
before any of them finished. This folder is everything recoverable from their
transcripts.

**Read `docs/workshop_module.md` §12 first — it is the specification. This folder
is raw material, not a deliverable.**

> Every figure in `notes/` is **unverified**. The agents were still gathering and
> cross-checking when they stopped, and none of them reached the point of writing
> a sourced fact into a YAML. Re-source anything before it reaches a card.

## What is here

```
drafts/    files the agents actually wrote, verbatim
notes/     their reasoning and their web-research results, in order
```

### drafts/

| file | what it is |
|---|---|
| `residential-heat__residential_heat.py` | a complete-looking topic module, 402 lines, 7 levers (`floor-area`, `renovation-rate`, `thermostat`, `hot-water`, `cooling`, `cooking`, `district-heat`). Never executed. |
| `international-mobility__probe.py`, `__calc.py`, `__nbrun.py` | throwaway scripts the aviation agent used to read the transport notebook and work out its numbers. Useful as a shortcut to the same values. |

The residential module is also parked in the tree as
`workshop_levers/_wip_residential_heat.py`. The loader skips modules whose name
begins with `_`, so it is preserved without breaking anything — a module with
shown levers but no content YAML would otherwise fail the content build.

To resume it: rename to `residential_heat.py`, run the buildings notebook, and
write `website/workshop/content/residential-heat.yaml`.

### notes/

| file | subject |
|---|---|
| `international-mobility-lead.md` | the aviation topic: which notebook variables, which levers, the long-haul "trips per lifetime" framing |
| `residential-heat-lead.md` | the residential topic: variables found, lever choices, the reasoning behind the module in `drafts/` |
| `tertiary-heat-lead.md` | the tertiary topic: variables found, lever thinking. No module was written. |
| `research-tertiary-europe.md` | EPBD text, European service-sector floor area and heating intensity |
| `research-cooling-degree-days.md` | cooling degree days, air-conditioning penetration |
| `research-lighting.md` | lighting regulation and lighting intensity |
| `research-night-lighting-*.md`, `research-floor-area-*.md` | further searches; two agents worked the same ground, hence the near-duplicate names |
| `research-eating-out.md` | catering / eating-out expenditure. Note: Statbel and Sciensano blocked the fetches, so this one is thin. |
| `research-aviation-benchmarks.md` | barely started — it died almost immediately |

## Why it failed, and what to do differently

The three topic agents each spawned their own web-research subagents, so a
three-agent dispatch became a fleet of eleven and exhausted the session budget
in about twenty minutes. None of them was close to finished.

On a retry:

- **tell the agents not to spawn subagents.** That single instruction was
  missing from the briefs and is what turned three agents into eleven.
- **dispatch them one at a time**, or at most two. The work is
  content-writing — three languages, eight levers, four sourced facts each —
  and it does not parallelise as cheaply as it looks.
- consider handing each agent the relevant `notes/` file, so the web research
  does not have to happen twice.

## What is *not* affected

The infrastructure work landed and is verified:

- `workshop_levers/` with one module per topic, and one generic
  `export_topics(sector, globals())` cell per notebook — the refactor that made
  parallel work possible in the first place. The inland levers came out
  byte-identical.
- `website/data/history_buildings.js` — 14 observed buildings series, exported
  once so the buildings topics have real curves.
- every page loads every sector's data; topics may declare their own interface
  strings; `reveal.js` no longer crashes when a poll fails.

`inland-mobility` is untouched and still passes everything.
