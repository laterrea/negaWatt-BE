"""Per-topic lever definitions for the interactive workshop.

One module per workshop topic. Two topics that share a notebook — inland and
international mobility both come out of the transport notebook, residential and
tertiary heat both out of the buildings notebook — can then be written, reviewed
and changed independently, instead of competing for one cell in a 140-cell
notebook.

A topic module declares three things:

    TOPIC  = "international-mobility"     # must match the content YAML's `topic`
    SECTOR = "transport"                  # which notebook computes its inputs
    ORDER  = 20                           # optional, only affects print order

    def build(ctx):
        return {"levers": [...],          # records from make_lever()
                "model": {...}}           # optional context quantities

``ctx`` is the notebook's ``globals()``, so a module reads the frames the
notebook has already computed and introduces no assumptions of its own. If a
module needs a quantity the notebook does not compute, that quantity belongs in
the notebook — not here.

The notebook then carries a single generic cell:

    from workshop_levers import export_topics
    export_topics("transport", globals(), title="Mobility & transport")

which writes ``website/data/levers_<sector>.js`` with every topic of that
sector. See docs/workshop_module.md.
"""
import importlib
import pkgutil

from nW_BE_demand_model_sub_functions import write_levers_js

__all__ = ["topic_modules", "export_topics", "need"]


def need(ctx, *names):
    """Fetch notebook variables, failing with a useful message if any is absent.

    Topic modules run against a notebook's namespace, so the common failure is a
    renamed or not-yet-computed variable. Saying which one, once, beats a
    KeyError from somewhere deep in the lever definitions.
    """
    missing = [n for n in names if n not in ctx]
    if missing:
        raise KeyError(
            "the notebook has not defined " + ", ".join(missing)
            + " — run the cells above the workshop export, or update this topic "
              "module if the variables were renamed"
        )
    return [ctx[n] for n in names]


def topic_modules(sector=None):
    """Every topic module, optionally filtered to one sector, in print order."""
    found = []
    for info in pkgutil.iter_modules(__path__):
        if info.name.startswith("_"):
            continue
        module = importlib.import_module(f"{__name__}.{info.name}")
        # notebooks run with %autoreload, but an explicit reload makes editing a
        # topic module and re-running the export cell reliable either way
        module = importlib.reload(module)
        if not hasattr(module, "TOPIC") or not hasattr(module, "build"):
            continue
        if sector is None or getattr(module, "SECTOR", None) == sector:
            found.append(module)
    return sorted(found, key=lambda m: (getattr(m, "ORDER", 100), m.TOPIC))


def export_topics(sector, ctx, title=None, out_path=None):
    """Build every topic of `sector` and write website/data/levers_<sector>.js.

    The model block is keyed by topic, so two topics of one sector cannot
    overwrite each other's context quantities.
    """
    modules = topic_modules(sector)
    if not modules:
        raise RuntimeError(f"no topic module declares SECTOR = {sector!r}")

    levers, model, owner = [], {}, {}
    for module in modules:
        result = module.build(ctx) or {}
        produced = list(result.get("levers") or [])
        for lever in produced:
            lid = lever["id"]
            if lid in owner:
                raise ValueError(
                    f"lever id {lid!r} is defined by both {owner[lid]!r} and "
                    f"{module.TOPIC!r}; ids must be unique within a sector"
                )
            owner[lid] = module.TOPIC
            if lever.get("topic") != module.TOPIC:
                raise ValueError(
                    f"lever {lid!r} in {module.__name__} declares topic "
                    f"{lever.get('topic')!r} but the module's TOPIC is {module.TOPIC!r}"
                )
        levers.extend(produced)
        if result.get("model"):
            model[module.TOPIC] = result["model"]
        shown = sum(1 for lever in produced if lever.get("shown"))
        print(f"[workshop] {module.TOPIC}: {len(produced)} levers ({shown} shown)")

    return write_levers_js(sector, levers, model=model, title=title,
                           out_path=out_path)
