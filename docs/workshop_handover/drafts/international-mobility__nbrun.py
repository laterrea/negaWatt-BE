"""Execute the transport notebook's code cells (except the two export cells)
into a namespace, so a lever module can be iterated on quickly.

Nothing is written to disk: cells 143 (website export + energy_totals CSV) and
145 (workshop export) are skipped.
"""
import json
import os
import re
import sys

ROOT = "/home/sylvain/svn/negaWatt-BE"


def load_globals(nb_name="nW_BE_demand_model_transports.ipynb", skip_from=143):
    os.chdir(ROOT)
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    nb = json.load(open(os.path.join(ROOT, nb_name)))
    g = {"__name__": "__main__", "__builtins__": __builtins__}
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        if i >= skip_from:
            continue
        src = "".join(cell["source"])
        # strip IPython magics / shell escapes
        src = "\n".join(ln for ln in src.split("\n")
                        if not re.match(r"^\s*[%!]", ln))
        try:
            exec(compile(src, f"<cell {i}>", "exec"), g)
        except Exception as exc:
            print(f"!! cell {i} raised {type(exc).__name__}: {exc}", file=sys.stderr)
            raise
    return g


if __name__ == "__main__":
    g = load_globals()
    print("ok, globals:", len(g))
