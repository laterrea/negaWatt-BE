# nW-BE - negaWatt Belgium

This repository contains the notebooks used by negaWatt Belgium to project sufficiency and efficiency assumptions on future energy demand levels.

<p align="center">
  <img src="nW_BE_logo_rectangle.png" width="700">
</p>

The documentation is still under construction...

## Environment setup

All notebooks are meant to run in the **`data_processing`** Conda environment.

From the repository root:

```bash
conda env create -f environment.yml
conda activate data_processing
python -m ipykernel install --user --name data_processing --display-name "Python (data_processing)"
```

The last command registers the kernel so Jupyter, VS Code, and Cursor can select **Python (data_processing)**.

To refresh an existing environment after pulling dependency changes:

```bash
conda env update -f environment.yml --prune
```

### Dependencies

| Package | Used for |
|---|---|
| `numpy`, `pandas`, `matplotlib` | Projections and plots in all notebooks |
| `scipy` | Projection helpers in `nW_BE_demand_model_sub_functions.py` |
| `openpyxl` | Reading JRC-IDEES Excel workbooks (industry notebook) |
| `jupyter`, `notebook`, `ipykernel` | Interactive and batch notebook execution |

The helper module `nW_BE_demand_model_sub_functions.py` must stay in the repository root (same folder as the notebooks).

## Running the notebooks

Open Jupyter from the repository root with the environment activated:

```bash
conda activate data_processing
jupyter notebook
```

Or run a single notebook headlessly (useful for CI or a full refresh):

```bash
conda activate data_processing
jupyter nbconvert --to notebook --execute --inplace nW_BE_demand_model_macro.ipynb
```

### Execution order

Several notebooks load shared parameters by running the macro notebook (`%run ./nW_BE_demand_model_macro.ipynb`). Recommended order:

1. `nW_BE_demand_model_macro.ipynb` — population, households, shared constants
2. `nW_BE_demand_data_aux.ipynb`, `nW_BE_demand_model_buildings.ipynb`, `nW_BE_demand_model_transports.ipynb` — each re-runs macro internally
3. `nW_BE_demand_model_industry.ipynb` — standalone (reads `data/jrc-idees-2021/` and writes `data/industry_output/`)

All five notebooks were verified to execute successfully in `data_processing` (Python 3.11).

## Notebooks

| Notebook | Sector |
|---|---|
| `nW_BE_demand_model_macro.ipynb` | Population, households, physical constants |
| `nW_BE_demand_model_buildings.ipynb` | Residential & tertiary buildings |
| `nW_BE_demand_model_transports.ipynb` | Passenger mobility & freight |
| `nW_BE_demand_model_industry.ipynb` | **Industry (new)** |
| `nW_BE_demand_data_aux.ipynb` | Auxiliary historical series |

### Industry notebook (`nW_BE_demand_model_industry.ipynb`)

This notebook documents and reconstructs the **industrial final-energy-demand** inputs of the scenario.
Unlike buildings and transport, industry is currently **not modelled bottom-up** in negaWatt-BE: PyPSA-Eur
imports the industrial FEC of every country directly from the external **CLEVER** scenario dashboard
(`clever_Industry_<year>.csv`, injected in `scripts/build_industrial_energy_demand_per_node.py`). The notebook:

1. explains how industry enters the negaWatt-BE / PyPSA-Eur pipeline (the carrier columns PyPSA reads);
2. loads the **JRC-IDEES-2021** statistical baseline for Belgium (production in kt, FEC by carrier in TWh);
3. makes the **CLEVER sufficiency / circularity / efficiency hypotheses explicit** — gross material-demand
   reductions, recycling rates and per-route energy intensities, and electrification/fuel-switch shares
   (extracted from the CLEVER reports — see `CLEVER/`), each annotated with its convergence corridor;
4. reconstructs the Belgian industry FEC trajectory (per sector and per carrier, 2020–2050) from those explicit
   levers — **deriving** the steel energy intensity from the recycling rate × route intensities (and showing
   where glass/paper cannot be derived) — and writes PyPSA-Eur-compatible CSVs to `data/industry_output/`;
5. **validates** the reconstruction against the dashboard (matches every carrier/year to < 0.01 TWh) and
   transparently flags the remaining gaps (JRC↔CLEVER baseline mismatch, cement/glass volumes, the unexplained
   2030 oil-feedstock spike, undocumented per-sector carrier split, …).

The CLEVER PDFs were converted to Markdown with `opendataloader-pdf`; the distilled, quantitative hypotheses
live in `CLEVER/CLEVER_industry_hypotheses.md`.

**Data folders**
- `data/jrc-idees-2021/BE/` — raw JRC-IDEES-2021 Belgium workbooks (Industry, EnergyBalance).
- `data/clever_dashboard_reference/` — CLEVER dashboard exports, kept **only** for validation.
- `data/industry_output/` — reconstructed `clever_Industry_<year>_BE_reconstructed.csv` files.