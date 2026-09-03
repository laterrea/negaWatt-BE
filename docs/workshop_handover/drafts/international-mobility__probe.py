import sys
sys.path.insert(0, "/tmp/claude-1000/-home-sylvain-svn-negaWatt-BE/0dead3ce-a8ac-4bab-be29-da0ba920056d/scratchpad")
from nbrun import load_globals
g = load_globals()

from nW_BE_demand_model_sub_functions import mode_totals_twh

years = g["years"]
Y0, Y1 = years[0], years[-1]
pop = g["population_dict"]
df_PM = g["df_PM"]
df_FT = g["df_FT"]
pm = mode_totals_twh(g["df_PM_TWh_all"])
ft = mode_totals_twh(g["df_FT_TWh_all"])

print("Y0,Y1 =", Y0, Y1, "| pop:", pop[Y0], pop[Y1])
print("kgoe_to_kWh =", g["kgoe_to_kWh"], " kgh2_to_kWh =", g.get("kgh2_to_kWh"))
print()
print("=== settings ===")
for n in ["pro_PM_spe", "pro_PM_spe_avi_lng", "pro_PM_spe_avi_srt",
          "sft_PM_rel_avi_srt_to_trn_cnv", "sft_PM_rel_avi_srt_to_trn_spd",
          "sft_PM_rel_avi_srt_to_cch",
          "occu_trgt_PM_avi_intra", "occu_trgt_PM_avi_extra",
          "redu_fuel_PM_avi_intra", "redu_fuel_PM_avi_extra",
          "sta_PM_avi_srt", "end_PM_avi_srt", "mid_PM_avi_srt",
          "ref_PM_spe", "ref_FT_spe"]:
    print(f"  {n:32s} = {g[n]!r}")

print()
print("=== ref_PM_mod_spe / rel ===")
for k in ("plane-intra EU", "plane-extra EU"):
    print(f"  {k}: spe={g['ref_PM_mod_spe'][k]:.4f} rel={g['ref_PM_mod_rel'][k]:.4f} abs={g['ref_PM_mod_abs'][k]}")
print("  trg spe:", {k: round(v, 4) for k, v in g["trg_PM_mod_spe"].items()})
print("  trg rel:", {k: round(v, 4) for k, v in g["trg_PM_mod_rel"].items()})
print("  trg_PM_mod_spe_avi_srt (2050 pre-shift baseline) =", g["trg_PM_mod_spe_avi_srt"])
print("  red_PM_rel =", g["red_PM_rel"])
print("  sft abs: cnv=%.4f spd=%.4f cch=%.4f" % (
    g["sft_PM_abs_avi_srt_to_trn_cnv"], g["sft_PM_abs_avi_srt_to_trn_spd"],
    g["sft_PM_abs_avi_srt_to_cch"]))

print()
print("=== df_PM aviation rows ===")
for m in ("plane-intra EU", "plane-extra EU", "car", "train-conventional",
          "train-high speed", "bus&coach"):
    for u in ("% of total", "pkm/person", "Gpkm"):
        print(f"  {m:20s} {u:12s} {float(df_PM.loc[(m,u),Y0]):14.4f} -> {float(df_PM.loc[(m,u),Y1]):14.4f}")

print()
print("=== mode TWh (passenger) ===")
for m in pm.index:
    print(f"  {m:22s} {float(pm.loc[m,Y0]):8.3f} -> {float(pm.loc[m,Y1]):8.3f}")
print("=== mode TWh (freight) ===")
for m in ft.index:
    print(f"  {m:24s} {float(ft.loc[m,Y0]):8.3f} -> {float(ft.loc[m,Y1]):8.3f}")

print()
print("=== per-powertrain TWh, aviation ===")
d = g["df_PM_avi_srt_TWh"]
print(d[["Mode", "Powertrain", Y0, Y1]].to_string())
d = g["df_PM_avi_lng_TWh"]
print(d[["Mode", "Powertrain", Y0, Y1]].to_string())

print()
print("=== fuel & occupancy frames ===")
print("cons intra:\n", g["cons_fuel_PM_avi_intra"][[Y0, Y1]])
print("occu intra:\n", g["occupancy_PM_avi_intra"][[Y0, Y1]])
print("cons extra:\n", g["cons_fuel_PM_avi_extra"][[Y0, Y1]])
print("occu extra:\n", g["occupancy_PM_avi_extra"][[Y0, Y1]])
print("carrier shares intra:\n", g["df_PM_avi_srt"][[Y0, Y1]])

print()
print("=== intensities kWh/pkm ===")
def inten(tbl, df, m, unit, y):
    a = float(df.loc[(m, unit), y])
    return float(tbl.loc[m, y]) / a if a else 0.0
for m in ("plane-intra EU", "plane-extra EU", "car", "train-conventional",
          "train-high speed", "bus&coach"):
    print(f"  {m:22s} {inten(pm, df_PM, m,'Gpkm',Y0):.5f} -> {inten(pm, df_PM, m,'Gpkm',Y1):.5f}")

print()
print("=== air freight (df_FT) ===")
for m in ("plane-intra EU", "plane-extra EU"):
    for u in ("% of total", "tkm/person", "Gtkm"):
        print(f"  {m:16s} {u:12s} {float(df_FT.loc[(m,u),Y0]):12.4f} -> {float(df_FT.loc[(m,u),Y1]):12.4f}")
print("  air freight TWh:", {m: (round(float(ft.loc[m, Y0]), 4), round(float(ft.loc[m, Y1]), 4))
                             for m in ("plane-intra EU", "plane-extra EU") if m in ft.index})
