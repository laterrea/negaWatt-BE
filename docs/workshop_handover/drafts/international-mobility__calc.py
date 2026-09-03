import math
import sys
sys.path.insert(0, "/tmp/claude-1000/-home-sylvain-svn-negaWatt-BE/0dead3ce-a8ac-4bab-be29-da0ba920056d/scratchpad")
from nbrun import load_globals
g = load_globals()
from nW_BE_demand_model_sub_functions import mode_totals_twh

years = g["years"]; Y0, Y1 = years[0], years[-1]
pop = {y: float(g["population_dict"][y]) for y in (Y0, Y1)}
df_PM, df_FT = g["df_PM"], g["df_FT"]
pm = mode_totals_twh(g["df_PM_TWh_all"]); ft = mode_totals_twh(g["df_FT_TWh_all"])
kgoe = g["kgoe_to_kWh"]

def act(df, m, u, y): return float(df.loc[(m, u), y])
def twh(t, m, y): return float(t.loc[m, y]) if m in t.index else 0.0
def inten(t, df, m, u, y):
    a = act(df, m, u, y); return twh(t, m, y)/a if a else 0.0

def gc(a, b):
    (la, lo), (lb, lob) = a, b
    la, lo, lb, lob = map(math.radians, (la, lo, lb, lob))
    return 6371.0088*2*math.asin(math.sqrt(math.sin((lb-la)/2)**2
        + math.cos(la)*math.cos(lb)*math.sin((lob-lo)/2)**2))

BRU = (50.9014, 4.4844)
print("great circle from BRU:")
for name, c in [("BCN", (41.2971, 2.0785)), ("PVG", (31.1434, 121.8052)),
                ("LAX", (33.9425, -118.4081)), ("FCO", (41.8003, 12.2389)),
                ("BER", (52.3667, 13.5033)), ("MAD", (40.4719, -3.5626))]:
    print(f"  BRU-{name}: {gc(BRU, c):8.1f} km   return {2*gc(BRU,c):8.1f}")

LIFE = 80.0
LNG_TRIP = 18000.0      # 9 000 km each way, notebook section 2.1 note
SRT_TRIP = 2160.0       # BRU-BCN return

print()
print("=== demand levers ===")
for m, trip, lab in [("plane-extra EU", LNG_TRIP, "long-haul"),
                     ("plane-intra EU", SRT_TRIP, "short-haul")]:
    s0, s1 = act(df_PM, m, "pkm/person", Y0), act(df_PM, m, "pkm/person", Y1)
    print(f"  {lab:11s} pkm/person {s0:9.3f} -> {s1:9.3f}  ({100*(s1/s0-1):+.1f}%)")
    print(f"              trips/life {s0/trip*LIFE:9.4f} -> {s1/trip*LIFE:9.4f}")
    print(f"              trips/year {s0/trip:9.4f} -> {s1/trip:9.4f}"
          f"   1 every {trip/s0:.1f} / {trip/s1:.1f} yr")
    print(f"              km/day     {s0/365:9.3f} -> {s1/365:9.3f}")

avi_pkm = {y: act(df_PM, "plane-intra EU", "pkm/person", y)
              + act(df_PM, "plane-extra EU", "pkm/person", y) for y in (Y0, Y1)}
print(f"  total air pkm/person {avi_pkm[Y0]:.1f} -> {avi_pkm[Y1]:.1f}"
      f"  = {avi_pkm[Y0]/365:.3f} -> {avi_pkm[Y1]/365:.3f} km/day"
      f"  ({100*(avi_pkm[Y1]/avi_pkm[Y0]-1):+.1f}%)")
print(f"  long-haul share of air pkm: {100*act(df_PM,'plane-extra EU','pkm/person',Y0)/avi_pkm[Y0]:.1f}%"
      f" -> {100*act(df_PM,'plane-extra EU','pkm/person',Y1)/avi_pkm[Y1]:.1f}%")

print()
print("=== TWh ===")
PAX = ["plane-intra EU", "plane-extra EU"]
pax_twh = {y: sum(twh(pm, m, y) for m in PAX) for y in (Y0, Y1)}
frt_twh = {y: sum(twh(ft, m, y) for m in PAX) for y in (Y0, Y1)}
tot = {y: pax_twh[y] + frt_twh[y] for y in (Y0, Y1)}
print(f"  passenger aviation {pax_twh[Y0]:.3f} -> {pax_twh[Y1]:.3f}")
print(f"  air freight        {frt_twh[Y0]:.3f} -> {frt_twh[Y1]:.3f}")
print(f"  TOTAL topic        {tot[Y0]:.3f} -> {tot[Y1]:.3f}")

# kerosene-only part of intra-EU passenger aviation
srt = g["df_PM_avi_srt_TWh"].set_index("Powertrain")
srt_kero = {y: float(srt.loc["liquid-kerosene", y]) for y in (Y0, Y1)}
srt_h2 = {y: float(srt.loc["hydrogen", y]) for y in (Y0, Y1)}
print(f"  intra-EU kerosene  {srt_kero[Y0]:.4f} -> {srt_kero[Y1]:.4f}"
      f"   hydrogen {srt_h2[Y0]:.4f} -> {srt_h2[Y1]:.4f}")

print()
print("=== fuel / occupancy ===")
ci, oi = g["cons_fuel_PM_avi_intra"], g["occupancy_PM_avi_intra"]
ce, oe = g["cons_fuel_PM_avi_extra"], g["occupancy_PM_avi_extra"]
print(f"  intra cons kerosene {float(ci.loc['liquid-kerosene',Y0]):.4f} -> {float(ci.loc['liquid-kerosene',Y1]):.4f} kWh/km")
print(f"  intra occu kerosene {float(oi.loc['liquid-kerosene',Y0]):.4f} -> {float(oi.loc['liquid-kerosene',Y1]):.4f} p")
print(f"  extra cons kerosene {float(ce.loc['liquid-kerosene',Y0]):.4f} -> {float(ce.loc['liquid-kerosene',Y1]):.4f} kWh/km")
print(f"  extra occu kerosene {float(oe.loc['liquid-kerosene',Y0]):.4f} -> {float(oe.loc['liquid-kerosene',Y1]):.4f} p")
print(f"  intra h2   cons {float(ci.loc['hydrogen',Y0]):.4f} kWh/km  occu {float(oi.loc['hydrogen',Y0]):.1f} p"
      f"  -> {float(ci.loc['hydrogen',Y1])/float(oi.loc['hydrogen',Y1]):.5f} kWh/pkm")

print()
print("=== notebook prose comments, converted (kgoe/100km -> kWh/km) ===")
for lab, v in [("intra 2000", 578.8), ("intra 2023", 633.0),
               ("extra 2000", 937.0), ("extra 2023", 629.5)]:
    print(f"  {lab}: {v} kgoe/100km = {v/100*kgoe:.3f} kWh/km")

print()
print("=== intensities kWh/pkm ===")
for m in ("plane-intra EU", "plane-extra EU", "car", "train-conventional",
          "train-high speed", "bus&coach"):
    print(f"  {m:20s} {inten(pm,df_PM,m,'Gpkm',Y0):.5f} -> {inten(pm,df_PM,m,'Gpkm',Y1):.5f}")
print("  car 2019 kWh/pkm vs plane-intra 2019:",
      round(inten(pm,df_PM,'plane-intra EU','Gpkm',Y0)/inten(pm,df_PM,'car','Gpkm',Y0), 3), "x")

print()
print("=== short-haul shift basket ===")
dest = {"train-conventional": g["sft_PM_rel_avi_srt_to_trn_cnv"],
        "train-high speed":   g["sft_PM_rel_avi_srt_to_trn_spd"],
        "bus&coach":          g["sft_PM_rel_avi_srt_to_cch"]}
w = sum(dest.values())
basket = sum(v/w*inten(pm, df_PM, m, "Gpkm", Y1) for m, v in dest.items())
print("  weights:", {k: round(v/w, 4) for k, v in dest.items()}, "sum", w)
print(f"  basket intensity 2050 = {basket:.6f} kWh/pkm; plane-intra = {inten(pm,df_PM,'plane-intra EU','Gpkm',Y1):.6f}")
per_unit_pkm = SRT_TRIP/LIFE
slope = per_unit_pkm*pop[Y1]/1e9*(inten(pm, df_PM, "plane-intra EU", "Gpkm", Y1) - basket)
print(f"  1 lever unit = {per_unit_pkm:.4f} pkm/person -> slope {slope:.6f} TWh per trip/life")
r0 = act(df_PM, "plane-intra EU", "pkm/person", Y0)/SRT_TRIP*LIFE
r1 = act(df_PM, "plane-intra EU", "pkm/person", Y1)/SRT_TRIP*LIFE
print(f"  contribution(target) vs ref = {(r1-r0)*slope:+.4f} TWh")

print()
print("=== slider checks ===")
def chk(name, ref, tgt, lo, hi, step):
    span = hi-lo
    edge = min(tgt-lo, hi-tgt)/span
    centre = abs((tgt-lo)/span - 0.5)
    dec = max(0, -int(math.floor(math.log10(step)))) if step < 1 else 0
    print(f"  {name:22s} ref={ref:9.3f} tgt={tgt:9.3f} [{lo:g},{hi:g}] step={step:g}"
          f" edge={edge:.3f} {'OK' if edge >= 0.12 else 'FAIL'}"
          f" centre={centre:.3f}{' (near centre!)' if centre < 0.05 else ''}"
          f" dec={dec} refin={'y' if lo <= ref <= hi else 'N'}")

chk("long-haul-flights", act(df_PM,'plane-extra EU','pkm/person',Y0)/LNG_TRIP*LIFE,
    act(df_PM,'plane-extra EU','pkm/person',Y1)/LNG_TRIP*LIFE, 2, 16, 0.25)
chk("short-haul-flights", r0, r1, 10, 80, 1)
chk("long-haul-load", float(oe.loc['liquid-kerosene',Y0]), float(oe.loc['liquid-kerosene',Y1]), 150, 270, 5)
chk("long-haul-fuel", float(ce.loc['liquid-kerosene',Y0]), float(ce.loc['liquid-kerosene',Y1]), 40, 90, 1)
chk("short-haul-load", float(oi.loc['liquid-kerosene',Y0]), float(oi.loc['liquid-kerosene',Y1]), 85, 190, 5)
chk("short-haul-fuel", float(ci.loc['liquid-kerosene',Y0]), float(ci.loc['liquid-kerosene',Y1]), 45, 95, 1)
chk("hydrogen-flights", 0.0, float(g["end_PM_avi_srt"]), 0, 35, 1)
chk("air-freight-fuel", 100.0, 100.0, 60, 130, 1)
chk("shift-to-rail", 100.0, 100*(dest['train-conventional']+dest['train-high speed'])/w, 40, 100, 1)

print()
print("=== leverage swings across slider range (single lever) ===")
def swing_prop(scaled, vT, lo, hi):  return abs(scaled*(hi-lo)/vT)
def swing_inv(scaled, vT, lo, hi):   return abs(scaled*vT*(1/hi-1/lo))
print(f"  long-haul-flights  {swing_prop(twh(pm,'plane-extra EU',Y1), act(df_PM,'plane-extra EU','pkm/person',Y1)/LNG_TRIP*LIFE, 2, 16):.3f} TWh")
print(f"  short-haul-flights {abs(slope)*(80-10):.3f} TWh")
print(f"  long-haul-load     {swing_inv(twh(pm,'plane-extra EU',Y1), float(oe.loc['liquid-kerosene',Y1]), 150, 270):.3f} TWh")
print(f"  long-haul-fuel     {swing_prop(twh(pm,'plane-extra EU',Y1), float(ce.loc['liquid-kerosene',Y1]), 40, 90):.3f} TWh")
print(f"  short-haul-load    {swing_inv(srt_kero[Y1], float(oi.loc['liquid-kerosene',Y1]), 85, 190):.3f} TWh")
print(f"  short-haul-fuel    {swing_prop(srt_kero[Y1], float(ci.loc['liquid-kerosene',Y1]), 45, 95):.3f} TWh")

print()
print("=== air freight ===")
for m in PAX:
    print(f"  {m:16s} tkm/person {act(df_FT,m,'tkm/person',Y0):9.3f} -> {act(df_FT,m,'tkm/person',Y1):9.3f}"
          f"   %tkm {act(df_FT,m,'% of total',Y0):.3f} -> {act(df_FT,m,'% of total',Y1):.3f}"
          f"   kWh/tkm {inten(ft,df_FT,m,'Gtkm',Y0):.4f} -> {inten(ft,df_FT,m,'Gtkm',Y1):.4f}")
print("  freight intensity tkm/person:", float(g['df_SUF'].loc['FT intensity [tkm/person]',Y0]),
      "->", float(g['df_SUF'].loc['FT intensity [tkm/person]',Y1]))

print()
print("=== whole-sector context ===")
allpm = {y: sum(twh(pm, m, y) for m in pm.index) for y in (Y0, Y1)}
allft = {y: sum(twh(ft, m, y) for m in ft.index) for y in (Y0, Y1)}
print(f"  all passenger {allpm[Y0]:.2f} -> {allpm[Y1]:.2f};  all freight {allft[Y0]:.2f} -> {allft[Y1]:.2f}")
print(f"  whole transport {allpm[Y0]+allft[Y0]:.2f} -> {allpm[Y1]+allft[Y1]:.2f}")
print(f"  topic share of 2050 transport: {100*tot[Y1]/(allpm[Y1]+allft[Y1]):.1f}%")
print(f"  topic share of 2019 transport: {100*tot[Y0]/(allpm[Y0]+allft[Y0]):.1f}%")
