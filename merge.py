"""
merge.py - Decoupled baseline model (FSO skin + MMC stiffener layout).

Builds the DECOUPLED panel model: the FSO and MMC optimizations are run
SEPARATELY and then merged, serving as the performance baseline against which
the sequential FSO-driven-MMC and MMC-driven-FSO approaches are compared (the
sequential approaches should beat the decoupled one).

It uses the SAME modelling as those approaches so the comparison is fair:

  * SKIN: one CQUAD4 per element covering the whole panel, with the FSO
    variable per-element thickness taken from an FSO model's .sh + .fem
    (exactly as main.py builds it). LAM = SMEAR, because a free-sized skin has
    per-ANGLE thicknesses but no known stacking ORDER.

  * STIFFENERS: separate COINCIDENT CQUAD4 shells (matching the DfM MMC-driven
    model, convert_stiffeners_to_flat_side): a duplicate CQUAD4 on the SAME
    four nodes as the skin element it stiffens, carrying only the stiffener
    thickness on its own single-ply property (no LAM entry). The stiffener
    LOCATIONS and local HEIGHTS come from an MMC run, read from EITHER an
    FSO_with_stiffeners.fem (already coincident shells) OR an MMC result whose
    stiffeners are integrated in the skin property (MAT1 / MAT8 / PLY_SCALING
    modes) - in that case the stiffener height is the per-element laminate
    thickness above the baseline skin. Either way the OUTPUT is the
    separate coincident flat-side shells.

MATERIALS: the skin MAT is copied verbatim from the FSO model; the stiffener
MAT1 is the CLT-homogenized isotropic equivalent of that same skin (the SAME
material the FSO-driven MMC export uses for its stiffeners), NOT whatever
material the MMC result stored. Nothing is hard-coded and the models stay
consistent.

Usage:
    python merge.py
"""

import os
import sys
import shutil
import subprocess

from helper import (
    parse_sh_file,
    parse_ply_stack_dvs,
    expand_sh_to_per_ply,
    _extract_base_plies_from_fem,
    _parse_pcomp_plies,
    _parse_pcompg_plies,
    homogenized_isotropic_from_skin,
    import_design_domain,
)

# =====================================================================
# Configuration
# =====================================================================
FSO_STEM = "FSOrawSMEARr"                 # FSO model stem (.fem + .sh): variable skin
MMC_RESULT = "MMConly2mm.fem"  # MMC run export: stiffener layout + heights
OUTPUT_FEM = "merged_decoupled.fem"
OPTISTRUCT_BAT = r"C:\Program Files\Altair\2025.1\hwsolvers\scripts\optistruct.bat"

fso_fem = FSO_STEM + ".fem"
fso_sh = FSO_STEM + ".sh"

for _f in (fso_fem, fso_sh, MMC_RESULT):
    if not os.path.isfile(_f):
        sys.exit(f"Required input not found: {_f}")

print(f"FSO skin model  : {fso_fem}  (+ {fso_sh})")
print(f"MMC stiffeners  : {MMC_RESULT}")
print()


# =====================================================================
# Small parsing helpers (materials + elements), nothing hard-coded
# =====================================================================
def _is_cont(line):
    return bool(line) and line.strip() and line[0] in (" ", "\t", "+", "*")


def _matf8(v):
    """8-char small-field value for a material modulus, keeping the MOST
    precision that fits the field (so the homogenized MAT1 matches the FSO-driven
    export's material exactly), with a scientific fallback for large magnitudes."""
    for prec in (6, 5, 4, 3, 2, 1):
        s = f"{v:.{prec}f}"
        if len(s) <= 8:
            return s.rjust(8)
    return f"{v:.2E}"[:8].rjust(8)


def _read_mat_card(fem_path, mid):
    """Return the verbatim lines (header + continuations) of the MAT1/MAT2/
    MAT8 card whose MID matches `mid`, or None. Copied as-is so the merged
    model uses exactly the same material as its source model."""
    mid = str(mid).strip()
    with open(fem_path, "r") as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if line[:4] in ("MAT1", "MAT2", "MAT8") and line[8:16].strip() == mid:
            card = [line]
            j = i + 1
            while j < len(lines) and _is_cont(lines[j]):
                card.append(lines[j])
                j += 1
            return card
    return None


def _read_cquad4(fem_path):
    """Return {eid: (pid, nodes_field)} and {node_signature: [eids]} for every
    CQUAD4, where nodes_field is the raw 4-node substring (cols 25-...)."""
    quads = {}
    sig_to_eids = {}
    with open(fem_path, "r") as f:
        for line in f:
            if not line.startswith("CQUAD4"):
                continue
            try:
                eid = int(line[8:16])
                pid = int(line[16:24])
            except ValueError:
                continue
            nodes_field = line[24:].rstrip("\r\n")
            nodes = tuple(nodes_field[k:k + 8].strip() for k in range(0, 32, 8))
            quads[eid] = (pid, nodes_field)
            sig_to_eids.setdefault(nodes, []).append(eid)
    return quads, sig_to_eids


def _read_all_props(fem_path):
    """Return {pid: [ply, ...]} for every PCOMP and PCOMPG, using the shared
    parsers so ply layouts are read identically to the rest of the code."""
    props = {}
    with open(fem_path, "r") as f:
        lines = f.readlines()
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        is_pcompg = line.startswith("PCOMPG")
        is_pcomp = line.startswith("PCOMP") and not is_pcompg
        if is_pcomp or is_pcompg:
            try:
                pid = int(line[8:16])
            except ValueError:
                i += 1
                continue
            cont = []
            j = i + 1
            while j < n and _is_cont(lines[j]):
                cont.append(lines[j])
                j += 1
            props[pid] = (_parse_pcompg_plies(cont) if is_pcompg
                          else _parse_pcomp_plies(cont))
            i = j
            continue
        i += 1
    return props


# =====================================================================
# 1. FSO variable skin (from the FSO .fem + .sh) - same as main.py
# =====================================================================
print("Parsing FSO variable skin ...")
base_plies = _extract_base_plies_from_fem(fso_fem)
if not base_plies:
    sys.exit(f"Could not extract skin plies from {fso_fem}")
angle_to_dv = parse_ply_stack_dvs(fso_fem)
elem_mults = parse_sh_file(fso_sh)

skin_mid = str(base_plies[0]["mid"]).strip()     # actual skin material (not hard-coded)
skin_angles = [p.get("theta", "0.0") for p in base_plies]
print(f"  Skin: {len(base_plies)} plies, angles {skin_angles}, MAT MID={skin_mid}")
if angle_to_dv:
    print(f"  DV->angle map (BALANCE-aware): {angle_to_dv}")

skin_mat_card = _read_mat_card(fso_fem, skin_mid)
if skin_mat_card is None:
    sys.exit(f"Skin material MID={skin_mid} not found in {fso_fem}")


# =====================================================================
# 2. MMC stiffener layout (locations + local heights) from the MMC result
# =====================================================================
# Two MMC-result formats are accepted; the merged output always uses the
# separate coincident flat-side stiffener shells regardless of which is read:
#
#   A. COINCIDENT SHELLS (an FSO_with_stiffeners.fem export): the stiffener is
#      already a duplicate CQUAD4 on the same nodes as its skin element, on a
#      single-ply stiffener PCOMP. Take the pair's higher-EID member.
#
#   B. STIFFENERS INTEGRATED IN THE SKIN PROPERTY (the MAT1 / MAT8 /
#      PLY_SCALING modes, e.g. an MMConly export): no duplicate elements; a
#      stiffened element's per-element laminate is simply THICKER than the
#      baseline (unstiffened) skin. The extra thickness IS the stiffener height
#      (H*stiffener_height) - true whether it is a dedicated extra ply or a
#      uniformly scaled stack, so this one rule covers all those modes.
print("\nParsing MMC stiffener layout ...")
mmc_quads, mmc_sig = _read_cquad4(MMC_RESULT)
mmc_props = _read_all_props(MMC_RESULT)

stiffener_thickness = {}     # {base_eid: stiffener thickness (H*height)}
stiff_mid = None
detection = "coincident stiffener shells"

# --- Method A: coincident stiffener shells ---
for nodes, eids in mmc_sig.items():
    if len(eids) < 2:
        continue
    eids_sorted = sorted(eids)
    stiff_pid = mmc_quads[eids_sorted[-1]][0]
    plies = mmc_props.get(stiff_pid, [])
    if len(plies) != 1:
        continue
    t = float(plies[0]["T"])
    if t <= 1e-8:
        continue
    stiffener_thickness[eids_sorted[0]] = t
    if stiff_mid is None:
        stiff_mid = str(plies[0]["mid"]).strip()

# --- Method B: stiffeners integrated in the skin property ---
if not stiffener_thickness:
    detection = "stiffeners integrated in the skin property"
    elem_total = {eid: sum(float(p["T"]) for p in mmc_props[pid])
                  for eid, (pid, _) in mmc_quads.items() if pid in mmc_props}
    if not elem_total:
        sys.exit(f"No PCOMP/PCOMPG properties found for the CQUAD4s in "
                 f"{MMC_RESULT}.")
    # baseline (unstiffened) skin = the thinnest per-element laminate
    base_eid_min = min(elem_total, key=elem_total.get)
    skin_baseline = elem_total[base_eid_min]
    n_skin = len(mmc_props[mmc_quads[base_eid_min][0]])
    for eid, total in elem_total.items():
        stiff_t = total - skin_baseline
        if stiff_t > 1e-3:
            stiffener_thickness[eid] = stiff_t
            if stiff_mid is None:
                plies = mmc_props[mmc_quads[eid][0]]
                # a dedicated stiffener ply sits AFTER the n_skin base plies;
                # ply-scaling has none, so fall back to the skin material.
                stiff_mid = str((plies[n_skin] if len(plies) > n_skin
                                 else plies[0])["mid"]).strip()

if not stiffener_thickness:
    sys.exit(f"No stiffeners found in {MMC_RESULT} (neither coincident shells "
             f"nor an integrated-skin thickness increase).")

n_stiffened = len(stiffener_thickness)
_ts = list(stiffener_thickness.values())
print(f"  Detected via: {detection}")
print(f"  Stiffened elements: {n_stiffened}")
print(f"  Stiffener thickness range: {min(_ts):.4f} - {max(_ts):.4f} mm")

# STIFFENER MATERIAL = the CLT-homogenized isotropic MAT1 of the UNDERLYING
# (FSO) skin - the SAME material the FSO-driven MMC export uses for its
# stiffeners - so the decoupled baseline is directly comparable. ONLY the
# stiffener HEIGHTS come from the MMC result; its own stiffener material (a
# different skin/steel/etc.) is irrelevant here.
_homog = homogenized_isotropic_from_skin(fso_fem, base_plies)
if _homog is None:
    sys.exit(f"Could not homogenize the skin material of {fso_fem} for the "
             f"stiffener MAT1.")
stiff_E, stiff_nu = _homog
stiff_G = stiff_E / (2.0 * (1.0 + stiff_nu))

# a free MAT id, not colliding with any material already in the FSO skin model
_used = set()
for _l in open(fso_fem):
    if _l[:4] in ("MAT1", "MAT2", "MAT8", "MAT9"):
        try:
            _used.add(int(_l[8:16]))
        except ValueError:
            pass
out_stiff_mid = str((max(_used) + 1) if _used else 1)
stiff_mat_card = [f"MAT1    {out_stiff_mid:>8}{_matf8(stiff_E)}{_matf8(stiff_G)}"
                  f"{stiff_nu:>8.4f}\n"]
print(f"  Stiffener MAT1 = homogenized skin: E={stiff_E:.1f}, G={stiff_G:.1f}, "
      f"nu={stiff_nu:.4f} (MID {out_stiff_mid})")


# =====================================================================
# 3. Detect the mesh + BC set ids from the FSO template (nothing hard-coded)
# =====================================================================
elem_ids, X, Y, bounds = import_design_domain(fso_fem)
elem_id_set = set(int(e) for e in elem_ids)
n_elem = len(elem_id_set)

with open(fso_fem, "r") as f:
    template_lines = f.readlines()

# max element id (over all shells) -> a safe, collision-free id offset for the
# coincident stiffener CQUAD4s and their properties.
_max_elem_id = 0
for line in template_lines:
    if line[:6] in ("CQUAD4", "CTRIA3", "CQUAD8", "CTRIA6"):
        try:
            _max_elem_id = max(_max_elem_id, int(line[8:16]))
        except ValueError:
            pass
STIFF_EID_OFFSET = 10 ** (len(str(_max_elem_id)) + 1)
STIFF_PID_OFFSET = STIFF_EID_OFFSET


def _stiff_pid(eid):
    return eid + STIFF_PID_OFFSET


# SPC / LOAD set ids from the template case control (fall back to bulk cards).
spc_id, load_id = None, None
for line in template_lines:
    s = line.strip().upper().replace(" ", "")
    if s.startswith("SPC=") and spc_id is None:
        try:
            spc_id = int(s.split("=", 1)[1])
        except ValueError:
            pass
    elif s.startswith("LOAD=") and load_id is None:
        try:
            load_id = int(s.split("=", 1)[1])
        except ValueError:
            pass
if spc_id is None:
    for line in template_lines:
        if line.startswith("SPC"):
            try:
                spc_id = int(line[8:16]); break
            except ValueError:
                pass
if load_id is None:
    for line in template_lines:
        if line[:5] in ("FORCE", "PLOAD", "MOMEN"):
            try:
                load_id = int(line[8:16]); break
            except ValueError:
                pass
print(f"\n  Mesh: {n_elem} elements | SPC set {spc_id} | LOAD set {load_id} | "
      f"stiffener id offset {STIFF_EID_OFFSET}")


# =====================================================================
# 4. Build the per-element properties (SMEAR skin + stiffener shells)
# =====================================================================
def _fmt8(val):
    s = f"{val:.4f}"
    return f"{s:>8}" if len(s) <= 8 else f"{val:.2E}"[:8].rjust(8)


def _skin_pcomp(eid):
    """Per-element variable-thickness skin PCOMP (LAM=SMEAR), FSO ply
    thicknesses, one ply per listed angle - identical to the export's skin."""
    raw_sh = elem_mults.get(eid, [])
    mults = (expand_sh_to_per_ply(raw_sh, base_plies, angle_to_dv)
             if raw_sh else [1.0] * len(base_plies))
    entries = [(skin_mid, (mults[k] if k < len(mults) else 1.0) * ply["T"],
                ply.get("theta", "0.0")) for k, ply in enumerate(base_plies)]
    out = [f"PCOMP   {eid:>8}" + " " * 48 + f"{'SMEAR':>8}\n"]
    for j in range(0, len(entries), 2):
        m1, t1, th1 = entries[j]
        cont = f"+       {m1:>8}{_fmt8(t1)}{th1:>8}{'YES':>8}"
        if j + 1 < len(entries):
            m2, t2, th2 = entries[j + 1]
            cont += f"{m2:>8}{_fmt8(t2)}{th2:>8}{'YES':>8}"
        out.append(cont + "\n")
    return out


def _stiffener_pcomp(eid):
    """Single-ply PCOMP (NO LAM entry - explicit stack) for the coincident
    flat-side stiffener shell: the MMC stiffener material at the MMC local
    thickness. Only the multi-ply skin uses LAM=SMEAR."""
    return [f"PCOMP   {_stiff_pid(eid):>8}\n",
            f"+       {out_stiff_mid:>8}{_fmt8(stiffener_thickness[eid])}"
            f"{'0.0':>8}{'YES':>8}\n"]


print("\nBuilding merged model ...")
pcomp_lines = ["$$\n$$  Per-element variable-thickness skin PCOMP (SMEAR)\n$$\n"]
for eid in sorted(elem_id_set):
    pcomp_lines.extend(_skin_pcomp(eid))
pcomp_lines.append("$$\n$$  Coincident flat-side stiffener shell PCOMP\n$$\n")
for eid in sorted(stiffener_thickness):
    pcomp_lines.extend(_stiffener_pcomp(eid))


# =====================================================================
# 5. Assemble the analysis deck
# =====================================================================
out = []
out.append("$$ Merged decoupled FSO skin + MMC stiffener analysis model\n")
out.append("$$ Generated by merge.py\n")
out.append("ESE(PUNCH) = ALL\n")
out.append("DISP(PUNCH) = ALL\n")
out.append("SUBCASE        1\n")
out.append("  LABEL merged_decoupled_analysis\n")
out.append("ANALYSIS STATICS\n")
out.append(f"  SPC = {spc_id:>8}\n")
out.append(f"  LOAD = {load_id:>8}\n")
out.append("BEGIN BULK\n")

# --- SET cards (needed by any SPC/LOAD that references them) ---
in_set = False
for line in template_lines:
    if line.startswith("SET "):
        out.append(line); in_set = True
    elif in_set and _is_cont(line):
        out.append(line)
    else:
        in_set = False

# --- Skin CQUAD4 (PID = EID) + coincident stiffener CQUAD4 (same nodes) ---
out.append("$$\n$$  Elements: skin (PID=EID) + coincident flat-side stiffeners\n$$\n")
for line in template_lines:
    if not line.startswith("CQUAD4"):
        continue
    try:
        eid = int(line[8:16])
    except ValueError:
        out.append(line); continue
    if eid not in elem_id_set:
        out.append(line); continue
    eid_field = f"{eid:>8}"
    nodes_field = line[24:]
    out.append("CQUAD4  " + eid_field + eid_field + nodes_field)  # skin: PID=EID
    if eid in stiffener_thickness:                                # coincident shell
        out.append("CQUAD4  " + f"{eid + STIFF_EID_OFFSET:>8}"
                   + f"{_stiff_pid(eid):>8}" + nodes_field)

# --- Properties ---
out.extend(pcomp_lines)

# --- Materials copied verbatim from their source models ---
out.append(f"$$\n$$  Skin material (MID {skin_mid}) from {fso_fem}\n$$\n")
out.extend(skin_mat_card)
out.append(f"$$\n$$  Stiffener MAT1 (MID {out_stiff_mid}) = homogenized FSO skin\n$$\n")
out.extend(stiff_mat_card)

# --- GRID + boundary conditions from the FSO template ---
out.append("$$\n$$  GRID + boundary conditions (from FSO template)\n$$\n")
for line in template_lines:
    if line.startswith("GRID"):
        out.append(line)
keep = False
for line in template_lines:
    if line[:6] in ("SPC   ", "SPCADD", "SPC1  ", "FORCE ", "PLOAD ", "PLOAD2",
                    "PLOAD4", "MOMENT", "GRAV  ", "LOAD  ") or line[:3] == "SPC" \
            or line[:5] in ("FORCE", "PLOAD", "MOMEN"):
        out.append(line); keep = True
    elif keep and _is_cont(line):
        out.append(line)
    else:
        keep = False

out.append("ENDDATA\n")

with open(OUTPUT_FEM, "w") as f:
    f.writelines(out)

print(f"  Wrote {OUTPUT_FEM}")
print(f"    {n_elem} skin elements (SMEAR variable thickness)")
print(f"    {n_stiffened} coincident flat-side stiffener shells")
print()


# =====================================================================
# 6. Run OptiStruct (if available)
# =====================================================================
if not os.path.isfile(OPTISTRUCT_BAT):
    print(f"OptiStruct not found at:\n  {OPTISTRUCT_BAT}\nSkipping solve.")
    sys.exit(0)

run_dir = os.path.join(os.path.dirname(os.path.abspath(OUTPUT_FEM)), "merge_run")
os.makedirs(run_dir, exist_ok=True)
for f in os.listdir(run_dir):
    try:
        os.unlink(os.path.join(run_dir, f))
    except OSError:
        pass
shutil.copy(OUTPUT_FEM, os.path.join(run_dir, OUTPUT_FEM))

print(f"Running OptiStruct on {OUTPUT_FEM} ...")
result = subprocess.run(
    [OPTISTRUCT_BAT, os.path.abspath(os.path.join(run_dir, OUTPUT_FEM)),
     "-optskip", "-nt", "18"],
    cwd=os.path.abspath(run_dir),
    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
)
if result.returncode != 0:
    print(f"OptiStruct error:\n{result.stdout[-2000:]}")
    sys.exit(1)

# --- Parse compliance (guard against a failed/licence-less solve) ---
jobname = OUTPUT_FEM.replace(".fem", "")
out_path = os.path.join(run_dir, f"{jobname}.out")
compliance = None
if os.path.exists(out_path):
    with open(out_path, "r") as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if "Subcase" in line and "Compliance" in line:
            for sub in lines[i + 1:i + 10]:
                parts = sub.split()
                if len(parts) >= 2 and parts[0].isdigit():
                    try:
                        compliance = float(parts[1]); break
                    except ValueError:
                        continue
            if compliance:
                break

print("\n" + "=" * 50)
print("  DECOUPLED (merged) ANALYSIS RESULT")
if compliance and compliance > 0.0:
    print(f"  Compliance = {compliance:.2f} N.mm")
else:
    print(f"  Compliance = INVALID ({compliance}) - solve failed (licence?);"
          f" do not use this run.")
print("=" * 50)
print("Baseline for comparison against FSO-driven and MMC-driven results.")
