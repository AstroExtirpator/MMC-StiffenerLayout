import os
import sys
import shutil

import numpy as np

from helper import *
from MMC_Classes import MMC_Problem
from optimizer import MMA, lagrangian_step

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

# ── Case ──────────────────────────────────────────────────────────────────────
# 1: MMC_LAM2  four fixed corners, central point load
# 2: MMC_LAM4  clamped edges, five point loads, twill-weave lamina
# 3: MMC_LAM5  clamped edges, five point loads, unidirectional lamina
# 4: MMC_LAM6  as 1, composite skin
# 5: Bulkhead  2 x 1 m flat pressure bulkhead (Sun et al. Sect. 4.3)
# 6: Bulkhead_compress     in-plane compression
# 7: Bulkhead_asymmetric   varying pressure
numerical_example = 5

# ── Stiffener modelling ───────────────────────────────────────────────────────
# how H*T_stiffener enters a PCOMP/PCOMPG laminate:
#   "MAT1"        dedicated one-sided ply with its own isotropic material
#   "MAT8"        as MAT1, but the ply uses the skin's MAT8 at theta = 0
#                 (STIFFENER_MATERIAL_MODE ignored)
#   "PLY_SCALING" all skin plies scaled uniformly, no separate stiffener ply
STIFFENER_MODEL = "MAT1"

# only used when STIFFENER_MODEL == "MAT1"
#   "HOMOGENIZED"  CLT-equivalent E, nu of the skin laminate
#   "MANUAL"       STIFFENER_MANUAL_E / _NU below
#   "TEMPLATE"     first MAT1 card in the template .fem
STIFFENER_MATERIAL_MODE = "HOMOGENIZED"
STIFFENER_MANUAL_E = 200.0
STIFFENER_MANUAL_NU = 0.3

STIFFENER_HEIGHT = 100.0         # fixed blade height T_stiffener (mm)

# ── Optimization ──────────────────────────────────────────────────────────────
#  0: no stiffeners (skin-only baseline)
#  1: 4x4 touching diagonal crosses      2: 4x4 separated diagonal crosses
#  5: vertical bars                      6: horizontal bars
#  7: 4x4 vertical + horizontal grid     8: 20x20 vertical + horizontal grid
#  9: six bars at even orientations     10: bulkhead grid (Sun et al. Fig. 4b)
INITIAL_LAYOUT = 10

OPTIMIZER = "MMA"                # "MMA" or "LAGRANGIAN"
MMA_MOVE_LIMIT = 0.0025          # max per-step change of a normalized variable
MOVE_LIMIT = 0.0025              # Lagrangian move limit
DAMPING = 0.5                    # Lagrangian oscillation damping

FINAL_VOL_TARGET = 0.15          # volume-fraction constraint V <= V_target
MAX_EVAL = 200                   # FEA evaluation budget

# Heaviside regularization band. Keep <= 1 so the peak reaches the full
# stiffener height; too small leaves elements with no sensitivity samples.
HEAVISIDE_EPS = 0.9
# Gradient-side band. Wider than HEAVISIDE_EPS removes the dead gradient core
# inside each member without shrinking the blade (a deliberately smoothed,
# inconsistent sensitivity). Set equal to HEAVISIDE_EPS for exact gradients.
SENS_EPS = 0.9

CONV_TOL = 1e-4                  # normalized design-change threshold
CONV_WINDOW = 10                 # consecutive steps below it before stopping

# "OFF", "ONE_PLANE_X", "ONE_PLANE_Y", "TWO_PLANE" (4-fold reflection),
# "FOUR_FOLD" (D4, square domains only)
SYMMETRY = "TWO_PLANE"

# False  free orientations
# "OPT"  theta removed from the optimization: snapped to the nearest axis at
#        initialization and frozen there, so orientation can only change by
#        material transfer between horizontal and vertical members
ORTHO_STIFFENERS = False

CLEANUP_RUN_DIR = True           # delete per-iteration solver artifacts

# ── FSO integration ───────────────────────────────────────────────────────────
# .sh from a prior skin free-size optimization; None = constant-skin templates
FSO_SH_FILE = None
T_BASE = 0.5                     # flat base thickness under stiffeners (mm)
# "ON_TOP"    stiffener added on the variable skin, skin unchanged
# "FLAT_BASE" skin under each footprint replaced by a flat T_BASE base
FSO_STIFFENER_INTEGRATION = "ON_TOP"

# ══════════════════════════════════════════════════════════════════════════════

if ORTHO_STIFFENERS not in (False, "OPT"):
    raise ValueError("ORTHO_STIFFENERS must be False or 'OPT'")
if ORTHO_STIFFENERS == "OPT" and INITIAL_LAYOUT in {1, 2, 9}:
    raise ValueError(
        f"ORTHO_STIFFENERS='OPT' needs an axis-aligned initial layout "
        f"(5, 6, 7, 8 or 10); layout {INITIAL_LAYOUT} is diagonal and "
        f"would collapse to horizontal bars.")

_use_flat_base = (FSO_SH_FILE is not None
                  and FSO_STIFFENER_INTEGRATION == "FLAT_BASE")
_t_base = T_BASE if _use_flat_base else None

INITIAL_COMPLIANCE = None
LAST_ESE_ARRAY = None
HISTORY_EVALS = []
HISTORY_COMPLIANCE = []
HISTORY_VOLUME = []

template, _initial_layout = select_template(numerical_example, FSO_SH_FILE,
                                            INITIAL_LAYOUT)

# bulkhead examples reproduce the paper: steel stiffeners as a separate MAT1 ply
if FSO_SH_FILE is None and numerical_example > 4:
    STIFFENER_MODEL = "MAT1"
    STIFFENER_MATERIAL_MODE = "TEMPLATE"
# SMEAR homogenizes the stacking sequence and erases the flexural anisotropy the
# layouts depend on, so always write the full explicit stack
USE_SMEAR = False

expanded_fem = "MMC_LAMe.fem"
run_folder = "optimization_run"
exec_cmd = r"C:\Program Files\Altair\2025.1\hwsolvers\scripts\optistruct.bat"

if not os.path.isfile(exec_cmd):
    print(f"OptiStruct not found at:\n  {exec_cmd}")
    print("No files will be written. Exiting.")
    sys.exit(0)

_log_fh = prepare_run_folder(run_folder)

# =============================================================================
#                    TEMPLATE EXPANSION & PROBLEM SETUP
# =============================================================================
_use_mat1 = (STIFFENER_MODEL == "MAT1")
_use_mat8 = (STIFFENER_MODEL == "MAT8")
_use_ply_scaling = (STIFFENER_MODEL == "PLY_SCALING")

_stiff_E_override, _stiff_nu_override = resolve_stiffener_material(
    STIFFENER_MATERIAL_MODE, _use_mat8, STIFFENER_MANUAL_E,
    STIFFENER_MANUAL_NU, template)

elem_ids, X, Y, bounds = import_design_domain(template)

expand_properties_for_mmc(template, expanded_fem, elem_ids,
                          sh_file_path=FSO_SH_FILE,
                          stiffener_E_override=_stiff_E_override,
                          stiffener_nu_override=_stiff_nu_override,
                          stiffener_model=STIFFENER_MODEL,
                          use_smear=USE_SMEAR)
shutil.copy(expanded_fem, os.path.join(run_folder, expanded_fem))

skin_t, prop_type, ply_info = detect_skin_thickness(template,
                                                    sh_file_path=FSO_SH_FILE)

problem = MMC_Problem(bounds=bounds, x_mesh=X, y_mesh=Y, elem_ids=elem_ids,
                      max_iter=MAX_EVAL,
                      skin_thickness=build_skin_thickness(prop_type, ply_info,
                                                          skin_t, elem_ids),
                      stiffener_height=STIFFENER_HEIGHT,
                      symmetry=SYMMETRY,
                      t_base=_t_base,
                      initial_layout=_initial_layout,
                      heaviside_eps=HEAVISIDE_EPS,
                      sens_eps=SENS_EPS,
                      ortho=(ORTHO_STIFFENERS == "OPT"))

_opt_stiffener_model = "MAT1" if _use_mat1 else ("MAT8" if _use_mat8 else
                                                 "PLY_SCALING")

fea = OptiStructOptimizer(
    template_path=os.path.join(run_folder, expanded_fem),
    exec_path=exec_cmd, problem=problem, work_dir=run_folder,
    prop_type=prop_type, ply_info=ply_info,
    stiffener_model=_opt_stiffener_model)

LB, UB, SCALE = build_design_bounds(problem, bounds)

_frozen_theta_idx = (np.arange(len(problem.components)) * 5 + 4
                     if ORTHO_STIFFENERS == "OPT"
                     else np.array([], dtype=int))

mma_optimizer = MMA(n=len(LB), move=MMA_MOVE_LIMIT) if OPTIMIZER == "MMA" else None


def run_fea(x_norm):
    """One FEA. Returns (compliance, vol_frac)."""
    global INITIAL_COMPLIANCE, LAST_ESE_ARRAY

    problem.set_design_variables(x_norm * SCALE + LB)
    jobname = f"iter_{fea.current_iter + 1}"
    fea.generate_input_file(problem, f"{jobname}.fem")

    try:
        fea.run_solver(f"{jobname}.fem")
        compliance, LAST_ESE_ARRAY, _ = fea.read_results(jobname)
        fea.current_iter += 1
    except Exception as e:
        print(f"  Solver Error: {e}")
        compliance = None

    # read_results returns 0.0 without raising when the solve produced nothing
    # (licence failure, empty .out, ...). Such a run must never win the "best
    # feasible" comparison, so penalize it instead of accepting C = 0.
    valid = (compliance is not None and np.isfinite(compliance)
             and compliance > 0.0)
    if not valid:
        penalty = (INITIAL_COMPLIANCE * 10.0
                   if INITIAL_COMPLIANCE and INITIAL_COMPLIANCE > 0.0 else 1e9)
        print(f"  WARNING: invalid/failed FEA for {jobname} (C = {compliance}) "
              f"-> penalizing to {penalty:.3e} so it can never be chosen as best.")
        compliance = penalty

    if INITIAL_COMPLIANCE is None and valid:
        INITIAL_COMPLIANCE = compliance

    vol_frac = problem.calculate_volume_fraction(current_iter=fea.current_iter)

    HISTORY_EVALS.append(fea.current_iter)
    HISTORY_COMPLIANCE.append(compliance)
    HISTORY_VOLUME.append(vol_frac)

    plot_iteration(problem, fea.current_iter,
                   save_path=os.path.join(run_folder,
                                          f"design_{fea.current_iter:03d}.png"))
    plot_components_vector(problem, os.path.join(run_folder, "components_live.png"),
                           title=f"Component layout (iteration {fea.current_iter})")
    plot_components_contour(problem, os.path.join(run_folder, "contour_live.png"),
                            title=f"Component contour (iteration {fea.current_iter})")
    return compliance, vol_frac


def get_vol_at(x_norm):
    """Volume fraction at x_norm without running FEA."""
    problem.set_design_variables(x_norm * SCALE + LB)
    return problem.calculate_volume_fraction(current_iter=fea.current_iter)


# =============================================================================
#                                MAIN LOOP
# =============================================================================
print("\n" + "=" * 60)
print(f"  MMC STIFFENER LAYOUT OPTIMIZATION")
if OPTIMIZER == "MMA":
    print(f"  Method: MMA (move limit = {MMA_MOVE_LIMIT})")
else:
    print(f"  Method: Lagrangian gradient + bisection "
          f"(move limit = {MOVE_LIMIT}, damping = {DAMPING})")
print(f"  Regularization: hard-max TDF + polynomial Heaviside "
      f"(eps = {HEAVISIDE_EPS})")
print(f"  Stiffener model = {_opt_stiffener_model}")
print(f"  VF target = {FINAL_VOL_TARGET * 100:.1f}% (surface area ratio)")
print(f"  Budget = {MAX_EVAL} FEA evaluations")
print(f"  Stiffener height = {STIFFENER_HEIGHT:.1f} mm (FIXED)")
print(f"  Stiffener material = {STIFFENER_MATERIAL_MODE}"
      f"{f' (E={_stiff_E_override:.1f}, nu={_stiff_nu_override:.4f})' if _stiff_E_override is not None else ''}")
print(f"  Symmetry = {SYMMETRY}"
      f"{f' ({len(problem.components)} independent -> {len(problem._get_symmetry_clones())} effective components)' if SYMMETRY != 'OFF' else ''}")
if FSO_SH_FILE is not None:
    if _use_flat_base:
        print(f"  Stiffener integration: FLAT_BASE (skin replaced by "
              f"T_BASE = {T_BASE:.2f} mm under stiffeners)")
    else:
        print(f"  Stiffener integration: ON_TOP (stiffeners added on the "
              f"variable skin, skin unchanged)")
    print(f"    delta range: [{np.min(problem.delta):.2f}, {np.max(problem.delta):.2f}] mm")
    print(f"    skin_t range: [{np.min(problem.skin_thickness):.4f}, "
          f"{np.max(problem.skin_thickness):.4f}] mm")
if SENS_EPS != HEAVISIDE_EPS:
    print(f"  Sensitivity smoothing: forward eps = {HEAVISIDE_EPS}, "
          f"gradient eps = {SENS_EPS}")
print("=" * 60)

x_current = (problem.get_design_variables() - LB) / SCALE
compliance, vol_frac = run_fea(x_current)

NO_STIFFENERS = (len(problem.components) == 0)

if FSO_SH_FILE is not None:
    plot_skin_thickness(problem, ply_info,
                        save_path=os.path.join(run_folder,
                                               "skin_thickness_FSO.png"))

print(f"\n  Initial: C = {compliance:.2f}, VF = {vol_frac:.4f} ({vol_frac * 100:.1f}%)")
print(f"  Target:  VF = {FINAL_VOL_TARGET:.4f} ({FINAL_VOL_TARGET * 100:.1f}%)\n")

best_overall_compliance = compliance
best_overall_x = x_current.copy()
best_overall_vol = vol_frac
best_overall_iter = fea.current_iter
best_feasible_compliance = np.inf
best_feasible_x = None
best_feasible_vol = None
best_feasible_iter = None
x_prev = x_current.copy()
converged_count = 0

if NO_STIFFENERS:
    best_feasible_compliance = compliance
    best_feasible_x = x_current.copy()
    best_feasible_vol = vol_frac
    best_feasible_iter = fea.current_iter
    print("\n  Skin-only mode: no stiffeners, skipping optimization loop.")

for step in (range(1, MAX_EVAL) if not NO_STIFFENERS else []):
    if fea.current_iter >= MAX_EVAL:
        break

    problem.set_design_variables(x_current * SCALE + LB)

    raw_dC = problem.calculate_analytical_gradients(
        LAST_ESE_ARRAY, current_iter=fea.current_iter)
    dC_dx = (raw_dC * SCALE) / max(INITIAL_COMPLIANCE, 1e-12)

    dv_real = problem.compute_vol_gradients_analytical(
        problem.get_design_variables(), current_iter=fea.current_iter)
    dV_dx = dv_real * SCALE

    if len(_frozen_theta_idx):
        dC_dx[_frozen_theta_idx] = 0.0
        dV_dx[_frozen_theta_idx] = 0.0

    if OPTIMIZER == "MMA":
        x_new = mma_optimizer.update(x_current, dC_dx,
                                     vol_frac - FINAL_VOL_TARGET, dV_dx)
    else:
        x_new = lagrangian_step(x_current, dC_dx, dV_dx, FINAL_VOL_TARGET,
                                MOVE_LIMIT, get_vol_at)
        x_new = (1.0 - DAMPING) * x_new + DAMPING * x_prev

    if len(_frozen_theta_idx):
        x_new[_frozen_theta_idx] = x_current[_frozen_theta_idx]

    design_change = np.max(np.abs(x_new - x_current))
    x_prev = x_current.copy()
    x_current = x_new

    compliance, vol_frac = run_fea(x_current)

    if compliance < best_overall_compliance:
        best_overall_compliance = compliance
        best_overall_x = x_current.copy()
        best_overall_vol = vol_frac
        best_overall_iter = fea.current_iter

    if vol_frac <= FINAL_VOL_TARGET + 0.005 and compliance < best_feasible_compliance:
        best_feasible_compliance = compliance
        best_feasible_x = x_current.copy()
        best_feasible_vol = vol_frac
        best_feasible_iter = fea.current_iter

    feasible = " <-- FEASIBLE" if vol_frac <= FINAL_VOL_TARGET + 0.005 else ""
    print(f"   Step {step:3d} | Eval {fea.current_iter:3d} | "
          f"C = {compliance:8.2f} | VF = {vol_frac:.4f} "
          f"(tgt {FINAL_VOL_TARGET:.4f}) | "
          f"dx = {design_change:.6f}{feasible}")

    plot_convergence(HISTORY_EVALS, HISTORY_COMPLIANCE,
                     os.path.join(run_folder,
                                  f"convergence_{fea.current_iter:03d}.png"))
    plot_paper_convergence(HISTORY_EVALS, HISTORY_COMPLIANCE, HISTORY_VOLUME,
                           FINAL_VOL_TARGET,
                           os.path.join(run_folder, "paper_convergence.png"))

    converged_count = converged_count + 1 if design_change < CONV_TOL else 0
    if converged_count >= CONV_WINDOW:
        print(f"\n  CONVERGED at step {step}: design change < {CONV_TOL} "
              f"for {CONV_WINDOW} consecutive steps.")
        break

# =============================================================================
#                          RESTORE BEST & EXPORT
# =============================================================================
print("\n" + "=" * 60)
if best_feasible_x is not None:
    chosen_compliance = best_feasible_compliance
    chosen_vol = best_feasible_vol
    chosen_iter = best_feasible_iter
    problem.set_design_variables(best_feasible_x * SCALE + LB)
    print(f"  BEST FEASIBLE: C = {chosen_compliance:.2f}, VF = {chosen_vol:.4f} "
          f"(iter {chosen_iter})")
else:
    chosen_compliance = best_overall_compliance
    chosen_vol = best_overall_vol
    chosen_iter = best_overall_iter
    problem.set_design_variables(best_overall_x * SCALE + LB)
    print(f"  BEST COMPLIANCE: C = {chosen_compliance:.2f}, VF = {chosen_vol:.4f} "
          f"(iter {chosen_iter}) -- not feasible")
print(f"  Total FEA evals: {fea.current_iter}")
print("=" * 60)

_equiv_E, _equiv_nu = resolve_export_material(_stiff_E_override,
                                              _stiff_nu_override,
                                              ply_info, template)
print(f"\n  Export stiffener material ({STIFFENER_MATERIAL_MODE}): "
      f"E = {_equiv_E:.1f}, nu = {_equiv_nu:.4f}")

export_fso_model_with_stiffeners(
    problem, run_folder,
    fso_template_path=template,
    prop_type=prop_type, ply_info=ply_info,
    equiv_E=_equiv_E, equiv_nu=_equiv_nu,
    t_base=_t_base,
    stiffener_model=_opt_stiffener_model,
    fso_integration=FSO_STIFFENER_INTEGRATION,
    output_filename="FSO_with_stiffeners_raw.fem")

# =============================================================================
#                              FINAL EVALUATION
# =============================================================================
final_comp, max_disp = evaluate_final_design(fea, problem, run_folder)
final_vf = problem.calculate_volume_fraction(current_iter=fea.current_iter)

print(f"\n  FINAL DESIGN (from iter {chosen_iter}):")
print(f"    Compliance = {final_comp:.2f} N.mm "
      f"(best raw feasible = {chosen_compliance:.2f})")
print(f"    Max displacement = {max_disp:.3f} mm")
print(f"    VF = {final_vf:.6f} (target = {FINAL_VOL_TARGET:.6f})")
print(f"    Stiffeners = {len(problem.components)}")
for i, c in enumerate(problem.components):
    print(f"    {i + 1:2d}: W={c.width:.2f}mm, L={c.L:.1f}mm, "
          f"theta={np.degrees(c.theta):.1f}deg")

save_final_plots(problem, run_folder, fea.current_iter, HISTORY_EVALS,
                 HISTORY_COMPLIANCE, HISTORY_VOLUME, FINAL_VOL_TARGET,
                 final_comp, final_vf)

# The optimizer minimises the thickness-interpolation compliance; this measures
# the exported coincident-shell model, which is the number comparable across the
# MMC-driven (DfM) and decoupled (merge.py) approaches.
print("\n" + "=" * 60)
print("  COINCIDENT-SHELL MODEL COMPLIANCE (cross-model comparison)")
print("=" * 60)
analyze_coincident_shell_compliance(fea, run_folder,
                                    "FSO_with_stiffeners_raw.fem",
                                    "FSO-driven MMC, RAW")
print("=" * 60)

if CLEANUP_RUN_DIR:
    cleanup_run_folder(run_folder, fea.current_iter, best_feasible_iter)

print("\n  Optimization Complete!")

sys.stdout = sys.stdout.terminal
_log_fh.close()
