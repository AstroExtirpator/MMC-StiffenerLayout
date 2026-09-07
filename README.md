# MMC-StiffenerLayout

This repository contains stiffener layout optimization code for stiffened composite panels, created as part of an MSc thesis titled "An Optimization and Design-for-Manufacturing Framework for Stiffened Composite Panels, Including Novel Morphological Filtering" by Flavio Claudio Padua at Delft University of Technology, in collaboration with Collins Aerospace.

The software optimizes the layout of blade stiffeners on a composite panel skin using the Method of Moving Morphable Components (MMC), driving an iterative feedback loop between Python and Altair OptiStruct. Stiffener geometry is described explicitly by a small set of geometric design variables rather than by an element density field, so the converged result is a set of discrete members with defined positions, lengths, widths and orientations — no grayscale interpretation step is required.

## Method

Each stiffener `i` is one morphable component carrying five design variables, `a_i = [x0, y0, L, W, theta]`: centre coordinates, half-length, width and orientation. The optimization proceeds as follows.

1. **Geometry mapping** — Each component's Topology Description Function (TDF) is a `p`-th order superellipse (`p = 6`), positive inside the member and negative outside. A hard maximum over all components aggregates them into a single global TDF, so overlapping members merge smoothly.

2. **Thickness interpolation** — A polynomial regularized Heaviside of band `±eps` maps the global TDF to an element density `H_e`, which sets the element thickness `T_e = T_skin + H_e * T_stiffener`. By default the stiffener enters the laminate as a dedicated one-sided ply with its own material, so it carries the parallel-axis bending stiffness of a real bonded blade; `STIFFENER_MODEL` selects the alternatives.

3. **Structural analysis** — Each iteration writes a per-element OptiStruct `.fem` deck, solves it as a linear static run, and parses the total compliance from the `.out` file and the element strain energies (ESE) from the `.pch` file.

4. **Analytical sensitivities** — `dC/da` follows the chain rule from the element strain energies through the thickness interpolation, the Heaviside slope and the winner-takes-all indicator of the hard maximum, down to the per-component TDF derivatives. Ties in the maximum split their weight equally, so crossing members in the initial grid are not starved of sensitivity.

5. **Design update** — The Method of Moving Asymptotes (MMA) minimizes compliance subject to `V <= V_target`, recovering the volume multiplier as part of its own subproblem. A projected-gradient scheme with bisection on the multiplier is retained as an alternative (`OPTIMIZER = "LAGRANGIAN"`), but MMA needs no problem-specific tuning and is used throughout the thesis.

The loop terminates on the FEA evaluation budget, or when the normalized design change stays below `CONV_TOL` for `CONV_WINDOW` consecutive steps. The best feasible design is restored, re-analysed with displacement output, and exported as a coincident-shell model so its compliance is directly comparable across integration schemes.

Optional features include mirror and four-fold symmetry enforcement (which also reduces the independent design variables), orientation freezing to the panel axes, and running the optimization on top of a variable-thickness skin taken from a prior free-size optimization (FSO).

## Repository Structure

| File | Purpose |
|------|---------|
| `main.py` | Configuration and main optimization loop |
| `MMC_Classes.py` | Component and problem classes: TDF, Heaviside mapping, initial layouts, symmetry, analytical sensitivities |
| `optimizer.py` | MMA and Lagrangian-bisection design update schemes |
| `helper.py` | FEM parsing, per-element property expansion, OptiStruct interface, plotting, model export |
| `merge.py` | Decoupled baseline: merges an independently optimized skin and stiffener layout into one model |

## Input Models

The optimization templates are OptiStruct `.fem` decks carrying the mesh, material, loads, boundary conditions and an ESE output request. The skin property is assigned to a single element; the Python script expands it to every element in the design domain.

| `numerical_example` | Template | Case |
|---|---|---|
| 1 | `MMC_LAM2.fem` | 200 x 200 mm square panel, four corners fixed, central out-of-plane point load, 2 mm isotropic skin |
| 2 | `MMC_LAM4.fem` | Clamped edges, five point loads, twill-weave lamina |
| 3 | `MMC_LAM5.fem` | Clamped edges, five point loads, unidirectional lamina |
| 4 | `MMC_LAM6.fem` | As case 1, composite skin |
| 5 | `Bulkhead.fem` | 2 x 1 m flat pressure bulkhead, uniform out-of-plane pressure (case study reference load case) |
| 6 | `Bulkhead_compress.fem` | Same panel, in-plane compression |
| 7 | `Bulkhead_asymmetric.fem` | Same panel, varying pressure |

Two further models support the integrated skin-stiffener schemes:

- `FSOrawSMEARr.fem` / `FSOrawSMEARr.sh` — a free-size optimized skin, solved without stiffeners attached. Setting `FSO_SH_FILE = "FSOrawSMEARr.sh"` in `main.py` runs the skin-driven scheme, optimizing the stiffener layout on top of this variable-thickness skin instead of a constant one.
- `MMConly2mm.fem` / `MMConly20mm.fem` — MMC layouts optimized on a flat skin, at 2 mm and 20 mm stiffener height.

`merge.py` combines one of each into the decoupled reference model, the baseline against which the stiffener-driven and skin-driven schemes are scored.

## Requirements

Python 3.9+ with numpy and matplotlib. The structural analysis requires a licensed Altair OptiStruct installation; the solver path is set by `exec_cmd` in `main.py` and `OPTISTRUCT_BAT` in `merge.py`.

```bash
pip install -r requirements.txt
```

## Usage

Edit the configuration constants at the top of `main.py`, then run:

```bash
python main.py
```

The shipped configuration reproduces the case study: the flat pressure bulkhead with the 72-component bulkhead grid as initial layout, 100 mm steel stiffeners, two-plane symmetry, MMA at a move limit of 0.0025, a 200 FEA evaluation budget and a requested stiffener volume fraction of 0.15.

Key settings:

| Constant | Meaning |
|---|---|
| `numerical_example` | Which template to optimize (see table above) |
| `INITIAL_LAYOUT` | Starting component configuration (`0` skin-only, `1`/`2` diagonal crosses, `5`/`6` vertical/horizontal bars, `7`/`8` orthogonal grids, `9` bars at even orientations, `10` bulkhead grid) |
| `STIFFENER_HEIGHT` | Fixed blade height, in mm |
| `STIFFENER_MODEL` | How the stiffener enters the laminate: `MAT1` (own isotropic ply), `MAT8` (skin material), `PLY_SCALING` (uniform skin scaling) |
| `STIFFENER_MATERIAL_MODE` | `HOMOGENIZED` (CLT equivalent of the skin), `MANUAL`, or `TEMPLATE` (first MAT1 card in the deck) |
| `FINAL_VOL_TARGET` | Volume-fraction constraint, as a panel surface-area ratio |
| `MAX_EVAL` | FEA evaluation budget |
| `SYMMETRY` | `OFF`, `ONE_PLANE_X`, `ONE_PLANE_Y`, `TWO_PLANE`, or `FOUR_FOLD` |
| `ORTHO_STIFFENERS` | `False` for free orientations, `"OPT"` to snap and freeze members to the panel axes |
| `HEAVISIDE_EPS` / `SENS_EPS` | Forward and gradient-side regularization bands |
| `FSO_SH_FILE` | A `.sh` from a prior skin free-size optimization; `None` uses the constant-skin templates |

To build a decoupled reference model, set `FSO_STEM` and `MMC_RESULT` at the top of `merge.py` and run:

```bash
python merge.py
```

## Outputs

Everything is written to the folder named by `run_folder` (default `optimization_run`), which is emptied at the start of each run:

- `run_log.txt` — full console log of the run
- `design_XXX.png`, `design_FINAL.png` — element thickness field per iteration
- `components_live.png`, `design_components_FINAL.png` — component outlines
- `contour_live.png`, `design_contour_FINAL.png` — global TDF contour
- `convergence_XXX.png`, `paper_convergence.png`, `specific_compliance_FINAL.png` — convergence history
- `eval_final.*` — re-analysis of the selected design, with displacement output
- `FSO_with_stiffeners_raw.fem` — the optimized panel exported as a coincident-shell model, ready for the free-size and design-for-manufacturing stages

With `CLEANUP_RUN_DIR = True`, per-iteration solver artifacts are deleted at the end of the run, keeping the first iteration, the last iteration, the selected best-feasible iteration and the final evaluation.

## Related

The design-for-manufacturing half of this thesis, which discretizes the free-size optimized skin into a manufacturable blended laminate, lives in [DfM-Skin](https://github.com/AstroExtirpator/DfM-Skin).
