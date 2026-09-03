import numpy as np
import sys
import re
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def parse_grid_line(line):
    """
    Parses a GRID card line, handling both comma-separated (Free Field)
    and Fixed-Width (Small Field) formats where numbers might touch.
    """
    line = line.strip()

    if ',' in line:
        parts = line.split(',')
        return [p.strip() for p in parts if p.strip()]
    else:
        def get_field(start, end):
            if len(line) > start:
                return line[start:min(end, len(line))].strip()
            return "0.0"

        card_name = get_field(0, 8)
        node_id = get_field(8, 16)
        cp_id = get_field(16, 24)
        x_str = get_field(24, 32)
        y_str = get_field(32, 40)
        z_str = get_field(40, 48)

        return [card_name, node_id, cp_id, x_str, y_str, z_str]


def clean_optistruct_float(val_str):
    """Converts OptiStruct 'short-form' scientific notation to standard python floats."""
    val_str = val_str.strip()
    if not val_str:
        return 0.0

    cleaned = re.sub(r'(?<=\d)(?=[+-])', 'E', val_str)

    try:
        return float(cleaned)
    except ValueError:
        return 0.0


def _parse_pcomp_plies(continuation_lines):
    """
    Parse ALL plies from PCOMP continuation lines.

    Each continuation line can hold up to 2 plies in Nastran Small Field format:
        +       MID1    T1      THETA1  SOUT1   MID2    T2      THETA2  SOUT2
        cols:   [8:16]  [16:24] [24:32] [32:40] [40:48] [48:56] [56:64] [64:72]

    Returns:
        list of dicts: [{'mid': str, 'T': float, 'theta': str, 'sout': str}, ...]
    """
    plies = []
    for line in continuation_lines:
        if len(line) < 24:
            continue
        prefix = line[:8]
        has_prefix = (prefix.strip().startswith('+') or
                      prefix.strip().startswith('*'))

        if has_prefix:
            # --- Ply 1: fields at [8:16] MID, [16:24] T, [24:32] THETA, [32:40] SOUT ---
            mid1 = line[8:16].strip() if len(line) > 8 else ''
            t1_str = line[16:24].strip() if len(line) > 16 else ''
            theta1 = line[24:32].strip() if len(line) > 24 else ''
            sout1 = line[32:40].strip() if len(line) > 32 else ''
            if t1_str:
                plies.append({'mid': mid1,
                              'T': clean_optistruct_float(t1_str),
                              'theta': theta1, 'sout': sout1})

            # --- Ply 2: fields at [40:48] MID, [48:56] T, [56:64] THETA, [64:72] SOUT ---
            mid2 = line[40:48].strip() if len(line) > 40 else ''
            t2_str = line[48:56].strip() if len(line) > 48 else ''
            theta2 = line[56:64].strip() if len(line) > 56 else ''
            sout2 = line[64:72].strip() if len(line) > 64 else ''
            if t2_str:
                plies.append({'mid': mid2,
                              'T': clean_optistruct_float(t2_str),
                              'theta': theta2, 'sout': sout2})
        else:
            # No prefix marker: MID at [0:8], T at [8:16], ...
            mid1 = line[0:8].strip()
            t1_str = line[8:16].strip() if len(line) > 8 else ''
            theta1 = line[16:24].strip() if len(line) > 16 else ''
            sout1 = line[24:32].strip() if len(line) > 24 else ''
            if t1_str:
                plies.append({'mid': mid1,
                              'T': clean_optistruct_float(t1_str),
                              'theta': theta1, 'sout': sout1})

            mid2 = line[32:40].strip() if len(line) > 32 else ''
            t2_str = line[40:48].strip() if len(line) > 40 else ''
            theta2 = line[48:56].strip() if len(line) > 48 else ''
            sout2 = line[56:64].strip() if len(line) > 56 else ''
            if t2_str:
                plies.append({'mid': mid2,
                              'T': clean_optistruct_float(t2_str),
                              'theta': theta2, 'sout': sout2})

    return plies


def _parse_pcompg_plies(continuation_lines):
    """
    Parse plies from PCOMPG continuation lines.

    PCOMPG has ONE ply per continuation line with a global ply ID:
        +       GPLYID  MID     T       THETA           SOUT
        cols:   [8:16]  [16:24] [24:32] [32:40] [40:48] [48:56]

    Without prefix marker (leading spaces count as continuation):
        GPLYID  MID     T       THETA           SOUT
        [0:8]   [8:16]  [16:24] [24:32] [32:40] [40:48]

    Returns:
        list of dicts: [{'gplyid': str, 'mid': str, 'T': float, 'theta': str, 'sout': str}, ...]
    """
    plies = []
    for line in continuation_lines:
        if len(line) < 24:
            continue
        prefix = line[:8]
        has_prefix = (prefix.strip().startswith('+') or
                      prefix.strip().startswith('*') or
                      prefix.strip() == '')

        if has_prefix:
            # GPLYID at [8:16], MID at [16:24], T at [24:32], THETA at [32:40],
            # skip [40:48], SOUT at [48:56]
            gplyid = line[8:16].strip() if len(line) > 8 else ''
            mid = line[16:24].strip() if len(line) > 16 else ''
            t_str = line[24:32].strip() if len(line) > 24 else ''
            theta = line[32:40].strip() if len(line) > 32 else ''
            sout = line[48:56].strip() if len(line) > 48 else ''
            if t_str:
                plies.append({'gplyid': gplyid, 'mid': mid,
                              'T': clean_optistruct_float(t_str),
                              'theta': theta, 'sout': sout})

    return plies


def parse_sh_file(sh_file_path):
    """
    Parse an OptiStruct .sh (shape/property) file from free-size optimization.

    The .sh file contains per-element, per-ply thickness multipliers (0..1).
    Structure:
        Line 1: version string
        Line 2: header (n_elements, n_iterations, ..., n_plies_per_element)
        Then for each element: a line with (element_id, n_values),
        followed by n_values lines of thickness multipliers.

    Returns:
        dict: {element_id: [ply1_mult, ply2_mult, ...], ...}
              Multipliers in range [0, 1] where 1 = full nominal ply thickness.
    """
    with open(sh_file_path, 'r') as f:
        lines = f.readlines()

    # Parse header
    header = lines[1].split()
    n_elements = int(header[0])
    n_plies = int(header[-1])

    # print(f"  Parsing .sh file: {n_elements} elements, {n_plies} plies per element")

    elem_thickness_map = {}
    i = 2  # start after header
    for _ in range(n_elements):
        if i >= len(lines):
            break
        parts = lines[i].strip().split()
        eid = int(parts[0])
        n_vals = int(parts[1])
        values = []
        for j in range(n_vals):
            values.append(float(lines[i + 1 + j].strip()))
        elem_thickness_map[eid] = values
        i += n_vals + 1

    # print(f"  Parsed {len(elem_thickness_map)} element thickness records")
    return elem_thickness_map


def parse_ply_stack_dvs(fem_file_path):
    """
    Parse PLY/STACK/DSIZE or PCOMP/DSIZE cards from a template to determine
    the design-variable-to-angle mapping used by free-size optimization.

    Handles two FSO template formats:
      1. PLY/STACK/DSIZE:  DSIZE references a STACK; PLY cards define angles.
      2. PCOMP/DSIZE:      DSIZE references a PCOMP; ply angles are in the PCOMP.

    In both cases, BALANCE constraints cause multiple angles (e.g. 45/-45)
    to share one DV.  The .sh file then has per-DV (not per-ply) multipliers.

    Returns:
        dict or None: {angle_str: dv_index, ...}
        e.g. {'0.0': 0, '90.0': 1, '45.0': 2, '-45.0': 2}
        Returns None if no DSIZE or no parseable angles found.
    """
    with open(fem_file_path, 'r') as f:
        lines = f.readlines()

    def _is_continuation(line):
        if not line or not line.strip():
            return False
        return line[0] in (' ', '\t', '+', '*')

    # --- Collect BALANCE pairs from DSIZE ---
    balanced_pairs = []
    dsize_ref_type = None
    dsize_ref_id = None
    for i, line in enumerate(lines):
        if line.startswith("DSIZE"):
            dsize_ref_type = line[16:24].strip()
            dsize_ref_id = line[24:32].strip()
            j = i + 1
            while j < len(lines):
                cl = lines[j]
                if not cl.strip() or cl.strip()[0] not in ('+', '*'):
                    break
                if "BALANCE" in cl:
                    toks = cl.split()
                    try:
                        bi = toks.index("BALANCE")
                        a1 = toks[bi + 1]
                        a2 = toks[bi + 2]
                        balanced_pairs.append((a1, a2))
                    except (ValueError, IndexError):
                        pass
                j += 1
            break

    if dsize_ref_type is None:
        return None

    # --- Case 1: DSIZE references STACK (PLY/STACK format) ---
    ply_angles_list = []
    if dsize_ref_type == "STACK":
        ply_angles = {}
        for line in lines:
            if line.startswith("PLY"):
                parts = line.split()
                if len(parts) >= 5:
                    try:
                        ply_id = int(parts[1])
                        theta = parts[4].strip()
                        ply_angles[ply_id] = theta
                    except (ValueError, IndexError):
                        continue

        stack_ply_ids = []
        for i, line in enumerate(lines):
            if line.startswith("STACK"):
                rest = line[16:]
                for tok in rest.split():
                    try:
                        stack_ply_ids.append(int(tok))
                    except ValueError:
                        pass
                j = i + 1
                while j < len(lines):
                    cl = lines[j]
                    if not cl.strip() or cl.strip()[0] not in ('+', '*'):
                        break
                    for tok in cl.strip()[1:].split():
                        try:
                            stack_ply_ids.append(int(tok))
                        except ValueError:
                            pass
                    j += 1
                break

        if not stack_ply_ids or not ply_angles:
            return None

        for pid in stack_ply_ids:
            angle = ply_angles.get(pid)
            if angle is not None:
                ply_angles_list.append(angle)

    # --- Case 2: DSIZE references PCOMP (no PLY/STACK) ---
    elif dsize_ref_type == "PCOMP":
        for i, line in enumerate(lines):
            if line.startswith("PCOMP") and not line.startswith("PCOMPG"):
                pid_str = line[8:16].strip()
                if pid_str == dsize_ref_id:
                    cont_lines = []
                    j = i + 1
                    while j < len(lines) and _is_continuation(lines[j]):
                        cont_lines.append(lines[j])
                        j += 1
                    plies = _parse_pcomp_plies(cont_lines)
                    for p in plies:
                        if p.get('theta'):
                            ply_angles_list.append(p['theta'])
                    break

    if not ply_angles_list:
        return None

    # --- Build angle_to_dv from ply angle sequence + BALANCE pairs ---
    angle_to_dv = {}
    dv_idx = 0
    seen_pairs = set()

    for angle in ply_angles_list:
        matched = False
        for a1, a2 in balanced_pairs:
            if angle == a1 or angle == a2:
                key = tuple(sorted([a1, a2]))
                if key not in seen_pairs:
                    seen_pairs.add(key)
                    angle_to_dv[a1] = dv_idx
                    angle_to_dv[a2] = dv_idx
                    dv_idx += 1
                matched = True
                break
        if not matched:
            if angle not in angle_to_dv:
                angle_to_dv[angle] = dv_idx
                dv_idx += 1

    return angle_to_dv if angle_to_dv else None


def expand_sh_to_per_ply(sh_mults, ply_info, angle_to_dv):
    """
    Expand per-DV .sh multipliers to per-ply multipliers using angle mapping.

    Args:
        sh_mults: list of N_DV multipliers from .sh file
        ply_info: list of ply dicts (each with 'theta' key)
        angle_to_dv: dict mapping angle_str -> dv_index (from parse_ply_stack_dvs)

    Returns:
        list of per-ply multipliers (same length as ply_info)
    """
    if angle_to_dv is None or not angle_to_dv:
        if len(sh_mults) == len(ply_info):
            return sh_mults
        return [sh_mults[i] if i < len(sh_mults) else 1.0
                for i in range(len(ply_info))]

    per_ply = []
    for ply in ply_info:
        theta = ply.get('theta', '')
        if not theta:
            per_ply.append(1.0)
            continue

        dv_idx = angle_to_dv.get(theta)
        if dv_idx is not None and dv_idx < len(sh_mults):
            per_ply.append(sh_mults[dv_idx])
        else:
            per_ply.append(1.0)

    return per_ply


def compute_equivalent_isotropic_from_pcomp(ply_info, E1, E2, G12, nu12):
    """
    Compute equivalent isotropic E and nu from a PCOMP layup using Classical
    Lamination Theory (CLT) A-matrix inversion.

    Uses the in-plane stiffness matrix [A] of the laminate, normalised by
    total thickness, to extract effective membrane moduli. The stiffener
    beam model needs a single (E, nu) pair; this function returns values
    that reproduce the laminate's in-plane stiffness for an isotropic
    PSHELL/MAT1.

    Args:
        ply_info: list of {'mid': str, 'T': float, 'theta': str, ...} dicts.
        E1, E2, G12, nu12: lamina engineering constants of the ply material.
                  These are REQUIRED and must be the ACTUAL constants of the
                  skin MAT8 being homogenized (read from the template — see
                  homogenized_isotropic_from_skin). No material is hardcoded.

    Returns:
        (E_equiv, nu_equiv)  in the same unit system as the input moduli.
    """
    nu21 = nu12 * E2 / E1

    # Reduced stiffness matrix Q for a single ply (ply axes)
    denom = 1.0 - nu12 * nu21
    Q11 = E1 / denom
    Q22 = E2 / denom
    Q12 = nu12 * E2 / denom
    Q66 = G12

    # Accumulate A-matrix (in-plane stiffness)
    A = np.zeros((3, 3))
    for ply in ply_info:
        t_k = ply['T']
        theta_deg = float(ply['theta']) if ply['theta'] else 0.0
        theta = np.radians(theta_deg)

        c = np.cos(theta)
        s = np.sin(theta)
        c2, s2, cs = c**2, s**2, c * s

        # Transform Q to global coordinates: Q_bar
        Qbar = np.zeros((3, 3))
        Qbar[0, 0] = Q11*c2**2 + 2*(Q12 + 2*Q66)*c2*s2 + Q22*s2**2
        Qbar[1, 1] = Q11*s2**2 + 2*(Q12 + 2*Q66)*c2*s2 + Q22*c2**2
        Qbar[0, 1] = (Q11 + Q22 - 4*Q66)*c2*s2 + Q12*(c2**2 + s2**2)
        Qbar[1, 0] = Qbar[0, 1]
        Qbar[0, 2] = (Q11 - Q12 - 2*Q66)*c*s*c2 + (Q12 - Q22 + 2*Q66)*c*s*s2
        Qbar[2, 0] = Qbar[0, 2]
        Qbar[1, 2] = (Q11 - Q12 - 2*Q66)*c*s*s2 + (Q12 - Q22 + 2*Q66)*c*s*c2
        Qbar[2, 1] = Qbar[1, 2]
        Qbar[2, 2] = (Q11 + Q22 - 2*Q12 - 2*Q66)*c2*s2 + Q66*(c2**2 + s2**2)

        A += Qbar * t_k

    # Normalise by total thickness -> effective membrane stiffness
    h = sum(p['T'] for p in ply_info)
    if h < 1e-12:
        raise ValueError("compute_equivalent_isotropic_from_pcomp: laminate "
                         "has zero total thickness; cannot homogenize.")

    A_norm = A / h  # units of stress (GPa or MPa depending on input)

    # Extract effective isotropic properties from A_norm
    # For an isotropic material: A11 = E/(1-nu^2), A12 = nu*E/(1-nu^2), A66 = E/(2*(1+nu))
    # Average x and y directions for near-isotropic estimate
    E_x = A_norm[0, 0] - A_norm[0, 1]**2 / A_norm[1, 1]
    E_y = A_norm[1, 1] - A_norm[0, 1]**2 / A_norm[0, 0]
    E_equiv = (E_x + E_y) / 2.0
    nu_equiv = A_norm[0, 1] / max(A_norm[0, 0], 1e-12)

    # Clamp to physical range
    nu_equiv = np.clip(nu_equiv, 0.05, 0.49)

    return float(E_equiv), float(nu_equiv)


def read_mat8_lamina_constants(template_path, mid):
    """(E1, E2, nu12, G12) of the MAT8 card with the given MID in the template,
    or None if there is no such MAT8. Reads the ACTUAL values from the file —
    nothing hardcoded."""
    mid = str(mid).strip()
    with open(template_path, 'r') as f:
        for line in f:
            if line.startswith("MAT8") and line[8:16].strip() == mid:
                suf = line[16:]
                return (clean_optistruct_float(suf[0:8]),    # E1
                        clean_optistruct_float(suf[8:16]),   # E2
                        clean_optistruct_float(suf[16:24]),  # nu12
                        clean_optistruct_float(suf[24:32]))  # G12
    return None


def read_mat1_constants(template_path, mid):
    """(E, nu) of the MAT1 card with the given MID in the template, or None."""
    mid = str(mid).strip()
    with open(template_path, 'r') as f:
        for line in f:
            if line.startswith("MAT1") and line[8:16].strip() == mid:
                return (clean_optistruct_float(line[16:24]),   # E
                        clean_optistruct_float(line[32:40]))   # nu
    return None


def homogenized_isotropic_from_skin(template_path, ply_list):
    """CLT-equivalent isotropic (E, nu) of the skin laminate, built entirely
    from the ACTUAL skin material in `template_path` — no hardcoded lamina
    constants for any panel.

      - composite (MAT8) skin: A-matrix homogenization of the `ply_list`
        layup using the MAT8's real E1/E2/nu12/G12.
      - isotropic (MAT1) skin: the MAT1's own (E, nu) (nothing to homogenize).

    `ply_list` is a list of {'mid','T','theta'} dicts (the skin plies); the
    skin material MID is taken from the first ply. Returns (E, nu), or None if
    the skin material cannot be found in the template."""
    if not ply_list:
        return None
    skin_mid = str(ply_list[0].get('mid', '')).strip()
    c8 = read_mat8_lamina_constants(template_path, skin_mid)
    if c8 is not None:
        E1, E2, nu12, G12 = c8
        return compute_equivalent_isotropic_from_pcomp(
            ply_list, E1=E1, E2=E2, G12=G12, nu12=nu12)
    c1 = read_mat1_constants(template_path, skin_mid)
    if c1 is not None:
        return c1   # isotropic skin: the equivalent IS the material itself
    return None


def read_template_mat1(fem_file_path, exclude_mids=(999999,)):
    """
    Read E and nu from the first MAT1 card in a template .fem file.

    Used when the stiffener material is defined directly in the template
    (e.g. the steel stiffeners of the bulkhead example) instead of being
    CLT-homogenized from the skin laminate.

    MAT1 Small Field format: MID [8:16], E [16:24], G [24:32], NU [32:40].
    If NU is blank but G is given, nu is recovered from G = E/(2(1+nu)).

    Returns:
        (E, nu) floats.
    Raises:
        ValueError if no usable MAT1 card is found.
    """
    with open(fem_file_path, 'r') as f:
        for line in f:
            if not line.startswith("MAT1"):
                continue
            mid_str = line[8:16].strip()
            try:
                mid = int(mid_str)
            except ValueError:
                continue
            if mid in exclude_mids:
                continue

            E = clean_optistruct_float(line[16:24]) if len(line) > 16 else 0.0
            g_str = line[24:32].strip() if len(line) > 24 else ''
            nu_str = line[32:40].strip() if len(line) > 32 else ''

            if nu_str:
                nu = clean_optistruct_float(nu_str)
            elif g_str and E > 0.0:
                G = clean_optistruct_float(g_str)
                nu = E / (2.0 * G) - 1.0
            else:
                nu = 0.3

            if E > 0.0:
                return E, nu

    raise ValueError(f"No usable MAT1 card found in {fem_file_path}")


def detect_skin_thickness(fem_file_path, sh_file_path=None):
    """
    Reads the template .fem file and determines the base skin thickness
    from the FIRST property card found (PSHELL or PCOMP).

    If sh_file_path is provided, parses the .sh file from a prior FSO run
    and returns prop_type='PCOMP_VARIABLE' with per-element thickness data.
    This works regardless of whether the template uses PSHELL or PCOMP.

    Supports:
      - PSHELL: reads the T field (cols 24-31)
      - PCOMP:  parses all plies (up to 2 per continuation line) and sums thicknesses
      - PCOMP_VARIABLE: .sh thickness multipliers per element (with PCOMP or PSHELL template)

    Returns:
        (skin_thickness, prop_type, ply_info)
        where:
          - For PSHELL: ply_info = None
          - For PCOMP:  ply_info = list of {'mid','T','theta','sout'} dicts
           - For PCOMP_VARIABLE: ply_info = {
                 'base_plies': [...],          # nominal ply definitions
                 'elem_multipliers': {eid: [m1, m2, ...], ...},  # per-element
                 'angle_to_dv': {...},         # angle -> DV index mapping
                 'equiv_E': float,             # equivalent isotropic E
                 'equiv_nu': float             # equivalent isotropic nu
             }
    """
    # ----------------------------------------------------------------
    # If a .sh file is provided, it takes priority: PCOMP_VARIABLE
    # ----------------------------------------------------------------
    if sh_file_path is not None:
        elem_mults = parse_sh_file(sh_file_path)

        sample_eid = next(iter(elem_mults))
        n_dvs = len(elem_mults[sample_eid])

        base_plies = _extract_base_plies_from_fem(fem_file_path)

        angle_to_dv = parse_ply_stack_dvs(fem_file_path)

        if base_plies is None or (len(base_plies) != n_dvs and angle_to_dv is None):
            base_plies = _infer_standard_layup(n_dvs, fem_file_path)

        ply_thicknesses = [p['T'] for p in base_plies]

        total_per_elem = {}
        for eid, raw_sh in elem_mults.items():
            expanded = expand_sh_to_per_ply(raw_sh, base_plies, angle_to_dv)
            total_per_elem[eid] = sum(m * t for m, t in zip(expanded, ply_thicknesses))
        median_t = float(np.median(list(total_per_elem.values())))

        # Compute equivalent isotropic properties from the base layup
        _homog = homogenized_isotropic_from_skin(fem_file_path, base_plies)
        if _homog is None:
            raise ValueError(f"Could not resolve skin material to homogenize "
                             f"in {fem_file_path}")
        equiv_E, equiv_nu = _homog

        # print(f"  Detected PCOMP_VARIABLE skin: "
        #       f"{len(base_plies)} plies, {len(elem_mults)} elements")
        # print(f"    Nominal ply thicknesses: {ply_thicknesses}")
        # print(f"    Total laminate range: "
        #       f"{min(total_per_elem.values()):.4f} - "
        #       f"{max(total_per_elem.values()):.4f}")
        # print(f"    Median total thickness: {median_t:.4f}")
        # print(f"    Equivalent isotropic: E = {equiv_E:.2f}, "
        #       f"nu = {equiv_nu:.4f}")

        variable_info = {
            'base_plies': base_plies,
            'elem_multipliers': elem_mults,
            'angle_to_dv': angle_to_dv,
            'equiv_E': equiv_E,
            'equiv_nu': equiv_nu,
        }
        return median_t, "PCOMP_VARIABLE", variable_info

    # ----------------------------------------------------------------
    # No .sh file: standard PSHELL / PCOMP detection
    # ----------------------------------------------------------------
    with open(fem_file_path, 'r') as f:
        lines = f.readlines()

    def _is_continuation(line):
        if not line or not line.strip():
            return False
        return line[0] in (' ', '\t', '+', '*')

    i = 0
    while i < len(lines):
        line = lines[i]

        # --- PSHELL ---
        if line.startswith("PSHELL"):
            pid_str = line[8:16].strip()
            if pid_str:
                t_str = line[24:32].strip()
                t_val = clean_optistruct_float(t_str) if t_str else 0.0
                # print(f"  Detected PSHELL skin (PID={pid_str}): T = {t_val:.4f}")
                return t_val, "PSHELL", None

        # --- PCOMPG (must check before PCOMP) ---
        elif line.startswith("PCOMPG"):
            pid_str = line[8:16].strip()
            if pid_str:
                cont_lines = []
                j = i + 1
                while j < len(lines) and _is_continuation(lines[j]):
                    cont_lines.append(lines[j])
                    j += 1

                base_plies = _parse_pcompg_plies(cont_lines)
                ply_thicknesses = [p['T'] for p in base_plies]
                total_t = sum(ply_thicknesses)

                return total_t, "PCOMPG", base_plies

        # --- PCOMP ---
        elif line.startswith("PCOMP"):
            pid_str = line[8:16].strip()
            if pid_str:
                cont_lines = []
                j = i + 1
                while j < len(lines) and _is_continuation(lines[j]):
                    cont_lines.append(lines[j])
                    j += 1

                base_plies = _parse_pcomp_plies(cont_lines)

                # Symmetric layup (LAM=SYM/SYSMEAR): plies listed are the
                # half-stack, so mirror them to get the full laminate
                lam_field = line[64:72].strip().upper() if len(line) > 64 else ''
                if lam_field.startswith('SY'):
                    base_plies = base_plies + [{**p} for p in reversed(base_plies)]

                ply_thicknesses = [p['T'] for p in base_plies]
                total_t = sum(ply_thicknesses)

                return total_t, "PCOMP", base_plies

        i += 1

    print("  Warning: No PSHELL or PCOMP found. Defaulting T_skin = 2.0")
    return 2.0, "UNKNOWN", None


def _extract_base_plies_from_fem(fem_file_path):
    """
    Try to extract PCOMP or PCOMPG ply definitions from a .fem file.
    Returns a list of ply dicts, or None if no composite property found.
    """
    with open(fem_file_path, 'r') as f:
        lines = f.readlines()

    def _is_continuation(line):
        if not line or not line.strip():
            return False
        return line[0] in (' ', '\t', '+', '*')

    for i, line in enumerate(lines):
        if line.startswith("PCOMPG"):
            # PCOMPG: one ply per continuation line with global ply ID
            pid_str = line[8:16].strip()
            if pid_str:
                cont_lines = []
                j = i + 1
                while j < len(lines) and _is_continuation(lines[j]):
                    cont_lines.append(lines[j])
                    j += 1
                plies = _parse_pcompg_plies(cont_lines)
                if plies:
                    return plies

        elif line.startswith("PCOMP") and not line.startswith("PCOMPG"):
            pid_str = line[8:16].strip()
            if pid_str:
                cont_lines = []
                header_line = line
                j = i + 1
                while j < len(lines) and _is_continuation(lines[j]):
                    cont_lines.append(lines[j])
                    j += 1
                plies = _parse_pcomp_plies(cont_lines)
                if plies:
                    lam_field = header_line[64:72].strip().upper() if len(header_line) > 64 else ''
                    if lam_field.startswith('SY'):
                        mirrored = [{**p} for p in reversed(plies)]
                        plies = plies + mirrored
                    return plies
    return None


def _infer_standard_layup(n_plies, fem_file_path):
    """
    Infer a standard symmetric balanced layup for a given number of plies.

    Reads the template .fem to get the total laminate thickness from
    PSHELL, PCOMP, or PCOMPG, then distributes it evenly across plies
    using standard aerospace angle conventions.
    """
    # Try to get total thickness from any property card
    total_t = None
    with open(fem_file_path, 'r') as f:
        for line in f:
            if line.startswith("PSHELL"):
                t_str = line[24:32].strip()
                if t_str:
                    total_t = clean_optistruct_float(t_str)
                break

    # If no PSHELL, try extracting from PCOMPG/PCOMP plies
    if total_t is None:
        plies = _extract_base_plies_from_fem(fem_file_path)
        if plies is not None:
            total_t = sum(p['T'] for p in plies)

    if total_t is None:
        total_t = 2.0  # fallback

    ply_t = total_t / n_plies

    # Standard angle sequence
    if n_plies == 8:
        angles = [0.0, 45.0, -45.0, 90.0, 90.0, -45.0, 45.0, 0.0]
    elif n_plies == 4:
        angles = [0.0, 45.0, -45.0, 90.0]
    else:
        base_angles = [0.0, 45.0, -45.0, 90.0]
        half = n_plies // 2
        angles = []
        for k in range(half):
            angles.append(base_angles[k % len(base_angles)])
        # Mirror for symmetry
        angles = angles + angles[::-1]
        # Trim or extend to exact count
        angles = angles[:n_plies]

    plies = []
    for angle in angles:
        plies.append({
            'mid': '1',
            'T': ply_t,
            'theta': f"{angle:.1f}",
            'sout': ''
        })

    # print(f"    Inferred layup: {[float(a) for a in angles]}")
    # print(f"    Ply thickness: {ply_t:.4f} mm (total: {total_t:.4f} mm)")

    return plies


def import_design_domain(fem_file_path):
    '''Read .fem file to extract design domain.'''
    nodes = {}
    elements = []

    print(f"Reading mesh from: {fem_file_path}...")

    with open(fem_file_path, 'r') as f:
        for line in f:
            if line.startswith('$'): continue

            if line.startswith('GRID'):
                parts = parse_grid_line(line)
                nid = int(parts[1])
                x = clean_optistruct_float(parts[3])
                y = clean_optistruct_float(parts[4])
                z = clean_optistruct_float(parts[5])
                nodes[nid] = np.array([x, y, z])

            if line.startswith('CQUAD4'):
                if ',' in line:
                    parts = line.replace(',', ' ').split()
                    eid = int(parts[1])
                    nids = [int(p) for p in parts[3:7]]
                else:
                    eid = int(line[8:16].strip())
                    nids = [
                        int(line[24:32].strip()),
                        int(line[32:40].strip()),
                        int(line[40:48].strip()),
                        int(line[48:56].strip()),
                    ]
                elements.append([eid] + nids)

            if line.startswith('CTRIA3'):
                if ',' in line:
                    parts = line.replace(',', ' ').split()
                    eid = int(parts[1])
                    nids = [int(p) for p in parts[3:6]]
                else:
                    eid = int(line[8:16].strip())
                    nids = [
                        int(line[24:32].strip()),
                        int(line[32:40].strip()),
                        int(line[40:48].strip()),
                    ]
                elements.append([eid] + nids)

    print(f"Imported {len(nodes)} nodes and {len(elements)} elements.")

    elem_ids = []
    elem_centers = []

    for elem in elements:
        eid = elem[0]
        node_ids = elem[1:]
        coords = [nodes[n] for n in node_ids if n in nodes]
        if len(coords) == len(node_ids):
            center = np.mean(coords, axis=0)
            elem_ids.append(eid)
            elem_centers.append(center)
        else:
            print(f"Warning: Element {eid} has missing nodes.")

    elem_ids = np.array(elem_ids)
    elem_centers = np.array(elem_centers)

    X = elem_centers[:, 0]
    Y = elem_centers[:, 1]

    domain_bounds = {
        'x_min': float(np.min(X)),
        'x_max': float(np.max(X)),
        'y_min': float(np.min(Y)),
        'y_max': float(np.max(Y))
    }
    return elem_ids, X, Y, domain_bounds


def expand_properties_for_mmc(template_path, output_path, elem_ids,
                               sh_file_path=None,
                               stiffener_E_override=None, stiffener_nu_override=None,
                                stiffener_model="MAT1",
                                use_smear=True):
    """Generate per-element property/material cards from a template FEM.

    use_smear: LAM field of the expanded per-element PCOMP cards.
      True  -> "SMEAR" (stacking sequence ignored; legacy behavior)
      False -> blank (full explicit stack written; stacking sequence,
               ply offsets and bending coupling are captured)
    """
    _use_mat1 = (stiffener_model == "MAT1")

    # print(f"Generating unique properties for {len(elem_ids)} elements...")

    with open(template_path, 'r') as f:
        lines = f.readlines()

    # --- Helper: detect Nastran continuation lines ---
    def _is_continuation(line):
        if not line or not line.strip():
            return False
        return line[0] in (' ', '\t', '+', '*')

    # --- Helper: replace ALL MIDs in a PCOMP ply continuation line ---
    def _replace_ply_mids(ply_line, new_mid):
        """Replace MID for BOTH plies (if present) on a continuation line."""
        if len(ply_line) < 8:
            return ply_line
        prefix = ply_line[:8]
        has_prefix = (prefix.strip().startswith('+') or
                      prefix.strip().startswith('*'))

        result = ply_line
        if has_prefix:
            # Ply 1 MID at [8:16]
            result = result[:8] + f"{new_mid:>8}" + result[16:]
            # Ply 2 MID at [40:48] (only if present)
            if len(result) > 48 and result[40:48].strip():
                result = result[:40] + f"{new_mid:>8}" + result[48:]
        else:
            # Ply 1 MID at [0:8]
            result = f"{new_mid:>8}" + result[8:]
            # Ply 2 MID at [32:40] (only if present)
            if len(result) > 40 and result[32:40].strip():
                result = result[:32] + f"{new_mid:>8}" + result[40:]

        return result

    # --- PASS 1: Detect property type and extract templates ---
    prop_type = None
    template_pid = None
    template_mid = None
    pshell_template_suffix = ""
    pcomp_header_line = ""
    pcomp_ply_lines = []
    mat_template_suffix = ""
    found_prop_template = False
    found_mat_template = False
    composite_mid = None        # MID actually used by composite plies
    composite_mat_suffix = ""   # MAT8 suffix for the real composite material

    i = 0
    while i < len(lines):
        line = lines[i]

        if not found_prop_template and line.startswith("PSHELL"):
            pid_str = line[8:16].strip()
            if pid_str:
                prop_type = "PSHELL"
                template_pid = pid_str
                pshell_template_suffix = line[16:]
                found_prop_template = True

        elif not found_prop_template and line.startswith("PCOMPG"):
            pid_str = line[8:16].strip()
            if pid_str:
                prop_type = "PCOMPG"
                template_pid = pid_str
                pcomp_header_line = line
                pcomp_ply_lines = []
                j = i + 1
                while j < len(lines) and _is_continuation(lines[j]):
                    pcomp_ply_lines.append(lines[j])
                    j += 1
                found_prop_template = True
                i = j
                continue

        elif not found_prop_template and line.startswith("PCOMP"):
            pid_str = line[8:16].strip()
            if pid_str:
                prop_type = "PCOMP"
                template_pid = pid_str
                pcomp_header_line = line
                pcomp_ply_lines = []
                j = i + 1
                while j < len(lines) and _is_continuation(lines[j]):
                    pcomp_ply_lines.append(lines[j])
                    j += 1
                found_prop_template = True
                i = j
                continue

        # Collect ALL MAT8 cards (we need to find the right one)
        if not found_mat_template and line.startswith("MAT8"):
            mid_str = line[8:16].strip()
            if mid_str:
                template_mid = mid_str
                mat_template_suffix = line[16:]
                found_mat_template = True

        i += 1

    if not found_prop_template:
        print("Error: Could not find any PSHELL, PCOMP, or PCOMPG property in template.")
        return
    if not found_mat_template:
        print("Error: Could not find any MAT8 material in template.")
        return

    # --- Parse plies and find the composite material ---
    ply_info = []
    if prop_type == "PCOMPG":
        ply_info = _parse_pcompg_plies(pcomp_ply_lines)
    elif prop_type == "PCOMP":
        ply_info = _parse_pcomp_plies(pcomp_ply_lines)

    # Expand symmetric layup (LAM=SYM, SYSMEAR, etc.)
    if prop_type == "PCOMP" and ply_info:
        lam_field = pcomp_header_line[64:72].strip().upper() if len(pcomp_header_line) > 64 else ''
        if lam_field.startswith('SY'):
            mirrored = []
            for p in reversed(ply_info):
                mirrored.append({**p})
            ply_info = ply_info + mirrored

    if ply_info:
        # Get the MID that the plies actually reference
        composite_mid = ply_info[0]['mid']

        # Find that specific MAT8 card in the file
        with open(template_path, 'r') as f2:
            for mat_line in f2:
                if mat_line.startswith("MAT8") and mat_line[8:16].strip() == composite_mid:
                    composite_mat_suffix = mat_line[16:]
                    break
    elif prop_type == "PSHELL" and template_mid:
        # For PSHELL, get the MID from the property card
        composite_mid = template_mid
        with open(template_path, 'r') as f2:
            for mat_line in f2:
                if mat_line.startswith("MAT8") and mat_line[8:16].strip() == composite_mid:
                    composite_mat_suffix = mat_line[16:]

    # ================================================================
    # PCOMP_VARIABLE: completely separate expansion path
    # ================================================================
    if sh_file_path is not None:
        elem_mults = parse_sh_file(sh_file_path)
        total_t = sum(p['T'] for p in ply_info) if ply_info else 2.0

        angle_to_dv = parse_ply_stack_dvs(template_path)
        if angle_to_dv:
            print(f"  PLY/STACK DV mapping: {angle_to_dv}")

        # CLT-equivalent isotropic of the ACTUAL skin laminate (its real MAT8)
        _homog = homogenized_isotropic_from_skin(template_path, ply_info)
        if _homog is None:
            raise ValueError(f"Could not resolve skin material to homogenize "
                             f"in {template_path}")
        equiv_E, equiv_nu = _homog
        if stiffener_E_override is not None:
            equiv_E = stiffener_E_override
        if stiffener_nu_override is not None:
            equiv_nu = stiffener_nu_override
        equiv_G = equiv_E / (2.0 * (1.0 + equiv_nu))
        stiffener_mat_id = 999999

        print(f"  Stiffener material:")
        if stiffener_E_override is not None:
            print(f"    MANUAL override: E={equiv_E:.1f}, G={equiv_G:.1f}, nu={equiv_nu:.4f}")
        else:
            print(f"    HOMOGENIZED CLT: E={equiv_E:.1f}, G={equiv_G:.1f}, nu={equiv_nu:.4f}")

        skin_mid = composite_mid

        # --- Diagnostic: show ply mapping for first element ---
        sample_eid = next(iter(elem_mults))
        sample_sh = elem_mults[sample_eid]
        sample_mults = expand_sh_to_per_ply(sample_sh, ply_info, angle_to_dv)
        print(f"  FSO ply mapping (element {sample_eid}):")
        for k in range(len(ply_info)):
            t_actual = ply_info[k]['T'] * sample_mults[k]
            print(f"    ply {k}: theta={ply_info[k]['theta']:>6s}, "
                  f"nominal={ply_info[k]['T']:.4f}, mult={sample_mults[k]:.4f}, "
                  f"actual_T={t_actual:.4f}")

        # --- Write expanded file ---
        output_lines = []
        i = 0
        while i < len(lines):
            line = lines[i]

            # 1. CQUAD4 / CTRIA3: set PID = EID
            if line.startswith("CQUAD4") or line.startswith("CTRIA3"):
                try:
                    eid_str = line[8:16]
                    eid = int(eid_str.strip())
                    if eid in elem_ids:
                        new_line = line[:16] + eid_str + line[24:]
                        output_lines.append(new_line)
                    else:
                        output_lines.append(line)
                except ValueError:
                    output_lines.append(line)

            # 2. Replace PCOMPG/PCOMP with per-element PCOMP using FSO thicknesses
            elif ((line.startswith("PCOMPG") or line.startswith("PCOMP"))
                  and line[8:16].strip() == template_pid):

                output_lines.append("$ --- EXPANDED PCOMP (FSO skin + stiffener ply) ---\n")
                for eid in elem_ids:
                    raw_sh = elem_mults.get(int(eid), [])
                    if raw_sh:
                        mults = expand_sh_to_per_ply(raw_sh, ply_info, angle_to_dv)
                    else:
                        mults = [1.0] * len(ply_info)

                    # PCOMP header (LAM field left blank; generate_input_file
                    # sets SMEAR dynamically for unstiffened elements)
                    output_lines.append(f"PCOMP   {eid:>8}\n")

                    for p_idx in range(0, len(ply_info), 2):
                        p1 = ply_info[p_idx]
                        m1 = mults[p_idx] if p_idx < len(mults) else 1.0
                        t1 = p1['T'] * m1

                        fields = f"+       {skin_mid:>8}{t1:<8.4f}"
                        fields += f"{p1['theta']:>8}" if p1['theta'] else "        "
                        fields += f"{p1['sout']:>8}" if p1['sout'] else "        "

                        if p_idx + 1 < len(ply_info):
                            p2 = ply_info[p_idx + 1]
                            m2 = mults[p_idx + 1] if (p_idx + 1) < len(mults) else 1.0
                            t2 = p2['T'] * m2
                            fields += f"{skin_mid:>8}{t2:<8.4f}"
                            fields += f"{p2['theta']:>8}" if p2['theta'] else "        "
                            fields += f"{p2['sout']:>8}" if p2['sout'] else "        "

                        output_lines.append(fields + "\n")

                    stiff_line = (f"+       {stiffener_mat_id:>8}{0.0:<8.4f}"
                                  f"{'0.0':>8}{'':>8}\n")
                    output_lines.append(stiff_line)

                j = i + 1
                while j < len(lines) and _is_continuation(lines[j]):
                    j += 1
                i = j
                continue

            # 3. Keep the single composite MAT8 card, remove others
            elif line.startswith("MAT8"):
                mid_str = line[8:16].strip()
                if mid_str == composite_mid:
                    output_lines.append(line)
                else:
                    output_lines.append(f"$ Removed MAT8 MID={mid_str}\n")

            elif line.startswith("MAT1"):
                output_lines.append(line)

            # 4. Remove DSIZE from template (not needed for MMC)
            elif line.startswith("DSIZE"):
                j = i + 1
                while j < len(lines) and _is_continuation(lines[j]):
                    j += 1
                i = j
                continue

            # 5. Pass through everything else
            else:
                output_lines.append(line)

            i += 1

        # Add the stiffener MAT1 if not already in the template. It MUST go
        # INSIDE the bulk data (before ENDDATA) — appending it at the end of the
        # file places it after ENDDATA, where OptiStruct ignores it, so every
        # per-element PCOMP referencing it fails with "ERROR 14: Missing MAT".
        has_mat1 = any("MAT1" in l and str(stiffener_mat_id) in l
                        for l in output_lines)
        if not has_mat1:
            def _fmt8(val):
                s = f"{val:.1f}"
                if len(s) <= 8:
                    return f"{s:>8}"
                s = f"{val:.3E}"
                return f"{s:>8}"

            mat1_card = (f"MAT1    {stiffener_mat_id:>8}"
                         f"{_fmt8(equiv_E)}{_fmt8(equiv_G)}"
                         f"{equiv_nu:>8.4f}\n")
            enddata_idx = next((k for k in range(len(output_lines) - 1, -1, -1)
                                if output_lines[k].lstrip().upper().startswith("ENDDATA")),
                               None)
            if enddata_idx is not None:
                output_lines.insert(enddata_idx, mat1_card)
            else:
                output_lines.append(mat1_card)

        with open(output_path, 'w') as f:
            f.writelines(output_lines)

        print(f"Successfully wrote expanded FEM to: {output_path}")
        return

    # ================================================================
    # Standard (non-FSO) expansion path
    # ================================================================
    # print(f"  Property type: {prop_type}")
    # if ply_info:
    #     print(f"  Number of plies: {len(ply_info)}")

    # --- Compute CLT-equivalent isotropic for stiffener material ---
    stiffener_mat_id = 999999
    if ply_info and composite_mat_suffix and len(composite_mat_suffix) >= 32:
        _E1_val = clean_optistruct_float(composite_mat_suffix[0:8])
        _E2_val = clean_optistruct_float(composite_mat_suffix[8:16])
        _nu12_val = clean_optistruct_float(composite_mat_suffix[16:24])
        _G12_val = clean_optistruct_float(composite_mat_suffix[24:32])
        equiv_E, equiv_nu = compute_equivalent_isotropic_from_pcomp(
            ply_info, E1=_E1_val, E2=_E2_val, G12=_G12_val, nu12=_nu12_val)
    else:
        # No composite (MAT8) skin -> isotropic PSHELL/MAT1 skin: the
        # "homogenized" equivalent is simply that skin MAT1's own (E, nu).
        _m1 = read_template_mat1(template_path)
        if _m1 is None:
            raise ValueError(f"Could not resolve skin material to homogenize "
                             f"in {template_path}")
        equiv_E, equiv_nu = _m1

    # MANUAL / TEMPLATE mode override (same as the FSO path)
    if stiffener_E_override is not None:
        equiv_E = stiffener_E_override
    if stiffener_nu_override is not None:
        equiv_nu = stiffener_nu_override
    equiv_G = equiv_E / (2.0 * (1.0 + equiv_nu))

    def _fmt8(val):
        s = f"{val:.1f}"
        if len(s) <= 8:
            return f"{s:>8}"
        s = f"{val:.3E}"
        return f"{s:>8}"

    # --- Fill blank G1Z/G2Z with G12 on the skin MAT8 ---
    # Explicit (non-SMEAR) stacks require transverse shear homogenization of
    # the laminate; blank shear moduli make that matrix singular (OptiStruct
    # error 1562).
    # MAT8 suffix fields (relative to line[16:]):
    #   E1 [0:8], E2 [8:16], NU12 [16:24], G12 [24:32], G1Z [32:40], G2Z [40:48]
    if not use_smear and mat_template_suffix:
        _suffix = mat_template_suffix.rstrip('\n')
        _g12_str = _suffix[24:32].strip() if len(_suffix) > 24 else ''
        if _g12_str:
            _suffix = _suffix.ljust(48)
            if not _suffix[32:40].strip():
                _suffix = _suffix[:32] + f"{_g12_str:>8}" + _suffix[40:]
            if not _suffix[40:48].strip():
                _suffix = _suffix[:40] + f"{_g12_str:>8}" + _suffix[48:]
            mat_template_suffix = _suffix + "\n"

    # --- PASS 2: Write new file (non-FSO path) ---
    elem_id_set = {int(e) for e in elem_ids}
    output_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]

        # 1. CQUAD4 / CTRIA3: set PID = EID
        if line.startswith("CQUAD4") or line.startswith("CTRIA3"):
            try:
                eid_str = line[8:16]
                eid = int(eid_str.strip())
                if eid in elem_ids:
                    new_line = line[:16] + eid_str + line[24:]
                    output_lines.append(new_line)
                else:
                    output_lines.append(line)
            except ValueError:
                output_lines.append(line)

        # 2. PSHELL expansion
        elif (prop_type == "PSHELL" and line.startswith("PSHELL")
              and line[8:16].strip() == template_pid):
            output_lines.append("$ --- EXPANDED PSHELL PROPERTIES ---\n")
            rest_of_suffix = pshell_template_suffix[8:]
            for eid in elem_ids:
                new_card = f"PSHELL  {eid:>8}{eid:>8}{rest_of_suffix}"
                output_lines.append(new_card)

        # 3. PCOMPG expansion (non-FSO)
        elif (prop_type == "PCOMPG" and line.startswith("PCOMPG")
              and line[8:16].strip() == template_pid):
            output_lines.append("$ --- EXPANDED PCOMPG PROPERTIES ---\n")
            for eid in elem_ids:
                new_header = "PCOMPG  " + f"{eid:>8}" + line[16:64] + "SMEAR   " + line[72:]
                output_lines.append(new_header)
                for ply_line in pcomp_ply_lines:
                    new_ply = _replace_ply_mids(ply_line, eid)
                    output_lines.append(new_ply)
                if _use_mat1:
                    stiff_line = (f"+       {stiffener_mat_id:>8}{0.0:<8.4f}"
                                  f"{'0.0':>8}{'':>8}\n")
                    output_lines.append(stiff_line)
            j = i + 1
            while j < len(lines) and _is_continuation(lines[j]):
                j += 1
            i = j
            continue

        # 4. PCOMP expansion (non-FSO)
        elif (prop_type == "PCOMP" and line.startswith("PCOMP")
              and not line.startswith("PCOMPG")
              and line[8:16].strip() == template_pid):
            output_lines.append("$ --- EXPANDED PCOMP PROPERTIES ---\n")
            lam_str = "SMEAR   " if use_smear else "        "
            # Build ply continuation lines from ply_info (SYM-expanded full
            # stack) so symmetric templates keep their full thickness.
            for eid in elem_ids:
                new_header = (line[:8] + f"{eid:>8}" + line[16:64]
                              + lam_str + line[72:])
                output_lines.append(new_header)
                for p_idx in range(0, len(ply_info), 2):
                    p1 = ply_info[p_idx]
                    fields = f"+       {eid:>8}{p1['T']:<8.4f}"
                    fields += f"{p1['theta']:>8}" if p1['theta'] else "        "
                    fields += f"{p1['sout']:>8}" if p1.get('sout') else "        "
                    if p_idx + 1 < len(ply_info):
                        p2 = ply_info[p_idx + 1]
                        fields += f"{eid:>8}{p2['T']:<8.4f}"
                        fields += f"{p2['theta']:>8}" if p2['theta'] else "        "
                        fields += f"{p2['sout']:>8}" if p2.get('sout') else "        "
                    output_lines.append(fields + "\n")
                if _use_mat1:
                    stiff_line = (f"+       {stiffener_mat_id:>8}{0.0:<8.4f}"
                                  f"{'0.0':>8}{'':>8}\n")
                    output_lines.append(stiff_line)
            j = i + 1
            while j < len(lines) and _is_continuation(lines[j]):
                j += 1
            i = j
            continue

        # 4. MAT8 expansion
        elif line.startswith("MAT8") and line[8:16].strip() == template_mid:
            output_lines.append("$ --- EXPANDED MAT8 MATERIALS ---\n")
            for eid in elem_ids:
                new_card = f"MAT8    {eid:>8}" + mat_template_suffix
                output_lines.append(new_card)
            # Insert stiffener MAT1 right after the MAT8 block
            # (only when stiffener is a separate MAT1 layer)
            if _use_mat1 and (prop_type in ("PCOMP", "PCOMPG") or prop_type == "PSHELL"):
                output_lines.append(f"MAT1    {stiffener_mat_id:>8}"
                                    f"{_fmt8(equiv_E)}{_fmt8(equiv_G)}"
                                    f"{equiv_nu:>8.4f}\n")

        # 5. Template MAT1 cards whose MID collides with the per-element
        #    MAT8 IDs (MID = EID) must be removed: material IDs share one
        #    namespace, and nothing references them after expansion.
        elif line.startswith("MAT1"):
            mid_str = line[8:16].strip()
            try:
                collides = int(mid_str) in elem_id_set
            except ValueError:
                collides = False
            if collides:
                output_lines.append(f"$ Removed MAT1 MID={mid_str} "
                                    f"(collides with expanded MAT8 IDs)\n")
                j = i + 1
                while j < len(lines) and _is_continuation(lines[j]):
                    j += 1
                i = j
                continue
            output_lines.append(line)

        # 6. Pass through everything else
        else:
            output_lines.append(line)

        i += 1

    with open(output_path, 'w') as f:
        f.writelines(output_lines)

    print(f"Successfully wrote expanded FEM to: {output_path}")


def _structured_grid(X, Y, values, decimals=6):
    """Reshape per-element values onto a regular lattice for imshow.

    Returns (array, extent) when the element centroids form a complete,
    evenly spaced rectangular grid with exactly one element per cell, and
    None otherwise (unstructured or graded meshes fall back to scatter).
    """
    Xr, Yr = np.round(X, decimals), np.round(Y, decimals)
    ux, uy = np.unique(Xr), np.unique(Yr)
    if len(ux) < 2 or len(uy) < 2 or len(ux) * len(uy) != len(values):
        return None
    dxs, dys = np.diff(ux), np.diff(uy)
    if not (np.allclose(dxs, dxs[0]) and np.allclose(dys, dys[0])):
        return None
    grid = np.full((len(uy), len(ux)), np.nan)
    grid[np.searchsorted(uy, Yr), np.searchsorted(ux, Xr)] = values
    if np.isnan(grid).any():          # duplicate centroids -> holes
        return None
    hx, hy = 0.5 * dxs[0], 0.5 * dys[0]
    return grid, [ux[0] - hx, ux[-1] + hx, uy[0] - hy, uy[-1] + hy]


def plot_iteration(problem, iteration, save_path=None):
    """Visualizes the current design by plotting the element Thickness distribution."""
    T_elem, _ = problem.get_element_properties(current_iter=iteration)

    X = problem.X
    Y = problem.Y

    plt.figure(figsize=(10, 8))

    # A structured mesh (the usual case: a rectangular panel meshed with a
    # regular CQUAD4 lattice) is drawn one image pixel per element. The plain
    # scatter path below uses a FIXED marker size, so on a fine mesh each
    # square comes out several times wider than the element pitch: markers
    # then overpaint their neighbours in element-ID order and the picture
    # picks up ragged bar edges, speckle and moire streaks that are NOT in
    # T_elem. imshow has no draw-order lottery, and is much faster than
    # scattering tens of thousands of markers every iteration.
    grid = _structured_grid(X, Y, T_elem)
    if grid is not None:
        sc = plt.imshow(grid[0], origin='lower', cmap='viridis',
                        interpolation='nearest', extent=grid[1])
    else:
        sc = plt.scatter(X, Y, c=T_elem, s=15, cmap='viridis', marker='s',
                         edgecolors='none')

    cbar = plt.colorbar(sc)
    skin_label = (f"{np.mean(problem.skin_thickness):.2f}"
                  if hasattr(problem.skin_thickness, '__len__')
                  else f"{problem.skin_thickness:.2f}")
    cbar.set_label(f"Element Thickness (mm)",
                   rotation=270, labelpad=20)

    plt.axis('equal')
    plt.title(f"Optimization Iteration {iteration}\nMax Thickness: {np.max(T_elem):.2f} mm")
    plt.xlabel("X (mm)")
    plt.ylabel("Y (mm)")
    plt.grid(True, linestyle=':', alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.savefig(f"design_iter_{iteration}.png", dpi=150, bbox_inches='tight')

    plt.close()


def plot_skin_thickness(problem, ply_info, save_path=None):
    """
    Plots the skin thickness distribution (without stiffeners).

    For PCOMP_VARIABLE: uses the per-element skin thickness already stored
    in problem.skin_thickness (which is a spatially varying array).
    For constant PCOMP/PSHELL: plots the uniform field (sanity check).
    """
    X = problem.X
    Y = problem.Y

    # problem.skin_thickness is always an array (per-element) since
    # MMC_Problem now broadcasts scalars to full arrays.
    skin_T = problem.skin_thickness

    plt.figure(figsize=(10, 8))

    sc = plt.scatter(X, Y, c=skin_T, s=15, cmap='YlOrRd', marker='s', edgecolors='none')

    cbar = plt.colorbar(sc)
    cbar.set_label("Skin Thickness (mm)", rotation=270, labelpad=20)

    plt.axis('equal')
    plt.title(f"FSO Skin Thickness Distribution\n"
              f"Min: {np.min(skin_T):.4f} mm | Max: {np.max(skin_T):.4f} mm | "
              f"Mean: {np.mean(skin_T):.4f} mm")
    plt.xlabel("X (mm)")
    plt.ylabel("Y (mm)")
    plt.grid(True, linestyle=':', alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.savefig("skin_thickness.png", dpi=150, bbox_inches='tight')

    plt.close()
    # print(f"  Skin thickness plot saved: {save_path or 'skin_thickness.png'}")


def plot_convergence(evals, compliances, save_path):
    plt.figure(figsize=(8, 5))

    plt.plot(evals, compliances, marker='o', markersize=4, linestyle='-',
             color='#1f77b4', linewidth=2)

    plt.title('Optimization Convergence History', fontsize=14)
    plt.xlabel('Solver Evaluation (Iteration)', fontsize=12)
    plt.ylabel('Real Compliance', fontsize=12)

    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_components_vector(problem, save_path, title="Component layout"):
    """
    Paper-style vector rendering of the component layout (cf. MMC1 Fig. 4c):
    each effective component (symmetry clones included) is drawn as its
    analytic superellipse outline, filled — smooth by construction, unlike
    the rasterized element-thickness plots. Handles tapered components.
    """
    fig, ax = plt.subplots(figsize=(10, 5.5))
    b = problem.bounds

    if problem.symmetry == "OFF":
        comps = list(problem.components)
    else:
        comps = [c for c, _, _ in problem._get_symmetry_clones()]

    tau = np.linspace(0.0, 2.0 * np.pi, 241)
    ct, st = np.cos(tau), np.sin(tau)
    for comp in comps:
        p = comp.p
        L = max(comp.L, 1e-6)
        xp = L * np.sign(ct) * np.abs(ct) ** (2.0 / p)
        t2 = getattr(comp, 't2', None)
        if t2 is None:
            f = np.full_like(xp, max(comp.width / 2.0, 1e-6))
        else:
            h1 = max(comp.width / 2.0, 1e-6)
            h2 = max(t2 / 2.0, 1e-6)
            f = np.maximum(0.5 * (h1 + h2) + 0.5 * (h2 - h1) * xp / L, 1e-6)
        yp = f * np.sign(st) * np.abs(st) ** (2.0 / p)
        c, s = np.cos(comp.theta), np.sin(comp.theta)
        ax.fill(comp.x0 + c * xp - s * yp, comp.y0 + s * xp + c * yp,
                facecolor='#8FE07A', edgecolor='#3E7A2E', linewidth=0.6,
                zorder=2)

    ax.add_patch(plt.Rectangle((b['x_min'], b['y_min']),
                               b['x_max'] - b['x_min'],
                               b['y_max'] - b['y_min'],
                               fill=False, edgecolor='black',
                               linewidth=1.0, zorder=3))
    ax.set_xlim(b['x_min'], b['x_max'])
    ax.set_ylim(b['y_min'], b['y_max'])
    ax.set_aspect('equal')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_title(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_components_contour(problem, save_path, title="Component contour",
                            grid_n=500):
    """
    Paper-style CONTOUR plot of the merged structure (cf. Sun et al. 2020
    Fig. 2c / Zhang et al. 2016 Table 2 "contour plot", i.e. the reference
    MMC code's contourf(Phi_max, [0,0])).

    Where plot_components_vector draws each component's superellipse
    separately (individual overlapping bars), this evaluates the UNION
    topology description function Phi_s(x) = max_i phi_i(x) on a fine regular
    grid and fills its zero-level set — showing the single merged material
    silhouette with the smooth analytic boundary the components actually
    produce after the max-union. Symmetry clones are included exactly as in
    the component/FEA path.
    """
    b = problem.bounds
    if problem.symmetry == "OFF":
        comps = list(problem.components)
    else:
        comps = [c for c, _, _ in problem._get_symmetry_clones()]

    # Fine regular grid over the domain (independent of the FE mesh, for a
    # smooth boundary), aspect-matched so cells stay roughly square.
    dw = b['x_max'] - b['x_min']
    dh = b['y_max'] - b['y_min']
    if dw >= dh:
        nx = int(grid_n)
        ny = max(int(round(grid_n * dh / dw)), 2)
    else:
        ny = int(grid_n)
        nx = max(int(round(grid_n * dw / dh)), 2)
    gx = np.linspace(b['x_min'], b['x_max'], nx)
    gy = np.linspace(b['y_min'], b['y_max'], ny)
    Xg, Yg = np.meshgrid(gx, gy)

    if comps:
        Phi_s = comps[0].calculate_phi(Xg, Yg)
        for comp in comps[1:]:
            np.maximum(Phi_s, comp.calculate_phi(Xg, Yg), out=Phi_s)
    else:
        Phi_s = np.full_like(Xg, -1.0)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    # Filled material region (Phi_s >= 0) in solid colour, void white —
    # the standard MMC contour presentation.
    ax.contourf(Xg, Yg, Phi_s, levels=[0.0, Phi_s.max() + 1e-9],
                colors=['#2E5E9E'])
    # Emphasise the explicit boundary (zero level set).
    if Phi_s.max() > 0.0 > Phi_s.min():
        ax.contour(Xg, Yg, Phi_s, levels=[0.0],
                   colors='#12315A', linewidths=0.8)

    ax.add_patch(plt.Rectangle((b['x_min'], b['y_min']), dw, dh,
                               fill=False, edgecolor='black',
                               linewidth=1.0, zorder=3))
    ax.set_xlim(b['x_min'], b['x_max'])
    ax.set_ylim(b['y_min'], b['y_max'])
    ax.set_aspect('equal')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_title(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_paper_convergence(evals, compliances_nmm, volumes, vol_target,
                           save_path, final_comp=None, final_vf=None):
    """
    Paper-style convergence plot (cf. MMC1 paper Figs. 3e / 4d):
      - left axis (red):  objective function in J (compliance N.mm / 1000)
      - right axis (blue): volume constraint function g = VF - vol_target
        (feasible when g <= 0), symmetric axis like the paper

    If final_comp/final_vf are given (the re-analysed FINAL design), one
    extra data point is appended to both curves after the last iteration.
    """
    evals = list(evals)
    compliances_nmm = list(compliances_nmm)
    volumes = list(volumes)
    if final_comp is not None and final_vf is not None and evals:
        evals = evals + [evals[-1] + 1]
        compliances_nmm = compliances_nmm + [final_comp]
        volumes = volumes + [final_vf]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    obj_J = np.asarray(compliances_nmm, dtype=float) / 10000.0
    ax1.plot(evals, obj_J, color='red', linewidth=1.5,
             label='Objective function')
    ax1.set_xlabel('Iteration step', fontsize=12)
    ax1.set_ylabel('Objective function (x 10J)', fontsize=12)
    ax1.tick_params(axis='y')
    ax1.set_ylim(bottom=0.0)
    ax1.set_xlim(0, max(evals) if len(evals) else 1)  # no x margins
    ax1.grid(True, linestyle=':', alpha=0.4)

    ax2 = ax1.twinx()
    g = np.asarray(volumes, dtype=float) - vol_target
    ax2.plot(evals, g, color='blue', linewidth=1.5,
             label='Volume constraint function')
    ax2.set_ylabel('Volume constraint function', fontsize=12)
    ax2.tick_params(axis='y')
    g_lim = max(float(np.max(np.abs(g))) * 1.1, 0.15)
    ax2.set_ylim(-g_lim, g_lim)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc='upper right', fontsize=9)

    plt.title('Convergence history', fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_specific_compliance(evals, compliances, volumes, save_path):
    plt.figure(figsize=(8, 5))

    spec_comp = np.array(compliances) / (np.array(volumes) * 100)

    plt.plot(evals, spec_comp, marker='s', markersize=4, linestyle='-',
             color='#d62728', linewidth=2)

    plt.title('Specific Compliance (C / VF%) vs Iterations\n(Lower is better)', fontsize=14)
    plt.xlabel('Solver Evaluation (Iteration)', fontsize=12)
    plt.ylabel('Compliance / Volume Fraction %', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.yscale('log')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# =============================================================================
#          RUN LOGGING, OPTISTRUCT INTERFACE AND EXPORT PLUMBING
# =============================================================================
import os
import subprocess


class Tee:
    """Mirror stdout to a log file (run_log.txt)."""

    def __init__(self, terminal, log_file):
        self.terminal = terminal
        self.log_file = log_file

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()


def cleanup_run_folder(run_folder, current_iter, best_feasible_iter=None):
    """
    Delete per-iteration solver artifacts (.fem/.h3d/.pch/.out etc.) from the
    run folder, keeping only: iter_1 (initial design), the last iteration
    (final raw design), the best-feasible iteration (the design that is
    exported, when different) and eval_final (its re-analysis). Plots and
    run_log.txt are always kept.
    """
    keep = {"iter_1", f"iter_{current_iter}", "eval_final"}
    if best_feasible_iter is not None:
        keep.add(f"iter_{best_feasible_iter}")
    n_del, bytes_del = 0, 0
    for f in os.listdir(run_folder):
        m = re.match(r"(iter_\d+|eval_[A-Za-z]+)", f)
        if m is None or m.group(1) in keep:
            continue
        p = os.path.join(run_folder, f)
        if os.path.isfile(p):
            try:
                bytes_del += os.path.getsize(p)
                os.remove(p)
                n_del += 1
            except OSError:
                pass
    print(f"\n  Run-folder cleanup: removed {n_del} solver files "
          f"({bytes_del / 1e6:.1f} MB); kept {sorted(keep)}")


class OptiStructOptimizer:
    """
    Feedback interface between the MMC problem and OptiStruct:
      generate_input_file  writes the per-iteration .fem deck (updated
                           element thicknesses / stiffener ply thicknesses)
      run_solver           launches a linear static OptiStruct run
      read_results         parses total compliance from the .out file and
                           the element strain energies (and displacements)
                           from the PUNCH .pch file
    """

    def __init__(self, template_path, exec_path, problem, work_dir=".",
                 prop_type="PSHELL", ply_info=None, stiffener_model="MAT1"):
        self.template_path = template_path
        self.exec_path = exec_path
        self.problem = problem
        self.work_dir = work_dir
        self.current_iter = 0
        self.prop_type = prop_type
        # "MAT1", "MAT8", "PLY_SCALING"
        self.stiffener_model = stiffener_model

        # Store original ply structure for PCOMP scaling
        self.ply_info = ply_info or []

        # Stiffener realized as a dedicated one-sided ply appended to the
        # skin stack:
        #   MAT1 -> ply material is the dedicated MAT1 (MID 999999)
        #   MAT8 -> ply material is the element's own skin MAT8
        #           (per-element clone, MID = element ID) at theta = 0
        #           (global laminate orientation)
        self._use_mat1_stiffener = (stiffener_model == "MAT1")
        self._use_mat8_stiffener = (stiffener_model == "MAT8")
        self._use_ply_stiffener = (self._use_mat1_stiffener
                                   or self._use_mat8_stiffener)

        # --- ISOTROPIC PROPERTIES ---
        # For PSHELL/PCOMP: hardcoded defaults
        # For PCOMP_VARIABLE: overridden from CLT equivalent
        if prop_type == "PCOMP_VARIABLE" and isinstance(ply_info, dict):
            self.E = ply_info['equiv_E']
            self.nu = ply_info['equiv_nu']
            self.G12 = self.E / (2.0 * (1.0 + self.nu))
            self.elem_multipliers = ply_info['elem_multipliers']
            self.angle_to_dv = ply_info.get('angle_to_dv')
            self.base_plies = ply_info['base_plies']
            self.elem_index_map = {int(eid): idx for idx, eid in enumerate(problem.elem_ids)}
            print(f"  Stiffener material (CLT equivalent): "
                  f"E = {self.E:.2f}, nu = {self.nu:.4f}, G = {self.G12:.2f}")
        else:
            self.E = 200.0
            self.nu = 0.3
            self.G12 = self.E / (2.0 * (1.0 + self.nu))
            self.elem_multipliers = None
            self.angle_to_dv = None
            self.base_plies = None
            self.elem_index_map = {}

    # --- Helper: detect Nastran continuation lines ---
    @staticmethod
    def _is_continuation(line):
        if not line or not line.strip():
            return False
        return line[0] in (' ', '\t', '+', '*')

    def _build_pcomp_ply_lines(self, mid, scale_factor):
        """
        Build PCOMP continuation lines with scaled ply thicknesses.

        Packs 2 plies per continuation line in Nastran Small Field format:
            +       MID1    T1      THETA1  SOUT1   MID2    T2      THETA2  SOUT2
        """
        cont_lines = []
        plies = self.ply_info
        for p_idx in range(0, len(plies), 2):
            ply1 = plies[p_idx]
            t1_scaled = ply1['T'] * scale_factor

            # Ply 1 fields
            fields = f"+       {mid:>8}{t1_scaled:<8.4f}"
            fields += f"{ply1['theta']:>8}" if ply1['theta'] else "        "
            fields += f"{ply1['sout']:>8}" if ply1['sout'] else "        "

            # Ply 2 (if exists)
            if p_idx + 1 < len(plies):
                ply2 = plies[p_idx + 1]
                t2_scaled = ply2['T'] * scale_factor
                fields += f"{mid:>8}{t2_scaled:<8.4f}"
                fields += f"{ply2['theta']:>8}" if ply2['theta'] else "        "
                fields += f"{ply2['sout']:>8}" if ply2['sout'] else "        "

            cont_lines.append(fields + "\n")
        return cont_lines

    def _build_pcompg_ply_lines(self, mid, scale_factor):
        """
        Build PCOMPG continuation lines with scaled ply thicknesses.

        PCOMPG has ONE ply per continuation line with global ply ID:
            +       GPLYID  MID     T       THETA           SOUT
        """
        cont_lines = []
        plies = self.ply_info
        for p_idx, ply in enumerate(plies):
            t_scaled = ply['T'] * scale_factor
            gplyid = p_idx + 1
            fields = f"+       {gplyid:>8}{mid:>8}{t_scaled:<8.4f}"
            fields += f"{ply['theta']:>8}" if ply.get('theta') else "        "
            fields += "        "
            fields += f"{ply.get('sout', ''):>8}" if ply.get('sout') else "        "
            cont_lines.append(fields + "\n")
        return cont_lines

    STIFFENER_MID = 999999

    @staticmethod
    def _make_stiffener_ply_line(thickness, mid=None):
        """Create a stiffener ply continuation line at theta = 0.
        mid=None uses the dedicated MAT1 (MID 999999); the MAT8 mode passes
        the element's own skin-material MID instead."""
        if mid is None:
            mid = OptiStructOptimizer.STIFFENER_MID
        return (f"+       {mid:>8}{thickness:<8.4f}"
                f"{'0.0':>8}{'':>8}\n")

    H_THRESHOLD = 0.01

    @staticmethod
    def _is_layup_symmetric(angles, thicknesses):
        """
        Check whether a layup is symmetric about the midplane.

        Symmetric means angle[i] == angle[N-1-i] AND
        thickness[i] ~= thickness[N-1-i] for all mirror pairs.
        """
        N = len(angles)
        if N < 2:
            return True
        for i in range(N // 2):
            j = N - 1 - i
            if angles[i] != angles[j]:
                return False
            t_max = max(abs(thicknesses[i]), abs(thicknesses[j]), 1e-30)
            if abs(thicknesses[i] - thicknesses[j]) > 1e-9 * t_max:
                return False
        return True

    def _write_elem_pcomp_variable(self, pid, H_elem, stiffener_T,
                                    problem, output_lines):
        """
        Write one per-element PCOMP for PCOMP_VARIABLE mode.

        Builds the skin layup first, checks symmetry, then writes the
        PCOMP header with the correct LAM field:
          - Asymmetric layup -> LAM = SYM   (mirror to restore symmetry)
          - Symmetric layup   -> LAM = blank (card already has all info)
        Never uses SMEAR or SYSMEAR so that ply stacking sequence
        effects are fully captured at every iteration.
        """
        mid = self.base_plies[0]['mid']
        n_plies = len(self.base_plies)

        skin_t = [0.0] * n_plies
        stiffened = H_elem >= self.H_THRESHOLD

        if not stiffened:
            raw_sh = self.elem_multipliers.get(pid, [])
            if raw_sh:
                mults = expand_sh_to_per_ply(
                    raw_sh, self.base_plies, self.angle_to_dv)
            else:
                mults = [1.0] * n_plies
            for k in range(n_plies):
                skin_t[k] = self.base_plies[k]['T'] * (
                    mults[k] if k < len(mults) else 1.0)
        else:
            elem_idx = self.elem_index_map.get(pid, 0)
            t_base_e = problem.t_base[elem_idx]
            ply_t = t_base_e / n_plies if n_plies > 0 else 0.0
            skin_t = [ply_t] * n_plies

        # LAM field: SMEAR for the bare skin (a membrane laminate, matching the
        # FSO PCOMP definition, total = sum of the listed plies); blank (explicit
        # stack) when a stiffener ply is present, so its one-sided offset is kept.
        # NEVER SYM: LAM=SYM mirrors the whole listed stack about the mid-plane,
        # which doubles BOTH the skin AND the stiffener ply thickness.
        lam = "        " if stiffened else "SMEAR   "
        output_lines.append(
            f"PCOMP   {pid:>8}" + " " * 48 + lam + "\n")

        for p_idx in range(0, n_plies, 2):
            ply1 = self.base_plies[p_idx]
            t1 = skin_t[p_idx]
            fields = f"+       {mid:>8}{t1:<8.4f}"
            fields += f"{ply1['theta']:>8}" if ply1.get('theta') else "        "
            fields += f"{ply1.get('sout', ''):>8}" if ply1.get('sout') else "        "

            if p_idx + 1 < n_plies:
                ply2 = self.base_plies[p_idx + 1]
                t2 = skin_t[p_idx + 1]
                fields += f"{mid:>8}{t2:<8.4f}"
                fields += f"{ply2['theta']:>8}" if ply2.get('theta') else "        "
                fields += f"{ply2.get('sout', ''):>8}" if ply2.get('sout') else "        "

            output_lines.append(fields + "\n")

        if stiffened:
            output_lines.append(self._make_stiffener_ply_line(stiffener_T))

    def generate_input_file(self, problem, filename):
        """
        Writes a new .fem file using strict 8-character Small Field formatting.
        Supports PSHELL, PCOMP, PCOMPG and PCOMP_VARIABLE property types.

        PSHELL: the element thickness entry is set to T_e.
        PCOMP/PCOMPG with MAT1 stiffener: the dedicated stiffener ply
        (MID=999999) thickness is set to H*stiffener_height; with
        PLY_SCALING all skin plies are scaled uniformly instead.
        """
        T_arr, _ = problem.get_element_properties(current_iter=self.current_iter + 1)

        thickness_map = {eid: T_arr[i] for i, eid in enumerate(problem.elem_ids)}

        if self.prop_type == "PCOMP_VARIABLE":
            H_arr = problem.compute_H_field(problem.X, problem.Y)
            H_map = {eid: H_arr[i] for i, eid in enumerate(problem.elem_ids)}
        else:
            H_map = {}

        output_lines = []

        with open(self.template_path, 'r') as f:
            lines = f.readlines()

        i = 0
        while i < len(lines):
            line = lines[i]

            if line.startswith("PSHELL"):
                try:
                    pid_str = line[8:16].strip()
                    mid1_str = line[16:24].strip()

                    if pid_str and int(pid_str) in thickness_map:
                        pid = int(pid_str)
                        mid1 = mid1_str.strip()
                        new_T = thickness_map[pid]
                        new_line = (f"PSHELL  {pid:>8}{mid1:>8}"
                                    f"{new_T:<8.4f}{line[32:]}")
                        output_lines.append(new_line)
                    else:
                        output_lines.append(line)
                except ValueError:
                    output_lines.append(line)

            elif line.startswith("PCOMPG"):
                try:
                    pid_str = line[8:16].strip()

                    if pid_str and int(pid_str) in thickness_map:
                        pid = int(pid_str)

                        if self.prop_type == "PCOMP_VARIABLE":
                            H_elem = H_map.get(pid, 0.0)
                            stiffener_T = H_elem * problem.stiffener_height

                            self._write_elem_pcomp_variable(
                                pid, H_elem, stiffener_T,
                                problem, output_lines)

                            j = i + 1
                            while j < len(lines) and self._is_continuation(lines[j]):
                                j += 1
                            i = j
                            continue
                        else:
                            new_T = thickness_map[pid]
                            orig_T = float(problem.skin_thickness[0])
                            if self._use_ply_stiffener:
                                stiff_T = new_T - orig_T
                                output_lines.append(line)
                                new_cont = self._build_pcompg_ply_lines(pid, 1.0)
                                output_lines.extend(new_cont)
                                stiff_mid = (pid if self._use_mat8_stiffener
                                             else None)
                                if stiff_T > 1e-6:
                                    stiff_line = self._make_stiffener_ply_line(
                                        stiff_T, mid=stiff_mid)
                                    output_lines.append(stiff_line)
                            else:
                                scale = new_T / orig_T if orig_T > 1e-12 else 1.0
                                output_lines.append(line)
                                new_cont = self._build_pcompg_ply_lines(pid, scale)
                                output_lines.extend(new_cont)

                            j = i + 1
                            while j < len(lines) and self._is_continuation(lines[j]):
                                j += 1
                            i = j
                            continue
                    else:
                        output_lines.append(line)
                except ValueError:
                    output_lines.append(line)

            elif line.startswith("PCOMP"):
                try:
                    pid_str = line[8:16].strip()

                    if pid_str and int(pid_str) in thickness_map:
                        pid = int(pid_str)

                        if self.prop_type == "PCOMP_VARIABLE":
                            H_elem = H_map.get(pid, 0.0)
                            stiffener_T = H_elem * problem.stiffener_height

                            self._write_elem_pcomp_variable(
                                pid, H_elem, stiffener_T,
                                problem, output_lines)

                            j = i + 1
                            while j < len(lines) and self._is_continuation(lines[j]):
                                j += 1
                            i = j
                            continue
                        else:
                            new_T = thickness_map[pid]
                            orig_T = float(problem.skin_thickness[0])
                            if self._use_ply_stiffener:
                                stiff_T = new_T - orig_T
                                output_lines.append(line)
                                skin_cont = self._build_pcomp_ply_lines(pid, 1.0)
                                output_lines.extend(skin_cont)
                                stiff_mid = (pid if self._use_mat8_stiffener
                                             else None)
                                if stiff_T > 1e-6:
                                    stiff_line = self._make_stiffener_ply_line(
                                        stiff_T, mid=stiff_mid)
                                    output_lines.append(stiff_line)
                            else:
                                scale = new_T / orig_T if orig_T > 1e-12 else 1.0
                                output_lines.append(line)
                                new_cont = self._build_pcomp_ply_lines(pid, scale)
                                output_lines.extend(new_cont)

                            j = i + 1
                            while j < len(lines) and self._is_continuation(lines[j]):
                                j += 1
                            i = j
                            continue
                    else:
                        output_lines.append(line)
                except ValueError:
                    output_lines.append(line)

            else:
                output_lines.append(line)

            i += 1

        out_path = os.path.join(self.work_dir, filename)
        with open(out_path, 'w') as f:
            f.writelines(output_lines)

    def run_solver(self, filename):
        """Calls OptiStruct (linear static analysis run)."""
        input_path = os.path.abspath(os.path.join(self.work_dir, filename))

        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Can't find file at {input_path}")

        if not os.path.isfile(self.exec_path):
            print(f"  [DRY-RUN] OptiStruct executable not found at {self.exec_path}")
            return

        cmd = [self.exec_path, input_path, "-optskip", "-nt", "18"]

        result = subprocess.run(
            cmd,
            cwd=os.path.abspath(self.work_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        if result.returncode != 0:
            print(f"OptiStruct Error Output:\n{result.stdout}")
            raise RuntimeError("OptiStruct finished with an error code.")

    def read_results(self, jobname):
        """Parses .out for Compliance and .pch for Strain Energies / Displacements."""
        out_path = os.path.join(self.work_dir, f"{jobname}.out")
        pch_path = os.path.join(self.work_dir, f"{jobname}.pch")

        compliance = 0.0
        ese_dict = {}
        max_disp = 0.0

        if os.path.exists(out_path):
            with open(out_path, 'r') as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                    if "Subcase" in line and "Compliance" in line:
                        for sub_line in lines[i + 1: i + 10]:
                            parts = sub_line.split()
                            if len(parts) >= 2 and parts[0].isdigit():
                                try:
                                    compliance = float(parts[1])
                                    break
                                except:
                                    continue
                        if compliance != 0.0: break

        if os.path.exists(pch_path):
            with open(pch_path, 'r') as f:
                capture_ese = False
                capture_disp = False
                for line in f:
                    # --- Element strain energies ---
                    if "$ELEMENT STRAIN ENERGIES" in line:
                        capture_ese = True
                        capture_disp = False
                        continue
                    if capture_ese and line.startswith("$") and "SUBCASE" not in line:
                        continue
                    if capture_ese and not line.startswith("$"):
                        try:
                            eid = int(line[0:10].strip())
                            val_str = line[18:].split()[0]
                            ese_dict[eid] = float(val_str)
                        except:
                            continue

                    # --- Displacements ---
                    if "$DISPLACEMENTS" in line:
                        capture_disp = True
                        capture_ese = False
                        continue
                    if capture_disp and line.startswith("$"):
                        continue
                    if capture_disp and not line.startswith("$"):
                        # PUNCH real displacement row (8-char fields):
                        #   NID[0:10] TYPE[10:18] T1[18:36] T2[36:54] T3[54:72]
                        # Each row is followed by a '-CONT-' row holding the
                        # rotations R1/R2/R3, which must NOT be read as
                        # translations.
                        # max_disp is the out-of-plane (Z) displacement: all
                        # panel meshes lie in the XY plane, so T3 is the
                        # out-of-plane component.
                        if line.startswith("-CONT-"):
                            continue
                        try:
                            tz = abs(float(line[54:72]))
                        except (ValueError, IndexError):
                            continue
                        if tz > max_disp:
                            max_disp = tz

        ese_array = np.zeros(len(self.problem.X))
        for i, eid in enumerate(self.problem.elem_ids):
            ese_array[i] = ese_dict.get(eid, 0.0)

        return compliance, ese_array, max_disp


# =============================================================================
#            FSO INTEGRATION EXPORT (stiffener-driven model)
# =============================================================================
# No ply material is hardcoded here: the skin material is copied verbatim from
# the MMC template, and the stiffener MAT1 is built from the (equiv_E, equiv_nu)
# that main.py resolves from the actual skin (see homogenized_isotropic_from_skin).


def export_fso_model_with_stiffeners(problem, run_dir, fso_template_path,
                                     prop_type="PSHELL", ply_info=None,
                                     equiv_E=None, equiv_nu=None,
                                     t_base=0.5, h_threshold=0.01,
                                     stiffener_model="MAT1",
                                     min_laminate_thickness=1.0,
                                     fso_integration="ON_TOP",
                                     output_filename="FSO_with_stiffeners.fem"):
    """
    Export the converged stiffener layout to a standalone .fem file that
    serves as input for the skin free-size optimization (the stiffener-driven
    integration model): the optimized stiffeners are frozen into the model
    and the FSO design space covers the remaining skin.

      - Variable skin (prop_type == PCOMP_VARIABLE): per-element PCOMP.
      - Constant skin (any template type):
          * non-stiffened elements -> one shared PCOMP (base [0/45/-45/90])
          * stiffened elements -> per-element PCOMPG with the same 4 base
            plies + one MAT1 stiffener ply with element-specific thickness
          * DSIZE acts only on the non-stiffened design domain
      - MAT8 for plies is always hardcoded AS4/3501-6 (MID=1).
    """
    output_path = os.path.join(run_dir, output_filename)
    is_variable_skin = (prop_type == "PCOMP_VARIABLE")
    stiffener_mat_id = 999999
    equiv_G = equiv_E / (2.0 * (1.0 + equiv_nu))

    # ------------------------------------------------------------------
    # 1. Compute per-element Heaviside and stiffener map
    # ------------------------------------------------------------------
    H = problem.compute_H_field(problem.X, problem.Y)

    stiff_height = problem.stiffener_height
    stiff_map = {}
    H_map_export = {}
    for i, eid in enumerate(problem.elem_ids):
        H_map_export[int(eid)] = float(H[i])
        if H[i] >= h_threshold:
            stiff_map[int(eid)] = float(H[i] * stiff_height)

    stiffened_eids = set(stiff_map.keys())
    all_eids_sorted = sorted(int(eid) for eid in problem.elem_ids)
    all_eids_set = set(all_eids_sorted)
    nonstiff_eids_sorted = [eid for eid in all_eids_sorted if eid not in stiffened_eids]
    n_stiffened = len(stiffened_eids)

    print(f"\n  Exporting FSO-with-stiffeners model")
    print(f"  {n_stiffened} / {len(problem.elem_ids)} elements stiffened")

    # ------------------------------------------------------------------
    # 2. Read template and detect base property card
    # ------------------------------------------------------------------
    with open(fso_template_path, 'r') as f:
        lines = f.readlines()

    def _is_continuation(line):
        if not line or not line.strip():
            return False
        return line[0] in (' ', '\t', '+', '*')

    def _fmt8(val):
        s = f"{val:.4f}"
        if len(s) <= 8:
            return f"{s:>8}"
        s = f"{val:.3f}"
        if len(s) <= 8:
            return f"{s:>8}"
        s = f"{val:.2E}"
        return f"{s:>8}"

    template_prop_type = None
    template_pid = None
    for line in lines:
        if line.startswith("PCOMPG"):
            pid = line[8:16].strip()
            if pid:
                template_prop_type = "PCOMPG"
                template_pid = pid
                break
        elif line.startswith("PCOMP") and not line.startswith("PCOMPG"):
            pid = line[8:16].strip()
            if pid:
                template_prop_type = "PCOMP"
                template_pid = pid
                break
        elif line.startswith("PSHELL"):
            pid = line[8:16].strip()
            if pid:
                template_prop_type = "PSHELL"
                template_pid = pid
                break

    if template_pid is None:
        raise ValueError("No PSHELL, PCOMP, or PCOMPG found in template")

    print(f"  Template property: {template_prop_type} PID={template_pid}")

    # ------------------------------------------------------------------
    # 2b. Resolve the ACTUAL skin and stiffener materials from the MMC
    #     template so the FSO model uses the SAME materials the MMC used
    #     (rather than a hard-coded AS4/3501-6). Works for PSHELL (MID field)
    #     and PCOMP/PCOMPG (ply MIDs).
    # ------------------------------------------------------------------
    def _extract_mat_card(mid):
        """(card_lines, 'MAT1'|'MAT8') for the given MID from the template,
        card + continuations. (None, None) if not found."""
        mid = str(mid).strip()
        for k, ln in enumerate(lines):
            if ((ln.startswith("MAT8") or ln.startswith("MAT1"))
                    and ln[8:16].strip() == mid):
                mtype = "MAT8" if ln.startswith("MAT8") else "MAT1"
                card = [ln]
                j = k + 1
                while j < len(lines) and _is_continuation(lines[j]):
                    card.append(lines[j]); j += 1
                return card, mtype
        return None, None

    # --- skin material MID ---
    skin_mid = None
    if template_prop_type in ("PCOMP", "PCOMPG"):
        if isinstance(ply_info, dict):          # PCOMP_VARIABLE
            bp = ply_info.get('base_plies') or []
            if bp:
                skin_mid = str(bp[0].get('mid', '')).strip()
        elif isinstance(ply_info, list) and ply_info:
            skin_mid = str(ply_info[0].get('mid', '')).strip()
        if not skin_mid:                        # fall back to first ply MID in template
            for k, ln in enumerate(lines):
                if ln[:8].strip() in ("PCOMP", "PCOMPG") and ln[8:16].strip() == str(template_pid):
                    j = k + 1
                    while j < len(lines) and _is_continuation(lines[j]):
                        cand = lines[j][8:16].strip()
                        if cand and cand.lstrip("-").isdigit():
                            skin_mid = cand; break
                        j += 1
                    break
    else:  # PSHELL: material in the MID1 field
        for ln in lines:
            if ln.startswith("PSHELL") and ln[8:16].strip() == str(template_pid):
                skin_mid = ln[16:24].strip(); break

    skin_mat_card, skin_mat_type = _extract_mat_card(skin_mid) if skin_mid else (None, None)
    if skin_mat_card is None:
        raise ValueError(
            f"FSO export: could not resolve the skin material (MID={skin_mid}) "
            f"from {fso_template_path}. The skin material must be copied from "
            f"the template - no material is hardcoded.")
    print(f"  FSO skin material: {skin_mat_type} MID={skin_mid} "
          f"(copied from the MMC template)")

    # --- stiffener material ---
    # PLY_SCALING / MAT8 stiffener modes are made of the SKIN material, so the
    # frozen stiffener ply reuses the skin MID. The MAT1 mode uses a dedicated
    # isotropic MAT1 built from equiv_E / equiv_nu (which main.py already sets
    # to the real MMC stiffener properties: e.g. the steel MAT1 for the
    # bulkhead).
    _stiff_uses_skin_mat = stiffener_model in ("PLY_SCALING", "MAT8")
    if _stiff_uses_skin_mat:
        stiff_mid = skin_mid
        stiff_mat_card = None
        print(f"  FSO stiffener material: same as skin ({skin_mat_type} "
              f"MID={skin_mid}) [{stiffener_model}]")
    else:
        stiff_mid = str(stiffener_mat_id)
        stiff_mat_card = [(f"MAT1    {stiffener_mat_id:>8}"
                           f"{_fmt8(equiv_E)}{_fmt8(equiv_G)}"
                           f"{equiv_nu:>8.4f}\n")]
        print(f"  FSO stiffener material: MAT1 MID={stiffener_mat_id} "
              f"E={equiv_E:.1f} nu={equiv_nu:.4f} [{stiffener_model}]")

    # Constant-skin FSO design space: one ply per UNIQUE angle, in the ACTUAL
    # skin material, with LAM = SMEAR. The starting total skin thickness must
    # equal the physical total of the template skin laminate - read straight
    # from the template plies, NOT from their individual thicknesses:
    #
    #   * the total is the SUM of the listed ply thicknesses;
    #   * if the template laminate is LAM = SYM / SYSMEAR the listed plies are
    #     only the half-stack (mirrored about the mid-plane), so the physical
    #     total is DOUBLED;
    #   * that total is then split equally over the number of unique angles,
    #     giving one ply per angle of thickness  total / n_unique_angles.
    #
    # This is generalized for any per-ply thickness, any number of plies, any
    # LAM setting, and any set of unique angles.
    def _read_skin_total_and_angles():
        if template_prop_type in ("PCOMP", "PCOMPG"):
            for k, ln in enumerate(lines):
                is_pcompg = ln.startswith("PCOMPG")
                is_pcomp = ln.startswith("PCOMP") and not is_pcompg
                if ((template_prop_type == "PCOMPG" and is_pcompg) or
                        (template_prop_type == "PCOMP" and is_pcomp)) and \
                        ln[8:16].strip() == str(template_pid):
                    cont = []
                    j = k + 1
                    while j < len(lines) and _is_continuation(lines[j]):
                        cont.append(lines[j])
                        j += 1
                    plies = (_parse_pcompg_plies(cont) if is_pcompg
                             else _parse_pcomp_plies(cont))
                    lam = ln[64:72].strip().upper() if len(ln) > 64 else ''
                    listed_total = sum(float(p['T']) for p in plies)
                    total = listed_total * (2.0 if lam.startswith('SY') else 1.0)
                    angles = list(dict.fromkeys(
                        str(p.get('theta', '0.0')).strip() for p in plies))
                    return total, (angles or ['0.0']), lam or '(none)'
        # PSHELL isotropic skin: build a quasi-isotropic design-space laminate
        # at the shell thickness used in the optimization.
        return (float(np.mean(problem.skin_thickness)),
                ['0.0', '45.0', '-45.0', '90.0'], 'PSHELL')

    total_skin_t, _skin_angles, _lam_used = _read_skin_total_and_angles()
    skin_ply_t = total_skin_t / len(_skin_angles)
    skin_ply_info = [{'mid': skin_mid, 'T': skin_ply_t, 'theta': th}
                     for th in _skin_angles]
    print(f"  FSO skin: template LAM={_lam_used}, total {total_skin_t:.4g} mm "
          f"over {len(_skin_angles)} unique angle(s) {_skin_angles} = "
          f"{skin_ply_t:.4g} mm/ply")

    # ------------------------------------------------------------------
    # 3. Helper builders
    # ------------------------------------------------------------------
    _H_THRESH = 0.01

    stiff_pid_offset = 100000

    def _stiff_pid(eid_int):
        return eid_int + stiff_pid_offset

    # Coincident FLAT-SIDE stiffener shells (matching the DfM MMC-driven model,
    # convert_stiffeners_to_flat_side): the stiffener is NOT a ply inside the
    # skin PCOMP but a SEPARATE CQUAD4 on the SAME four nodes as the skin
    # element (EID + stiff_eid_offset), pointing to its own single-ply property
    # of thickness H*stiffener_height. Skin and stiffener shells share nodes, so
    # their stiffnesses add in parallel. The skin keeps its FSO per-element
    # variable thickness (LAM=SMEAR), untouched — the flat-side approach never
    # replaces the skin under the stiffener (fso_integration is not used here).
    _max_elem_id = max(all_eids_set) if all_eids_set else 0
    for _ln in lines:
        if _ln[:6] in ("CQUAD4", "CTRIA3", "CQUAD8", "CTRIA6"):
            try:
                _max_elem_id = max(_max_elem_id, int(_ln[8:16]))
            except ValueError:
                pass
    stiff_eid_offset = 10 ** (len(str(_max_elem_id)) + 1)

    def _build_skin_elem(eid_int):
        """Per-element variable-thickness skin PCOMP (LAM=SMEAR), skin plies
        only — the FSO per-ply thicknesses, unchanged. Any stiffener is a
        separate coincident shell, never a ply here."""
        base = ply_info['base_plies']
        mid = skin_mid                      # actual MMC skin material
        angle_to_dv = ply_info.get('angle_to_dv')
        raw_sh = ply_info['elem_multipliers'].get(eid_int, [])
        mults = (expand_sh_to_per_ply(raw_sh, base, angle_to_dv)
                 if raw_sh else [1.0] * len(base))
        skin_entries = [(mid, (mults[k] if k < len(mults) else 1.0) * ply['T'],
                         ply.get('theta', '0.0')) for k, ply in enumerate(base)]
        result = [f"PCOMP   {eid_int:>8}" + " " * 48 + f"{'SMEAR':>8}\n"]
        for j in range(0, len(skin_entries), 2):
            mid1, t1, th1 = skin_entries[j]
            cont = f"+       {mid1:>8}{_fmt8(t1)}{th1:>8}{'YES':>8}"
            if j + 1 < len(skin_entries):
                mid2, t2, th2 = skin_entries[j + 1]
                cont += f"{mid2:>8}{_fmt8(t2)}{th2:>8}{'YES':>8}"
            result.append(cont + "\n")
        return result

    def _build_stiffener_shell_prop(eid_int):
        """Single-ply PCOMP (NO LAM entry - explicit stack) for the coincident
        flat-side stiffener shell: the resolved stiffener material at thickness
        H*stiffener_height. Only the multi-ply skin uses LAM=SMEAR."""
        pid = _stiff_pid(eid_int)
        t = stiff_map[eid_int]
        return [f"PCOMP   {pid:>8}\n",
                f"+       {stiff_mid:>8}{_fmt8(t)}{'0.0':>8}{'YES':>8}\n"]

    def _emit_constant_skin_cards(out):
        if nonstiff_eids_sorted:
            # LAM = SMEAR only. The LAM field is cols 65-72, so everything
            # before it must stay blank: an earlier version also wrote SYSMEAR
            # into the GE field (cols 57-64), which expects a damping NUMBER,
            # and OptiStruct rejected the card with "ERROR 1485: unrecognized
            # label" -> the PCOMP was dropped, its continuation lines became
            # orphans ("ERROR 1490") and the whole deck failed to solve.
            out.append(f"PCOMP   {template_pid:>8}{'':>48}{'SMEAR':>8}\n")
            for j in range(0, len(skin_ply_info), 2):
                sp1 = skin_ply_info[j]
                cont = f"+       {sp1['mid']:>8}{_fmt8(sp1['T'])}{sp1['theta']:>8}{'YES':>8}"
                if j + 1 < len(skin_ply_info):
                    sp2 = skin_ply_info[j + 1]
                    cont += f"{sp2['mid']:>8}{_fmt8(sp2['T'])}{sp2['theta']:>8}{'YES':>8}"
                out.append(cont + "\n")

        if stiffened_eids:
            out.append("$ --- Per-element PCOMPG for stiffened elements ---\n")
            for eid_int in sorted(stiffened_eids):
                pid = _stiff_pid(eid_int)
                out.append(f"PCOMPG  {pid:>8}\n")
                # Skin under the stiffener = the SAME full skin laminate as the
                # unstiffened elements (sp['T'] per angle, summing to the total
                # optimization skin thickness). The stiffener ply below is added
                # on top and left untouched.
                for gplyid, sp in enumerate(skin_ply_info, start=1):
                    out.append(
                        f"{'':8}{gplyid:>8}{sp['mid']:>8}"
                        f"{_fmt8(sp['T'])}{sp['theta']:>8}{'YES':>8}\n")
                stiff_gply = 900000 + eid_int
                out.append(
                    f"{'':8}{stiff_gply:>8}{stiff_mid:>8}"
                    f"{_fmt8(stiff_map[eid_int])}{'0.0':>8}{'YES':>8}\n")

    # ------------------------------------------------------------------
    # 4. Main processing loop
    # ------------------------------------------------------------------
    skip_dsize_cont = False
    output_lines = []
    property_written = False
    desglb_seen = False
    begin_bulk_seen = False

    i = 0
    while i < len(lines):
        line = lines[i]

        # Remove old DSIZE blocks from template
        if line.startswith("DSIZE"):
            skip_dsize_cont = True
            i += 1
            continue
        if skip_dsize_cont and _is_continuation(line):
            i += 1
            continue
        if skip_dsize_cont:
            skip_dsize_cont = False

        # Always rebuild final ENDDATA section ourselves
        if line.strip() == "ENDDATA":
            i += 1
            continue

        # Rebuild objective setup in case control
        if line.startswith("DESGLB"):
            output_lines.append("DESGLB        2\n")
            desglb_seen = True
            i += 1
            continue
        if "DESOBJ" in line and not line.lstrip().startswith("$"):
            i += 1
            continue
        if line.strip() == "BEGIN BULK":
            if not desglb_seen:
                output_lines.append("DESGLB        2\n")
            output_lines.append("  DESOBJ(MIN)=1\n")
            output_lines.append(line)
            begin_bulk_seen = True
            i += 1
            continue

        # Remove any pre-existing ply/stack/pcompp to avoid duplicates
        if line.startswith("PLY") or line.startswith("STACK") or line.startswith("PCOMPP"):
            i += 1
            continue

        # Remove previous optimization response/constraint cards
        if line.startswith("DRESP1") or line.startswith("DCONSTR") or line.startswith("DCONADD"):
            i += 1
            continue

        # SPCs are kept EXACTLY as in the source template: the export uses
        # the run's own template, so the FSO sees the same BCs the stiffener
        # layout was optimized for.

        # CQUAD4 property reassignment
        if line.startswith("CQUAD4"):
            try:
                eid = int(line[8:16].strip())
                if eid not in all_eids_set:
                    output_lines.append(line)
                elif is_variable_skin:
                    # skin element -> its own per-element variable skin PID (=eid)
                    output_lines.append(line[:16] + f"{eid:>8}" + line[24:])
                    # stiffened -> add a COINCIDENT flat-side stiffener shell on
                    # the SAME four nodes (line[24:]), its own EID and PID.
                    if eid in stiffened_eids:
                        output_lines.append(
                            "CQUAD4  " + f"{eid + stiff_eid_offset:>8}"
                            + f"{_stiff_pid(eid):>8}" + line[24:])
                elif eid in stiffened_eids:
                    output_lines.append(line[:16] + f"{_stiff_pid(eid):>8}" + line[24:])
                else:
                    output_lines.append(line[:16] + f"{int(template_pid):>8}" + line[24:])
            except ValueError:
                output_lines.append(line)
            i += 1
            continue

        # Replace the template base property card
        is_template_prop = (
            (template_prop_type == "PSHELL" and line.startswith("PSHELL") and line[8:16].strip() == template_pid)
            or (template_prop_type == "PCOMPG" and line.startswith("PCOMPG") and line[8:16].strip() == template_pid)
            or (template_prop_type == "PCOMP" and line.startswith("PCOMP") and not line.startswith("PCOMPG") and line[8:16].strip() == template_pid)
        )
        if is_template_prop:
            if is_variable_skin:
                output_lines.append("$ --- PER-ELEMENT variable-thickness skin PCOMP (SMEAR) ---\n")
                for eid_int in all_eids_sorted:
                    output_lines.extend(_build_skin_elem(eid_int))
                if stiffened_eids:
                    output_lines.append("$ --- Coincident flat-side stiffener shell PCOMP ---\n")
                    for eid_int in sorted(stiffened_eids):
                        output_lines.extend(_build_stiffener_shell_prop(eid_int))
            else:
                _emit_constant_skin_cards(output_lines)
            property_written = True
            j = i + 1
            while j < len(lines) and _is_continuation(lines[j]):
                j += 1
            i = j
            continue

        # Drop ALL template materials (MAT1/MAT8, with continuations): the
        # exact skin + stiffener materials used by the MMC are re-emitted
        # below, so the FSO uses the same materials regardless of what else
        # the template carried.
        if line.startswith("MAT8") or line.startswith("MAT1"):
            j = i + 1
            while j < len(lines) and _is_continuation(lines[j]):
                j += 1
            i = j
            continue

        output_lines.append(line)
        i += 1

    # If no template property was replaced, still emit generated properties
    if not property_written:
        if is_variable_skin:
            output_lines.append("$ --- PER-ELEMENT variable-thickness skin PCOMP (SMEAR) ---\n")
            for eid_int in all_eids_sorted:
                output_lines.extend(_build_skin_elem(eid_int))
            if stiffened_eids:
                output_lines.append("$ --- Coincident flat-side stiffener shell PCOMP ---\n")
                for eid_int in sorted(stiffened_eids):
                    output_lines.extend(_build_stiffener_shell_prop(eid_int))
        else:
            _emit_constant_skin_cards(output_lines)

    # Emit the ACTUAL skin + stiffener materials (resolved from the MMC
    # template above). When the stiffener is made of the skin material
    # (PLY_SCALING / MAT8 modes) only the skin card is written, so the two
    # share one MID; otherwise the dedicated stiffener MAT1 is written too.
    output_lines.append(f"$ --- SKIN MATERIAL ({skin_mat_type} MID {skin_mid}) "
                        f"from MMC template ---\n")
    output_lines.extend(skin_mat_card)
    if stiff_mat_card is not None:
        output_lines.append(f"$ --- STIFFENER MATERIAL (MAT1 MID {stiff_mid}) "
                            f"from MMC ---\n")
        output_lines.extend(stiff_mat_card)

    if not begin_bulk_seen:
        output_lines.append("DESGLB        2\n")
        output_lines.append("  DESOBJ(MIN)=1\n")
        output_lines.append("BEGIN BULK\n")
        begin_bulk_seen = True

    # DSIZE only for the non-stiffened PCOMP design domain (constant skin case)
    if (not is_variable_skin) and nonstiff_eids_sorted:
        output_lines.append(f"DSIZE   {'1':>8}{'PCOMP':>8}{int(template_pid):>8}\n")
        output_lines.append(
            f"+       COMP    LAMTHK  {_fmt8(min_laminate_thickness)}"
            f"{'':>24}\n")
        output_lines.append("+       COMP    BALANCE 45.0    -45.0           BYANG   \n")

    # Hardcoded optimization responses and constraints
    output_lines.append("DRESP1  1       optirespCOMP                                    \n")
    output_lines.append("DRESP1  2       optirespVOLFRAC                                 \n")
    output_lines.append("DCONSTR        1       20.0     0.6                             \n")
    output_lines.append("DCONADD        2       1\n")

    output_lines.append("ENDDATA\n")

    with open(output_path, 'w') as f:
        f.writelines(output_lines)

    print(f"\n  Wrote {output_path}")
    if is_variable_skin:
        print(f"    {n_stiffened} coincident flat-side stiffener shells "
              f"(SMEAR skin + separate stiffener CQUAD4)")
    else:
        print(f"    {n_stiffened} elements with stiffener ply")


# =============================================================================
#                            RUN ORCHESTRATION
# =============================================================================

_TEMPLATES = {1: "MMC_LAM2.fem", 2: "MMC_LAM4.fem", 3: "MMC_LAM5.fem",
              4: "MMC_LAM6.fem", 5: "Bulkhead.fem", 6: "Bulkhead_compress.fem",
              7: "Bulkhead_asymmetric.fem"}


def select_template(numerical_example, fso_sh_file, initial_layout):
    if fso_sh_file is not None:
        # the .fem matching the .sh holds the base layup and free-size DSIZE,
        # so the per-DV multipliers map back to per-ply thicknesses
        template = os.path.splitext(fso_sh_file)[0] + ".fem"
        if not os.path.exists(template):
            raise FileNotFoundError(
                f"FSO run: expected the free-size template '{template}' next to "
                f"the .sh file '{fso_sh_file}' (same basename), but it is missing.")
        return template, initial_layout
    if numerical_example not in _TEMPLATES:
        raise ValueError(f"Unknown numerical example: {numerical_example}")
    return _TEMPLATES[numerical_example], initial_layout


def prepare_run_folder(run_folder):
    """Create/empty the run folder and tee stdout into its log. Returns the
    log file handle (close it at the end of the run)."""
    if not os.path.exists(run_folder):
        os.makedirs(run_folder)
    else:
        for f in os.listdir(run_folder):
            try:
                os.unlink(os.path.join(run_folder, f))
            except OSError:
                pass
    fh = open(os.path.join(run_folder, "run_log.txt"), 'w')
    sys.stdout = Tee(sys.stdout, fh)
    return fh


def resolve_stiffener_material(material_mode, use_mat8, manual_E, manual_nu,
                               template):
    """(E, nu) overrides for the stiffener ply. MAT8 mode uses the skin's own
    material, HOMOGENIZED is resolved later inside the expansion."""
    if use_mat8 or material_mode == "HOMOGENIZED":
        return None, None
    if material_mode == "MANUAL":
        return manual_E, manual_nu
    E, nu = read_template_mat1(template)
    print(f"Stiffener material from template MAT1: E = {E:.1f}, nu = {nu:.4f}")
    return E, nu


def build_skin_thickness(prop_type, ply_info, skin_t, elem_ids):
    """Per-element skin thickness for a free-size skin, scalar otherwise."""
    if prop_type != "PCOMP_VARIABLE":
        return skin_t
    base_t = [p['T'] for p in ply_info['base_plies']]
    elem_mults = ply_info['elem_multipliers']
    angle_to_dv = ply_info.get('angle_to_dv')
    out = np.zeros(len(elem_ids))
    for i, eid in enumerate(elem_ids):
        raw_sh = elem_mults.get(int(eid), [])
        expanded = (expand_sh_to_per_ply(raw_sh, ply_info['base_plies'],
                                         angle_to_dv)
                    if raw_sh else [1.0] * len(base_t))
        out[i] = sum(m * t for m, t in zip(expanded, base_t))
    return out


def build_design_bounds(problem, bounds):
    """Bounds for [x0, y0, L, W, theta] per component. Returns (LB, UB, SCALE)."""
    x_min, x_max = bounds['x_min'], bounds['x_max']
    y_min, y_max = bounds['y_min'], bounds['y_max']
    max_L = max(x_max - x_min, y_max - y_min)
    margin = max_L * 0.5

    w_lb, w_ub = 1e-4, 100.0

    b = []
    for _ in problem.components:
        b += [(x_min - margin, x_max + margin),
              (y_min - margin, y_max + margin),
              (0.001, max_L * 1.5),
              (w_lb, w_ub),
              (-np.pi, np.pi)]
    LB = np.array([v[0] for v in b])
    UB = np.array([v[1] for v in b])
    SCALE = UB - LB
    return LB, UB, np.where(SCALE == 0, 1.0, SCALE)


def resolve_export_material(stiff_E, stiff_nu, ply_info, template):
    """(E, nu) written into the FSO-with-stiffeners export."""
    if stiff_E is not None:
        return stiff_E, stiff_nu
    if isinstance(ply_info, list):
        plies = ply_info
    elif isinstance(ply_info, dict):
        plies = ply_info.get('base_plies')
    else:
        plies = None
    homog = homogenized_isotropic_from_skin(template, plies) if plies else None
    if homog is None:
        homog = read_template_mat1(template)
    return homog


def analyze_coincident_shell_compliance(fea, run_folder, fem_filename, label):
    """Solve an exported FSO_with_stiffeners*.fem as a pure static analysis so
    the result is measured on the same model as the DfM and decoupled
    baselines. Returns the compliance, or None if it was skipped/failed."""
    src = os.path.join(run_folder, fem_filename)
    if not os.path.exists(src):
        print(f"  [{label}] export not found: {fem_filename}")
        return None
    analysis_fem = fem_filename.replace(".fem", "_analysis.fem")
    with open(src) as fh:
        lines = fh.readlines()
    with open(os.path.join(run_folder, analysis_fem), "w") as fh:
        for line in lines:
            if line.strip().upper().startswith(("DESOBJ", "DESGLB", "DESSUB",
                                                "DESVAR")):
                continue
            fh.write(line)
    jobname = analysis_fem.replace(".fem", "")
    try:
        fea.run_solver(analysis_fem)
        c, _, _ = fea.read_results(jobname)
    except Exception as e:
        print(f"  [{label}] analysis failed: {e}")
        return None
    if not (np.isfinite(c) and c > 0.0):
        print(f"  [{label}] INVALID compliance ({c}) - solve skipped/failed "
              f"(licence?); do not use.")
        return None
    print(f"  [{label}] coincident-shell compliance = {c:.2f} N.mm")
    return c


def evaluate_final_design(fea, problem, run_folder, jobname="eval_final"):
    """Solve the current design with DISP output. Returns (C, max_disp)."""
    fea.generate_input_file(problem, f"{jobname}.fem")
    fem_path = os.path.join(run_folder, f"{jobname}.fem")
    fem_lines = open(fem_path).readlines()
    inserted = False
    with open(fem_path, 'w') as fh:
        for line in fem_lines:
            fh.write(line)
            if not inserted and line.startswith("ESE(PUNCH)"):
                fh.write("DISP(PUNCH) = ALL\n")
                inserted = True
    fea.run_solver(f"{jobname}.fem")
    c, _, d = fea.read_results(jobname)
    return c, d


def save_final_plots(problem, run_folder, current_iter, evals, compliances,
                     volumes, vol_target, final_comp, final_vf):
    plot_iteration(problem, current_iter,
                   save_path=os.path.join(run_folder, "design_FINAL.png"))
    plot_components_vector(problem,
                           os.path.join(run_folder, "design_components_FINAL.png"),
                           title="Final component layout")
    plot_components_contour(problem,
                            os.path.join(run_folder, "design_contour_FINAL.png"),
                            title="Final component contour")
    if not evals:
        return
    plot_convergence(evals, compliances,
                     os.path.join(run_folder, "convergence_FINAL.png"))
    plot_paper_convergence(evals, compliances, volumes, vol_target,
                           os.path.join(run_folder, "paper_convergence.png"),
                           final_comp=final_comp, final_vf=final_vf)
    plot_specific_compliance(evals, compliances, [max(v, 1e-6) for v in volumes],
                             os.path.join(run_folder,
                                          "specific_compliance_FINAL.png"))
