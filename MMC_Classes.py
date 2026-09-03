import numpy as np


class MMC_Component:
    """
    One morphable component (stiffener), described by the explicit geometric
    design vector a_i = [x0, y0, L, W, theta]:
      (x0, y0)  centre coordinates
      L         half-length
      width     width W (the TDF uses the half-width W/2)
      theta     orientation angle, measured from the horizontal axis
    The component boundary is the p-th order superellipse of the TDF
      phi_i(x, y) = 1 - (|x'|/L)^p - (|y'|/(W/2))^p
    with (x', y') the point coordinates in the component's local frame.
    """

    def __init__(self, x0, y0, L, width, theta, p=6):
        self.x0 = float(x0)
        self.y0 = float(y0)
        self.L = float(L)
        self.width = float(width)
        self.theta = float(theta)
        self.p = p

    def calculate_phi(self, X_mesh, Y_mesh):
        """TDF phi_i at the given points: > 0 inside, = 0 on the boundary,
        < 0 outside the component."""
        dx = X_mesh - self.x0
        dy = Y_mesh - self.y0
        c, s = np.cos(self.theta), np.sin(self.theta)
        x_prime = c * dx + s * dy
        y_prime = -s * dx + c * dy
        safe_L = max(self.L, 1e-4)
        half_W = max(self.width / 2.0, 1e-6)
        term1 = (np.abs(x_prime) / safe_L) ** self.p
        term2 = (np.abs(y_prime) / half_W) ** self.p
        return 1.0 - term1 - term2

    def calculate_derivatives(self, X_mesh, Y_mesh):
        """Analytical derivatives of phi_i with respect to the five design
        variables [x0, y0, L, W, theta], stacked as a (5, n_points) array."""
        dx = X_mesh - self.x0
        dy = Y_mesh - self.y0
        c, s = np.cos(self.theta), np.sin(self.theta)
        x_prime = c * dx + s * dy
        y_prime = -s * dx + c * dy
        half_L = max(self.L, 1e-4)
        half_W = max(self.width / 2.0, 1e-6)
        p = self.p
        dphi_dx_prime = -p * (np.abs(x_prime) / half_L) ** (p - 1) * np.sign(x_prime) / half_L
        dphi_dy_prime = -p * (np.abs(y_prime) / half_W) ** (p - 1) * np.sign(y_prime) / half_W
        dphi_dx0 = dphi_dx_prime * (-c) + dphi_dy_prime * (s)
        dphi_dy0 = dphi_dx_prime * (-s) + dphi_dy_prime * (-c)
        dphi_dL = (np.abs(x_prime) / half_L) ** p * (p / half_L)
        dphi_dW = (np.abs(y_prime) / half_W) ** p * (p / half_W) * 0.5
        dx_prime_dtheta = -s * dx + c * dy
        dy_prime_dtheta = -c * dx - s * dy
        dphi_dtheta = dphi_dx_prime * dx_prime_dtheta + dphi_dy_prime * dy_prime_dtheta
        return np.array([dphi_dx0, dphi_dy0, dphi_dL, dphi_dW, dphi_dtheta])


class MMC_Problem:
    """
    The MMC stiffener-layout optimization problem:

        min  C(a)   s.t.  V(a) <= V_target,  a_lb <= a <= a_ub

    Geometry mapping (per element e, evaluated at the element centroid):
        Phi_s = max_i phi_i                      (global TDF, hard max)
        H_e   = H_eps(Phi_s)                     (polynomial regularized
                                                  Heaviside, band +-eps)
        T_e   = T_skin + H_e * T_stiffener       (thickness interpolation)

    Sensitivities follow the analytical chain rule
        dC/da = sum_e dC/dT_e * dT_e/dH_e * dH_e/dPhi_s * dPhi_s/dphi_i
                       * dphi_i/da
    with dC/dT_e obtained from the element strain energies (ESE) of the
    OptiStruct run, and dPhi_s/dphi_i the winner-takes-all indicator of the
    hard max.

    The optional symmetry enforcement mode mirrors every component across
    the selected mid-plane(s); a component whose centre lies ON an active
    plane is self-symmetric: it is pinned to the plane with an axis-aligned
    angle and counted once (no clone).
    """

    def __init__(self, bounds, x_mesh, y_mesh, elem_ids, max_iter=200,
                 skin_thickness=2.0, stiffener_height=20.0,
                 symmetry="OFF", t_base=None, initial_layout=2,
                 heaviside_eps=0.9, sens_eps=None, ortho=False):
        self.bounds = bounds
        self.X = x_mesh
        self.Y = y_mesh
        self.elem_ids = elem_ids
        self.components = []
        self.max_iter = max_iter
        _valid_sym = ("OFF", "ONE_PLANE", "ONE_PLANE_X", "ONE_PLANE_Y",
                      "TWO_PLANE", "FOUR_FOLD")
        if symmetry not in _valid_sym:
            raise ValueError(
                f"Unknown SYMMETRY '{symmetry}'; valid options: {_valid_sym}")
        if symmetry == "FOUR_FOLD":
            # D4 (square) symmetry only makes sense on a square domain: the
            # diagonal mirrors and 90 deg rotations map the domain onto itself
            # only when the two side lengths are equal.
            _dw = bounds['x_max'] - bounds['x_min']
            _dh = bounds['y_max'] - bounds['y_min']
            if abs(_dw - _dh) > 1e-6 * max(_dw, _dh, 1.0):
                raise ValueError(
                    f"SYMMETRY='FOUR_FOLD' (D4 square symmetry) requires a "
                    f"square design domain, but this one is {_dw:.4g} x "
                    f"{_dh:.4g}. Use TWO_PLANE / ONE_PLANE_* for a rectangular "
                    f"panel.")
        self.symmetry = symmetry
        self.initial_layout = initial_layout
        # ortho: restrict all component orientations to 0/90 deg. Components
        # are snapped to the nearest axis at initialization and re-snapped
        # on every design update (safety net; in OPT mode main.py also
        # freezes the theta design variables so they never move).
        self.ortho = ortho
        self.n_var = 5
        # Regularization parameter eps of the polynomial Heaviside: the +-eps
        # band of the (dimensionless) TDF must stay resolvable by the mesh —
        # the TDF slope at a member boundary is p/half_width per mm, so the
        # band is ~2*eps*half_width/p mm wide physically.
        self.heaviside_eps = float(heaviside_eps)
        # Sensitivity smoothing: dH/dPhi in the gradients uses sens_eps, while
        # the forward H (thickness field, volume) uses heaviside_eps. sens_eps
        # >= heaviside_eps widens the gradient band (removes the dead gradient
        # core, evens out per-component sensitivities) without changing the
        # physical field - a smoothed, deliberately inconsistent sensitivity
        # (cf. Sigmund's sensitivity filtering). None => same as heaviside_eps
        # (exact gradients).
        self.sens_eps = (float(heaviside_eps) if sens_eps is None
                         else float(sens_eps))

        skin_t_input = np.asarray(skin_thickness, dtype=float).ravel()
        if skin_t_input.size == 1:
            self.skin_thickness = np.full(len(elem_ids), skin_t_input[0])
        else:
            self.skin_thickness = skin_t_input

        self.stiffener_height = float(stiffener_height)

        if t_base is None:
            self.t_base = self.skin_thickness.copy()
        else:
            t_base_input = np.asarray(t_base, dtype=float).ravel()
            if t_base_input.size == 1:
                self.t_base = np.full(len(elem_ids), t_base_input[0])
            else:
                self.t_base = t_base_input

        self._update_delta()
        self._initialize_components()
        self._restrict_to_fundamental_domain()
        self._enforce_ortho()
        self._mark_on_plane_components()
        self._enforce_on_plane_pins()

    def _restrict_to_fundamental_domain(self):
        """When a mirror symmetry is active, keep only the components whose
        centre lies in the fundamental domain — the LEFT half (x <= cx) for an
        x-mirror, the UPPER half (y >= cy) for a y-mirror, the upper-left
        quadrant for TWO_PLANE. The mirror clones generated in
        _get_symmetry_clones regenerate the rest, so this halves/quarters the
        independent design variables without changing the effective layout.

        This makes EVERY initial layout symmetry-aware generically: previously
        only layouts 2 and 10 hand-coded the restriction, so enabling symmetry
        on any other layout silently double/quadruple-counted the design
        (it generated the full domain AND cloned it). Components already on a
        mirror plane are kept — they become self-symmetric pins.
        """
        if self.symmetry == "OFF" or not self.components:
            return
        cx = (self.bounds['x_min'] + self.bounds['x_max']) / 2.0
        cy = (self.bounds['y_min'] + self.bounds['y_max']) / 2.0
        dw = self.bounds['x_max'] - self.bounds['x_min']
        dh = self.bounds['y_max'] - self.bounds['y_min']
        tol_x = 1e-6 * max(dw, 1.0)
        tol_y = 1e-6 * max(dh, 1.0)

        if self.symmetry == "FOUR_FOLD":
            # D4 fundamental domain: one OCTANT of the square. Keep the
            # upper-left wedge between the vertical mid-line and the
            # anti-diagonal to the top-left corner: u <= 0, v >= 0, v >= -u
            # (u = x0-cx, v = y0-cy). The eight D4 images fill the rest.
            tol = 1e-6 * max(dw, dh, 1.0)
            kept = []
            for c in self.components:
                u = c.x0 - cx
                v = c.y0 - cy
                if u <= tol and v >= -tol and (v + u) >= -tol:
                    kept.append(c)
            self.components = kept
            return

        mirrors_x = self.symmetry in ("ONE_PLANE", "ONE_PLANE_X", "TWO_PLANE")
        mirrors_y = self.symmetry in ("ONE_PLANE_Y", "TWO_PLANE")
        kept = []
        for c in self.components:
            if mirrors_x and c.x0 > cx + tol_x:
                continue   # right half comes from the x-mirror clone
            if mirrors_y and c.y0 < cy - tol_y:
                continue   # lower half comes from the y-mirror clone
            kept.append(c)
        self.components = kept

    def _enforce_ortho(self):
        """Snap every component orientation to the nearest axis (0/90 deg).
        No-op unless the problem was built with ortho=True."""
        if not self.ortho:
            return
        for c in self.components:
            c.theta = 0.0 if abs(np.cos(c.theta)) >= abs(np.sin(c.theta)) \
                else np.pi / 2.0

    def _update_delta(self):
        # delta = dT/dH: the thickness added at H = 1 (equals the stiffener
        # height when t_base == skin thickness)
        self.delta = np.maximum(
            self.stiffener_height + self.t_base - self.skin_thickness, 0.0)

    # ------------------------------------------------------------------
    # Symmetry enforcement
    # ------------------------------------------------------------------
    def _mark_on_plane_components(self):
        """
        Components that are SELF-SYMMETRIC under an active mirror plane are
        pinned: their centre stays on the plane, their (already axis-aligned)
        angle stays snapped, and no clone is generated for that plane — the
        clone would coincide with the component and be double-counted.

        Self-symmetric requires BOTH: centre on the plane AND an axis-aligned
        orientation (0/90 deg) — only then does the mirror map the bar onto
        itself. A DIAGONAL bar centred on the plane is NOT self-symmetric
        (its mirror is the opposite diagonal, a genuinely distinct member)
        and must be cloned normally, never pinned or angle-snapped.
        """
        cx = (self.bounds['x_min'] + self.bounds['x_max']) / 2.0
        cy = (self.bounds['y_min'] + self.bounds['y_max']) / 2.0
        dw = self.bounds['x_max'] - self.bounds['x_min']
        dh = self.bounds['y_max'] - self.bounds['y_min']
        tol_x = 1e-6 * max(dw, 1.0)
        tol_y = 1e-6 * max(dh, 1.0)
        ang_tol = 1e-3  # rad-equivalent: min(|sin|,|cos|) below this = axis-aligned

        mirrors_x = self.symmetry in ("ONE_PLANE", "ONE_PLANE_X", "TWO_PLANE")
        mirrors_y = self.symmetry in ("ONE_PLANE_Y", "TWO_PLANE")
        for c in self.components:
            axis_aligned = min(abs(np.sin(c.theta)),
                               abs(np.cos(c.theta))) < ang_tol
            c.pin_x = bool(mirrors_x and axis_aligned
                           and abs(c.x0 - cx) < tol_x)
            c.pin_y = bool(mirrors_y and axis_aligned
                           and abs(c.y0 - cy) < tol_y)

    def _enforce_on_plane_pins(self):
        """Keep self-symmetric components on their mirror plane with an
        axis-aligned angle. No-op when nothing is pinned."""
        cx = (self.bounds['x_min'] + self.bounds['x_max']) / 2.0
        cy = (self.bounds['y_min'] + self.bounds['y_max']) / 2.0
        for c in self.components:
            pin_x = getattr(c, 'pin_x', False)
            pin_y = getattr(c, 'pin_y', False)
            if pin_x:
                c.x0 = cx
            if pin_y:
                c.y0 = cy
            if pin_x or pin_y:
                c.theta = 0.0 if abs(np.cos(c.theta)) >= abs(np.sin(c.theta)) \
                    else np.pi / 2.0

    @staticmethod
    def _sym_jac(m00, m01, m10, m11, th_sign):
        """5x5 chain Jacobian J = d(clone vars)/d(source vars) for a symmetry
        operation whose position map is the 2x2 [[m00,m01],[m10,m11]] acting on
        (x0-cx, y0-cy), that leaves L and W unchanged, and maps theta with
        derivative th_sign (-1 for a reflection theta->const-theta, +1 for a
        rotation theta->theta+const)."""
        J = np.eye(5)
        J[0, 0] = m00; J[0, 1] = m01
        J[1, 0] = m10; J[1, 1] = m11
        J[4, 4] = th_sign
        return J

    # The eight D4 (square dihedral) operations as
    # (m00, m01, m10, m11, theta_map, theta_sign). The position map acts on
    # (u, v) = (x0-cx, y0-cy); the diagonal mirrors and the +-90 deg rotations
    # swap u<->v, which is why the chain has to be a full matrix, not a
    # per-variable sign flip.
    _D4_OPS = [
        (1, 0, 0, 1, (lambda t: t), 1.0),                        # identity
        (-1, 0, 0, 1, (lambda t: np.pi - t), -1.0),              # reflect x=cx
        (1, 0, 0, -1, (lambda t: -t), -1.0),                     # reflect y=cy
        (-1, 0, 0, -1, (lambda t: np.pi + t), 1.0),              # rotate 180
        (0, 1, 1, 0, (lambda t: np.pi / 2 - t), -1.0),           # reflect y=x
        (0, -1, -1, 0, (lambda t: -np.pi / 2 - t), -1.0),        # reflect y=-x
        (0, -1, 1, 0, (lambda t: t + np.pi / 2), 1.0),           # rotate +90
        (0, 1, -1, 0, (lambda t: t - np.pi / 2), 1.0),           # rotate -90
    ]

    def _get_symmetry_clones(self):
        """
        Return list of (component, J, source_index) for all effective
        components: each source plus its mirror/rotation images. J is the 5x5
        chain Jacobian d(clone vars)/d(source vars) (see _sym_jac), so
        _get_symmetry_clones' images propagate their phi-derivatives back to
        the source design variables via J^T in calculate_analytical_sensitivities.

        ONE_PLANE_X ("ONE_PLANE" alias): mirror across x = cx.
        ONE_PLANE_Y:                     mirror across y = cy.
        TWO_PLANE:                       both planes (4-fold reflection).
        FOUR_FOLD:                       full D4 square symmetry (8 images:
          the two axis mirrors, the two diagonal mirrors, and the 90/180/270
          rotations). Images that coincide with an already-listed one (a
          source lying on a mirror line has a smaller orbit) are dropped so a
          coincident bar is never double-counted.
        """
        cx = (self.bounds['x_min'] + self.bounds['x_max']) / 2.0
        cy = (self.bounds['y_min'] + self.bounds['y_max']) / 2.0
        dw = self.bounds['x_max'] - self.bounds['x_min']
        dh = self.bounds['y_max'] - self.bounds['y_min']
        ptol = 1e-6 * max(dw, dh, 1.0)
        atol = 1e-6

        clones = []
        for i, comp in enumerate(self.components):
            clones.append((comp, np.eye(5), i))
            pin_x = getattr(comp, 'pin_x', False)
            pin_y = getattr(comp, 'pin_y', False)

            if self.symmetry in ("ONE_PLANE", "ONE_PLANE_X"):
                if not pin_x:
                    m = MMC_Component(2 * cx - comp.x0, comp.y0, comp.L,
                                      comp.width, np.pi - comp.theta, comp.p)
                    clones.append((m, self._sym_jac(-1, 0, 0, 1, -1.0), i))

            elif self.symmetry == "ONE_PLANE_Y":
                if not pin_y:
                    m = MMC_Component(comp.x0, 2 * cy - comp.y0, comp.L,
                                      comp.width, -comp.theta, comp.p)
                    clones.append((m, self._sym_jac(1, 0, 0, -1, -1.0), i))

            elif self.symmetry == "TWO_PLANE":
                if not pin_x:
                    m = MMC_Component(2 * cx - comp.x0, comp.y0, comp.L,
                                      comp.width, np.pi - comp.theta, comp.p)
                    clones.append((m, self._sym_jac(-1, 0, 0, 1, -1.0), i))
                if not pin_y:
                    m = MMC_Component(comp.x0, 2 * cy - comp.y0, comp.L,
                                      comp.width, -comp.theta, comp.p)
                    clones.append((m, self._sym_jac(1, 0, 0, -1, -1.0), i))
                if not (pin_x or pin_y):
                    m = MMC_Component(2 * cx - comp.x0, 2 * cy - comp.y0,
                                      comp.L, comp.width,
                                      np.pi + comp.theta, comp.p)
                    clones.append((m, self._sym_jac(-1, 0, 0, -1, 1.0), i))

            elif self.symmetry == "FOUR_FOLD":
                u = comp.x0 - cx
                v = comp.y0 - cy
                seen = [(comp.x0, comp.y0, comp.theta)]
                for (m00, m01, m10, m11, th_fn, th_sign) in self._D4_OPS[1:]:
                    nx = cx + m00 * u + m01 * v
                    ny = cy + m10 * u + m11 * v
                    nth = th_fn(comp.theta)
                    dup = False
                    for (sx, sy, sth) in seen:
                        da = abs((nth - sth) % np.pi)
                        da = min(da, np.pi - da)
                        if abs(nx - sx) < ptol and abs(ny - sy) < ptol and da < atol:
                            dup = True
                            break
                    if dup:
                        continue
                    seen.append((nx, ny, nth))
                    m = MMC_Component(nx, ny, comp.L, comp.width, nth, comp.p)
                    clones.append((m, self._sym_jac(m00, m01, m10, m11, th_sign), i))

        return clones

    # ------------------------------------------------------------------
    # Initial layouts
    # ------------------------------------------------------------------
    def _initialize_components(self):

        x_min, x_max = self.bounds['x_min'], self.bounds['x_max']
        y_min, y_max = self.bounds['y_min'], self.bounds['y_max']
        cx = (x_min + x_max) / 2.0
        cy = (y_min + y_max) / 2.0
        dw = x_max - x_min
        dh = y_max - y_min

        if self.initial_layout == 0:
            # No stiffeners: empty initial layout, skip optimization
            pass

        elif self.initial_layout == 1:
            # 4x4 grid of touching diagonal crosses (Sun et al. Fig. 2b, ex. 1
            # initial design: 32 components). Same structure as layout 2 but
            # each bar spans its full cell diagonal (x1.05 overlap), so
            # neighbouring crosses connect into a continuous lattice.
            n_cols = 4
            n_rows = 4
            cell_w = dw / n_cols
            cell_h = dh / n_rows
            cell_diag = np.sqrt(cell_w ** 2 + cell_h ** 2)
            comp_L = cell_diag * 0.5 * 1.05
            init_width = 4.0
            theta1 = np.arctan2(cell_h, cell_w)
            theta2 = -np.arctan2(cell_h, cell_w)

            for i in range(n_cols):
                for j in range(n_rows):
                    cx_ij = x_min + (i + 0.5) * cell_w
                    cy_ij = y_min + (j + 0.5) * cell_h
                    self.components.append(MMC_Component(cx_ij, cy_ij, comp_L, init_width, theta1))
                    self.components.append(MMC_Component(cx_ij, cy_ij, comp_L, init_width, theta2))

        elif self.initial_layout == 2:
            # Grid of 4x4 diagonal crosses (Sun et al. ex. 2/3, Fig. 3b:
            # 32 components, crosses NOT touching)
            n_cols = 4
            n_rows = 4
            cell_w = dw / n_cols
            cell_h = dh / n_rows
            cell_diag = np.sqrt(cell_w ** 2 + cell_h ** 2)
            comp_L = cell_diag * 0.4
            init_width = 4.0
            theta1 = np.pi / 4.0
            theta2 = -np.pi / 4.0

            # Always generate the full grid; _restrict_to_fundamental_domain
            # trims it to the fundamental domain when a symmetry is active.
            for i in range(n_cols):
                for j in range(n_rows):
                    cx_ij = x_min + (i + 0.5) * cell_w
                    cy_ij = y_min + (j + 0.5) * cell_h
                    self.components.append(MMC_Component(cx_ij, cy_ij, comp_L, init_width, theta1))
                    self.components.append(MMC_Component(cx_ij, cy_ij, comp_L, init_width, theta2))

        elif self.initial_layout == 5:
            # Vertical bars spanning panel height, evenly spaced across width
            n_bars = 5
            spacing = dw / (n_bars + 1)
            comp_L = dh * 1.05
            init_width = 48.0
            theta = np.pi / 2.0

            for i in range(1, n_bars + 1):
                x0 = x_min + i * spacing
                y0 = cy
                self.components.append(MMC_Component(x0, y0, comp_L, init_width, theta))

        elif self.initial_layout == 6:
            # Horizontal bars spanning panel width, evenly spaced across height
            n_bars = 5
            spacing = dh / (n_bars + 1)
            comp_L = dw * 1.05
            init_width = 48.0
            theta = 0.0

            for i in range(1, n_bars + 1):
                x0 = cx
                y0 = y_min + i * spacing
                self.components.append(MMC_Component(x0, y0, comp_L, init_width, theta))

        elif self.initial_layout == 7:
            # Grid of vertical + horizontal bars (4 per side)
            n_bars = 4
            spacing_w = dw / (n_bars + 1)
            spacing_h = dh / (n_bars + 1)
            init_width = 48.0

            for i in range(1, n_bars + 1):
                x0 = x_min + i * spacing_w
                self.components.append(MMC_Component(x0, cy, dh * 1.05, init_width, np.pi / 2.0))
            for i in range(1, n_bars + 1):
                y0 = y_min + i * spacing_h
                self.components.append(MMC_Component(cx, y0, dw * 1.05, init_width, 0.0))

        elif self.initial_layout == 8:
            # Grid of vertical + horizontal bars (20 per side)
            n_bars = 20
            spacing_w = dw / (n_bars + 1)
            spacing_h = dh / (n_bars + 1)
            init_width = 12.0

            for i in range(1, n_bars + 1):
                x0 = x_min + i * spacing_w
                self.components.append(MMC_Component(x0, cy, dh * 1.05, init_width, np.pi / 2.0))
            for i in range(1, n_bars + 1):
                y0 = y_min + i * spacing_h
                self.components.append(MMC_Component(cx, y0, dw * 1.05, init_width, 0.0))

        elif self.initial_layout == 9:
            # Six long bars at evenly spaced orientations — minimum setup
            L_diag = np.sqrt(dw ** 2 + dh ** 2)
            comp_L = L_diag * 1.05
            init_width = 48.0
            orientations = [0.0, np.pi / 6, np.pi / 3, np.pi / 2, 2 * np.pi / 3, 5 * np.pi / 6]

            for theta in orientations:
                # Place each bar through the panel centre
                self.components.append(MMC_Component(cx, cy, comp_L, init_width, theta))

        elif self.initial_layout == 10:
            # Bulkhead grid (Sun et al., Fig. 4b): 9 vertical grid lines at
            # x = x_min + i*dw/10 and 4 horizontal grid lines at
            # y = y_min + j*dh/5. One vertical + one horizontal component is
            # centered at each of the 36 intersections (72 components total).
            # Half-lengths are 1.05x the intersection spacing so neighbouring
            # segments overlap into continuous full-length grid lines.
            #
            # Symmetry support: OFF, ONE_PLANE_X/ONE_PLANE, ONE_PLANE_Y and
            # TWO_PLANE. Only the half (or quadrant) of the intersections on
            # one side of each active mirror plane is generated; the mirrors
            # recreate the rest — always the same 72 effective components.
            # The centre grid column lies ON the x-mirror plane: those
            # components are kept and handled as self-symmetric via the
            # on-plane pins (no clone, centre pinned, axis-aligned angle).
            _mirrors_x = self.symmetry in ("ONE_PLANE", "ONE_PLANE_X", "TWO_PLANE")
            _mirrors_y = self.symmetry in ("ONE_PLANE_Y", "TWO_PLANE")

            n_v = 9   # vertical grid lines
            n_h = 4   # horizontal grid lines
            spacing_x = dw / (n_v + 1)
            spacing_y = dh / (n_h + 1)
            # 18.5 mm makes the initial grid start ~AT the volume target
            # (VF ~= 0.15) at eps = 0.9, i.e. FEASIBLE from step 1. This
            # avoids the initial volume-reduction phase entirely: at an
            # infeasible start, reducing volume and reducing compliance are
            # directly opposed (any volume removal raises compliance), so MMA
            # climbs compliance destructively no matter how the constraint
            # multiplier is capped. Starting feasible, the constraint is
            # inactive and MMA simply minimizes compliance at fixed volume.
            init_width = 30
            L_vert = spacing_y * 1.05
            L_horz = spacing_x * 1.05
            tiny = 1e-9 * max(dw, dh)

            for i in range(1, n_v + 1):
                x0 = x_min + i * spacing_x
                if _mirrors_x and x0 > cx + tiny:
                    continue  # right half comes from the mirror
                for j in range(1, n_h + 1):
                    y0 = y_min + j * spacing_y
                    if _mirrors_y and y0 <= cy - tiny:
                        continue  # lower half comes from the mirror
                    self.components.append(MMC_Component(x0, y0, L_vert, init_width, np.pi / 2.0))
                    self.components.append(MMC_Component(x0, y0, L_horz, init_width, 0.0))

        else:
            raise ValueError(f"Unknown initial layout: {self.initial_layout}")

    # ------------------------------------------------------------------
    # Design vector access
    # ------------------------------------------------------------------
    def get_design_variables(self):
        dv = []
        for c in self.components:
            dv.extend([c.x0, c.y0, c.L, c.width, c.theta])
        return np.array(dv)

    def set_design_variables(self, x):
        idx = 0
        for c in self.components:
            c.x0, c.y0, c.L, c.width, c.theta = x[idx:idx + 5]
            idx += 5
        self._enforce_ortho()
        self._enforce_on_plane_pins()

    # ------------------------------------------------------------------
    # TDF -> Heaviside -> element density mapping and its sensitivities
    # ------------------------------------------------------------------
    def calculate_analytical_sensitivities(self, X_mesh, Y_mesh, current_iter):
        """
        Evaluate, at the element centroids:
          Phi_G          global TDF (hard max over all effective components)
          H              element density from the polynomial regularized
                         Heaviside H_eps(Phi_G)
          dH_dphiG       dH/dPhi_s at heaviside_eps (CONSISTENT with H above;
                         for the volume constraint gradient)
          dH_dphiG_sens  dH/dPhi_s at sens_eps (SMOOTHED, wider band; for the
                         compliance objective gradient only)
          weighted_dphi  (n_comp, 5, n_elem) tensor combining the hard max's
                         winner-takes-all indicator dPhi_s/dphi_i with
                         dphi_i/da (symmetry-clone chain factors included)
        """
        n_comp = len(self.components)
        n_elem = len(X_mesh)
        n_var = self.n_var

        if n_comp == 0:
            Phi_G = np.full(n_elem, -np.inf)
            H = np.zeros(n_elem)
            dH_dphiG = np.zeros(n_elem)
            weighted_dphi = np.zeros((0, n_var, n_elem))
            return Phi_G, H, dH_dphiG, dH_dphiG, weighted_dphi

        if self.symmetry == "OFF":
            all_entries = [(comp, np.eye(n_var), i)
                           for i, comp in enumerate(self.components)]
        else:
            all_entries = self._get_symmetry_clones()

        n_all = len(all_entries)

        phi_array = np.zeros((n_all, n_elem))
        dphi_dvars_all = np.zeros((n_all, n_var, n_elem))
        for k, (comp, _, _) in enumerate(all_entries):
            phi_array[k] = comp.calculate_phi(X_mesh, Y_mesh)
            dphi_dvars_all[k] = comp.calculate_derivatives(X_mesh, Y_mesh)

        # Global TDF: hard max; dPhi_s/dphi_i is the winner indicator.
        # At TIES (e.g. two crossing bars of the initial grids share exact
        # phi values over their whole overlap region) the weight is split
        # EQUALLY among all tied components — the symmetric subgradient
        # choice. A one-hot argmax would systematically hand the whole
        # overlap sensitivity to the lower-index component, starving one
        # bar family from iteration 1.
        Phi_G = np.max(phi_array, axis=0)
        TIE_TOL = 1e-9
        max_weights = (phi_array >= Phi_G[None, :] - TIE_TOL).astype(float)
        max_weights /= np.sum(max_weights, axis=0)[None, :]

        # Forward Heaviside (physical thickness field / volume): heaviside_eps.
        eps_f = self.heaviside_eps
        x = Phi_G
        H = np.where(x > eps_f, 1.0,
                     np.where(x < -eps_f, 0.0,
                              0.75 * (x / eps_f - x ** 3 / (3.0 * eps_f ** 3))
                              + 0.5))
        # CONSISTENT Heaviside slope at heaviside_eps: the exact dH/dPhi of the
        # forward field H above. Used for the VOLUME gradient, which MUST stay
        # consistent with the volume VALUE (V = mean H at heaviside_eps) or the
        # constraint pushes the design the wrong way.
        dH_dphiG = np.where(np.abs(x) <= eps_f,
                            0.75 * (1.0 / eps_f - x ** 2 / eps_f ** 3),
                            0.0)
        # SMOOTHED Heaviside slope at sens_eps (>= eps_f): a deliberately wider,
        # inconsistent band used only for the COMPLIANCE (objective) gradient,
        # to remove the dead gradient core / even out per-component sensitivities
        # (sensitivity filtering). NOT used for the volume constraint.
        eps_s = self.sens_eps
        dH_dphiG_sens = np.where(np.abs(x) <= eps_s,
                                 0.75 * (1.0 / eps_s - x ** 2 / eps_s ** 3),
                                 0.0)

        # weighted_dphi[src] = sum over that source's clones of
        # max_weight * d(phi_clone)/d(source vars). The chain rule
        # d(phi_clone)/d(source_w) = sum_v d(phi_clone)/d(clone_v) * J[v, w]
        # is exactly J^T @ dphi_dvars, where J = d(clone vars)/d(source vars).
        # J is the identity for the source itself, a diagonal sign-flip for the
        # plane mirrors, and a permutation-with-signs for FOUR_FOLD's diagonal
        # mirrors and rotations (which swap x0<->y0).
        weighted_dphi = np.zeros((n_comp, n_var, n_elem))
        for k, (_, J, src_idx) in enumerate(all_entries):
            weighted_dphi[src_idx] += max_weights[k] * (J.T @ dphi_dvars_all[k])

        # dH_dphiG  = consistent slope (heaviside_eps) -> VOLUME gradient
        # dH_dphiG_sens = smoothed slope (sens_eps)    -> COMPLIANCE gradient
        return Phi_G, H, dH_dphiG, dH_dphiG_sens, weighted_dphi

    def compute_H_field(self, X_mesh, Y_mesh):
        """
        Element density field H only (no derivative tensors) — the cheap
        path for volume fraction, plotting and FEA input generation. Matches
        the H returned by calculate_analytical_sensitivities exactly.
        """
        n_elem = len(X_mesh)
        if len(self.components) == 0:
            return np.zeros(n_elem)

        if self.symmetry == "OFF":
            comps = list(self.components)
        else:
            comps = [c for c, _, _ in self._get_symmetry_clones()]

        phi_array = np.stack([c.calculate_phi(X_mesh, Y_mesh) for c in comps])
        x = np.max(phi_array, axis=0)
        eps = self.heaviside_eps
        return np.where(x > eps, 1.0,
                        np.where(x < -eps, 0.0,
                                 0.75 * (x / eps - x ** 3 / (3.0 * eps ** 3))
                                 + 0.5))

    def get_element_properties(self, current_iter):
        """Element thickness field T_e = T_skin*(1-H) + (t_base + h)*H,
        i.e. the interpolation T_e = T_skin + H*T_stiffener when
        t_base == skin thickness (standard runs), generalized so FSO-driven
        runs can substitute a replacement base thickness under stiffeners."""
        H = self.compute_H_field(self.X, self.Y)
        T = (1.0 - H) * self.skin_thickness + H * (self.t_base + self.stiffener_height)
        return T, None

    def calculate_volume_fraction(self, current_iter):
        """V = mean of H over all mesh elements (the fraction of the panel
        surface covered by stiffeners, since the height is fixed)."""
        H = self.compute_H_field(self.X, self.Y)
        return float(np.mean(H))

    # ------------------------------------------------------------------
    # Sensitivities of compliance and volume
    # ------------------------------------------------------------------
    def calculate_analytical_gradients(self, ese_array, current_iter):
        """dC/da for every design variable: the chain rule of Eq. (3.9),
        with dC/dT_e taken from the element strain energies."""
        _, H, _, dH_dphiG_sens, weighted_dphi = \
            self.calculate_analytical_sensitivities(self.X, self.Y, current_iter)
        T_elems = (1.0 - H) * self.skin_thickness + H * (self.t_base + self.stiffener_height)
        dC_dH = -6.0 * ese_array * self.delta / np.maximum(T_elems, 1e-6)

        # Compliance (objective) gradient uses the SMOOTHED slope (sens_eps).
        chain_multiplier = dC_dH * dH_dphiG_sens

        n_var = self.n_var
        n_comp_vars = len(self.components) * n_var
        gradients = np.zeros(n_comp_vars)

        for i in range(n_comp_vars):
            c_id = i // n_var
            comp_idx = i % n_var
            gradients[i] = np.sum(chain_multiplier * weighted_dphi[c_id, comp_idx])

        return gradients

    def compute_vol_gradients_analytical(self, current_dv, current_iter):
        """dV/da for every design variable: same chain as the compliance,
        with dC/dT_e * dT_e/dH_e replaced by dV/dH_e = 1/N_elements.

        check_volume_gradient() finite-differences it."""
        _, _, dH_dphiG, _, weighted_dphi = \
            self.calculate_analytical_sensitivities(self.X, self.Y, current_iter)
        dV_dH = 1.0 / len(self.X)
        # Volume (constraint) gradient uses the CONSISTENT slope (heaviside_eps)
        # so it matches the volume VALUE; smoothing it would make MMA drive the
        # volume the wrong way.
        chain_multiplier = dV_dH * dH_dphiG

        n_var = self.n_var
        n_comp_vars = len(self.components) * n_var
        gradients = np.zeros(n_comp_vars)

        for i in range(n_comp_vars):
            c_id = i // n_var
            comp_idx = i % n_var
            gradients[i] = np.sum(
                chain_multiplier * weighted_dphi[c_id, comp_idx])

        return gradients

    def check_volume_gradient(self, current_iter=1, step=1e-3, n_probe=12,
                              seed=0):
        """Finite-difference audit of compute_vol_gradients_analytical.

        Probes a random subset of design variables and returns
        (max_rel_err, mean_rel_err, worst_variable_index).
        """
        x0 = self.get_design_variables().astype(float).copy()
        g = self.compute_vol_gradients_analytical(x0, current_iter)
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(x0), size=min(n_probe, len(x0)), replace=False)
        errs = []
        for i in idx:
            h = step * max(abs(x0[i]), 1.0)
            xp = x0.copy(); xp[i] += h
            self.set_design_variables(xp)
            vp = self.calculate_volume_fraction(current_iter)
            xm = x0.copy(); xm[i] -= h
            self.set_design_variables(xm)
            vm = self.calculate_volume_fraction(current_iter)
            fd = (vp - vm) / (2.0 * h)
            scale = max(abs(fd), abs(g[i]), 1e-12)
            errs.append(abs(fd - g[i]) / scale)
        self.set_design_variables(x0)
        errs = np.array(errs)
        return float(errs.max()), float(errs.mean()), int(idx[int(errs.argmax())])
