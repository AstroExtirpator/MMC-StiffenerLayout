import numpy as np


class MMA:
    """
    Method of Moving Asymptotes (Svanberg 1987) for a single inequality
    constraint, operating on a normalized design vector x in [0, 1]^n.

    This is the standard update scheme used in the MMC literature
    (Guo et al. 2014; Zhang et al. 2016). The convex subproblem is solved
    exactly in the dual: with one constraint, the primal minimizer has a
    closed form per variable and the multiplier is found by bisection.

    The key practical property vs. a fixed-step gradient scheme: per-variable
    asymptotes tighten (x asydecr) for variables that oscillate between
    iterations and relax (x asyincr) for variables moving monotonically,
    so period-2 oscillation dies out and the design settles.
    """

    def __init__(self, n, move=0.05, asyinit=0.2, asyincr=1.2, asydecr=0.7,
                 albefa=0.1, asymin=0.001, c=1000.0):
        self.n = n
        self.move = move          # per-iteration move limit (fraction of range)
        self.asyinit = asyinit    # initial asymptote distance
        self.asyincr = asyincr    # asymptote relaxation factor
        self.asydecr = asydecr    # asymptote tightening factor
        self.albefa = albefa      # bound offset from asymptotes
        self.asymin = asymin      # minimum asymptote distance (fraction)
        # Svanberg's constraint penalty c: the FINITE upper bound on the
        # constraint multiplier lambda. When the constraint cannot be met
        # within the move limits (far infeasible), lambda saturates at c and
        # the artificial variable absorbs the residual violation, so the step
        # still balances the objective instead of bulldozing the constraint.
        # An UNbounded lambda (the old lam_hi -> 1e12) is the c -> infinity
        # limit: it enforces the volume constraint with effectively infinite
        # penalty and drives compliance up destructively during the initial
        # volume reduction. Smaller c => gentler constraint enforcement.
        self.c = c
        self.xmin = np.zeros(n)
        self.xmax = np.ones(n)
        self.low = None
        self.upp = None
        self.xold1 = None
        self.xold2 = None
        self.iter = 0

    def update(self, x, df0dx, g, dgdx):
        """
        One MMA design update.

        Args:
            x:     current design, shape (n,), in [0, 1]
            df0dx: objective gradient at x, shape (n,)
            g:     constraint value at x (feasible when g <= 0)
            dgdx:  constraint gradient at x, shape (n,)

        Returns:
            x_new, shape (n,), within move limits and [0, 1]
        """
        self.iter += 1
        x = np.asarray(x, dtype=float)
        df0dx = np.asarray(df0dx, dtype=float)
        dgdx = np.asarray(dgdx, dtype=float)
        rng = self.xmax - self.xmin  # = 1

        # --- Asymptote update ---
        if self.iter <= 2 or self.xold2 is None:
            low = x - self.asyinit * rng
            upp = x + self.asyinit * rng
        else:
            osc = (x - self.xold1) * (self.xold1 - self.xold2)
            fac = np.ones(self.n)
            fac[osc > 0.0] = self.asyincr
            fac[osc < 0.0] = self.asydecr
            low = x - fac * (self.xold1 - self.low)
            upp = x + fac * (self.upp - self.xold1)
            low = np.clip(low, x - 10.0 * rng, x - self.asymin * rng)
            upp = np.clip(upp, x + self.asymin * rng, x + 10.0 * rng)

        # --- Feasible bounds for the subproblem ---
        alfa = np.maximum.reduce([self.xmin,
                                  low + self.albefa * (x - low),
                                  x - self.move * rng])
        beta = np.minimum.reduce([self.xmax,
                                  upp - self.albefa * (upp - x),
                                  x + self.move * rng])
        beta = np.maximum(beta, alfa)

        # --- p/q coefficients (Svanberg's convex separable approximation) ---
        ux = upp - x
        xl = x - low
        df0p = np.maximum(df0dx, 0.0)
        df0m = np.maximum(-df0dx, 0.0)
        raa0 = 1e-5
        p0 = ux ** 2 * (1.001 * df0p + 0.001 * df0m + raa0)
        q0 = xl ** 2 * (0.001 * df0p + 1.001 * df0m + raa0)

        dgp = np.maximum(dgdx, 0.0)
        dgm = np.maximum(-dgdx, 0.0)
        p1 = ux ** 2 * dgp
        q1 = xl ** 2 * dgm
        # Subproblem constraint: sum(p1/(upp-y) + q1/(y-low)) <= b
        b = np.sum(p1 / ux + q1 / xl) - g

        def x_of(lam):
            P = p0 + lam * p1
            Q = q0 + lam * q1
            sp = np.sqrt(P)
            sq = np.sqrt(Q)
            y = (low * sp + upp * sq) / (sp + sq)
            return np.clip(y, alfa, beta)

        def g_sub(lam):
            y = x_of(lam)
            return np.sum(p1 / (upp - y) + q1 / (y - low)) - b

        # --- Dual bisection on the constraint multiplier, CAPPED at c ---
        if g_sub(0.0) <= 0.0:
            lam = 0.0                       # constraint inactive
        elif g_sub(self.c) > 0.0:
            lam = self.c                    # infeasible even at max penalty:
                                            # saturate lambda (Svanberg's y var
                                            # absorbs the residual violation)
        else:
            lam_lo, lam_hi = 0.0, self.c
            for _ in range(100):
                lam = 0.5 * (lam_lo + lam_hi)
                if g_sub(lam) > 0.0:
                    lam_lo = lam
                else:
                    lam_hi = lam
            lam = lam_hi
        x_new = x_of(lam)

        self.xold2 = self.xold1
        self.xold1 = x.copy()
        self.low = low
        self.upp = upp
        return x_new


def lagrangian_step(x_norm, dC_dx, dV_dx, vol_target, move_limit, vol_at_fn):
    """
    One design update of the projected-gradient scheme on the Lagrangian
        L = C + lambda * (V - V_target)
    with explicit move limits and bisection on the multiplier lambda: a step
    is taken in the direction of steepest descent of L with respect to the
    normalized design variables, clipped to the move limit, and lambda is
    found by bisection based on how much volume penalization is needed to
    balance compliance minimization.

    Args:
        x_norm:     current normalized design, shape (n,), in [0, 1]
        dC_dx:      normalized compliance gradient at x_norm
        dV_dx:      normalized volume gradient at x_norm
        vol_target: volume-fraction constraint value
        move_limit: max per-step change of any normalized design variable
        vol_at_fn:  callable x_norm -> volume fraction (no FEA required)

    Returns:
        x_new, shape (n,), within move limits and [0, 1]
    """
    lam_low = 0.0
    lam_high = 1e8

    dC_scale = max(np.max(np.abs(dC_dx)), 1e-12)
    dV_scale = max(np.max(np.abs(dV_dx)), 1e-12)

    dC_n = dC_dx / dC_scale
    dV_n = dV_dx / dV_scale

    x_best = x_norm.copy()
    x_lowest_vol = x_norm.copy()
    lowest_vol = np.inf
    best_gap = np.inf

    for bisect_iter in range(60):
        lam = (lam_low + lam_high) / 2.0

        dL = dC_n + (lam * dV_scale / dC_scale) * dV_n

        raw_step = -dL

        step = np.clip(raw_step, -move_limit, move_limit)

        x_trial = np.clip(x_norm + step, 0.0, 1.0)

        vol_trial = vol_at_fn(x_trial)

        if vol_trial < lowest_vol:
            lowest_vol = vol_trial
            x_lowest_vol = x_trial.copy()

        if vol_trial > vol_target:
            lam_low = lam
        else:
            lam_high = lam
            gap = abs(vol_trial - vol_target)
            if gap < best_gap:
                best_gap = gap
                x_best = x_trial.copy()

        if abs(vol_trial - vol_target) < 0.001:
            x_best = x_trial.copy()
            break

    return x_best if best_gap < np.inf else x_lowest_vol
