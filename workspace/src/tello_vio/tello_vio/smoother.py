"""Fixed-lag factor-graph smoother (sliding-window MAP estimation).

The ESKF in :mod:`tello_vio.eskf` is a *filter*: it linearises once, at the
current estimate, and never revisits that choice. That is cheap and it is what
runs in the real-time path. Its cost is that an early linearisation error --
made when the attitude or scale was still poorly known -- is baked in forever.

A smoother instead keeps the last ``N`` keyframes as free variables and
re-solves the whole window every time new data arrives, so an early mistake
gets corrected once later measurements disambiguate it. That is why VINS-Mono,
OKVIS and ORB-SLAM3 are all optimisation-based. The price is compute, bounded
here by fixing the window length -- hence *fixed-lag*.

What makes this a real fixed-lag smoother and not a naive sliding window
--------------------------------------------------------------------------
When a keyframe leaves the window you cannot simply delete it: the
measurements that touched it carry information about the keyframes that
*remain*, and dropping them throws that information away, which makes the
estimate both worse and over-confident about being worse. The correct operation
is **marginalisation**: eliminate the departing variables from the information
matrix by Schur complement, and keep the resulting dense Gaussian as a prior on
their neighbours::

    H_prior = H_kk - H_km H_mm^-1 H_mk
    b_prior = b_k  - H_km H_mm^-1 b_m

:meth:`FixedLagSmoother.marginalise_oldest` implements exactly this, and
``test_smoother.py`` verifies the property that matters: solving the window
*after* marginalisation gives the same answer as solving the full batch. That
equivalence is the whole point, and it is easy to get wrong in a way no smoke
test detects.

The marginalisation prior is held at the linearisation point where it was
created (a first-estimates-Jacobian convention). Re-linearising a prior around
a moving estimate silently injects information that was never measured and is
the standard route to an over-confident, inconsistent estimator.

Variable layout
---------------
Each keyframe contributes 15 error dimensions, ordered::

    [ dtheta(3) | dp(3) | dv(3) | db_a(3) | db_g(3) ]

Note this is *not* the preintegration module's ordering (which is
``[dtheta, dv, dp]``, following Forster); :meth:`ImuFactor.evaluate` remaps
between them explicitly rather than relying on the two happening to agree.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import lie
from .lie import Exp, I3, Log, right_jacobian_inv, skew
from .preintegration import PreintegratedImu

#: Per-keyframe error-state dimension and intra-block slices.
KF_DIM = 15
S_TH = slice(0, 3)
S_P = slice(3, 6)
S_V = slice(6, 9)
S_BA = slice(9, 12)
S_BG = slice(12, 15)


@dataclass
class KeyframeState:
    """Nominal state of one keyframe. Errors live in the tangent space."""

    stamp: float
    R: np.ndarray = field(default_factory=lambda: np.eye(3))
    p: np.ndarray = field(default_factory=lambda: np.zeros(3))
    v: np.ndarray = field(default_factory=lambda: np.zeros(3))
    ba: np.ndarray = field(default_factory=lambda: np.zeros(3))
    bg: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def copy(self) -> "KeyframeState":
        return KeyframeState(self.stamp, self.R.copy(), self.p.copy(),
                             self.v.copy(), self.ba.copy(), self.bg.copy())

    def boxplus(self, d: np.ndarray) -> None:
        """Retract a 15-vector increment onto the manifold, in place."""
        self.R = self.R @ Exp(d[S_TH])
        self.p = self.p + d[S_P]
        self.v = self.v + d[S_V]
        self.ba = self.ba + d[S_BA]
        self.bg = self.bg + d[S_BG]

    def boxminus(self, other: "KeyframeState") -> np.ndarray:
        d = np.zeros(KF_DIM)
        d[S_TH] = Log(other.R.T @ self.R)
        d[S_P] = self.p - other.p
        d[S_V] = self.v - other.v
        d[S_BA] = self.ba - other.ba
        d[S_BG] = self.bg - other.bg
        return d


# --------------------------------------------------------------------------- #
# Factors
# --------------------------------------------------------------------------- #

class Factor:
    """A residual block. ``keys`` names the keyframes it touches."""

    keys: tuple

    def evaluate(self, states: dict):
        """Return ``(r, [J per key], Lambda)``: residual, Jacobians, information."""
        raise NotImplementedError

    def robust_weight(self, r: np.ndarray, Lambda: np.ndarray) -> float:
        """Huber weight on the Mahalanobis norm. 1.0 disables robustification.

        Vision produces outliers that survive RANSAC; a squared loss lets a
        single one of them drag the entire window. Huber caps each residual's
        influence at linear growth beyond the threshold, which is enough to keep
        the solution sane without the non-convexity of a hard redescending loss.
        """
        return 1.0


@dataclass
class ImuFactor(Factor):
    """Preintegrated inertial constraint between consecutive keyframes."""

    i: int
    j: int
    pim: PreintegratedImu

    @property
    def keys(self):
        return (self.i, self.j)

    def evaluate(self, states):
        si, sj = states[self.i], states[self.j]
        r = self.pim.residual(si.R, si.v, si.p, sj.R, sj.v, sj.p, si.ba, si.bg)
        Ji_f, Jj_f, Jb = self.pim.jacobians(si.R, si.v, si.p, sj.R, sj.v, sj.p,
                                            si.ba, si.bg)
        # Preintegration orders state columns [dtheta | dv | dp]; this module
        # orders them [dtheta | dp | dv | dba | dbg]. Remap explicitly.
        Ji = np.zeros((9, KF_DIM))
        Jj = np.zeros((9, KF_DIM))
        Ji[:, S_TH] = Ji_f[:, 0:3]
        Ji[:, S_V] = Ji_f[:, 3:6]
        Ji[:, S_P] = Ji_f[:, 6:9]
        Ji[:, S_BA] = Jb[:, 0:3]
        Ji[:, S_BG] = Jb[:, 3:6]
        Jj[:, S_TH] = Jj_f[:, 0:3]
        Jj[:, S_V] = Jj_f[:, 3:6]
        Jj[:, S_P] = Jj_f[:, 6:9]
        return r, [Ji, Jj], self.pim.information()


@dataclass
class BiasRandomWalkFactor(Factor):
    """Ties consecutive keyframes' biases with the IMU's random-walk model."""

    i: int
    j: int
    cov6: np.ndarray

    @property
    def keys(self):
        return (self.i, self.j)

    def evaluate(self, states):
        si, sj = states[self.i], states[self.j]
        r = np.concatenate([sj.ba - si.ba, sj.bg - si.bg])
        Ji = np.zeros((6, KF_DIM))
        Jj = np.zeros((6, KF_DIM))
        Ji[0:3, S_BA] = -I3
        Ji[3:6, S_BG] = -I3
        Jj[0:3, S_BA] = I3
        Jj[3:6, S_BG] = I3
        Lam = np.linalg.inv(self.cov6 + np.eye(6) * 1e-12)
        return r, [Ji, Jj], Lam


@dataclass
class AttitudeFactor(Factor):
    """Unary gravity-referenced attitude from the drone's AHRS."""

    i: int
    q_meas: np.ndarray
    std_rp: float
    std_yaw: float

    @property
    def keys(self):
        return (self.i,)

    def evaluate(self, states):
        s = states[self.i]
        R_meas = lie.quat_to_rot(self.q_meas)
        r = Log(R_meas.T @ s.R)
        J = np.zeros((3, KF_DIM))
        J[:, S_TH] = right_jacobian_inv(r)
        Sigma_w = np.diag([self.std_rp ** 2, self.std_rp ** 2, self.std_yaw ** 2])
        Sigma_b = s.R.T @ Sigma_w @ s.R
        return r, [J], np.linalg.inv(Sigma_b)


@dataclass
class BodyVelocityFactor(Factor):
    """Unary metric body-frame velocity (the Tello's optical-flow estimate)."""

    i: int
    v_body: np.ndarray
    std: float

    @property
    def keys(self):
        return (self.i,)

    def evaluate(self, states):
        s = states[self.i]
        h = s.R.T @ s.v
        r = h - self.v_body
        J = np.zeros((3, KF_DIM))
        J[:, S_V] = s.R.T
        J[:, S_TH] = -skew(h)
        return r, [J], np.eye(3) / self.std ** 2


@dataclass
class HeightFactor(Factor):
    """Unary world-z constraint (barometer with a known bias, or ToF)."""

    i: int
    z_meas: float
    std: float

    @property
    def keys(self):
        return (self.i,)

    def evaluate(self, states):
        s = states[self.i]
        r = np.array([s.p[2] - self.z_meas])
        J = np.zeros((1, KF_DIM))
        J[0, S_P.start + 2] = 1.0
        return r, [J], np.array([[1.0 / self.std ** 2]])


@dataclass
class PositionFactor(Factor):
    """Unary world-position constraint.

    Used to anchor the window's gauge freedom (a visual-inertial graph with no
    absolute position measurement is invariant to a global translation, so the
    Hessian is rank deficient by 3 and Cholesky fails), and as the injection
    point for any absolute position source you may add later -- a fiducial
    marker, a UWB anchor, a loop-closed SLAM pose.
    """

    i: int
    p_meas: np.ndarray
    std: float

    @property
    def keys(self):
        return (self.i,)

    def evaluate(self, states):
        s = states[self.i]
        r = s.p - np.asarray(self.p_meas, dtype=np.float64).reshape(3)
        J = np.zeros((3, KF_DIM))
        J[:, S_P] = I3
        return r, [J], np.eye(3) / self.std ** 2


@dataclass
class VisualRelativeFactor(Factor):
    """Relative rotation + translation *bearing* between two keyframes.

    Same geometry as :meth:`tello_vio.eskf.ErrorStateKF.update_visual_relative`,
    expressed as a residual block. As there, the translation contributes only 2
    degrees of freedom (a direction on the unit sphere) because monocular
    vision cannot observe its magnitude.
    """

    i: int
    j: int
    R_c1c2: np.ndarray
    t_dir_c1: np.ndarray
    R_BC: np.ndarray
    p_BC: np.ndarray
    rot_std: float
    dir_std: float
    use_translation: bool = True
    min_baseline: float = 0.02
    huber_delta: float = 2.5

    @property
    def keys(self):
        return (self.i, self.j)

    def evaluate(self, states):
        si, sj = states[self.i], states[self.j]
        R_BC, p_BC = self.R_BC, self.p_BC

        M = R_BC @ self.R_c1c2 @ R_BC.T
        A = si.R.T @ sj.R
        r_rot = Log(M.T @ A)
        JrInv = right_jacobian_inv(r_rot)

        Ji = np.zeros((3, KF_DIM))
        Jj = np.zeros((3, KF_DIM))
        Ji[:, S_TH] = -JrInv @ sj.R.T @ si.R
        Jj[:, S_TH] = JrInv

        w = si.R.T @ (sj.p - si.p)
        d = R_BC.T @ (w - p_BC + A @ p_BC)
        baseline = float(np.linalg.norm(d))

        if not self.use_translation or baseline < self.min_baseline:
            return r_rot, [Ji, Jj], np.eye(3) / self.rot_std ** 2

        t = np.asarray(self.t_dir_c1, dtype=np.float64)
        nt = float(np.linalg.norm(t))
        if nt < 1e-9:
            return r_rot, [Ji, Jj], np.eye(3) / self.rot_std ** 2
        t = t / nt

        u = d / baseline
        B = _tangent_basis(t)
        r_dir = B.T @ (u - t)

        Dd_i = np.zeros((3, KF_DIM))
        Dd_j = np.zeros((3, KF_DIM))
        Dd_j[:, S_P] = R_BC.T @ si.R.T
        Dd_i[:, S_P] = -R_BC.T @ si.R.T
        Dd_i[:, S_TH] = R_BC.T @ (skew(w) + skew(A @ p_BC))
        Dd_j[:, S_TH] = R_BC.T @ (-A @ skew(p_BC))
        Du = (I3 - np.outer(u, u)) / baseline

        Ji_full = np.vstack([Ji, B.T @ Du @ Dd_i])
        Jj_full = np.vstack([Jj, B.T @ Du @ Dd_j])
        r = np.concatenate([r_rot, r_dir])
        Lam = np.diag(np.concatenate([
            np.full(3, 1.0 / self.rot_std ** 2),
            np.full(2, 1.0 / self.dir_std ** 2),
        ]))
        return r, [Ji_full, Jj_full], Lam

    def robust_weight(self, r, Lambda):
        e = float(np.sqrt(max(0.0, r @ Lambda @ r)))
        if e <= self.huber_delta:
            return 1.0
        return self.huber_delta / e


@dataclass
class MarginalPriorFactor(Factor):
    """Dense Gaussian prior left behind by marginalising older keyframes.

    Held at a *fixed* linearisation point (``anchor``). Evaluating it at the
    current estimate uses ``delta = x [-] anchor``, contributing ``H`` to the
    Hessian and ``b - H delta`` to the gradient -- the standard
    first-estimates-Jacobian treatment.
    """

    key_list: tuple
    H: np.ndarray
    b: np.ndarray
    anchor: dict

    @property
    def keys(self):
        return self.key_list

    def evaluate(self, states):
        raise NotImplementedError("prior contributes directly; see _accumulate")


def _tangent_basis(u: np.ndarray) -> np.ndarray:
    u = np.asarray(u, dtype=np.float64).reshape(3)
    a = np.zeros(3)
    a[int(np.argmin(np.abs(u)))] = 1.0
    b1 = np.cross(u, a)
    b1 /= np.linalg.norm(b1)
    return np.column_stack([b1, np.cross(u, b1)])


# --------------------------------------------------------------------------- #
# Solver
# --------------------------------------------------------------------------- #

class FixedLagSmoother:
    """Sliding-window MAP estimator over keyframe states.

    Compute scales as ``O((15 N)^3)`` for the dense solve. At the default
    ``N = 10`` that is a 150x150 Cholesky -- a few hundred microseconds in
    NumPy, comfortably inside a keyframe interval. Growing the window is the
    knob to trade accuracy for CPU; past roughly ``N = 25`` a sparse solver
    starts to be worth the complexity, and past that you want incremental
    factorisation (iSAM2) rather than re-solving from scratch.
    """

    def __init__(self, window: int = 10, robust: bool = True):
        self.window = int(window)
        self.robust = bool(robust)
        self.states: dict = {}
        self.factors: list = []
        self.prior: MarginalPriorFactor | None = None
        self._order: list = []
        self.last_report: dict = {}

    # ------------------------------------------------------------------ #

    def add_keyframe(self, key: int, state: KeyframeState) -> None:
        self.states[key] = state
        self._order.append(key)

    def add_factor(self, factor: Factor) -> None:
        self.factors.append(factor)

    @property
    def keys(self) -> list:
        return list(self._order)

    # ------------------------------------------------------------------ #

    def _index(self):
        return {k: n * KF_DIM for n, k in enumerate(self._order)}

    def _accumulate(self, lam: float = 0.0):
        """Build the damped normal equations ``(H + lam I) d = b`` and the cost."""
        idx = self._index()
        n = len(self._order) * KF_DIM
        H = np.zeros((n, n))
        b = np.zeros(n)
        cost = 0.0

        for f in self.factors:
            if any(k not in idx for k in f.keys):
                continue
            r, Js, Lam = f.evaluate(self.states)
            w = f.robust_weight(r, Lam) if self.robust else 1.0
            Lw = Lam * w
            cost += 0.5 * w * float(r @ Lam @ r)
            for a, ka in enumerate(f.keys):
                ia = idx[ka]
                JaT_L = Js[a].T @ Lw
                b[ia:ia + KF_DIM] -= JaT_L @ r
                for c, kc in enumerate(f.keys):
                    ic = idx[kc]
                    H[ia:ia + KF_DIM, ic:ic + KF_DIM] += JaT_L @ Js[c]

        if self.prior is not None:
            live = [k for k in self.prior.key_list if k in idx]
            if len(live) == len(self.prior.key_list):
                delta = np.concatenate([
                    self.states[k].boxminus(self.prior.anchor[k])
                    for k in self.prior.key_list
                ])
                rows = np.concatenate([
                    np.arange(idx[k], idx[k] + KF_DIM) for k in self.prior.key_list
                ])
                Hp, bp = self.prior.H, self.prior.b
                H[np.ix_(rows, rows)] += Hp
                b[rows] += bp - Hp @ delta
                cost += 0.5 * float(delta @ Hp @ delta) - float(bp @ delta)

        if lam > 0.0:
            H = H + lam * np.diag(np.maximum(np.diag(H), 1e-9))
        return H, b, cost

    def solve(self, iterations: int = 8, tol: float = 1e-8) -> dict:
        """Levenberg-Marquardt on the manifold. Returns a small diagnostic dict."""
        if not self._order:
            return {"iterations": 0, "cost": 0.0}

        lam = 1e-6
        _, _, cost = self._accumulate()
        report = {"cost0": cost, "iterations": 0, "cost": cost, "status": "ok"}

        for it in range(iterations):
            H, b, cost = self._accumulate(lam)
            try:
                # Cholesky is ~2x faster than a general solve and, more
                # usefully, *fails loudly* if H is not positive definite --
                # which is the signal that the window is under-constrained
                # (e.g. no measurement yet anchors yaw or scale).
                L = np.linalg.cholesky(H)
                d = np.linalg.solve(L.T, np.linalg.solve(L, b))
            except np.linalg.LinAlgError:
                lam *= 10.0
                if lam > 1e8:
                    report["status"] = "indefinite"
                    break
                continue

            backup = {k: s.copy() for k, s in self.states.items()}
            for n, k in enumerate(self._order):
                self.states[k].boxplus(d[n * KF_DIM:(n + 1) * KF_DIM])

            _, _, new_cost = self._accumulate()
            if new_cost < cost:
                lam = max(1e-9, lam * 0.3)
                report["iterations"] = it + 1
                report["cost"] = new_cost
                if abs(cost - new_cost) < tol * max(1.0, abs(cost)):
                    break
                cost = new_cost
            else:
                self.states = backup
                lam *= 10.0
                if lam > 1e8:
                    report["status"] = "stalled"
                    break

        report["dim"] = len(self._order) * KF_DIM
        self.last_report = report
        return report

    # ------------------------------------------------------------------ #

    def marginalise_oldest(self) -> None:
        """Schur-complement the oldest keyframe out, keeping its information.

        Only factors that *touch* the departing keyframe are eliminated; the
        rest stay as-is. Folding untouched factors into the prior would
        double-count them on the next solve.
        """
        if len(self._order) <= 1:
            return
        drop = self._order[0]

        touching = [f for f in self.factors if drop in f.keys]
        keep_factors = [f for f in self.factors if drop not in f.keys]

        # Variables entangled with the departing one, in a stable order.
        connected = []
        for f in touching:
            for k in f.keys:
                if k != drop and k not in connected:
                    connected.append(k)
        if self.prior is not None and drop in self.prior.key_list:
            for k in self.prior.key_list:
                if k != drop and k not in connected:
                    connected.append(k)
            touching_prior = True
        else:
            touching_prior = False
            connected = [k for k in connected]

        order = [drop] + connected
        idx = {k: n * KF_DIM for n, k in enumerate(order)}
        n_tot = len(order) * KF_DIM
        H = np.zeros((n_tot, n_tot))
        b = np.zeros(n_tot)

        for f in touching:
            r, Js, Lam = f.evaluate(self.states)
            w = f.robust_weight(r, Lam) if self.robust else 1.0
            Lw = Lam * w
            for a, ka in enumerate(f.keys):
                ia = idx[ka]
                JaT_L = Js[a].T @ Lw
                b[ia:ia + KF_DIM] -= JaT_L @ r
                for c, kc in enumerate(f.keys):
                    ic = idx[kc]
                    H[ia:ia + KF_DIM, ic:ic + KF_DIM] += JaT_L @ Js[c]

        if touching_prior:
            rows = np.concatenate([
                np.arange(idx[k], idx[k] + KF_DIM) for k in self.prior.key_list
            ])
            delta = np.concatenate([
                self.states[k].boxminus(self.prior.anchor[k])
                for k in self.prior.key_list
            ])
            H[np.ix_(rows, rows)] += self.prior.H
            b[rows] += self.prior.b - self.prior.H @ delta
        elif self.prior is not None:
            # Prior does not touch the departing keyframe: carry it forward.
            keep_factors = keep_factors

        m = KF_DIM                                  # dimensions being removed
        Hmm, Hmk = H[:m, :m], H[:m, m:]
        Hkk = H[m:, m:]
        bm, bk = b[:m], b[m:]

        # Hmm can be singular when the departing keyframe is only partly
        # observed; a pseudo-inverse marginalises the observed subspace and
        # leaves the unobserved one alone, which is the conservative choice.
        Hmm_inv = np.linalg.pinv(0.5 * (Hmm + Hmm.T), rcond=1e-10)
        H_prior = Hkk - Hmk.T @ Hmm_inv @ Hmk
        b_prior = bk - Hmk.T @ Hmm_inv @ bm
        H_prior = 0.5 * (H_prior + H_prior.T)

        # Numerical round-off can leave tiny negative eigenvalues; clamp them so
        # later Cholesky factorisations stay valid.
        w_eig, V = np.linalg.eigh(H_prior)
        w_eig = np.maximum(w_eig, 0.0)
        H_prior = V @ np.diag(w_eig) @ V.T

        anchor = {k: self.states[k].copy() for k in connected}
        new_prior = MarginalPriorFactor(tuple(connected), H_prior, b_prior, anchor)

        if touching_prior or self.prior is None:
            self.prior = new_prior if connected else None
        else:
            # Two independent priors on disjoint variable sets: keep the new
            # one only if the old one has expired with its variables.
            self.prior = new_prior if connected else self.prior

        self.factors = keep_factors
        self.states.pop(drop, None)
        self._order.pop(0)

    def enforce_window(self) -> None:
        while len(self._order) > self.window:
            self.marginalise_oldest()

    # ------------------------------------------------------------------ #

    def covariance(self, key: int) -> np.ndarray:
        """Marginal 15x15 covariance of one keyframe, or ``None`` if singular."""
        H, _, _ = self._accumulate()
        try:
            C = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            return None
        i = self._index()[key]
        return C[i:i + KF_DIM, i:i + KF_DIM]

    def latest(self) -> KeyframeState:
        return self.states[self._order[-1]]
