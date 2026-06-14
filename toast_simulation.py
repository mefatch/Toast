from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


PAPER_ANGULAR_VELOCITIES = [
    # overhang_cm, measured omega_rad_s, standard_deviation_rad_s
    (0.10, 6.93, 0.26),
    (0.30, 8.80, 0.24),
    (0.55, 10.03, 0.34),
    (0.85, 10.90, 0.36),
    (1.10, 11.57, 0.37),
    (1.30, 12.20, 0.67),
    (1.60, 12.47, 0.62),
    (1.85, 12.57, 0.37),
    (2.10, 12.17, 0.31),
    (2.30, 12.43, 0.55),
]

PAPER_LANDINGS = [
    # overhang_cm, observed final side
    (0.10, "Down"),
    (0.30, "Down"),
    (0.63, "Down"),
    (0.79, "Down"),
    (0.81, "Up"),
    (0.85, "Up"),
    (1.10, "Up"),
    (1.30, "Up"),
    (1.60, "Up"),
    (1.85, "Up"),
    (2.10, "Up"),
    (2.35, "Up"),
    (2.80, "Up"),
    (3.03, "Down"),
    (3.28, "Down"),
    (3.40, "Down"),
    (3.70, "Down"),
]


@dataclass(frozen=True)
class ToastParams:
    """Physical parameters for the 2D rigid-board model."""

    half_length_m: float = 0.051
    thickness_m: float = 0.013
    mu_static: float = 0.32
    mu_kinetic: float = 0.24
    table_height_m: float = 0.76
    g_m_s2: float = 9.8
    mass_kg: float | None = None

    @property
    def inertia_per_mass_m2(self) -> float:
        # Moment of inertia about the center of mass, divided by mass.
        # The board length is 2a and thickness is b.
        a = self.half_length_m
        b = self.thickness_m
        return a * a / 3.0 + b * b / 12.0


@dataclass(frozen=True)
class SolverParams:
    rtol: float = 1e-9
    atol: float = 1e-11
    max_step_s: float = 5e-4
    sample_dt_s: float = 1e-3


@dataclass(frozen=True)
class SimulationResult:
    overhang_m: float
    slip_time_s: float | None
    slip_angle_rad: float | None
    slip_omega_rad_s: float | None
    liftoff_time_s: float
    liftoff_alpha_rad: float
    liftoff_alpha_dot_rad_s: float
    liftoff_y_m: float
    liftoff_y_dot_m_s: float
    landing_time_after_liftoff_s: float
    total_landing_time_s: float
    landing_alpha_rad: float
    landing_side: str
    r_liftoff_m: float
    rdot_liftoff_m_s: float
    theta_liftoff_rad: float
    theta_dot_liftoff_rad_s: float
    friction_sign: float
    liftoff_phase: str


def rad_to_deg(angle_rad: float) -> float:
    return angle_rad * 180.0 / math.pi


def deg_to_rad(angle_deg: float) -> float:
    return angle_deg * math.pi / 180.0


def initial_geometry(overhang_m: float, p: ToastParams) -> tuple[float, float]:
    """Return r0 and beta0 from Fig. 1 of the paper."""
    if overhang_m <= 0.0:
        raise ValueError("overhang must be positive")
    if overhang_m >= p.half_length_m:
        raise ValueError(
            "overhang must be smaller than the half length "
            f"({100.0 * p.half_length_m:.3f} cm)"
        )
    r0 = math.hypot(overhang_m, p.thickness_m / 2.0)
    beta0 = math.atan2(p.thickness_m / 2.0, overhang_m)
    return r0, beta0


def beta_from_r(r_m: float, p: ToastParams) -> float:
    z = (p.thickness_m / 2.0) / r_m
    if z >= 1.0:
        raise ValueError("invalid slipping geometry: r must stay above b/2")
    return math.asin(z)


def beta_dot_from_r(r_m: float, rdot_m_s: float, p: ToastParams) -> float:
    beta = beta_from_r(r_m, p)
    return -rdot_m_s * p.thickness_m / (2.0 * r_m * r_m * math.cos(beta))


def static_theta_ddot(
    theta_rad: float, overhang_m: float, p: ToastParams
) -> float:
    """Equation (13): angular acceleration before slipping."""
    r0, beta0 = initial_geometry(overhang_m, p)
    denominator = p.inertia_per_mass_m2 + r0 * r0
    return r0 * p.g_m_s2 * math.cos(theta_rad - beta0) / denominator


def static_forces_per_mass(
    theta_rad: float, theta_dot_rad_s: float, overhang_m: float, p: ToastParams
) -> tuple[float, float, float]:
    """Return normal/mass, friction/mass, theta_ddot during no-slip contact.

    The forces are obtained from Eqs. (10) and (11) after Eq. (13) gives
    theta_ddot.  Positive friction is the sign convention of Fig. 1.
    """
    r0, beta0 = initial_geometry(overhang_m, p)
    s = math.sin(beta0)
    c = math.cos(beta0)
    psi = theta_rad - beta0
    theta_ddot = static_theta_ddot(theta_rad, overhang_m, p)

    matrix = np.array([[s, -c], [-c, -s]], dtype=float)
    rhs = np.array(
        [
            -r0 * theta_dot_rad_s * theta_dot_rad_s
            - p.g_m_s2 * math.sin(psi),
            r0 * theta_ddot - p.g_m_s2 * math.cos(psi),
        ],
        dtype=float,
    )
    normal, friction = np.linalg.solve(matrix, rhs)
    return float(normal), float(friction), float(theta_ddot)


def slip_accelerations(
    r_m: float,
    rdot_m_s: float,
    theta_rad: float,
    theta_dot_rad_s: float,
    overhang_m: float,
    p: ToastParams,
    friction_sign: float,
) -> tuple[float, float, float, float]:
    """Solve Eqs. (4), (5), (7), and (9) in the slipping phase.

    Returns r_ddot, theta_ddot, normal/mass, beta_ddot.
    Kinetic friction is friction_sign * mu_kinetic * normal.
    """
    _, beta0 = initial_geometry(overhang_m, p)
    beta = beta_from_r(r_m, p)
    s = math.sin(beta)
    c = math.cos(beta)
    tan_beta = s / c
    psi = theta_rad - beta0
    mu = friction_sign * p.mu_kinetic
    inertia = p.inertia_per_mass_m2

    beta_ddot_without_rddot = (
        rdot_m_s
        * rdot_m_s
        * s
        * (1.0 + c * c)
        / (r_m * r_m * c * c * c)
    )

    matrix = np.array(
        [
            [1.0, 0.0, -(s - mu * c)],
            [0.0, r_m, c + mu * s],
            [-inertia * tan_beta / r_m, inertia, -(r_m * c + mu * p.thickness_m / 2.0)],
        ],
        dtype=float,
    )
    rhs = np.array(
        [
            r_m * theta_dot_rad_s * theta_dot_rad_s
            + p.g_m_s2 * math.sin(psi),
            p.g_m_s2 * math.cos(psi) - 2.0 * rdot_m_s * theta_dot_rad_s,
            -inertia * beta_ddot_without_rddot,
        ],
        dtype=float,
    )
    r_ddot, theta_ddot, normal = np.linalg.solve(matrix, rhs)
    beta_ddot = beta_ddot_without_rddot - r_ddot * tan_beta / r_m
    return float(r_ddot), float(theta_ddot), float(normal), float(beta_ddot)


def y_cm(
    r_m: float, theta_rad: float, overhang_m: float, p: ToastParams
) -> float:
    """Center-of-mass height relative to the table edge, from Sec. V."""
    _, beta0 = initial_geometry(overhang_m, p)
    return -r_m * math.sin(theta_rad - beta0)


def y_cm_dot(
    r_m: float,
    rdot_m_s: float,
    theta_rad: float,
    theta_dot_rad_s: float,
    overhang_m: float,
    p: ToastParams,
) -> float:
    _, beta0 = initial_geometry(overhang_m, p)
    psi = theta_rad - beta0
    return -rdot_m_s * math.sin(psi) - r_m * math.cos(psi) * theta_dot_rad_s


def classify_landing(total_alpha_rad: float) -> str:
    alpha_mod = total_alpha_rad % (2.0 * math.pi)
    return "Down" if math.pi / 2.0 < alpha_mod < 3.0 * math.pi / 2.0 else "Up"


def landing_after_liftoff(
    alpha0_rad: float,
    alpha_dot_rad_s: float,
    y0_m: float,
    ydot0_m_s: float,
    p: ToastParams,
) -> tuple[float, float, str]:
    """Return free-fall time, total alpha at floor contact, and side."""

    def nearest_edge_minus_floor(t_s: float) -> float:
        y_edge = (
            y0_m
            + ydot0_m_s * t_s
            - 0.5 * p.g_m_s2 * t_s * t_s
            - p.half_length_m * abs(math.sin(alpha0_rad + alpha_dot_rad_s * t_s))
        )
        return y_edge + p.table_height_m

    if nearest_edge_minus_floor(0.0) <= 0.0:
        t_floor = 0.0
    else:
        hi = 1.0
        while nearest_edge_minus_floor(hi) > 0.0:
            hi *= 2.0
            if hi > 30.0:
                raise RuntimeError("could not bracket floor contact time")
        t_floor = brentq(nearest_edge_minus_floor, 0.0, hi, xtol=1e-12)

    total_alpha = alpha0_rad + alpha_dot_rad_s * t_floor
    return float(t_floor), float(total_alpha), classify_landing(total_alpha)


def simulate_overhang(
    overhang_m: float,
    p: ToastParams,
    solver: SolverParams,
    keep_solutions: bool = False,
):
    """Simulate one overhang.

    Returns (SimulationResult, solution_bundle).  solution_bundle is None
    unless keep_solutions=True.
    """
    r0, beta0 = initial_geometry(overhang_m, p)
    friction_sign = 1.0

    def static_ode(_t, state):
        theta, theta_dot = state
        return [theta_dot, static_theta_ddot(theta, overhang_m, p)]

    def slip_event(_t, state):
        theta, theta_dot = state
        normal, friction, _ = static_forces_per_mass(theta, theta_dot, overhang_m, p)
        if normal <= 0.0:
            return -1.0
        return abs(friction) / normal - p.mu_static

    slip_event.terminal = True
    slip_event.direction = 1

    def static_liftoff_event(_t, state):
        theta, theta_dot = state
        normal, _friction, _ = static_forces_per_mass(theta, theta_dot, overhang_m, p)
        return normal

    static_liftoff_event.terminal = True
    static_liftoff_event.direction = -1

    static_sol = solve_ivp(
        static_ode,
        (0.0, 2.0),
        [0.0, 0.0],
        events=[slip_event, static_liftoff_event],
        dense_output=True,
        rtol=solver.rtol,
        atol=solver.atol,
        max_step=solver.max_step_s,
    )

    slipped = len(static_sol.t_events[0]) > 0
    lifted_without_slip = len(static_sol.t_events[1]) > 0
    if not slipped and not lifted_without_slip:
        raise RuntimeError("no slip or liftoff event found during static phase")

    if lifted_without_slip and (
        not slipped or static_sol.t_events[1][0] <= static_sol.t_events[0][0]
    ):
        liftoff_time = float(static_sol.t_events[1][0])
        theta, theta_dot = map(float, static_sol.y_events[1][0])
        alpha = theta
        alpha_dot = theta_dot
        y0 = y_cm(r0, theta, overhang_m, p)
        ydot0 = y_cm_dot(r0, 0.0, theta, theta_dot, overhang_m, p)
        t_floor, landing_alpha, landing_side = landing_after_liftoff(
            alpha, alpha_dot, y0, ydot0, p
        )
        result = SimulationResult(
            overhang_m=overhang_m,
            slip_time_s=None,
            slip_angle_rad=None,
            slip_omega_rad_s=None,
            liftoff_time_s=liftoff_time,
            liftoff_alpha_rad=alpha,
            liftoff_alpha_dot_rad_s=alpha_dot,
            liftoff_y_m=y0,
            liftoff_y_dot_m_s=ydot0,
            landing_time_after_liftoff_s=t_floor,
            total_landing_time_s=liftoff_time + t_floor,
            landing_alpha_rad=landing_alpha,
            landing_side=landing_side,
            r_liftoff_m=r0,
            rdot_liftoff_m_s=0.0,
            theta_liftoff_rad=theta,
            theta_dot_liftoff_rad_s=theta_dot,
            friction_sign=friction_sign,
            liftoff_phase="no_slip",
        )
        bundle = {"static": static_sol, "slip": None} if keep_solutions else None
        return result, bundle

    slip_time = float(static_sol.t_events[0][0])
    slip_theta, slip_theta_dot = map(float, static_sol.y_events[0][0])
    slip_normal, slip_friction, _ = static_forces_per_mass(
        slip_theta, slip_theta_dot, overhang_m, p
    )
    if slip_normal <= 0.0:
        raise RuntimeError("static phase reached nonpositive normal before slip")
    friction_sign = 1.0 if slip_friction >= 0.0 else -1.0

    def slip_ode(_t, state):
        r, rdot, theta, theta_dot = state
        r_ddot, theta_ddot, _normal, _beta_ddot = slip_accelerations(
            r, rdot, theta, theta_dot, overhang_m, p, friction_sign
        )
        return [rdot, r_ddot, theta_dot, theta_ddot]

    def slip_liftoff_event(_t, state):
        r, rdot, theta, theta_dot = state
        _r_ddot, _theta_ddot, normal, _beta_ddot = slip_accelerations(
            r, rdot, theta, theta_dot, overhang_m, p, friction_sign
        )
        return normal

    slip_liftoff_event.terminal = True
    slip_liftoff_event.direction = -1

    slip_sol = solve_ivp(
        slip_ode,
        (slip_time, slip_time + 2.0),
        [r0, 0.0, slip_theta, slip_theta_dot],
        events=slip_liftoff_event,
        dense_output=True,
        rtol=solver.rtol,
        atol=solver.atol,
        max_step=solver.max_step_s,
    )

    if len(slip_sol.t_events[0]) == 0:
        raise RuntimeError("normal force did not reach zero during slipping phase")

    liftoff_time = float(slip_sol.t_events[0][0])
    r_lift, rdot_lift, theta_lift, theta_dot_lift = map(
        float, slip_sol.y_events[0][0]
    )
    beta_lift = beta_from_r(r_lift, p)
    beta_dot_lift = beta_dot_from_r(r_lift, rdot_lift, p)
    alpha_lift = theta_lift - beta0 + beta_lift
    alpha_dot_lift = theta_dot_lift + beta_dot_lift
    y0 = y_cm(r_lift, theta_lift, overhang_m, p)
    ydot0 = y_cm_dot(
        r_lift, rdot_lift, theta_lift, theta_dot_lift, overhang_m, p
    )
    t_floor, landing_alpha, landing_side = landing_after_liftoff(
        alpha_lift, alpha_dot_lift, y0, ydot0, p
    )

    result = SimulationResult(
        overhang_m=overhang_m,
        slip_time_s=slip_time,
        slip_angle_rad=slip_theta,
        slip_omega_rad_s=slip_theta_dot,
        liftoff_time_s=liftoff_time,
        liftoff_alpha_rad=alpha_lift,
        liftoff_alpha_dot_rad_s=alpha_dot_lift,
        liftoff_y_m=y0,
        liftoff_y_dot_m_s=ydot0,
        landing_time_after_liftoff_s=t_floor,
        total_landing_time_s=liftoff_time + t_floor,
        landing_alpha_rad=landing_alpha,
        landing_side=landing_side,
        r_liftoff_m=r_lift,
        rdot_liftoff_m_s=rdot_lift,
        theta_liftoff_rad=theta_lift,
        theta_dot_liftoff_rad_s=theta_dot_lift,
        friction_sign=friction_sign,
        liftoff_phase="slip",
    )
    bundle = {"static": static_sol, "slip": slip_sol} if keep_solutions else None
    return result, bundle


def force_newtons(force_per_mass: float, p: ToastParams) -> float | None:
    if p.mass_kg is None:
        return None
    return force_per_mass * p.mass_kg


def build_time_series_rows(
    result: SimulationResult,
    solutions,
    p: ToastParams,
    solver: SolverParams,
) -> list[dict[str, float | str | None]]:
    rows: list[dict[str, float | str | None]] = []
    overhang_m = result.overhang_m
    r0, beta0 = initial_geometry(overhang_m, p)
    static_sol = solutions["static"]
    slip_sol = solutions["slip"]

    def append_row(
        phase: str,
        t_s: float,
        r_m: float | None,
        rdot_m_s: float | None,
        theta_rad: float | None,
        theta_dot_rad_s: float | None,
        alpha_rad: float,
        alpha_dot_rad_s: float,
        normal_per_mass: float,
        friction_per_mass: float,
        y_center_m: float,
        y_edge_m: float,
    ) -> None:
        rows.append(
            {
                "phase": phase,
                "t_s": t_s,
                "overhang_cm": 100.0 * overhang_m,
                "r_m": r_m,
                "rdot_m_s": rdot_m_s,
                "theta_deg": None if theta_rad is None else rad_to_deg(theta_rad),
                "theta_dot_rad_s": theta_dot_rad_s,
                "alpha_deg": rad_to_deg(alpha_rad),
                "alpha_dot_rad_s": alpha_dot_rad_s,
                "normal_per_mass_m_s2": normal_per_mass,
                "friction_per_mass_m_s2": friction_per_mass,
                "normal_N": force_newtons(normal_per_mass, p),
                "friction_N": force_newtons(friction_per_mass, p),
                "y_cm_m": y_center_m,
                "y_nearest_edge_m": y_edge_m,
            }
        )

    static_end = (
        result.slip_time_s
        if result.slip_time_s is not None
        else result.liftoff_time_s
    )
    static_times = sample_times(0.0, static_end, solver.sample_dt_s)
    for t_s in static_times:
        theta, theta_dot = map(float, static_sol.sol(t_s))
        normal, friction, _ = static_forces_per_mass(
            theta, theta_dot, overhang_m, p
        )
        alpha = theta
        alpha_dot = theta_dot
        y_center = y_cm(r0, theta, overhang_m, p)
        y_edge = y_center - p.half_length_m * abs(math.sin(alpha))
        append_row(
            "no_slip",
            t_s,
            r0,
            0.0,
            theta,
            theta_dot,
            alpha,
            alpha_dot,
            normal,
            friction,
            y_center,
            y_edge,
        )

    if slip_sol is not None:
        slip_times = sample_times(result.slip_time_s, result.liftoff_time_s, solver.sample_dt_s)
        for t_s in slip_times:
            r, rdot, theta, theta_dot = map(float, slip_sol.sol(t_s))
            beta = beta_from_r(r, p)
            beta_dot = beta_dot_from_r(r, rdot, p)
            normal = slip_accelerations(
                r, rdot, theta, theta_dot, overhang_m, p, result.friction_sign
            )[2]
            friction = result.friction_sign * p.mu_kinetic * normal
            alpha = theta - beta0 + beta
            alpha_dot = theta_dot + beta_dot
            y_center = y_cm(r, theta, overhang_m, p)
            y_edge = y_center - p.half_length_m * abs(math.sin(alpha))
            append_row(
                "slip",
                t_s,
                r,
                rdot,
                theta,
                theta_dot,
                alpha,
                alpha_dot,
                normal,
                friction,
                y_center,
                y_edge,
            )

    fall_times = sample_times(
        0.0, result.landing_time_after_liftoff_s, solver.sample_dt_s
    )
    for tau_s in fall_times:
        alpha = result.liftoff_alpha_rad + result.liftoff_alpha_dot_rad_s * tau_s
        y_center = (
            result.liftoff_y_m
            + result.liftoff_y_dot_m_s * tau_s
            - 0.5 * p.g_m_s2 * tau_s * tau_s
        )
        y_edge = y_center - p.half_length_m * abs(math.sin(alpha))
        append_row(
            "free_fall",
            result.liftoff_time_s + tau_s,
            None,
            None,
            None,
            None,
            alpha,
            result.liftoff_alpha_dot_rad_s,
            0.0,
            0.0,
            y_center,
            y_edge,
        )

    return rows


def sample_times(start: float | None, stop: float, step: float) -> np.ndarray:
    if start is None:
        raise ValueError("sample start time is None")
    if stop < start:
        raise ValueError("sample stop time must be >= start time")
    if stop == start:
        return np.array([start], dtype=float)
    n = max(1, int(math.ceil((stop - start) / step)))
    times = np.linspace(start, stop, n + 1)
    return np.unique(np.round(times, decimals=12))


def summary_row(result: SimulationResult) -> dict[str, float | str | None]:
    return {
        "overhang_cm": 100.0 * result.overhang_m,
        "slip_time_s": result.slip_time_s,
        "slip_angle_deg": None
        if result.slip_angle_rad is None
        else rad_to_deg(result.slip_angle_rad),
        "slip_omega_rad_s": result.slip_omega_rad_s,
        "liftoff_phase": result.liftoff_phase,
        "liftoff_time_s": result.liftoff_time_s,
        "liftoff_angle_deg": rad_to_deg(result.liftoff_alpha_rad),
        "freefall_angular_velocity_rad_s": result.liftoff_alpha_dot_rad_s,
        "liftoff_y_m": result.liftoff_y_m,
        "liftoff_y_dot_m_s": result.liftoff_y_dot_m_s,
        "freefall_time_s": result.landing_time_after_liftoff_s,
        "total_landing_time_s": result.total_landing_time_s,
        "landing_total_angle_deg": rad_to_deg(result.landing_alpha_rad),
        "landing_total_angle_mod_deg": rad_to_deg(
            result.landing_alpha_rad % (2.0 * math.pi)
        ),
        "landing_side": result.landing_side,
        "r_liftoff_m": result.r_liftoff_m,
        "rdot_liftoff_m_s": result.rdot_liftoff_m_s,
        "theta_liftoff_deg": rad_to_deg(result.theta_liftoff_rad),
        "theta_dot_liftoff_rad_s": result.theta_dot_liftoff_rad_s,
        "friction_sign": result.friction_sign,
    }


def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"no rows to write to {path}")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_single(
    rows: list[dict[str, object]],
    result: SimulationResult,
    p: ToastParams,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.array([float(row["t_s"]) for row in rows])
    alpha = np.array([float(row["alpha_deg"]) for row in rows])
    omega = np.array([float(row["alpha_dot_rad_s"]) for row in rows])
    y_edge = np.array([float(row["y_nearest_edge_m"]) for row in rows])
    normal = np.array([float(row["normal_per_mass_m_s2"]) for row in rows])

    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    axes[0].plot(t, alpha, lw=2)
    axes[0].axvline(result.liftoff_time_s, color="tab:red", ls="--", lw=1)
    axes[0].set_ylabel("alpha (deg)")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(t, omega, lw=2, color="tab:orange")
    axes[1].axvline(result.liftoff_time_s, color="tab:red", ls="--", lw=1)
    axes[1].set_ylabel("alpha dot (rad/s)")
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(t, y_edge, lw=2, label="nearest edge")
    axes[2].plot(t, normal, lw=1.5, label="normal/mass")
    axes[2].axhline(
        -p.table_height_m,
        color="0.2",
        ls=":",
        lw=1,
        label=f"floor for {100.0 * p.table_height_m:.0f} cm",
    )
    axes[2].set_xlabel("time (s)")
    axes[2].set_ylabel("m or m/s^2")
    axes[2].grid(True, alpha=0.25)
    axes[2].legend(loc="best")

    fig.suptitle(f"Overhang {100.0 * result.overhang_m:.3f} cm: {result.landing_side}")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_sweep(results: list[SimulationResult], path_prefix: Path) -> None:
    path_prefix.parent.mkdir(parents=True, exist_ok=True)
    overhang = np.array([100.0 * r.overhang_m for r in results])
    omega = np.array([r.liftoff_alpha_dot_rad_s for r in results])
    landing_angle = np.array([rad_to_deg(r.landing_alpha_rad % (2.0 * math.pi)) for r in results])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(overhang, omega, lw=2, label="simulation")
    exp_x = np.array([row[0] for row in PAPER_ANGULAR_VELOCITIES])
    exp_y = np.array([row[1] for row in PAPER_ANGULAR_VELOCITIES])
    exp_err = np.array([row[2] for row in PAPER_ANGULAR_VELOCITIES])
    ax.errorbar(exp_x, exp_y, yerr=exp_err, fmt="o", ms=4, capsize=3, label="paper data")
    ax.set_xlabel("overhang d0 (cm)")
    ax.set_ylabel("free-fall angular velocity (rad/s)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path_prefix.with_name(path_prefix.name + "_angular_velocity.png"), dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(overhang, landing_angle, lw=2, label="simulation")
    ax.axhspan(90.0, 270.0, color="tab:orange", alpha=0.12, label="butter-side down")
    ax.axhline(90.0, color="0.4", ls=":", lw=1)
    ax.axhline(270.0, color="0.4", ls=":", lw=1)
    down_x = [x for x, side in PAPER_LANDINGS if side == "Down"]
    up_x = [x for x, side in PAPER_LANDINGS if side == "Up"]
    ax.scatter(down_x, [90.0] * len(down_x), marker="v", color="tab:red", label="observed down")
    ax.scatter(up_x, [270.0] * len(up_x), marker="^", color="tab:blue", label="observed up")
    ax.set_xlabel("overhang d0 (cm)")
    ax.set_ylabel("landing angle modulo 360 deg")
    ax.set_ylim(0.0, 360.0)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path_prefix.with_name(path_prefix.name + "_landing_angle.png"), dpi=180)
    plt.close(fig)


def print_result(result: SimulationResult) -> None:
    print(f"overhang: {100.0 * result.overhang_m:.3f} cm")
    if result.slip_time_s is not None:
        print(
            "slip: "
            f"t={result.slip_time_s:.6f} s, "
            f"alpha={rad_to_deg(result.slip_angle_rad):.3f} deg, "
            f"omega={result.slip_omega_rad_s:.5f} rad/s"
        )
    else:
        print("slip: none before liftoff")
    print(
        "liftoff: "
        f"t={result.liftoff_time_s:.6f} s, "
        f"alpha={rad_to_deg(result.liftoff_alpha_rad):.3f} deg, "
        f"omega_free={result.liftoff_alpha_dot_rad_s:.5f} rad/s"
    )
    print(
        "landing: "
        f"free_fall={result.landing_time_after_liftoff_s:.6f} s, "
        f"alpha_total={rad_to_deg(result.landing_alpha_rad):.3f} deg, "
        f"side={result.landing_side}"
    )


def build_params(args) -> tuple[ToastParams, SolverParams]:
    mass_kg = None if args.mass_g is None else args.mass_g / 1000.0
    params = ToastParams(
        half_length_m=args.length_cm / 200.0,
        thickness_m=args.thickness_cm / 100.0,
        mu_static=args.mu_s,
        mu_kinetic=args.mu_k,
        table_height_m=args.height_cm / 100.0,
        g_m_s2=args.g,
        mass_kg=mass_kg,
    )
    solver = SolverParams(
        rtol=args.rtol,
        atol=args.atol,
        max_step_s=args.max_step_ms / 1000.0,
        sample_dt_s=args.sample_dt_ms / 1000.0,
    )
    return params, solver


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--length-cm", type=float, default=10.2, help="board length 2a")
    parser.add_argument("--thickness-cm", type=float, default=1.3, help="board thickness b")
    parser.add_argument("--mu-s", type=float, default=0.32, help="static friction coefficient")
    parser.add_argument("--mu-k", type=float, default=0.24, help="kinetic friction coefficient")
    parser.add_argument("--height-cm", type=float, default=76.0, help="table height")
    parser.add_argument("--g", type=float, default=9.8, help="gravitational acceleration")
    parser.add_argument("--mass-g", type=float, default=None, help="optional board mass for force in N")
    parser.add_argument("--rtol", type=float, default=1e-9, help="ODE relative tolerance")
    parser.add_argument("--atol", type=float, default=1e-11, help="ODE absolute tolerance")
    parser.add_argument("--max-step-ms", type=float, default=0.5, help="maximum ODE step")
    parser.add_argument("--sample-dt-ms", type=float, default=1.0, help="CSV sampling interval")
    parser.add_argument("--no-plots", action="store_true", help="skip PNG plots")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simulate the tumbling-toast experiment from Bacon, Heald, and James."
    )
    subparsers = parser.add_subparsers(dest="command")

    single = subparsers.add_parser("single", help="simulate one overhang")
    add_common_arguments(single)
    single.add_argument("--overhang-cm", type=float, default=1.10)
    single.add_argument("--out-dir", type=Path, default=Path("results"))

    sweep = subparsers.add_parser("sweep", help="sweep overhang values")
    add_common_arguments(sweep)
    sweep.add_argument("--start-cm", type=float, default=0.05)
    sweep.add_argument("--stop-cm", type=float, default=None)
    sweep.add_argument("--step-cm", type=float, default=0.05)
    sweep.add_argument("--out-dir", type=Path, default=Path("results"))

    paper = subparsers.add_parser("paper", help="simulate the overhangs tabulated in the paper")
    add_common_arguments(paper)
    paper.add_argument("--out-dir", type=Path, default=Path("results"))

    args = parser.parse_args()
    if args.command is None:
        # Running the file without arguments gives a useful default product.
        args = parser.parse_args(["sweep"])
    return args


def run_single(args) -> None:
    params, solver = build_params(args)
    overhang_m = args.overhang_cm / 100.0
    result, bundle = simulate_overhang(
        overhang_m, params, solver, keep_solutions=True
    )
    rows = build_time_series_rows(result, bundle, params, solver)

    safe_name = f"single_{args.overhang_cm:.3f}cm".replace(".", "p")
    summary_path = args.out_dir / f"{safe_name}_summary.csv"
    series_path = args.out_dir / f"{safe_name}_timeseries.csv"
    write_csv(summary_path, [summary_row(result)])
    write_csv(series_path, rows)
    if not args.no_plots:
        plot_single(rows, result, params, args.out_dir / f"{safe_name}.png")

    print_result(result)
    print(f"saved: {summary_path}")
    print(f"saved: {series_path}")


def run_sweep(args) -> None:
    params, solver = build_params(args)
    stop_cm = (
        args.stop_cm
        if args.stop_cm is not None
        else 100.0 * params.half_length_m - 0.01
    )
    overhangs_cm = np.arange(args.start_cm, stop_cm + 0.5 * args.step_cm, args.step_cm)
    overhangs_cm = overhangs_cm[overhangs_cm < 100.0 * params.half_length_m]
    results = [
        simulate_overhang(d_cm / 100.0, params, solver, keep_solutions=False)[0]
        for d_cm in overhangs_cm
    ]

    summary_path = args.out_dir / "sweep_summary.csv"
    write_csv(summary_path, [summary_row(result) for result in results])
    if not args.no_plots:
        plot_sweep(results, args.out_dir / "sweep")

    print(f"simulated {len(results)} overhangs")
    print(f"saved: {summary_path}")
    down_ranges = landing_ranges(results, "Down")
    if down_ranges:
        ranges_text = ", ".join(f"{lo:.2f}-{hi:.2f} cm" for lo, hi in down_ranges)
        print(f"butter-side down ranges in this sweep: {ranges_text}")


def run_paper(args) -> None:
    params, solver = build_params(args)
    results = []
    rows = []
    for overhang_cm, measured, sigma in PAPER_ANGULAR_VELOCITIES:
        result, _bundle = simulate_overhang(
            overhang_cm / 100.0, params, solver, keep_solutions=False
        )
        results.append(result)
        row = summary_row(result)
        row["paper_measured_omega_rad_s"] = measured
        row["paper_sigma_rad_s"] = sigma
        row["omega_error_rad_s"] = result.liftoff_alpha_dot_rad_s - measured
        rows.append(row)

    out_path = args.out_dir / "paper_table_velocity_comparison.csv"
    write_csv(out_path, rows)
    if not args.no_plots:
        plot_sweep(results, args.out_dir / "paper_points")

    errors = np.array([row["omega_error_rad_s"] for row in rows], dtype=float)
    rms = float(np.sqrt(np.mean(errors * errors)))
    print(f"saved: {out_path}")
    print(f"RMS omega difference vs Table I measurements: {rms:.3f} rad/s")


def landing_ranges(
    results: list[SimulationResult], side: str
) -> list[tuple[float, float]]:
    ranges: list[tuple[float, float]] = []
    start: float | None = None
    previous: float | None = None
    for result in results:
        x = 100.0 * result.overhang_m
        if result.landing_side == side and start is None:
            start = x
        if result.landing_side != side and start is not None:
            ranges.append((start, previous if previous is not None else x))
            start = None
        previous = x
    if start is not None and previous is not None:
        ranges.append((start, previous))
    return ranges


def main() -> None:
    args = parse_args()
    if args.command == "single":
        run_single(args)
    elif args.command == "sweep":
        run_sweep(args)
    elif args.command == "paper":
        run_paper(args)
    else:
        raise ValueError(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()
