"""Controller comparison and payload identification. Writes results/controllers.md and plots."""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from armctl import load_ur5e, site_pose  # noqa: E402
from armctl.controllers import ComputedTorque  # noqa: E402
from armctl.experiments import (PAYLOAD_KG, SITE, TORQUE_NOISE_NM, WN, ZETA,  # noqa: E402
                                identify_from_log, joint_errors, rms, run_joint_suite,
                                run_task_suite, task_errors)
from armctl.identification import PARAM_NAMES, params_from_body  # noqa: E402
from armctl.sim import Simulation  # noqa: E402

RESULTS = ROOT / "results"
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # validated categorical slots 1-4, fixed order

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK_2,
    "axes.titlecolor": INK, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
    "font.size": 9.5, "legend.frameon": False, "legend.labelcolor": INK_2,
    "lines.linewidth": 2.0, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
})

NOMINAL, PAYLOAD = 0.0, PAYLOAD_KG
CONDITIONS = {NOMINAL: "Exact model", PAYLOAD: f"Unmodelled {PAYLOAD_KG:g} kg payload"}


def fmt_small(x, unit):
    return f"< 0.0001{unit}" if x < 1e-4 else f"{x:.3g}{unit}"


def ee_position_errors(robot, log, traj):
    """End-effector position error (m) implied by joint tracking error."""
    return np.array([np.linalg.norm(site_pose(robot, q, SITE)[1] - site_pose(robot, traj(t)[0], SITE)[1])
                     for t, q in zip(log["t"], log["q"])])


def error_panels(results, names, series_err, ylabel, title, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, cond in zip(axes, (NOMINAL, PAYLOAD)):
        for color, name in zip(SERIES, names):
            if (cond, name) in results:
                log = results[(cond, name)]
                ax.plot(log["t"], np.maximum(series_err(log), 1e-5), color=color, label=name)
        ax.set_yscale("log")
        ax.set_title(CONDITIONS[cond], loc="left")
        ax.set_xlabel("time (s)")
    axes[0].set_ylabel(ylabel)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    RESULTS.mkdir(exist_ok=True)
    lines = ["# Controller comparison", "",
             f"All controllers share the same nominal bandwidth (ωn = {WN:g} rad/s, ζ = {ZETA:g}); "
             "PD gains are scaled by the diagonal of M(q) (or of the task inertia Λ) at the home pose, "
             "so differences come from model-based compensation, not larger gains. "
             "Simulated at 1 kHz in MuJoCo, torques clipped to the UR5e ratings (150 N·m / 28 N·m).", ""]

    # ---- Joint space --------------------------------------------------------------
    traj, robot, results, pi_hat, cond, robot_id = run_joint_suite(seed=0)
    names = ["PD", "PD + gravity comp.", "Computed torque", "Computed torque + identified payload"]
    lines += ["## Joint-space tracking", "",
              "Three 1.5 s rest-to-rest moves through all six joints (up to 2.75 rad/s and 5.7 rad/s²; the UR5e limit is 3.14 rad/s).", "",
              "| Model | Controller | RMS joint error | Max joint error | RMS tool-point error | RMS torque | Saturated |",
              "|---|---|---|---|---|---|---|"]
    for (payload, name), log in results.items():
        e = np.degrees(joint_errors(log, traj))
        ee = ee_position_errors(robot, log, traj) * 1e3
        lines.append(f"| {CONDITIONS[payload]} | {name} | {rms(e):.3g}° | {np.abs(e).max():.3g}° | "
                     f"{rms(ee):.3g} mm | {rms(log['tau']):.3g} N·m | {100 * log['saturated'].mean():.0f}% |")
    error_panels(results, names, lambda log: np.degrees(np.linalg.norm(joint_errors(log, traj), axis=1)),
                 "joint error norm (deg, log)", "Joint-space tracking error",
                 RESULTS / "joint_tracking.png")

    # ---- Payload identification ------------------------------------------------------
    _, true_robot = load_ur5e(PAYLOAD_KG)
    pb = true_robot.bodies[-1]
    pi_true = params_from_body(pb.mass, pb.p0 + pb.R0 @ pb.com, pb.R0 @ pb.I_com @ pb.R0.T)
    lines += ["", "## Payload identification", "",
              "Dynamics are linear in a body's inertial parameters, so the payload's "
              "10 parameters (mass, first moments, inertia about the flange origin) follow from "
              "least squares on one logged run of the move above: τ_measured − RNEA_nominal = Y(q, q̇, q̈) π. "
              f"Measured torques carry {TORQUE_NOISE_NM} N·m Gaussian noise and q̈ comes from "
              f"numerically differentiating q̇, as on a real robot. Regressor condition number: {cond:.0f}.", "",
              "| Parameter | True | Estimated |", "|---|---|---|"]
    for k in range(4):
        lines.append(f"| {PARAM_NAMES[k]} | {pi_true[k]:.4f} | {pi_hat[k]:.4f} |")
    com_err = np.linalg.norm(pi_hat[1:4] / pi_hat[0] - pi_true[1:4] / pi_true[0]) * 1e3
    lines += ["", f"Mass error {abs(pi_hat[0] - pi_true[0]) * 1e3:.1f} g, centre-of-mass error {com_err:.1f} mm. "
              "The 6 inertia terms are estimated too but are weakly excited by this move and matter little here.", "",
              "### Sensitivity to torque-sensor noise", "",
              "| Torque noise (std. dev.) | Mass error | COM error | Computed-torque RMS error after identification |",
              "|---|---|---|---|"]
    sim = Simulation(load_ur5e(PAYLOAD_KG)[0])
    excite = results[(PAYLOAD, "Computed torque")]
    for noise in (0.0, 0.5, 1.0, 2.0):
        # Same seed as run_joint_suite, so the 0.5 N·m row reproduces the table above.
        pi_n, _, robot_n = identify_from_log(robot, excite, sim.dt, np.random.default_rng(0), noise=noise)
        sim.reset(traj(0)[0])
        log = sim.run(ComputedTorque(traj, robot_n, WN, ZETA), traj.duration + 0.5)
        com_n = np.linalg.norm(pi_n[1:4] / pi_n[0] - pi_true[1:4] / pi_true[0]) * 1e3
        lines.append(f"| {noise:g} N·m | {abs(pi_n[0] - PAYLOAD_KG) * 1e3:.1f} g | {com_n:.1f} mm | "
                     f"{rms(np.degrees(joint_errors(log, traj))):.3g}° |")

    # ---- Task space ----------------------------------------------------------------
    ctraj, _, tresults = run_task_suite(robot_id)
    tnames = ["Jacobian-transpose PD", "Operational space", "Operational space + identified payload"]
    lines += ["", "## Task-space tracking", "",
              "Two laps of a vertical circle of 15 cm radius in 4 s at fixed tool orientation (peak ≈ 0.9 m/s).", "",
              "| Model | Controller | RMS position error | Max position error | RMS orientation error | RMS torque |",
              "|---|---|---|---|---|---|"]
    task_err = {}
    for (payload, name), log in tresults.items():
        pos, rot = task_errors(log, ctraj, robot)
        task_err[(payload, name)] = pos
        lines.append(f"| {CONDITIONS[payload]} | {name} | {rms(pos) * 1e3:.3g} mm | {pos.max() * 1e3:.3g} mm | "
                     f"{fmt_small(rms(np.degrees(rot)), '°')} | {rms(log['tau']):.3g} N·m |")
    error_panels(tresults, tnames, lambda log: task_errors(log, ctraj, robot)[0] * 1e3,
                 "tool position error (mm, log)", "Task-space tracking error", RESULTS / "task_tracking.png")

    # Tool path in the circle's plane, with the payload: shows the sag the model error causes.
    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    ts = np.linspace(0, ctraj.duration, 400)
    ref = np.array([ctraj(t)[1] for t in ts])
    ax.plot(ref[:, 0] * 100, ref[:, 2] * 100, color=AXIS, linewidth=6, solid_capstyle="round", label="Reference")
    for color, name in zip(SERIES, tnames):
        log = tresults[(PAYLOAD, name)]
        path = np.array([site_pose(robot, q, SITE)[1] for q in log["q"][::10]])
        ax.plot(path[:, 0] * 100, path[:, 2] * 100, color=color, label=name)
    ax.set_aspect("equal")
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("z (cm)")
    ax.set_title(f"Tool path with an unmodelled {PAYLOAD_KG:g} kg payload", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    fig.tight_layout()
    fig.savefig(RESULTS / "circle_path.png", dpi=150)
    plt.close(fig)

    lines += ["", "![Joint tracking](joint_tracking.png)", "", "![Task tracking](task_tracking.png)", "",
              "![Tool path](circle_path.png)"]
    (RESULTS / "controllers.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
