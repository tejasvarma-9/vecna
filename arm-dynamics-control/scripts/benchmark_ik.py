"""Benchmark Newton, damped least squares and Levenberg-Marquardt IK. Writes results/ik_benchmark.md."""

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from armctl import UR5E_HOME, load_ur5e, site_pose, solve_ik  # noqa: E402

N_GENERAL = 300
N_SINGULAR = 200
RESTARTS = 9
METHODS = {"newton": "Newton (pseudo-inverse)", "dls": "Damped least squares (λ = 0.1)",
           "lm": "Levenberg-Marquardt, error-damped"}


def solve_all(robot, problems, method, restarts, seed):
    rng = np.random.default_rng(seed)
    results, times = [], []
    for q_init, R_d, p_d in problems:
        t0 = time.perf_counter()
        results.append(solve_ik(robot, "ee", R_d, p_d, q_init, method=method,
                                restarts=restarts, rng=rng))
        times.append(time.perf_counter() - t0)
    return results, 1e3 * float(np.mean(times))


def main():
    rng = np.random.default_rng(0)
    _, robot = load_ur5e()

    # General: reachable targets (FK of a random configuration), first attempt from home.
    general = [(UR5E_HOME, *site_pose(robot, robot.random_q(rng), "ee")) for _ in range(N_GENERAL)]

    # Near the wrist singularity (wrist_2 ~ 0 lines up the wrist_1 and wrist_3 axes),
    # starting from a nearby configuration, as when tracking a path through it.
    singular = []
    for _ in range(N_SINGULAR):
        q_t = robot.random_q(rng)
        q_t[4] = rng.uniform(-0.01, 0.01)
        singular.append((q_t + rng.normal(scale=0.2, size=6), *site_pose(robot, q_t, "ee")))

    lines = ["# Inverse kinematics benchmark", "",
             "UR5e-style arm, full 6D pose. Converged means within 0.01 mm and 0.006°, "
             "in at most 100 iterations per attempt, with joint limits enforced by clamping. "
             "Targets are reachable by construction (forward kinematics of a random configuration).", "",
             f"## Random reachable targets (n = {N_GENERAL})", "",
             f"First attempt starts from the home pose; up to {RESTARTS} restarts from random configurations.", "",
             "| Method | Converged, 1 attempt | Converged, ≤ 10 attempts | Median iterations (1st attempt) "
             "| Mean time per target | 99th pct largest step |",
             "|---|---|---|---|---|---|"]
    for key, name in METHODS.items():
        res, ms = solve_all(robot, general, key, RESTARTS, seed=1)
        first = [r for r in res if r.success and r.attempts == 1]
        lines.append(
            f"| {name} | {100 * len(first) / len(res):.1f}% | "
            f"{100 * np.mean([r.success for r in res]):.1f}% | "
            f"{np.median([r.iterations for r in first]):.0f} | {ms:.0f} ms | "
            f"{np.percentile([r.max_step for r in res], 99):.3g} rad |")
        print(lines[-1])

    lines += ["", f"## Targets within 0.01 rad of the wrist singularity (n = {N_SINGULAR})", "",
              "One attempt each, starting 0.2 rad (std. dev.) away from the target configuration.", "",
              "| Method | Converged | Median final position error | Median final orientation error "
              "| 99th pct largest step |",
              "|---|---|---|---|---|"]
    for key, name in METHODS.items():
        res, _ = solve_all(robot, singular, key, 0, seed=2)
        lines.append(
            f"| {name} | {100 * np.mean([r.success for r in res]):.1f}% | "
            f"{1e3 * np.median([r.pos_error for r in res]):.2g} mm | "
            f"{np.degrees(np.median([r.rot_error for r in res])):.2g}° | "
            f"{np.percentile([r.max_step for r in res], 99):.3g} rad |")
        print(lines[-1])

    lines += ["",
              "*Largest step* is the biggest joint update ‖Δq‖ in any single iteration. The "
              "pseudo-inverse divides by the smallest singular value of J, so near a singularity it "
              "commands joint jumps of 10⁹ rad or more: on hardware that is a fault, even when the "
              "iteration eventually converges. Damping bounds every step; the price is slower "
              "convergence in the near-singular direction, which is why the damped methods finish "
              "some singular targets at sub-millimetre rather than 0.01 mm accuracy.",
              "Error-dependent damping gets most of both: heavy damping while the error is large, "
              "almost none once it is small."]
    out = ROOT / "results" / "ik_benchmark.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
