"""Validate kinematics and dynamics against MuJoCo; time both. Writes results/validation.md."""

import sys
import time
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from armctl import aba, crba, load_random_tree, load_ur5e, rnea  # noqa: E402
from armctl.validation import CHECKS, compare_random_states  # noqa: E402

N_STATES = 1000
N_TREES = 50
UNITS = {"fk": "m / -", "jacobian": "-", "jdot_qdot": "m/s², rad/s²",
         "rnea": "N·m", "crba": "kg·m²", "aba": "rad/s²"}
NAMES = {"fk": "Forward kinematics (site pose)", "jacobian": "Geometric Jacobian",
         "jdot_qdot": "J̇·q̇ (analytic)", "rnea": "Inverse dynamics (RNEA)",
         "crba": "Mass matrix (CRBA)", "aba": "Forward dynamics (ABA)"}


def per_call_us(fn, n=2000):
    fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n * 1e6


def main():
    rng = np.random.default_rng(0)
    rows = {}
    model, robot = load_ur5e()
    rows["UR5e-style arm"] = compare_random_states(model, robot, "ee", rng, N_STATES)
    model_p, robot_p = load_ur5e(payload_kg=3.0)
    rows["UR5e-style arm + 3 kg payload"] = compare_random_states(model_p, robot_p, "ee", rng, N_STATES)

    tree_worst = dict.fromkeys(CHECKS, 0.0)
    for _ in range(N_TREES):
        n = int(rng.integers(4, 13))
        m, r = load_random_tree(rng, n)
        for k, v in compare_random_states(m, r, f"s{n - 1}", rng, 20).items():
            tree_worst[k] = max(tree_worst[k], v)
    rows[f"{N_TREES} random branching trees (4-12 bodies)"] = tree_worst

    # Timing on the UR5e-style arm: this package (pure Python/NumPy) vs MuJoCo (C).
    q, qd, qdd, tau = robot.random_q(rng), *rng.normal(size=(3, robot.nv))
    data = mujoco.MjData(model)
    data.qpos[:], data.qvel[:], data.qacc[:] = q, qd, qdd

    def mj_rnea():
        mujoco.mj_inverse(model, data)

    def mj_fwd():
        mujoco.mj_forward(model, data)

    timing = [
        ("RNEA", per_call_us(lambda: rnea(robot, q, qd, qdd)), per_call_us(mj_rnea, 20000), "`mj_inverse`"),
        ("CRBA", per_call_us(lambda: crba(robot, q)), None, ""),
        ("ABA", per_call_us(lambda: aba(robot, q, qd, tau)), per_call_us(mj_fwd, 20000), "`mj_forward`"),
    ]

    lines = ["# Validation against MuJoCo", "",
             f"Worst-case absolute error over random states ({N_STATES} per arm model, "
             f"20 per random tree). Random trees have branching, arbitrary joint axes, "
             "joint axes offset from the body origin, rotated inertias and armature.", "",
             "| Quantity | Units | " + " | ".join(rows) + " |",
             "|---|---|" + "---|" * len(rows)]
    for k in CHECKS:
        lines.append(f"| {NAMES[k]} | {UNITS[k]} | "
                     + " | ".join(f"{rows[r][k]:.1e}" for r in rows) + " |")
    lines += ["", "MuJoCo references: `site_xpos/site_xmat`, `mj_jacSite`, `mj_jacDot`, "
              "`mj_inverse` (`qfrc_inverse`), `mj_fullM`, `mj_forward` (`qacc`).", "",
              "## Timing (6-DOF arm, per call)", "",
              "| Algorithm | This package (Python) | MuJoCo (C) |", "|---|---|---|"]
    for name, ours, theirs, which in timing:
        lines.append(f"| {name} | {ours:.0f} µs | "
                     + (f"{theirs:.1f} µs ({which}, does more work)" if theirs else "-") + " |")
    lines += ["", "The Python implementations favour readability over speed; the gap to C "
              "is interpreter overhead, not algorithmic (all are O(n) or O(n²))."]
    out = ROOT / "results" / "validation.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
