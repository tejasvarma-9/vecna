# Validation against MuJoCo

Worst-case absolute error over random states (1000 per arm model, 20 per random tree). Random trees have branching, arbitrary joint axes, joint axes offset from the body origin, rotated inertias and armature.

| Quantity | Units | UR5e-style arm | UR5e-style arm + 3 kg payload | 50 random branching trees (4-12 bodies) |
|---|---|---|---|---|
| Forward kinematics (site pose) | m / - | 2.1e-15 | 2.1e-15 | 4.2e-15 |
| Geometric Jacobian | - | 2.9e-15 | 2.4e-15 | 8.4e-15 |
| J̇·q̇ (analytic) | m/s², rad/s² | 7.8e-14 | 6.4e-14 | 8.5e-13 |
| Inverse dynamics (RNEA) | N·m | 1.6e-13 | 1.8e-13 | 5.2e-12 |
| Mass matrix (CRBA) | kg·m² | 1.0e-14 | 1.9e-14 | 1.5e-13 |
| Forward dynamics (ABA) | rad/s² | 6.8e-13 | 6.5e-13 | 2.7e-12 |

MuJoCo references: `site_xpos/site_xmat`, `mj_jacSite`, `mj_jacDot`, `mj_inverse` (`qfrc_inverse`), `mj_fullM`, `mj_forward` (`qacc`).

## Timing (6-DOF arm, per call)

| Algorithm | This package (Python) | MuJoCo (C) |
|---|---|---|
| RNEA | 298 µs | 2.8 µs (`mj_inverse`, does more work) |
| CRBA | 167 µs | - |
| ABA | 394 µs | 3.0 µs (`mj_forward`, does more work) |

The Python implementations favour readability over speed; the gap to C is interpreter overhead, not algorithmic (all are O(n) or O(n²)).
