# Inverse kinematics benchmark

UR5e-style arm, full 6D pose. Converged means within 0.01 mm and 0.006°, in at most 100 iterations per attempt, with joint limits enforced by clamping. Targets are reachable by construction (forward kinematics of a random configuration).

## Random reachable targets (n = 300)

First attempt starts from the home pose; up to 9 restarts from random configurations.

| Method | Converged, 1 attempt | Converged, ≤ 10 attempts | Median iterations (1st attempt) | Mean time per target | 99th pct largest step |
|---|---|---|---|---|---|
| Newton (pseudo-inverse) | 11.7% | 41.0% | 6 | 254 ms | 1.17e+10 rad |
| Damped least squares (λ = 0.1) | 58.0% | 87.7% | 18 | 78 ms | 12.1 rad |
| Levenberg-Marquardt, error-damped | 71.0% | 100.0% | 13 | 26 ms | 0.703 rad |

## Targets within 0.01 rad of the wrist singularity (n = 200)

One attempt each, starting 0.2 rad (std. dev.) away from the target configuration.

| Method | Converged | Median final position error | Median final orientation error | 99th pct largest step |
|---|---|---|---|---|
| Newton (pseudo-inverse) | 69.0% | 0.0018 mm | 6.1e-06° | 2.29e+09 rad |
| Damped least squares (λ = 0.1) | 15.5% | 0.086 mm | 0.0027° | 0.876 rad |
| Levenberg-Marquardt, error-damped | 50.5% | 0.01 mm | 0.00044° | 0.51 rad |

*Largest step* is the biggest joint update ‖Δq‖ in any single iteration. The pseudo-inverse divides by the smallest singular value of J, so near a singularity it commands joint jumps of 10⁹ rad or more: on hardware that is a fault, even when the iteration eventually converges. Damping bounds every step; the price is slower convergence in the near-singular direction, which is why the damped methods finish some singular targets at sub-millimetre rather than 0.01 mm accuracy.
Error-dependent damping gets most of both: heavy damping while the error is large, almost none once it is small.
