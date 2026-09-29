# Interview prep: `arm-dynamics-control`

Private notes for Indira; don't publish them alongside the project.

## Read this first

Robotics interviewers pick one resume project and dig until they find the edge of
your understanding. If that edge is "I don't know, the code does it", the project
counts *against* you. Before this project goes on the resume:

1. You can derive everything in sections A–C below on a whiteboard, without notes.
2. You have run every script, broken something on purpose, and understood why the tests caught it.
3. You have built at least one extension from section E yourself, so part of the repo
   is unambiguously your own work.
4. If asked, you describe honestly how it was built: "I used AI assistance for parts of
   the implementation, then verified every algorithm against MuJoCo and extended it
   with X." That answer is acceptable at most companies today. Being caught unable to
   explain your own code is not.

Never claim: a real robot (it's simulation), "the UR5e" (say *UR5e-style parameters*),
or real-time performance (the Python RNEA takes about 300 µs).

## 7-day plan (placements are running now)

| Day | Study | Do |
|---|---|---|
| 1 | Spatial vectors, FK, Jacobian (`docs/THEORY.md` §1–2) | Derive the 2-link planar Jacobian by hand; find its singularity (sin q₂ = 0) |
| 2 | RNEA (§4) | Derive 2-link M(q), C, g with Lagrange; check against `rnea` on a 2-link MJCF you write |
| 3 | CRBA, ABA (§5–6) | Explain why ABA is O(n) and when you'd use it over solving M q̈ = τ − h |
| 4 | Control (§7) | PD+g Lyapunov proof on paper; computed-torque error dynamics; OSC derivation |
| 5 | IK (§3) | UR singularities (wrist, elbow, shoulder); DLS derivation from the cost function |
| 6 | Identification (§8) | Why linear in π; persistent excitation; what the condition number tells you |
| 7 | Mock interview | Have a friend ask the section D questions; time the 2-minute walkthrough |

## A. 30-second pitch

"I implemented manipulator dynamics and control from scratch (the recursive
Newton–Euler, composite rigid body and articulated body algorithms) for a 6-DOF
UR5e-style arm, and verified them against MuJoCo to machine precision. Then I compared
five torque controllers in closed-loop simulation. The interesting result was that
model-based controllers fall apart with an unmodelled payload, even underperforming
simple PD. So I added least-squares payload identification from joint torques, which
brought operational-space tracking from 25 mm back to 0.13 mm."

## B. 2-minute walkthrough (problem → method → evidence → insight)

1. **Why:** model-based control (computed torque, operational space) is what lets
   manipulators track fast and comply softly, and it depends on a correct dynamics model.
2. **Dynamics:** spatial-vector RNEA for inverse dynamics, CRBA for M(q), ABA for forward
   dynamics. All three are validated against MuJoCo's C code on the arm and on 50 random
   branching trees (so it isn't overfit to one robot), plus physics checks: M symmetric
   positive definite, power balance.
3. **Control:** PD, PD+g, computed torque, Jacobian-transpose, operational space. All
   gains matched to the same 20 rad/s bandwidth so the comparison is fair.
4. **Finding:** with an exact model, computed torque is ~300× better than PD. With a
   3 kg payload the model is wrong, and OSC becomes worse than J-transpose.
5. **Fix:** dynamics are linear in inertial parameters, so the payload's parameters come
   out of least squares on one logged move. With realistic torque noise, mass is accurate
   to 1 g and COM to 2 mm, and tracking is restored.
6. **IK side result:** error-damped Levenberg–Marquardt with restarts solves 100% of random
   targets; the pseudo-inverse solves 41% and takes 10¹⁰ rad steps near singularities.

## C. Numbers to know cold

| | |
|---|---|
| Validation error vs MuJoCo | within 1e-11 (RNEA, CRBA, ABA, FK, J, J̇q̇) |
| Joint tracking, exact model | PD 1.49°, PD+g 0.25°, CTC 0.0053° RMS |
| With 3 kg unmodelled payload | CTC 1.41° → 0.018° after identification |
| Task tracking | J-transpose 7.7 mm, OSC 0.12 mm; payload: 19.3 / 25.3 mm → 0.13 mm |
| Identification | mass error ~1 g, COM error ~2 mm at 0.5 N·m torque noise |
| IK | LM + restarts 100% (~26 ms/target); DLS 88%; pseudo-inverse 41% |
| Bandwidth | ωn = 20 rad/s, ζ = 1; sim at 1 kHz; torque limits 150/28 N·m |

## D. Questions you will get, with the answer's core

1. **Why spatial vectors instead of 3D Newton–Euler?** One 6D equation per body instead of
   separate force and moment equations. Transforms compose cleanly, and the same primitives
   serve RNEA, CRBA and ABA.
2. **Complexity of RNEA / CRBA / ABA?** O(n), O(n²) (chain), O(n). A dense solve of
   M q̈ = τ − h is O(n³), which is why ABA wins for large n.
3. **How is gravity handled?** As an upward base acceleration a₀ = −g. The equivalence
   principle means no per-body gravity term is needed.
4. **Properties of M(q)?** Symmetric positive definite; configuration-dependent;
   Ṁ − 2C is skew-symmetric (with a Christoffel-consistent C). That skew-symmetry is the
   key step in the PD+g stability proof.
5. **Prove PD + gravity compensation is stable.** V = ½q̇ᵀMq̇ + ½eᵀKₚe gives
   V̇ = −q̇ᵀK_d q̇ ≤ 0, and LaSalle's principle gives convergence. The proof covers
   regulation only, not tracking.
6. **Why does PD+g still show 0.25° tracking error?** While tracking, the uncompensated
   inertial and Coriolis terms act as a disturbance.
7. **What does computed torque do to the closed loop?** It makes the error dynamics
   linear and decoupled, ë + K_d ė + Kₚ e = 0. It is implemented as one RNEA call.
8. **When does computed torque fail?** Model error (the payload), torque saturation,
   unmodelled friction, time delay and sensor noise, and heavy compute on slow hardware.
9. **Derive OSC.** Start from ẍ = Jq̈ + J̇q̇ and τ = Jᵀ F + h, which gives
   F = Λ(ẍ_cmd − J̇q̇) with Λ = (J M⁻¹ Jᵀ)⁻¹.
10. **What happens to OSC near singularities?** Λ blows up, because J M⁻¹ Jᵀ loses rank.
    In practice you damp or clamp it, as in IK.
11. **What's the null space, and why doesn't this arm have one?** It's the set of joint
    motions that don't move the task. A 6-joint arm with a 6D task has none; a 7-DOF arm
    uses the null space for posture or obstacle avoidance.
12. **Why is the pseudo-inverse dangerous near singularities?** Its gain is 1/σ, so it
    commands huge joint velocities.
13. **Why DLS?** It minimises ‖JΔq − e‖² + λ²‖Δq‖², which bounds the gain by 1/(2λ).
    The cost is accuracy near singularities.
14. **Why error-damped LM?** Damping ∝ ‖e‖² is heavy far from the target and vanishes
    near it, so it gets both robustness and fast final convergence.
15. **UR5e singularities?** Wrist: q₅ = 0 (wrist 1 and 3 axes align). Elbow: arm
    fully stretched (q₃ = 0). Shoulder: wrist centre on the base axis.
16. **Why are dynamics linear in inertial parameters?** Newton–Euler is linear in the
    spatial inertia, and the inertia is linear in (m, mc, I about the origin).
17. **Why inertia about the origin, not the COM?** About the COM, the parallel-axis term
    m c×c×ᵀ is nonlinear in the parameters.
18. **What's persistent excitation?** The trajectory must make Y full rank and well
    conditioned. Here the condition number is 12; the inertia terms are weakly excited.
19. **Why numerically differentiate q̇ instead of using the simulator's q̈?** A real robot
    doesn't measure acceleration. Differentiation amplifies noise, so on hardware you'd
    low-pass filter it.
20. **How would you do this online?** Recursive least squares, or Slotine–Li adaptive
    control, which updates π̂ from the tracking error and keeps Lyapunov stability.
21. **How did you make the controller comparison fair?** Gains are scaled by M_ii (or
    Λ_ii) at the home pose, giving every controller the same 20 rad/s bandwidth.
22. **How do you know your dynamics are right?** Against an independent implementation
    (MuJoCo), on random trees, plus physics identities (power balance, RNEA∘ABA = I).
23. **What would change on real hardware?** Friction and motor models, a torque-sensing
    method (current → torque), filtering, control-loop timing (a C++ port), and safety
    limits on velocity and torque.

## E. Make it yours (do at least one before listing the project)

Each is 1–3 days and gives you something to say that only you could say.

1. **ROS 2 wrapper** (most JDs list ROS): a node that runs the sim, publishes
   `sensor_msgs/JointState`, subscribes to a target `geometry_msgs/PoseStamped`, and runs
   the OSC controller. Adds ROS 2 to the resume honestly.
2. **C++ port of RNEA** with Eigen plus pybind11. Benchmark it against the Python version
   and MuJoCo, and aim for under 10 µs. Adds real C++ to the resume.
3. **7-DOF Franka Panda** from MuJoCo Menagerie. You'll need to extend `Robot.from_mujoco`
   for its finger slide joints, or strip the hand, then add null-space posture control
   to OSC.
4. **Friction:** add Coulomb plus viscous joint friction to the sim, show what it does to
   computed torque, then identify the friction coefficients with the same least-squares
   machinery.
5. **Adaptive control (Slotine–Li):** use `payload_regressor` to update π̂ online during
   tracking instead of batch identification.

## F. Whiteboard warm-ups

- 2-link planar arm (lengths l₁, l₂, point masses m₁, m₂ at the tips): write the FK, the
  Jacobian, det J = l₁l₂ sin q₂, M(q), and g(q).
- Given M, C, g, write the computed-torque law and the resulting error dynamics.
- Why a quintic and not a cubic for rest-to-rest motion? (Six boundary conditions.)
- Given the SVD of J, write the pseudo-inverse and DLS solutions and compare their gains.
