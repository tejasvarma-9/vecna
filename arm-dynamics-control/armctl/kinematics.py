"""Forward kinematics, geometric Jacobian, Jdot*qdot, and inverse kinematics.

Task-space vectors here are [linear; angular] in world coordinates, matching
MuJoCo's (jacp, jacr) pair so the two can be compared directly.
"""

from dataclasses import dataclass

import numpy as np

from .spatial import axis_angle_to_rot, rot_log


def local_pose(body, q):
    """Pose (R, p) of a body in its parent's frame at configuration q."""
    j = body.joint
    if j is None:
        return body.R0, body.p0
    Rj = axis_angle_to_rot(j.axis, q[j.dof] - j.q0)
    # Rotate about an axis through the anchor, not through the body origin.
    return body.R0 @ Rj, body.p0 + body.R0 @ (j.anchor - Rj @ j.anchor)


def forward_kinematics(robot, q):
    """World poses [(R, p), ...] of every body, in robot.bodies order."""
    poses = []
    for b in robot.bodies:
        R, p = local_pose(b, q)
        if b.parent >= 0:
            Rp, pp = poses[b.parent]
            R, p = Rp @ R, pp + Rp @ p
        poses.append((R, p))
    return poses


def site_pose(robot, q, site, poses=None):
    """World orientation and position of a named site."""
    s = robot.sites[site]
    if poses is None:
        poses = forward_kinematics(robot, q)
    Rb, pb = poses[s.body]
    return Rb @ s.R, pb + Rb @ s.pos


def _chain_axes(robot, poses, body):
    """(dof, world axis, world anchor) for every joint between the root and `body`, root first."""
    out = []
    for i in reversed(robot.chain(body)):
        j = robot.bodies[i].joint
        if j is not None:
            R, p = poses[i]
            out.append((j.dof, R @ j.axis, p + R @ j.anchor))
    return out


def jacobian(robot, q, site, poses=None):
    """6 x nv geometric Jacobian of a site, world coordinates, rows [linear; angular]."""
    if poses is None:
        poses = forward_kinematics(robot, q)
    _, pe = site_pose(robot, q, site, poses)
    J = np.zeros((6, robot.nv))
    for dof, z, a in _chain_axes(robot, poses, robot.sites[site].body):
        J[:3, dof] = np.cross(z, pe - a)
        J[3:, dof] = z
    return J


def jacobian_dot_qdot(robot, q, qd, site, poses=None):
    """Jdot @ qd, the task acceleration produced by joint velocity alone (qdd = 0).

    Differentiates each Jacobian column analytically:
      d/dt z_k        = w_k x z_k                    (axis is fixed in its own body)
      d/dt [z_k x r]  = zdot_k x r + z_k x (v_e - v_ak),  r = p_e - a_k
    where w_k is body k's angular velocity and v_ak the velocity of joint k's anchor.
    """
    if poses is None:
        poses = forward_kinematics(robot, q)
    _, pe = site_pose(robot, q, site, poses)
    axes = _chain_axes(robot, poses, robot.sites[site].body)
    ve = sum((np.cross(z, pe - a) * qd[dof] for dof, z, a in axes), np.zeros(3))

    out = np.zeros(6)
    w = np.zeros(3)
    parents = []                          # (z, anchor, rate) of joints above the current one
    for dof, z, a in axes:
        rate = qd[dof]
        w = w + z * rate
        va = sum((np.cross(zm, a - am) * rm for zm, am, rm in parents), np.zeros(3))
        zdot = np.cross(w, z)
        out[:3] += (np.cross(zdot, pe - a) + np.cross(z, ve - va)) * rate
        out[3:] += zdot * rate
        parents.append((z, a, rate))
    return out


def pose_error(R, p, R_d, p_d):
    """[position error; rotation-vector error] taking (R, p) to (R_d, p_d), world frame."""
    return np.concatenate([p_d - p, rot_log(R_d @ R.T)])


@dataclass
class IKResult:
    q: np.ndarray
    success: bool
    iterations: int      # total over all attempts
    attempts: int
    pos_error: float     # metres
    rot_error: float     # radians
    max_step: float      # largest joint-space step norm taken, radians


def _ik_attempt(robot, site, R_d, p_d, q, method, damping, max_iters, tol_pos, tol_rot):
    lower, upper = robot.lower, robot.upper
    q = np.clip(np.asarray(q, dtype=float), lower, upper)
    n = robot.nv
    max_step = 0.0
    for it in range(max_iters + 1):
        poses = forward_kinematics(robot, q)
        R, p = site_pose(robot, q, site, poses)
        e = pose_error(R, p, R_d, p_d)
        pos_err, rot_err = np.linalg.norm(e[:3]), np.linalg.norm(e[3:])
        if pos_err < tol_pos and rot_err < tol_rot:
            return q, True, it, pos_err, rot_err, max_step
        if it == max_iters:
            break
        J = jacobian(robot, q, site, poses)
        if method == "newton":
            dq = np.linalg.pinv(J) @ e
        elif method == "dls":
            dq = J.T @ np.linalg.solve(J @ J.T + damping ** 2 * np.eye(6), e)
        elif method == "lm":
            lam = 0.5 * e @ e + 1e-4
            dq = np.linalg.solve(J.T @ J + lam * np.eye(n), J.T @ e)
        else:
            raise ValueError(f"unknown IK method {method!r}")
        max_step = max(max_step, float(np.linalg.norm(dq)))
        q = np.clip(q + dq, lower, upper)
    return q, False, it, pos_err, rot_err, max_step


def solve_ik(robot, site, R_d, p_d, q_init, method="lm", damping=0.1, max_iters=100,
             tol_pos=1e-5, tol_rot=1e-4, restarts=0, rng=None):
    """Iterative 6D inverse kinematics.

    method:
      "newton"  dq = pinv(J) e. Quadratic convergence near a solution, but the step
                blows up as J loses rank (it divides by the smallest singular value).
      "dls"     damped least squares, minimises |J dq - e|^2 + damping^2 |dq|^2.
                Bounded steps, but a fixed damping also slows the final convergence.
      "lm"      Levenberg-Marquardt with error-dependent damping (Sugihara, 2011):
                lambda = |e|^2 / 2 + 1e-4. Heavily damped far from the target, almost
                undamped near it, so it is both robust and fast to converge.

    Iterative IK is local: from a poor start it can stall in a local minimum or
    against a joint limit. `restarts` extra attempts from random configurations
    (drawn with `rng`) make it robust in practice.
    """
    total_iters, best = 0, None
    for attempt in range(restarts + 1):
        q0 = q_init if attempt == 0 else robot.random_q(rng)
        q, ok, it, pos_err, rot_err, step = _ik_attempt(
            robot, site, R_d, p_d, q0, method, damping, max_iters, tol_pos, tol_rot)
        total_iters += it
        max_step = step if best is None else max(best.max_step, step)
        if best is None or ok or pos_err + rot_err < best.pos_error + best.rot_error:
            best = IKResult(q, ok, total_iters, attempt + 1, pos_err, rot_err, max_step)
        best.iterations, best.attempts, best.max_step = total_iters, attempt + 1, max_step
        if ok:
            break
    return best
