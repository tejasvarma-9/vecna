"""Rigid-body dynamics of kinematic trees: RNEA, CRBA and ABA.

Equation of motion:  M(q) qdd + h(q, qd) = tau,   h = C(q, qd) qd + g(q)

  rnea(q, qd, qdd)  -> tau                 inverse dynamics,        O(n)
  crba(q)           -> M(q)                joint-space inertia,     O(n^2)
  aba(q, qd, tau)   -> qdd                 forward dynamics,        O(n)

All three use spatial vectors in body coordinates (see spatial.py) and follow
Featherstone, "Rigid Body Dynamics Algorithms" (2008), chapters 5-7. Joint
armature (reflected rotor inertia) is included, matching MuJoCo's model.
Gravity enters as a fictitious upward acceleration of the base, a0 = -g.
"""

import numpy as np

from .kinematics import local_pose
from .spatial import crf, crm, plucker


def parent_transforms(robot, q):
    """Motion transforms i_X_parent(i) for every body at configuration q."""
    Xup = []
    for b in robot.bodies:
        R, p = local_pose(b, q)
        Xup.append(plucker(R.T, p))
    return Xup


def _base_acceleration(robot, gravity):
    a0 = np.zeros(6)
    if gravity:
        a0[3:] = -robot.gravity
    return a0


def body_motion(robot, q, qd, qdd, gravity=True, Xup=None):
    """Spatial velocity and acceleration of every body (body coordinates), root to leaves.

    With gravity=True the accelerations include the fictitious base acceleration -g,
    which is what the Newton-Euler equations need.
    """
    if Xup is None:
        Xup = parent_transforms(robot, q)
    n = len(robot.bodies)
    v, a = [None] * n, [None] * n
    a0 = _base_acceleration(robot, gravity)
    zero = np.zeros(6)
    for i, b in enumerate(robot.bodies):
        vp = v[b.parent] if b.parent >= 0 else zero
        ap = a[b.parent] if b.parent >= 0 else a0
        v[i] = Xup[i] @ vp
        a[i] = Xup[i] @ ap
        if b.joint is not None:
            vJ = b.joint.S * qd[b.joint.dof]
            v[i] = v[i] + vJ
            a[i] = a[i] + b.joint.S * qdd[b.joint.dof] + crm(v[i]) @ vJ
    return v, a


def rnea(robot, q, qd, qdd, gravity=True, Xup=None):
    """Inverse dynamics: the joint torques that produce acceleration qdd."""
    if Xup is None:
        Xup = parent_transforms(robot, q)
    n = len(robot.bodies)

    # Forward pass: velocities and accelerations, root to leaves; then each
    # body's net force from Newton-Euler, f = I a + v x* I v.
    v, a = body_motion(robot, q, qd, qdd, gravity, Xup)
    f = [b.inertia @ a[i] + crf(v[i]) @ (b.inertia @ v[i]) for i, b in enumerate(robot.bodies)]

    # Backward pass: accumulate forces from leaves to the root, project onto joints.
    tau = np.zeros(robot.nv)
    for i in reversed(range(n)):
        b = robot.bodies[i]
        if b.joint is not None:
            k = b.joint.dof
            tau[k] = b.joint.S @ f[i] + b.joint.armature * qdd[k]
        if b.parent >= 0:
            f[b.parent] = f[b.parent] + Xup[i].T @ f[i]
    return tau


def bias_forces(robot, q, qd):
    """h(q, qd) = C(q, qd) qd + g(q): RNEA with zero acceleration."""
    return rnea(robot, q, qd, np.zeros(robot.nv))


def gravity_torques(robot, q):
    """g(q): RNEA at rest."""
    return rnea(robot, q, np.zeros(robot.nv), np.zeros(robot.nv))


def crba(robot, q, Xup=None):
    """Joint-space inertia matrix M(q) via the Composite Rigid Body Algorithm."""
    if Xup is None:
        Xup = parent_transforms(robot, q)
    bodies = robot.bodies
    Ic = [b.inertia.copy() for b in bodies]
    for i in reversed(range(len(bodies))):
        p = bodies[i].parent
        if p >= 0:
            Ic[p] += Xup[i].T @ Ic[i] @ Xup[i]

    M = np.zeros((robot.nv, robot.nv))
    for i, b in enumerate(bodies):
        if b.joint is None:
            continue
        ki = b.joint.dof
        F = Ic[i] @ b.joint.S                 # force needed to accelerate subtree i along joint i
        M[ki, ki] = b.joint.S @ F + b.joint.armature
        j = i
        while bodies[j].parent >= 0:          # walk towards the root
            F = Xup[j].T @ F
            j = bodies[j].parent
            if bodies[j].joint is not None:
                kj = bodies[j].joint.dof
                M[ki, kj] = M[kj, ki] = bodies[j].joint.S @ F
    return M


def aba(robot, q, qd, tau, gravity=True, Xup=None):
    """Forward dynamics via the Articulated Body Algorithm: qdd = M^-1 (tau - h)."""
    if Xup is None:
        Xup = parent_transforms(robot, q)
    bodies = robot.bodies
    n = len(bodies)
    zero = np.zeros(6)
    v, c, IA, pA = [None] * n, [None] * n, [None] * n, [None] * n

    # Pass 1: velocities, velocity-product accelerations, rigid-body bias forces.
    for i, b in enumerate(bodies):
        vp = v[b.parent] if b.parent >= 0 else zero
        v[i] = Xup[i] @ vp
        c[i] = zero
        if b.joint is not None:
            vJ = b.joint.S * qd[b.joint.dof]
            v[i] = v[i] + vJ
            c[i] = crm(v[i]) @ vJ
        IA[i] = b.inertia.copy()
        pA[i] = crf(v[i]) @ (b.inertia @ v[i])

    # Pass 2: articulated inertias and bias forces, leaves to root.
    U, d, u = [None] * n, [None] * n, [None] * n
    for i in reversed(range(n)):
        b = bodies[i]
        if b.joint is not None:
            S = b.joint.S
            U[i] = IA[i] @ S
            d[i] = S @ U[i] + b.joint.armature
            u[i] = tau[b.joint.dof] - S @ pA[i]
            Ia = IA[i] - np.outer(U[i], U[i]) / d[i]
            pa = pA[i] + Ia @ c[i] + U[i] * (u[i] / d[i])
        else:
            Ia = IA[i]
            pa = pA[i] + Ia @ c[i]
        if b.parent >= 0:
            IA[b.parent] = IA[b.parent] + Xup[i].T @ Ia @ Xup[i]
            pA[b.parent] = pA[b.parent] + Xup[i].T @ pa

    # Pass 3: accelerations, root to leaves.
    qdd = np.zeros(robot.nv)
    a = [None] * n
    a0 = _base_acceleration(robot, gravity)
    for i, b in enumerate(bodies):
        ap = a[b.parent] if b.parent >= 0 else a0
        a[i] = Xup[i] @ ap + c[i]
        if b.joint is not None:
            k = b.joint.dof
            qdd[k] = (u[i] - U[i] @ a[i]) / d[i]
            a[i] = a[i] + b.joint.S * qdd[k]
    return qdd
