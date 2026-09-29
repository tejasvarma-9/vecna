"""Payload identification by least squares.

Key fact: rigid-body dynamics are linear in each body's 10 inertial parameters
    pi = [m, m*cx, m*cy, m*cz, Ixx, Iyy, Izz, Ixy, Ixz, Iyz]
(mass, first moments, and the inertia tensor about the body *origin*, not the COM;
about the COM the dependence would be nonlinear). So an unknown payload rigidly
attached to a known body adds torques
    tau_payload = Y(q, qd, qdd) @ pi
and pi can be recovered from logged motion and torques with ordinary least squares:
    tau_measured - rnea_nominal(q, qd, qdd) = Y(q, qd, qdd) @ pi
"""

import numpy as np

from .dynamics import body_motion, parent_transforms, rnea
from .robot import Body, Robot
from .spatial import crf, skew

PARAM_NAMES = ["m", "m*cx", "m*cy", "m*cz", "Ixx", "Iyy", "Izz", "Ixy", "Ixz", "Iyz"]


def inertia_from_params(pi):
    """6x6 spatial inertia (about the body origin) of the parameter vector pi. Linear in pi."""
    m, h = pi[0], np.asarray(pi[1:4])
    xx, yy, zz, xy, xz, yz = pi[4:]
    I = np.zeros((6, 6))
    I[:3, :3] = [[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]]
    I[:3, 3:] = skew(h)
    I[3:, :3] = skew(h).T
    I[3:, 3:] = m * np.eye(3)
    return I


def params_from_body(mass, com, I_com):
    """Inertial parameter vector of a body given mass, COM and inertia about the COM."""
    C = skew(com)
    Io = I_com + mass * C @ C.T          # parallel-axis theorem, to the body origin
    return np.array([mass, *(mass * np.asarray(com)),
                     Io[0, 0], Io[1, 1], Io[2, 2], Io[0, 1], Io[0, 2], Io[1, 2]])


_BASIS = [inertia_from_params(e) for e in np.eye(10)]


def payload_regressor(robot, q, qd, qdd, body):
    """nv x 10 matrix Y with tau_payload = Y @ pi for a payload welded to `body`."""
    Xup = parent_transforms(robot, q)
    v, a = body_motion(robot, q, qd, qdd, Xup=Xup)
    vb, ab = v[body], a[body]
    # Newton-Euler force of each unit-parameter "payload", as columns of a 6x10 matrix.
    F = np.column_stack([B @ ab + crf(vb) @ (B @ vb) for B in _BASIS])
    Y = np.zeros((robot.nv, 10))
    i = body
    while i >= 0:                          # carry the force up the chain, projecting on each joint
        b = robot.bodies[i]
        if b.joint is not None:
            Y[b.joint.dof] = b.joint.S @ F
        if b.parent >= 0:
            F = Xup[i].T @ F
        i = b.parent
    return Y


def identify_payload(robot, body, q, qd, qdd, tau):
    """Batch least-squares estimate of the payload parameters from logged data.

    q, qd, qdd, tau: (N, nv) arrays. Returns (pi_hat, condition number of the stacked regressor).
    A large condition number means the trajectory did not excite some parameters.
    """
    Ys, rs = [], []
    for qk, qdk, qddk, tk in zip(q, qd, qdd, tau):
        Ys.append(payload_regressor(robot, qk, qdk, qddk, body))
        rs.append(tk - rnea(robot, qk, qdk, qddk))
    Y, r = np.vstack(Ys), np.concatenate(rs)
    pi_hat, *_ = np.linalg.lstsq(Y, r, rcond=None)
    return pi_hat, np.linalg.cond(Y)


def with_payload(robot, body, pi):
    """Copy of `robot` with a welded body of parameters pi attached to `body`."""
    m = pi[0]
    com = np.asarray(pi[1:4]) / m
    Io = inertia_from_params(pi)[:3, :3]
    C = skew(com)
    I_com = Io - m * C @ C.T
    payload = Body("payload_estimate", body, np.eye(3), np.zeros(3), m, com, I_com, None)
    return Robot(robot.bodies + [payload], list(robot.sites.values()), robot.gravity)
