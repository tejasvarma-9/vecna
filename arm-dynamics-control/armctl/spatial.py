"""Rotation utilities and 6D spatial vector algebra.

Spatial conventions follow Featherstone, "Rigid Body Dynamics Algorithms" (2008):
  * motion vectors (velocity, acceleration) are [angular; linear]
  * force vectors (wrenches) are [moment; force]
  * everything is expressed in body-fixed coordinates unless stated otherwise

Task-space quantities in kinematics.py use the robotics convention [linear; angular]
instead, because that is what position/orientation errors look like.
"""

import numpy as np


def skew(v):
    """3x3 matrix such that skew(a) @ b == np.cross(a, b)."""
    return np.array([[0.0, -v[2], v[1]],
                     [v[2], 0.0, -v[0]],
                     [-v[1], v[0], 0.0]])


def quat_to_rot(q):
    """Rotation matrix from a (w, x, y, z) quaternion (MuJoCo's ordering)."""
    w, x, y, z = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def axis_angle_to_rot(axis, angle):
    """Rodrigues' formula for a unit axis."""
    K = skew(axis)
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def rot_log(R):
    """Rotation vector (axis * angle) of R, i.e. the inverse of axis_angle_to_rot.

    Handles the two numerically delicate cases: angle ~ 0 and angle ~ pi.
    """
    # vee = 2 sin(theta) * axis. atan2 of (sin, cos) stays accurate at every angle,
    # unlike arccos(cos), which loses about half the digits near 0 and pi.
    vee = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    cos_theta = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arctan2(0.5 * np.linalg.norm(vee), cos_theta)
    if theta < 1e-7:
        return 0.5 * vee
    if np.pi - theta < 1e-4:
        # Near pi, sin(theta) ~ 0, so recover the axis from the symmetric part:
        # (R + R^T)/2 = cos(theta) I + (1 - cos(theta)) a a^T.
        aaT = (0.5 * (R + R.T) - cos_theta * np.eye(3)) / (1.0 - cos_theta)
        k = int(np.argmax(np.diag(aaT)))
        axis = aaT[:, k] / np.linalg.norm(aaT[:, k])
        # Fix the sign so the (small) antisymmetric part agrees with the axis.
        if axis @ vee < 0:
            axis = -axis
        return theta * axis
    return theta / (2.0 * np.sin(theta)) * vee


def plucker(E, p):
    """Motion transform B_X_A.

    E: 3x3 rotation taking A coordinates to B coordinates (= R_AB^T).
    p: position of B's origin, in A coordinates.
    """
    X = np.zeros((6, 6))
    X[:3, :3] = E
    X[3:, 3:] = E
    X[3:, :3] = -E @ skew(p)
    return X


def crm(v):
    """Spatial cross product for motion vectors: crm(v) @ m == v x m."""
    X = np.zeros((6, 6))
    X[:3, :3] = skew(v[:3])
    X[3:, :3] = skew(v[3:])
    X[3:, 3:] = skew(v[:3])
    return X


def crf(v):
    """Spatial cross product for force vectors: crf(v) @ f == v x* f."""
    return -crm(v).T


def spatial_inertia(mass, com, I_com):
    """6x6 spatial inertia about the body origin.

    mass: scalar, com: centre of mass in body coordinates,
    I_com: 3x3 rotational inertia about the COM in body coordinates.
    """
    C = skew(com)
    I = np.zeros((6, 6))
    I[:3, :3] = I_com + mass * C @ C.T
    I[:3, 3:] = mass * C
    I[3:, :3] = mass * C.T
    I[3:, 3:] = mass * np.eye(3)
    return I
