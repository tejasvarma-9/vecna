"""Torque controllers. Each is a callable (t, q, qd) -> tau.

Joint space (track q_d(t)):
  JointPD              tau = Kp e + Kd edot                       no model
  JointPDGravity       tau = Kp e + Kd edot + g(q)                gravity model only
  ComputedTorque       tau = M(q)(qdd_d + Kp e + Kd edot) + h     full model

Task space (track an end-effector pose):
  JacobianTransposePD  tau = J^T (Kp e + Kd edot) + g(q)          gravity model only
  OperationalSpace     tau = J^T Lambda (xdd_d + Kp e + Kd edot - Jdot qd) + h   full model

The controllers only ever see the model they are constructed with. Passing a
robot that differs from the simulated one (e.g. no payload) is how the
robustness experiments inject model error.
"""

import numpy as np

from .dynamics import crba, gravity_torques, parent_transforms, rnea
from .kinematics import forward_kinematics, jacobian, jacobian_dot_qdot, pose_error, site_pose


class JointPD:
    name = "PD"

    def __init__(self, traj, Kp, Kd):
        self.traj, self.Kp, self.Kd = traj, np.asarray(Kp), np.asarray(Kd)

    def feedback(self, t, q, qd):
        q_d, qd_d, _ = self.traj(t)
        return self.Kp * (q_d - q) + self.Kd * (qd_d - qd)

    def __call__(self, t, q, qd):
        return self.feedback(t, q, qd)


class JointPDGravity(JointPD):
    name = "PD + gravity comp."

    def __init__(self, traj, Kp, Kd, robot):
        super().__init__(traj, Kp, Kd)
        self.robot = robot

    def __call__(self, t, q, qd):
        return self.feedback(t, q, qd) + gravity_torques(self.robot, q)


class ComputedTorque:
    """Feedback linearisation: closed loop becomes e'' + Kd e' + Kp e = 0 per joint.

    M(q) v + h(q, qd) is exactly inverse dynamics evaluated at qdd = v, so the
    whole control law is a single O(n) RNEA call; M is never formed.
    """
    name = "Computed torque"

    def __init__(self, traj, robot, wn=20.0, zeta=1.0):
        self.traj, self.robot = traj, robot
        self.Kp, self.Kd = wn**2, 2 * zeta * wn

    def __call__(self, t, q, qd):
        q_d, qd_d, qdd_d = self.traj(t)
        v = qdd_d + self.Kp * (q_d - q) + self.Kd * (qd_d - qd)
        return rnea(self.robot, q, qd, v)


class JacobianTransposePD:
    name = "Jacobian-transpose PD"

    def __init__(self, traj, robot, site, Kp, Kd):
        self.traj, self.robot, self.site = traj, robot, site
        self.Kp, self.Kd = np.asarray(Kp), np.asarray(Kd)

    def __call__(self, t, q, qd):
        R_d, p_d, xd_d, _ = self.traj(t)
        poses = forward_kinematics(self.robot, q)
        R, p = site_pose(self.robot, q, self.site, poses)
        J = jacobian(self.robot, q, self.site, poses)
        F = self.Kp * pose_error(R, p, R_d, p_d) + self.Kd * (xd_d - J @ qd)
        return J.T @ F + gravity_torques(self.robot, q)


class OperationalSpace:
    """Khatib's operational-space control for a 6D end-effector task.

    With Lambda = (J M^-1 J^T)^-1 and an exact model, the task acceleration equals
    the commanded one: xdd = xdd_d + Kp e + Kd edot. The arm has 6 joints and the
    task is 6D, so there is no null space to control here (a 7-DOF arm would add one).
    """
    name = "Operational space"

    def __init__(self, traj, robot, site, wn=20.0, zeta=1.0):
        self.traj, self.robot, self.site = traj, robot, site
        self.Kp, self.Kd = wn**2, 2 * zeta * wn

    def __call__(self, t, q, qd):
        R_d, p_d, xd_d, xdd_d = self.traj(t)
        robot = self.robot
        Xup = parent_transforms(robot, q)
        poses = forward_kinematics(robot, q)
        R, p = site_pose(robot, q, self.site, poses)
        J = jacobian(robot, q, self.site, poses)
        M = crba(robot, q, Xup)
        h = rnea(robot, q, qd, np.zeros(robot.nv), Xup=Xup)
        xdd_cmd = xdd_d + self.Kp * pose_error(R, p, R_d, p_d) + self.Kd * (xd_d - J @ qd)
        Lambda = np.linalg.inv(J @ np.linalg.solve(M, J.T))
        F = Lambda @ (xdd_cmd - jacobian_dot_qdot(robot, q, qd, self.site, poses))
        return J.T @ F + h


def bandwidth_matched_joint_gains(robot, q, wn=20.0, zeta=1.0):
    """PD gains giving each joint roughly the same bandwidth as computed torque at pose q.

    Kp_i = M_ii(q) wn^2, Kd_i = M_ii(q) 2 zeta wn. This makes the comparison fair:
    every controller has the same nominal stiffness, so the differences come from
    model-based compensation, not from one controller simply having bigger gains.
    """
    m = np.diag(crba(robot, q))
    return m * wn**2, m * 2 * zeta * wn


def bandwidth_matched_task_gains(robot, q, site, wn=20.0, zeta=1.0):
    """Task-space analogue: scale by the diagonal of the task-space inertia Lambda(q)."""
    J = jacobian(robot, q, site)
    lam = np.diag(np.linalg.inv(J @ np.linalg.solve(crba(robot, q), J.T)))
    return lam * wn**2, lam * 2 * zeta * wn
