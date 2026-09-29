"""Compare this package's algorithms against MuJoCo's C implementations (the ground truth)."""

import mujoco
import numpy as np

from .dynamics import aba, crba, rnea
from .kinematics import jacobian, jacobian_dot_qdot, site_pose

CHECKS = ("fk", "jacobian", "jdot_qdot", "rnea", "crba", "aba")


def compare_state(model, data, robot, site, q, qd, qdd, tau):
    """Max absolute error of each quantity against MuJoCo at one state."""
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)
    nv = model.nv
    data.qpos[:], data.qvel[:], data.ctrl[:] = q, qd, 0.0
    data.qfrc_applied[:] = tau
    mujoco.mj_forward(model, data)          # kinematics, M, bias, and qacc = M^-1 (tau - h)

    R, p = site_pose(robot, q, site)
    jacp, jacr = np.zeros((3, nv)), np.zeros((3, nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, sid)
    jdp, jdr = np.zeros((3, nv)), np.zeros((3, nv))
    mujoco.mj_jacDot(model, data, jdp, jdr, data.site_xpos[sid], model.site_bodyid[sid])
    M = np.zeros((nv, nv))
    mujoco.mj_fullM(model, data, M)
    qacc = data.qacc.copy()

    data.qacc[:] = qdd
    mujoco.mj_inverse(model, data)           # qfrc_inverse = M qdd + h

    return {
        "fk": max(np.abs(p - data.site_xpos[sid]).max(),
                  np.abs(R - data.site_xmat[sid].reshape(3, 3)).max()),
        "jacobian": np.abs(jacobian(robot, q, site) - np.vstack([jacp, jacr])).max(),
        "jdot_qdot": np.abs(jacobian_dot_qdot(robot, q, qd, site)
                            - np.concatenate([jdp @ qd, jdr @ qd])).max(),
        "rnea": np.abs(rnea(robot, q, qd, qdd) - data.qfrc_inverse).max(),
        "crba": np.abs(crba(robot, q) - M).max(),
        "aba": np.abs(aba(robot, q, qd, tau) - qacc).max(),
    }


def compare_random_states(model, robot, site, rng, n_states):
    """Worst-case error of each check over n_states random (q, qd, qdd, tau)."""
    data = mujoco.MjData(model)
    worst = dict.fromkeys(CHECKS, 0.0)
    for _ in range(n_states):
        q = robot.random_q(rng)
        qd, qdd = rng.normal(scale=2.0, size=(2, robot.nv))
        tau = rng.normal(scale=10.0, size=robot.nv)
        for k, v in compare_state(model, data, robot, site, q, qd, qdd, tau).items():
            worst[k] = max(worst[k], float(v))
    return worst
