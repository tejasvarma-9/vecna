import numpy as np
import pytest

from armctl import UR5E_HOME, jacobian, jacobian_dot_qdot, load_ur5e, site_pose, solve_ik
from armctl.spatial import axis_angle_to_rot, rot_log


@pytest.mark.parametrize("angle", [0.0, 1e-9, 0.3, 2.0, np.pi - 1e-6, np.pi])
def test_rot_log_inverts_rodrigues(angle):
    rng = np.random.default_rng(0)
    for _ in range(20):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        w = rot_log(axis_angle_to_rot(axis, angle))
        # At exactly pi, axis and -axis give the same rotation.
        assert np.allclose(axis_angle_to_rot(w / max(np.linalg.norm(w), 1e-300), np.linalg.norm(w)),
                           axis_angle_to_rot(axis, angle), atol=1e-9)
        assert np.linalg.norm(w) == pytest.approx(angle, abs=1e-9)


def test_jacobian_matches_finite_differences():
    _, robot = load_ur5e()
    rng = np.random.default_rng(1)
    q = robot.random_q(rng)
    J = jacobian(robot, q, "ee")
    R0, p0 = site_pose(robot, q, "ee")
    h = 1e-7
    for k in range(robot.nv):
        dq = np.zeros(robot.nv)
        dq[k] = h
        R1, p1 = site_pose(robot, q + dq, "ee")
        assert np.allclose((p1 - p0) / h, J[:3, k], atol=1e-6)
        assert np.allclose(rot_log(R1 @ R0.T) / h, J[3:, k], atol=1e-6)


def test_jacobian_dot_matches_finite_differences():
    _, robot = load_ur5e()
    rng = np.random.default_rng(2)
    q, qd = robot.random_q(rng), rng.normal(size=6)
    h = 1e-6
    fd = (jacobian(robot, q + h * qd, "ee") - jacobian(robot, q - h * qd, "ee")) / (2 * h) @ qd
    assert np.allclose(jacobian_dot_qdot(robot, q, qd, "ee"), fd, atol=1e-7)


def test_lm_ik_reaches_random_targets():
    _, robot = load_ur5e()
    rng = np.random.default_rng(3)
    for _ in range(20):
        R_d, p_d = site_pose(robot, robot.random_q(rng), "ee")
        res = solve_ik(robot, "ee", R_d, p_d, UR5E_HOME, method="lm", restarts=9, rng=rng)
        assert res.success
        R, p = site_pose(robot, res.q, "ee")
        assert np.linalg.norm(p - p_d) < 1e-5
        assert np.linalg.norm(rot_log(R_d @ R.T)) < 1e-4
