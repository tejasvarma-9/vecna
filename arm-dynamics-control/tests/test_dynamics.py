import numpy as np
import pytest

from armctl import aba, crba, forward_kinematics, load_random_tree, load_ur5e, rnea
from armctl.validation import compare_random_states

TOL = 1e-9


@pytest.mark.parametrize("payload", [0.0, 3.0])
def test_ur5e_matches_mujoco(payload):
    model, robot = load_ur5e(payload)
    worst = compare_random_states(model, robot, "ee", np.random.default_rng(0), 50)
    assert max(worst.values()) < TOL, worst


def test_random_trees_match_mujoco():
    rng = np.random.default_rng(1)
    for _ in range(10):
        n = int(rng.integers(3, 10))
        model, robot = load_random_tree(rng, n)
        worst = compare_random_states(model, robot, f"s{n - 1}", rng, 5)
        assert max(worst.values()) < TOL, worst


def test_mass_matrix_symmetric_positive_definite():
    _, robot = load_ur5e()
    rng = np.random.default_rng(2)
    for _ in range(20):
        M = crba(robot, robot.random_q(rng))
        assert np.allclose(M, M.T)
        assert np.linalg.eigvalsh(M).min() > 0


def test_forward_and_inverse_dynamics_are_inverses():
    _, robot = load_ur5e()
    rng = np.random.default_rng(3)
    for _ in range(20):
        q, qd, tau = robot.random_q(rng), rng.normal(size=6), rng.normal(size=6) * 20
        assert np.allclose(rnea(robot, q, qd, aba(robot, q, qd, tau)), tau, atol=1e-9)


def test_power_balance():
    """d/dt (kinetic + potential energy) equals the power input qd . tau."""
    _, robot = load_ur5e()
    rng = np.random.default_rng(4)
    q, qd, tau = robot.random_q(rng), rng.normal(size=6), rng.normal(size=6) * 10
    qdd = aba(robot, q, qd, tau)

    def energy(q, qd):
        # Kinetic energy (M includes rotor armature) plus potential energy of every COM.
        V = sum(-b.mass * robot.gravity @ (p + R @ b.com)
                for b, (R, p) in zip(robot.bodies, forward_kinematics(robot, q)))
        return 0.5 * qd @ crba(robot, q) @ qd + V

    h = 1e-6
    dE = (energy(q + h * qd, qd + h * qdd) - energy(q - h * qd, qd - h * qdd)) / (2 * h)
    assert dE == pytest.approx(qd @ tau, rel=1e-6, abs=1e-6)
