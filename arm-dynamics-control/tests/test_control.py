import numpy as np
import pytest

from armctl import UR5E_HOME, load_ur5e, rnea
from armctl.controllers import ComputedTorque, OperationalSpace
from armctl.experiments import SITE, circle_trajectory, joint_errors, joint_trajectory, rms, task_errors
from armctl.identification import (identify_payload, params_from_body, payload_regressor,
                                   with_payload)
from armctl.sim import Simulation
from armctl.trajectory import quintic_blend


def test_quintic_boundary_conditions():
    assert np.allclose(quintic_blend(0.0), (0, 0, 0))
    assert np.allclose(quintic_blend(1.0), (1, 0, 0))
    s = np.linspace(0, 1, 101)
    b, db, _ = quintic_blend(s)
    assert np.allclose(np.gradient(b, s), db, atol=1e-3)
    assert np.all(np.diff(b) >= 0)


def test_trajectory_derivatives_consistent():
    traj = joint_trajectory()
    h = 1e-6
    for t in np.linspace(0.1, traj.duration - 0.1, 17):
        q0, qd0, qdd0 = traj(t)
        assert np.allclose((traj(t + h)[0] - traj(t - h)[0]) / (2 * h), qd0, atol=1e-5)
        assert np.allclose((traj(t + h)[1] - traj(t - h)[1]) / (2 * h), qdd0, atol=1e-4)


def test_computed_torque_tracks_with_exact_model():
    model, robot = load_ur5e()
    traj = joint_trajectory()
    sim = Simulation(model)
    sim.reset(UR5E_HOME)
    log = sim.run(ComputedTorque(traj, robot), traj.duration)
    assert rms(np.degrees(joint_errors(log, traj))) < 0.02


def test_operational_space_tracks_with_exact_model():
    model, robot = load_ur5e()
    traj = circle_trajectory(robot)
    sim = Simulation(model)
    sim.reset(UR5E_HOME)
    log = sim.run(OperationalSpace(traj, robot, SITE), traj.duration)
    pos, rot = task_errors(log, traj, robot)
    assert rms(pos) < 0.5e-3 and rms(rot) < 1e-3


def _true_payload_params():
    _, with_pl = load_ur5e(3.0)
    pb = with_pl.bodies[-1]
    return pb.parent, params_from_body(pb.mass, pb.p0 + pb.R0 @ pb.com, pb.R0 @ pb.I_com @ pb.R0.T), with_pl


def test_payload_regressor_is_exact():
    _, robot = load_ur5e()
    flange, pi, with_pl = _true_payload_params()
    rng = np.random.default_rng(0)
    for _ in range(10):
        q, qd, qdd = robot.random_q(rng), rng.normal(size=6), rng.normal(size=6)
        Y = payload_regressor(robot, q, qd, qdd, flange)
        assert np.allclose(rnea(robot, q, qd, qdd) + Y @ pi, rnea(with_pl, q, qd, qdd), atol=1e-10)
        assert np.allclose(rnea(with_payload(robot, flange, pi), q, qd, qdd),
                           rnea(with_pl, q, qd, qdd), atol=1e-10)


def test_identification_recovers_payload_from_noiseless_data():
    _, robot = load_ur5e()
    flange, pi, with_pl = _true_payload_params()
    rng = np.random.default_rng(1)
    q = np.array([robot.random_q(rng) for _ in range(40)])
    qd, qdd = rng.normal(size=(2, 40, 6))
    tau = np.array([rnea(with_pl, *x) for x in zip(q, qd, qdd)])
    pi_hat, _ = identify_payload(robot, flange, q, qd, qdd, tau)
    assert pi_hat == pytest.approx(pi, abs=1e-8)
