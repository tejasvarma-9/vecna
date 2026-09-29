"""Experiment definitions shared by the scripts, tests and demo renderer."""

import numpy as np

from .controllers import (ComputedTorque, JacobianTransposePD, JointPD, JointPDGravity,
                          OperationalSpace, bandwidth_matched_joint_gains,
                          bandwidth_matched_task_gains)
from .identification import identify_payload, with_payload
from .kinematics import pose_error, site_pose
from .models import UR5E_HOME, load_ur5e
from .sim import Simulation
from .trajectory import CircleTrajectory, JointTrajectory

SITE = "ee"
PAYLOAD_KG = 3.0            # UR5e is rated for 5 kg
TORQUE_NOISE_NM = 0.5       # std. dev. of simulated joint-torque measurement noise
WN, ZETA = 20.0, 1.0        # closed-loop bandwidth (rad/s) and damping for every controller


def joint_trajectory():
    """Three rest-to-rest moves that exercise all six joints (up to 2.75 rad/s and 5.7 rad/s^2)."""
    offsets = [[1.0, 0.5, -0.8, 0.6, 0.8, 1.0],
               [-0.8, -0.3, 0.6, -0.8, -0.6, -1.2]]
    waypoints = [UR5E_HOME] + [UR5E_HOME + np.array(o) for o in offsets] + [UR5E_HOME]
    return JointTrajectory(waypoints, [1.5, 1.5, 1.5])


def circle_trajectory(robot):
    """Two laps of a vertical circle of 15 cm radius at fixed tool orientation, peak ~0.9 m/s."""
    R, p = site_pose(robot, UR5E_HOME, SITE)
    return CircleTrajectory(p, R, radius=0.15, e1=[1, 0, 0], e2=[0, 0, 1], period=2.0, laps=2)


def joint_errors(log, traj):
    """(N, nv) tracking error in radians."""
    return log["q"] - np.array([traj(t)[0] for t in log["t"]])


def task_errors(log, traj, robot):
    """(N,) position error [m] and (N,) orientation error [rad] of the end effector."""
    pos, rot = [], []
    for t, q in zip(log["t"], log["q"]):
        R_d, p_d, _, _ = traj(t)
        R, p = site_pose(robot, q, SITE)
        e = pose_error(R, p, R_d, p_d)
        pos.append(np.linalg.norm(e[:3]))
        rot.append(np.linalg.norm(e[3:]))
    return np.array(pos), np.array(rot)


def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


def identify_from_log(robot, log, dt, rng, noise=TORQUE_NOISE_NM, stride=10):
    """Identify the payload on the tool flange from one logged run.

    Uses what a real robot would have: noisy joint torques and accelerations from
    numerically differentiating joint velocities (not the simulator's true qacc).
    """
    qdd = np.gradient(log["qd"], dt, axis=0)
    tau = log["tau"] + rng.normal(scale=noise, size=log["tau"].shape)
    sl = slice(1, -1, stride)
    flange = next(i for i, b in enumerate(robot.bodies) if b.name == "wrist_3_link")
    pi_hat, cond = identify_payload(robot, flange, log["q"][sl], log["qd"][sl], qdd[sl], tau[sl])
    return pi_hat, cond, with_payload(robot, flange, pi_hat)


def run_joint_suite(seed=0):
    """PD, PD+g, computed torque; nominal and with an unmodelled payload; then identify it."""
    rng = np.random.default_rng(seed)
    _, robot = load_ur5e()
    traj = joint_trajectory()
    duration = traj.duration + 0.5
    Kp, Kd = bandwidth_matched_joint_gains(robot, UR5E_HOME, WN, ZETA)
    results = {}
    for payload in (0.0, PAYLOAD_KG):
        sim = Simulation(load_ur5e(payload)[0])
        for ctrl in (JointPD(traj, Kp, Kd), JointPDGravity(traj, Kp, Kd, robot),
                     ComputedTorque(traj, robot, WN, ZETA)):
            sim.reset(UR5E_HOME)
            results[(payload, ctrl.name)] = sim.run(ctrl, duration)
    # Identify the payload from the computed-torque run, then re-run with the updated model.
    sim = Simulation(load_ur5e(PAYLOAD_KG)[0])
    pi_hat, cond, robot_id = identify_from_log(robot, results[(PAYLOAD_KG, ComputedTorque.name)], sim.dt, rng)
    sim.reset(UR5E_HOME)
    results[(PAYLOAD_KG, "Computed torque + identified payload")] = sim.run(
        ComputedTorque(traj, robot_id, WN, ZETA), duration)
    return traj, robot, results, pi_hat, cond, robot_id


def run_task_suite(robot_id):
    """Jacobian-transpose PD vs operational space; nominal, payload, identified payload.

    robot_id is the model with the payload identified during the joint-space
    suite: identification is a one-off calibration move, and the circle itself
    (fixed tool orientation) would excite the payload's inertia poorly.
    """
    _, robot = load_ur5e()
    traj = circle_trajectory(robot)
    Kp, Kd = bandwidth_matched_task_gains(robot, UR5E_HOME, SITE, WN, ZETA)
    results = {}
    for payload in (0.0, PAYLOAD_KG):
        sim = Simulation(load_ur5e(payload)[0])
        for ctrl in (JacobianTransposePD(traj, robot, SITE, Kp, Kd),
                     OperationalSpace(traj, robot, SITE, WN, ZETA)):
            sim.reset(UR5E_HOME)
            results[(payload, ctrl.name)] = sim.run(ctrl, traj.duration)
    sim = Simulation(load_ur5e(PAYLOAD_KG)[0])
    sim.reset(UR5E_HOME)
    results[(PAYLOAD_KG, "Operational space + identified payload")] = sim.run(
        OperationalSpace(traj, robot_id, SITE, WN, ZETA), traj.duration)
    return traj, robot, results
