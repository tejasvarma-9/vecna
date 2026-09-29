"""Reference trajectories with analytic velocities and accelerations."""

import numpy as np


def quintic_blend(s):
    """Rest-to-rest blend b(s) on s in [0, 1] with b, b', b'' = (0, 0, 0) -> (1, 0, 0).

    Six boundary conditions need six coefficients, hence a 5th-order polynomial:
    b(s) = 10 s^3 - 15 s^4 + 6 s^5. Returns (b, db/ds, d2b/ds2).
    """
    s = np.clip(s, 0.0, 1.0)
    return (10 * s**3 - 15 * s**4 + 6 * s**5,
            30 * s**2 - 60 * s**3 + 30 * s**4,
            60 * s - 180 * s**2 + 120 * s**3)


class JointTrajectory:
    """Rest-to-rest quintic segments through joint-space waypoints."""

    def __init__(self, waypoints, durations):
        self.waypoints = [np.asarray(w, dtype=float) for w in waypoints]
        self.durations = list(durations)
        assert len(self.durations) == len(self.waypoints) - 1
        self.starts = np.concatenate([[0.0], np.cumsum(self.durations)])

    @property
    def duration(self):
        return float(self.starts[-1])

    def __call__(self, t):
        """(q, qd, qdd) at time t; holds the final waypoint afterwards."""
        k = int(np.clip(np.searchsorted(self.starts, t, side="right") - 1, 0, len(self.durations) - 1))
        T = self.durations[k]
        q0, q1 = self.waypoints[k], self.waypoints[k + 1]
        b, db, ddb = quintic_blend((t - self.starts[k]) / T)
        dq = q1 - q0
        return q0 + b * dq, db / T * dq, ddb / T**2 * dq


class CircleTrajectory:
    """End-effector circle at fixed orientation, starting and ending at rest.

    p(t) = c + r (cos th e1 + sin th e2), where the phase th(t) follows a quintic
    blend from 0 to 2*pi*laps, so velocity and acceleration are zero at both ends.
    Returns (R_d, p_d, xd_d, xdd_d) with task vectors ordered [linear; angular].
    """

    def __init__(self, start, R, radius, e1, e2, period, laps=1, hold=0.5):
        self.e1 = np.asarray(e1, dtype=float) / np.linalg.norm(e1)
        self.e2 = np.asarray(e2, dtype=float) / np.linalg.norm(e2)
        self.center = np.asarray(start, dtype=float) - radius * self.e1   # th = 0 is `start`
        self.R, self.radius, self.hold = R, radius, hold
        self.T = period * laps
        self.total_angle = 2 * np.pi * laps

    @property
    def duration(self):
        return self.hold + self.T + self.hold

    def __call__(self, t):
        b, db, ddb = quintic_blend((t - self.hold) / self.T)
        th = self.total_angle * b
        thd = self.total_angle * db / self.T
        thdd = self.total_angle * ddb / self.T**2
        radial = np.cos(th) * self.e1 + np.sin(th) * self.e2
        tangent = -np.sin(th) * self.e1 + np.cos(th) * self.e2
        p = self.center + self.radius * radial
        v = self.radius * thd * tangent
        a = self.radius * (thdd * tangent - thd**2 * radial)
        zero = np.zeros(3)
        return self.R, p, np.concatenate([v, zero]), np.concatenate([a, zero])
