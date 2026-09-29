"""MuJoCo simulation loop for torque-controlled arms.

MuJoCo plays the role of the real robot: the controller computes torques from
its own model, the torques are clipped to the actuator limits, and MuJoCo
integrates the true dynamics (which may include an unmodelled payload).
"""

import mujoco
import numpy as np


class Simulation:
    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.data = mujoco.MjData(model)
        self.dt = model.opt.timestep
        # Actuator a drives dof act_dof[a]; limits are the ctrlrange (gear 1 motors).
        self.act_dof = np.array([model.jnt_dofadr[model.actuator_trnid[a, 0]] for a in range(model.nu)])
        self.limits = np.full(model.nv, np.inf)
        for a, dof in enumerate(self.act_dof):
            if model.actuator_ctrllimited[a]:
                self.limits[dof] = np.max(np.abs(model.actuator_ctrlrange[a]))

    def reset(self, q, qd=None):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = q
        self.data.qvel[:] = 0.0 if qd is None else qd
        mujoco.mj_forward(self.model, self.data)

    def run(self, controller, duration, on_step=None):
        """Run closed loop for `duration` seconds at the model timestep. Returns a log dict."""
        n = int(round(duration / self.dt))
        nv = self.model.nv
        log = {k: np.zeros((n, nv)) for k in ("q", "qd", "tau_cmd", "tau")}
        log["t"] = np.arange(n) * self.dt
        d = self.data
        for k in range(n):
            t = k * self.dt
            q, qd = d.qpos.copy(), d.qvel.copy()
            tau_cmd = controller(t, q, qd)
            tau = np.clip(tau_cmd, -self.limits, self.limits)
            d.ctrl[:] = tau[self.act_dof]
            log["q"][k], log["qd"][k], log["tau_cmd"][k], log["tau"][k] = q, qd, tau_cmd, tau
            if on_step is not None:
                on_step(k, t, d)
            mujoco.mj_step(self.model, d)
        log["saturated"] = np.any(np.abs(log["tau_cmd"]) > self.limits + 1e-9, axis=1)
        return log
