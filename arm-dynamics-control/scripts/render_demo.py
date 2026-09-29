"""Render results/demo.gif: operational-space control with the payload unmodelled vs identified.

Needs an OpenGL backend; on a headless Linux machine run with MUJOCO_GL=osmesa (or egl).
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from armctl import UR5E_HOME, load_ur5e  # noqa: E402
from armctl.controllers import ComputedTorque, OperationalSpace  # noqa: E402
from armctl.experiments import (PAYLOAD_KG, SITE, WN, ZETA, circle_trajectory,  # noqa: E402
                                identify_from_log, joint_trajectory)
from armctl.sim import Simulation  # noqa: E402

W, H, FPS = 360, 300, 25
REF_RGBA = np.array([0.55, 0.55, 0.5, 0.9], dtype=np.float32)
TRAIL_RGBA = np.array([0.92, 0.41, 0.2, 1.0], dtype=np.float32)


def add_sphere(scene, pos, radius, rgba):
    if scene.ngeom >= scene.maxgeom:
        return
    mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.array([radius, 0, 0]), np.asarray(pos, dtype=float),
                        np.eye(3).flatten(), rgba)
    scene.ngeom += 1


def render_run(model, controller, traj, label):
    sim = Simulation(model)
    sim.reset(UR5E_HOME)
    renderer = mujoco.Renderer(model, H, W)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [-0.18, 0.3, 0.4]
    cam.distance, cam.azimuth, cam.elevation = 1.3, -115.0, -12.0
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, SITE)
    ref = [traj(t)[1] for t in np.linspace(traj.hold, traj.hold + traj.T / 2, 60)]
    every = int(round(1.0 / (FPS * sim.dt)))
    frames, trail = [], []

    def on_step(k, t, data):
        if k % every:
            return
        trail.append(data.site_xpos[sid].copy())
        renderer.update_scene(data, cam)
        for p in ref:
            add_sphere(renderer.scene, p, 0.005, REF_RGBA)
        for p in trail[-40:]:
            add_sphere(renderer.scene, p, 0.006, TRAIL_RGBA)
        img = Image.fromarray(renderer.render())
        ImageDraw.Draw(img).text((10, 8), label, fill=(20, 20, 20))
        frames.append(img)

    sim.run(controller, traj.duration, on_step=on_step)
    renderer.close()
    return frames


def main():
    model, robot = load_ur5e(PAYLOAD_KG)
    _, nominal = load_ur5e()
    traj = circle_trajectory(nominal)

    # Calibration move with the nominal model, then least-squares payload identification.
    jtraj = joint_trajectory()
    sim = Simulation(model)
    sim.reset(UR5E_HOME)
    log = sim.run(ComputedTorque(jtraj, nominal, WN, ZETA), jtraj.duration + 0.5)
    _, _, identified = identify_from_log(nominal, log, sim.dt, np.random.default_rng(0))

    left = render_run(model, OperationalSpace(traj, nominal, SITE, WN, ZETA), traj,
                      f"Payload unmodelled ({PAYLOAD_KG:g} kg)")
    right = render_run(model, OperationalSpace(traj, identified, SITE, WN, ZETA), traj,
                       "Payload identified from torques")
    frames = []
    for a, b in zip(left, right):
        f = Image.new("RGB", (2 * W, H))
        f.paste(a, (0, 0))
        f.paste(b, (W, 0))
        frames.append(f.quantize(colors=128, method=Image.Quantize.MEDIANCUT))
    out = ROOT / "results" / "demo.gif"
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=int(1000 / FPS), loop=0,
                   optimize=True)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {len(frames)} frames)")


if __name__ == "__main__":
    main()
