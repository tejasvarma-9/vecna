"""Model loading: the UR5e-style arm (optionally carrying a payload) and random kinematic trees."""

from pathlib import Path

import mujoco
import numpy as np

from .robot import Robot

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
UR5E_XML = MODEL_DIR / "ur5e_style.xml"
UR5E_HOME = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])


def ur5e_xml(payload_kg=0.0, payload_offset=0.05):
    """UR5e-style MJCF, with a welded point-like payload `payload_offset` m past the tool flange."""
    xml = UR5E_XML.read_text()
    if payload_kg > 0:
        # Treat the payload as a 10 cm cube of uniform density.
        i = payload_kg * (0.1 ** 2) / 6.0
        body = (f'<body name="payload" pos="0 {0.1 + payload_offset} 0">'
                f'<inertial pos="0 0 0" mass="{payload_kg}" diaginertia="{i} {i} {i}"/>'
                f'<geom type="box" size="0.05 0.05 0.05" rgba="0.85 0.35 0.2 1"/></body>')
        xml = xml.replace("<!-- PAYLOAD -->", body)
    return xml


def load_ur5e(payload_kg=0.0):
    """(MjModel, Robot) pair for the UR5e-style arm."""
    model = mujoco.MjModel.from_xml_string(ur5e_xml(payload_kg))
    return model, Robot.from_mujoco(model)


def _random_quat(rng):
    q = rng.normal(size=4)
    return q / np.linalg.norm(q)


def random_tree_xml(rng, n_bodies=7, branch_prob=0.3):
    """MJCF for a random kinematic tree, used to test the algorithms beyond one robot.

    Every body gets a random parent (so the tree can branch), random offset and
    orientation, a random hinge axis through a random (non-zero) anchor, a random
    mass/COM/rotated inertia, and random armature. Nothing about the UR5e layout
    (zero anchors, axis-aligned joints, a serial chain) is assumed.
    """
    parents = [-1]
    for i in range(1, n_bodies):
        parents.append(int(rng.integers(0, i)) if rng.random() < branch_prob else i - 1)

    def fmt(v):
        return " ".join(f"{x:.6f}" for x in v)

    def body_xml(i):
        moments = rng.uniform(0.8, 1.2, size=3) * rng.uniform(0.01, 0.1)   # satisfies triangle inequality
        children = "".join(body_xml(c) for c in range(n_bodies) if parents[c] == i)
        return (
            f'<body name="b{i}" pos="{fmt(rng.uniform(-0.3, 0.3, 3))}" quat="{fmt(_random_quat(rng))}">'
            f'<inertial pos="{fmt(rng.uniform(-0.1, 0.1, 3))}" quat="{fmt(_random_quat(rng))}" '
            f'mass="{rng.uniform(0.5, 3.0):.6f}" diaginertia="{fmt(moments)}"/>'
            f'<joint name="j{i}" type="hinge" axis="{fmt(rng.normal(size=3))}" '
            f'pos="{fmt(rng.uniform(-0.1, 0.1, 3))}" armature="{rng.uniform(0, 0.2):.6f}"/>'
            f'<site name="s{i}" pos="{fmt(rng.uniform(-0.2, 0.2, 3))}" quat="{fmt(_random_quat(rng))}"/>'
            f'{children}</body>')

    return ('<mujoco model="random_tree"><option gravity="0 0 -9.81">'
            '<flag contact="disable"/></option><worldbody>'
            f'{body_xml(0)}</worldbody></mujoco>')


def load_random_tree(rng, n_bodies=7):
    model = mujoco.MjModel.from_xml_string(random_tree_xml(rng, n_bodies))
    return model, Robot.from_mujoco(model)
