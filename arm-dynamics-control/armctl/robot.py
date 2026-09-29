"""Kinematic-tree robot description, built from a compiled MuJoCo model.

MuJoCo is used here only as an XML parser: every algorithm in this package
(kinematics, RNEA, CRBA, ABA, controllers) runs on the arrays extracted below,
never on MuJoCo's own solvers. MuJoCo's solvers are used only as ground truth
in the tests and in scripts/validate_dynamics.py.

Supported: kinematic trees of rigid bodies connected by revolute (hinge) joints,
at most one joint per body, plus welded (joint-less) bodies such as a payload.
"""

from dataclasses import dataclass, field

import mujoco
import numpy as np

from .spatial import quat_to_rot, spatial_inertia


@dataclass
class Joint:
    name: str
    dof: int                 # index into q / qd / tau
    axis: np.ndarray         # unit axis, body coordinates
    anchor: np.ndarray       # point on the axis, body coordinates
    q0: float                # angle at which the body sits at its model pose
    armature: float          # reflected rotor inertia added to M[dof, dof]
    limits: tuple | None     # (lower, upper) or None
    S: np.ndarray = field(init=False)   # motion subspace, see __post_init__

    def __post_init__(self):
        # Rotating about an axis through `anchor` also moves the body origin,
        # with velocity anchor x axis per unit joint rate.
        self.S = np.concatenate([self.axis, np.cross(self.anchor, self.axis)])


@dataclass
class Body:
    name: str
    parent: int              # index into Robot.bodies, -1 for the world
    R0: np.ndarray           # orientation in the parent frame when q == q0
    p0: np.ndarray           # origin in the parent frame when q == q0
    mass: float
    com: np.ndarray          # body coordinates
    I_com: np.ndarray        # 3x3, about the COM, body coordinates
    joint: Joint | None
    inertia: np.ndarray = field(init=False)   # 6x6 spatial inertia about the origin

    def __post_init__(self):
        self.inertia = spatial_inertia(self.mass, self.com, self.I_com)


@dataclass
class Site:
    name: str
    body: int
    pos: np.ndarray          # body coordinates
    R: np.ndarray            # orientation in body coordinates


class Robot:
    def __init__(self, bodies, sites, gravity):
        self.bodies = bodies
        self.sites = {s.name: s for s in sites}
        self.gravity = np.asarray(gravity, dtype=float)
        self.nv = sum(b.joint is not None for b in bodies)
        self.joints = sorted((b.joint for b in bodies if b.joint), key=lambda j: j.dof)
        # Ancestor chain (body indices, leaf first) for each body, used by the Jacobian.
        self._chains = []
        for i in range(len(bodies)):
            chain, j = [], i
            while j >= 0:
                chain.append(j)
                j = bodies[j].parent
            self._chains.append(chain)

    def chain(self, body):
        return self._chains[body]

    @property
    def lower(self):
        return np.array([j.limits[0] if j.limits else -np.inf for j in self.joints])

    @property
    def upper(self):
        return np.array([j.limits[1] if j.limits else np.inf for j in self.joints])

    def random_q(self, rng, span=np.pi):
        """Uniform sample inside the joint limits, clipped to [-span, span]."""
        lo = np.maximum(self.lower, -span)
        hi = np.minimum(self.upper, span)
        return rng.uniform(lo, hi)

    @classmethod
    def from_mujoco(cls, model: mujoco.MjModel):
        if model.nq != model.nv:
            raise NotImplementedError("only hinge-joint models are supported")
        bodies = []
        for b in range(1, model.nbody):            # body 0 is the world
            njnt = model.body_jntnum[b]
            if njnt > 1:
                raise NotImplementedError(f"body {b} has {njnt} joints; at most one supported")
            joint = None
            if njnt == 1:
                j = model.body_jntadr[b]
                if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
                    raise NotImplementedError("only hinge joints are supported")
                dof = model.jnt_dofadr[j]
                axis = model.jnt_axis[j] / np.linalg.norm(model.jnt_axis[j])
                limits = tuple(model.jnt_range[j]) if model.jnt_limited[j] else None
                joint = Joint(
                    name=mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or f"joint{j}",
                    dof=int(dof), axis=axis, anchor=model.jnt_pos[j].copy(),
                    q0=float(model.qpos0[model.jnt_qposadr[j]]),
                    armature=float(model.dof_armature[dof]), limits=limits)
            Ri = quat_to_rot(model.body_iquat[b])
            bodies.append(Body(
                name=mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or f"body{b}",
                parent=int(model.body_parentid[b]) - 1,
                R0=quat_to_rot(model.body_quat[b]), p0=model.body_pos[b].copy(),
                mass=float(model.body_mass[b]), com=model.body_ipos[b].copy(),
                I_com=Ri @ np.diag(model.body_inertia[b]) @ Ri.T, joint=joint))

        sites = [Site(name=mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, s) or f"site{s}",
                      body=int(model.site_bodyid[s]) - 1, pos=model.site_pos[s].copy(),
                      R=quat_to_rot(model.site_quat[s]))
                 for s in range(model.nsite)]

        return cls(bodies, sites, model.opt.gravity.copy())
