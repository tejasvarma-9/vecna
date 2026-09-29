# Resume update for robotics placements

## 1. Replace the duplicate project entries

Delete these two Projects entries. They repeat the Experience section almost word for word:

- ~~UI Automation Framework for Adobe Bridge~~ (same as the Adobe internship)
- ~~In-Hand Dexterous Rotation Research~~ (same as the RRC role)

Put this in first position under Projects:

**Manipulator Dynamics, Identification & Control from Scratch** | *Python, NumPy, MuJoCo* | [GitHub link]
- Implemented rigid-body dynamics (RNEA, CRBA, Articulated Body Algorithm) and kinematics for a 6-DOF UR5e-style arm from first principles; matched MuJoCo to within 1e-11 across 2,000 states and 50 random branching robots.
- Compared five torque controllers at equal bandwidth in 1 kHz closed-loop simulation; computed-torque control cut joint tracking error from 1.49° (PD) to 0.005° RMS.
- Identified an unknown 3 kg payload's mass (±1 g) and centre of mass (±2 mm) by least squares on noisy joint torques, restoring operational-space tracking from 25 mm to 0.13 mm RMS.
- Built a Levenberg–Marquardt IK solver with random restarts that solved 100% of 300 random 6D targets, vs 41% for pseudo-inverse Newton, with bounded steps near singularities.

If an extension from `INTERVIEW_PREP.md` §E gets done, add it as a fifth bullet in the
same shape (what was built, then the number). For example: "Ported RNEA to C++ (Eigen,
pybind11): 300 µs → X µs per call."

LaTeX, if the resume uses the usual `\resumeProjectHeading` / `\resumeItem` macros:

```latex
\resumeProjectHeading
  {\textbf{Manipulator Dynamics, Identification \& Control from Scratch} $|$ \emph{Python, NumPy, MuJoCo}}{}
  \resumeItemListStart
    \resumeItem{Implemented rigid-body dynamics (RNEA, CRBA, Articulated Body Algorithm) and kinematics for a 6-DOF UR5e-style arm from first principles; matched MuJoCo to within $10^{-11}$ across 2,000 states and 50 random branching robots.}
    \resumeItem{Compared five torque controllers at equal bandwidth in 1\,kHz closed-loop simulation; computed-torque control cut joint tracking error from 1.49\textdegree{} (PD) to 0.005\textdegree{} RMS.}
    \resumeItem{Identified an unknown 3\,kg payload's mass ($\pm$1\,g) and centre of mass ($\pm$2\,mm) by least squares on noisy joint torques, restoring operational-space tracking from 25\,mm to 0.13\,mm RMS.}
    \resumeItem{Built a Levenberg--Marquardt IK solver with random restarts that solved 100\% of 300 random 6D targets, vs 41\% for pseudo-inverse Newton, with bounded steps near singularities.}
  \resumeItemListEnd
```

(`\textdegree` needs `\usepackage{textcomp}`, which is loaded by default in recent LaTeX.)

## 2. Rewrite the RRC research bullets (only she can fill these in)

Right now they say "contributed to" and "worked on", which tells a robotics interviewer
nothing. Use this shape, with her real details:

- Implemented [method, e.g. RL policy / trajectory optimisation / impedance controller] for in-hand cube reorientation on [hand / simulator, e.g. Allegro, LEAP, Isaac Gym, MuJoCo]; achieved [X]% success over [N] trials / [Y]° orientation error.
- Built [specific pipeline piece, e.g. state estimation from ..., sim-to-real ..., contact-rich planner ...] in [tools]; [measurable effect].

If the lab uses ROS, PyTorch or a simulator, those belong in Skills. At the moment the
resume lists none of them.

## 3. Other fixes from the earlier review

- Add a phone number to the header; move "IIIT Hyderabad" out of the header (it's already under Education).
- Show the LinkedIn and GitHub URLs as text, so they survive printing.
- Merge the two TA roles into one line.
- Drop Relevant Coursework (two lines of little signal) to make room.
- Add `\hyphenpenalty=10000` so words aren't split across lines (it breaks ATS keyword matching).
- Skills: add MuJoCo. Remove anything she can't talk about for 5 minutes (XFLR5, Fusion360 and Simulink are candidates for a robotics-software role).
- Move the Adobe Playwright work lower, or rephrase it towards software-engineering rigour. For robotics roles it's the least relevant item, and it takes up the most space.
