# Theory notes

The derivations behind each module, in the notation the code uses.
Primary references: Featherstone (2008) for dynamics, Lynch & Park (2017) and
Siciliano et al. (2009) for kinematics and control.

## 1. Spatial vectors (`spatial.py`)

A spatial **motion** vector stacks angular and linear velocity, $\hat v = [\omega;\ v_O]$,
where $v_O$ is the velocity of the body-fixed point currently at the frame origin $O$.
A spatial **force** vector stacks moment and force, $\hat f = [n_O;\ f]$.

Changing coordinates from frame $A$ to frame $B$, where $B$ has rotation $E$ (A→B) and
origin $r$ (in A coordinates), uses the Plücker transform

$$
{}^B X_A = \begin{bmatrix} E & 0 \\ -E\,[r]_\times & E \end{bmatrix},
\qquad {}^A X_B^{*} = ({}^B X_A)^\top \text{ for forces.}
$$

The second identity is why the backward pass of every algorithm uses `Xup[i].T`.

The spatial cross products are $\hat v \times = \begin{bmatrix}[\omega]_\times & 0\\ [v]_\times & [\omega]_\times\end{bmatrix}$
(motion) and $\hat v \times^{*} = -(\hat v\times)^\top$ (force). The spatial inertia of a body
with mass $m$, centre of mass $c$ and rotational inertia $\bar I_c$ about the COM is

$$
I = \begin{bmatrix} \bar I_c + m[c]_\times[c]_\times^\top & m[c]_\times \\ m[c]_\times^\top & m\mathbb 1 \end{bmatrix}.
$$

A revolute joint with unit axis $a$ through the point $p$ (both in child coordinates) has
motion subspace $S = [a;\ p \times a]$. The second block is non-zero when the axis does
not pass through the origin, which the random-tree tests exercise.

## 2. Kinematics (`kinematics.py`)

**Forward kinematics** composes each body's fixed offset with its joint rotation about
an axis through the anchor $p$: $R = R_0 R_j(q)$, $\ t = t_0 + R_0 (p - R_j p)$.

**Geometric Jacobian** of a point $x_e$, column $k$ for revolute joint $k$ with world axis
$z_k$ and anchor $a_k$: $J_k = [z_k \times (x_e - a_k);\ z_k]$, rows ordered [linear; angular].

**$\dot J\dot q$** is needed by operational-space control, because $\ddot x = J\ddot q + \dot J \dot q$.
Each axis is fixed in its own body, so $\dot z_k = \omega_k \times z_k$, and

$$
\frac{d}{dt}\big(z_k \times (x_e - a_k)\big) = \dot z_k \times (x_e - a_k) + z_k \times (\dot x_e - \dot a_k),
$$

with $\omega_k = \sum_{m \le k} z_m \dot q_m$ and $\dot a_k = \sum_{m<k} z_m \times (a_k - a_m)\dot q_m$. The code sums these and
checks the result against both MuJoCo's `mj_jacDot` and a finite difference.

**Orientation error** uses the rotation vector $e_R = \log(R_d R^\top)$ (world frame), which
`rot_log` computes with `atan2` so it stays accurate near 0 and π.

## 3. Inverse kinematics

Each iteration solves a linearised problem $J\,\Delta q \approx e$.

- **Newton / pseudo-inverse:** $\Delta q = J^{+} e$. With $J = U\Sigma V^\top$, the gain along
  each singular direction is $1/\sigma_i$, which is unbounded as $\sigma_i \to 0$.
- **Damped least squares:** minimise $\|J\Delta q - e\|^2 + \lambda^2\|\Delta q\|^2$, giving
  $\Delta q = J^\top (JJ^\top + \lambda^2 I)^{-1} e$ with gain $\sigma_i/(\sigma_i^2 + \lambda^2) \le 1/(2\lambda)$.
  The step stays bounded, but convergence is slow along directions where $\sigma_i \ll \lambda$.
- **Levenberg–Marquardt, error-damped (Sugihara 2011):** $(J^\top J + \lambda I)\Delta q = J^\top e$
  with $\lambda = \tfrac12\|e\|^2 + \bar\lambda$. Damping is large far from the target and
  vanishes near it, recovering Newton's fast final convergence.

IK is a local method, so random restarts handle local minima and starts that clamp
against joint limits.

## 4. Inverse dynamics: RNEA (`dynamics.rnea`)

Forward pass, root to leaves ($\lambda$ = parent):

$$
\hat v_i = {}^iX_\lambda \hat v_\lambda + S_i\dot q_i,\qquad
\hat a_i = {}^iX_\lambda \hat a_\lambda + S_i\ddot q_i + \hat v_i \times S_i \dot q_i,\qquad
\hat f_i = I_i \hat a_i + \hat v_i \times^{*} I_i \hat v_i .
$$

Backward pass, leaves to root: $\tau_i = S_i^\top \hat f_i + b_i \ddot q_i$ (armature $b_i$), then
$\hat f_\lambda \mathrel{+}= {}^iX_\lambda^\top \hat f_i$.

Gravity enters as a fictitious upward base acceleration, $\hat a_0 = [0;\ -g]$, so no body
needs a separate gravity term. Cost is O(n).

## 5. Mass matrix: CRBA (`dynamics.crba`)

The composite inertia of the subtree rooted at $i$ is
$I^c_i = I_i + \sum_{j \in \text{children}(i)} {}^jX_i^\top I^c_j\, {}^jX_i$. Then, with $F = I^c_i S_i$,
$M_{ii} = S_i^\top F + b_i$, and walking $j$ up towards the root carrying
$F \leftarrow {}^jX_{\lambda(j)}^\top F$ gives $M_{ij} = M_{ji} = S_j^\top F$. Cost is O(n²) for a chain.

## 6. Forward dynamics: ABA (`dynamics.aba`)

The Articulated Body Algorithm solves $M\ddot q = \tau - h$ in O(n) without forming M.

1. Compute velocities, $c_i = \hat v_i \times S_i\dot q_i$, and bias forces $p_i = \hat v_i \times^{*} I_i \hat v_i$.
2. From the leaves to the root, set $U_i = I^A_i S_i$, $d_i = S_i^\top U_i + b_i$ and $u_i = \tau_i - S_i^\top p^A_i$.
   Pass $I^a = I^A_i - U_iU_i^\top/d_i$ and $p^a = p^A_i + I^a c_i + U_i u_i/d_i$ to the parent through ${}^iX_\lambda$.
3. From the root to the leaves, set $\hat a' = {}^iX_\lambda \hat a_\lambda + c_i$, $\ \ddot q_i = (u_i - U_i^\top \hat a')/d_i$ and $\hat a_i = \hat a' + S_i \ddot q_i$.

Armature enters only through $d_i$. Consistency check: `rnea(q, qd, aba(q, qd, tau)) == tau`.

## 7. Control (`controllers.py`)

Model: $M(q)\ddot q + C(q,\dot q)\dot q + g(q) = \tau$, with $h = C\dot q + g$ and $e = q_d - q$.

**PD + gravity compensation.** For a constant set-point, take
$V = \tfrac12 \dot q^\top M \dot q + \tfrac12 e^\top K_p e$. Using the skew-symmetry of $\dot M - 2C$,
the law $\tau = K_p e - K_d\dot q + g$ gives $\dot V = -\dot q^\top K_d \dot q \le 0$, and LaSalle's principle
gives $e \to 0$. The guarantee covers regulation only: while *tracking* a moving
reference, the uncompensated $M\ddot q_d + C\dot q$ appears as a disturbance. That is the
0.25° residual in the results.

**Computed torque.** $\tau = M(q)(\ddot q_d + K_d\dot e + K_p e) + h$ substituted into the
dynamics gives $M(\ddot e + K_d\dot e + K_p e) = 0$, i.e. decoupled linear error dynamics per joint. With
$K_p = \omega_n^2$, $K_d = 2\zeta\omega_n$, every joint behaves like a critically damped second-order
system. Implementation note: the right-hand side *is* RNEA evaluated at
$\ddot q = v$, so one O(n) call suffices. With a wrong model
$(\hat M, \hat h)$, the error dynamics gain a disturbance $M^{-1}\big[(\hat M - M)v + \hat h - h\big]$, which is
what the payload experiment exposes.

**Operational-space control (Khatib 1987).** From $\ddot x = J\ddot q + \dot J\dot q$ and
$\ddot q = M^{-1}(\tau - h)$, the choice $\tau = J^\top F + h$ gives $\ddot x = \Lambda^{-1}F + \dot J \dot q$ with
$\Lambda = (J M^{-1} J^\top)^{-1}$. Choosing $F = \Lambda(\ddot x_{cmd} - \dot J\dot q)$ yields $\ddot x = \ddot x_{cmd} = \ddot x_d + K_d\dot e_x + K_p e_x$.
With 6 joints and a 6D task, $J$ is square, so there is no null space.

**Jacobian-transpose PD.** $\tau = J^\top(K_p e_x + K_d \dot e_x) + g$ is a Cartesian spring-damper.
It is stable for regulation by the same Lyapunov argument, but it ignores the arm's
inertia and couples task directions through $\Lambda$.

**Fair gains.** PD gains are $K_{p,i} = M_{ii}(q_0)\omega_n^2$ and $K_{d,i} = M_{ii}(q_0)\,2\zeta\omega_n$ (task space: use
$\Lambda_{ii}$). At the home pose every controller then has the same local bandwidth.

## 8. Payload identification (`identification.py`)

The Newton–Euler force $\hat f = I\hat a + \hat v \times^{*} I \hat v$ is linear in $I$. Parametrising $I$ by
$\pi = [m,\ mc,\ \bar I_O]$ (10 numbers, with inertia about the body **origin**) makes $I$ linear in $\pi$.
The COM form $\bar I_c + m[c]_\times[c]_\times^\top$ is not linear, which is why the origin is used.
So a payload welded to body $b$ adds

$$
\tau_{payload} = Y(q,\dot q,\ddot q)\,\pi,
$$

where column $k$ of $Y$ is the joint torque produced by the unit-parameter inertia $I(e_k)$
moving with body $b$, carried to the root through ${}^iX_\lambda^\top$ and projected on each $S_j$.
Stacking $N$ samples of $\tau_{meas} - \text{RNEA}_{nominal} = Y\pi$ gives an overdetermined linear
system, solved by least squares. The condition number of the stacked $Y$ measures how
well the trajectory excites the parameters.

## 9. Trajectories (`trajectory.py`)

Rest-to-rest motion must satisfy six boundary conditions (position, velocity and
acceleration at both ends), so the lowest-order polynomial that works is quintic:
$b(s) = 10s^3 - 15s^4 + 6s^5$. Peak speed is $1.875\,\Delta q/T$ and peak acceleration
$5.77\,\Delta q/T^2$. Zero boundary acceleration means no torque step at the start or end
of each move.
