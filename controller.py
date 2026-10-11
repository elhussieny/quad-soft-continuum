"""Geometric SE(3) tracking controller for the quadrotor + continuum robot.

Outer loop: PID on position -> desired force vector (uses total mass incl. the arm).
Inner loop: attitude error on SO(3) (Lee et al., 2010) -> body torques.
Mixer:      [T, tau_x, tau_y, tau_z] -> four rotor thrusts, built from the
            actuator gears in the MJCF so geometry/signs never go out of sync.
"""
import numpy as np
import mujoco

ROTORS = ["rotor1", "rotor2", "rotor3", "rotor4"]
TENDONS = ["tendon_a", "tendon_b", "tendon_c"]


def _vee(S):
    return np.array([S[2, 1], S[0, 2], S[1, 0]])


class QuadController:
    def __init__(self, model, data,
                 kp=(6.0, 6.0, 10.0), kd=(4.0, 4.0, 6.0), ki=(1.0, 1.0, 2.0),
                 kR=(4.0, 4.0, 0.8), kw=(0.4, 0.4, 0.15),
                 max_tilt=np.deg2rad(35)):
        self.m, self.d = model, data
        self.body = model.body("quad").id
        self.jnt_qpos = model.jnt_qposadr[model.joint("root").id]
        self.jnt_qvel = model.jnt_dofadr[model.joint("root").id]
        self.mass = model.body_subtreemass[self.body]   # drone + soft arm
        self.g = -model.opt.gravity[2]
        self.kp, self.kd, self.ki = map(np.asarray, (kp, kd, ki))
        self.kR, self.kw = np.asarray(kR), np.asarray(kw)
        self.max_tilt = max_tilt
        self.int_err = np.zeros(3)
        self.rotor_ids = [model.actuator(n).id for n in ROTORS]
        self.tendon_ids = [model.actuator(n).id for n in TENDONS]
        self.thrust_max = model.actuator_ctrlrange[self.rotor_ids, 1]
        self._build_mixer()

    def _build_mixer(self):
        """Allocation matrix A: wrench = A @ thrusts (body frame)."""
        A = np.zeros((4, 4))
        for j, aid in enumerate(self.rotor_ids):
            site = self.m.actuator_trnid[aid, 0]
            p = self.m.site_pos[site]
            gear = self.m.actuator_gear[aid]
            A[0, j] = gear[2]              # thrust
            A[1, j] = p[1] * gear[2]       # roll  = y * F
            A[2, j] = -p[0] * gear[2]      # pitch = -x * F
            A[3, j] = gear[5]              # yaw drag torque
        self.A_inv = np.linalg.inv(A)

    def state(self):
        qp, qv = self.jnt_qpos, self.jnt_qvel
        pos = self.d.qpos[qp:qp + 3].copy()
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, self.d.qpos[qp + 3:qp + 7])
        vel = self.d.qvel[qv:qv + 3].copy()        # world frame
        omega = self.d.qvel[qv + 3:qv + 6].copy()  # body frame
        return pos, R.reshape(3, 3), vel, omega

    def compute(self, p_des, v_des=np.zeros(3), a_des=np.zeros(3), yaw_des=0.0, dt=0.002):
        pos, R, vel, omega = self.state()

        # ---- position loop ----
        e_p, e_v = p_des - pos, v_des - vel
        self.int_err = np.clip(self.int_err + e_p * dt, -1.0, 1.0)
        acc = self.kp * e_p + self.kd * e_v + self.ki * self.int_err + a_des
        # tilt limit
        horiz = acc[:2]
        lim = np.tan(self.max_tilt) * (acc[2] + self.g)
        if np.linalg.norm(horiz) > lim:
            acc[:2] = horiz / np.linalg.norm(horiz) * lim
        F = self.mass * (acc + np.array([0, 0, self.g]))

        # ---- desired attitude ----
        b3 = F / np.linalg.norm(F)
        b1c = np.array([np.cos(yaw_des), np.sin(yaw_des), 0.0])
        b2 = np.cross(b3, b1c); b2 /= np.linalg.norm(b2)
        R_des = np.column_stack((np.cross(b2, b3), b2, b3))

        T = F @ R[:, 2]
        e_R = 0.5 * _vee(R_des.T @ R - R.T @ R_des)
        tau = -self.kR * e_R - self.kw * omega

        thrusts = self.A_inv @ np.concatenate(([T], tau))
        return np.clip(thrusts, 0.0, self.thrust_max)

    def apply(self, thrusts, tendon_tension=(0.0, 0.0, 0.0)):
        self.d.ctrl[self.rotor_ids] = thrusts
        self.d.ctrl[self.tendon_ids] = tendon_tension


def bend_to_tensions(direction, magnitude, t_max=6.0):
    """Map a desired bending direction (rad, in body xy-plane) and magnitude in
    [0, 1] to the three tendon tensions (tendons at 0, 120, 240 deg)."""
    angles = np.deg2rad([0.0, 120.0, 240.0])
    # Least-norm tensions producing the bending moment, plus a common-mode offset
    # (which adds no net moment) so the slackest tendon is at zero: tendons only pull.
    # max(c) - min(c) <= sqrt(3), so dividing by sqrt(3) keeps every tension in
    # [0, t_max]; the net pull is then sqrt(3)/2*magnitude*t_max in every direction.
    c = np.cos(direction - angles)
    t = magnitude * t_max * (c - c.min()) / np.sqrt(3.0)
    return np.clip(t, 0.0, t_max)
