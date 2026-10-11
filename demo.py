"""Demo: quadrotor flying with a tendon-driven soft continuum robot underneath.

Mission (loops in the interactive viewer):
    0 - 3 s   hover at 1.0 m
    3 - 6 s   climb to 1.6 m
    6 - 22 s  figure-eight (the soft arm swings passively)
   22 - 32 s  hover while the arm sweeps a cone (tendon actuation)
   32 - 36 s  return to start

Usage:
    mjpython demo.py                 # interactive viewer (macOS needs mjpython)
    python  demo.py --headless       # no GUI: saves plots + video to ./results
"""
import argparse
import os
import time

import mujoco
import numpy as np

from controller import QuadController, bend_to_tensions

HERE = os.path.dirname(os.path.abspath(__file__))
XML = os.path.join(HERE, "quad_continuum.xml")
MISSION_T = 36.0
HOME = np.array([0.0, 0.0, 1.0])
CRUISE_Z = 1.6


def smooth(s):
    """Min-jerk blend 0 -> 1 on s in [0, 1]."""
    s = np.clip(s, 0.0, 1.0)
    return 10 * s**3 - 15 * s**4 + 6 * s**5


def reference(t):
    """Returns (position, yaw, tendon tensions) of the mission at time t."""
    t = t % MISSION_T
    tension = np.zeros(3)
    yaw = 0.0
    if t < 3.0:
        p = HOME.copy()
    elif t < 6.0:
        p = HOME + np.array([0, 0, (CRUISE_Z - HOME[2]) * smooth((t - 3.0) / 3.0)])
    elif t < 22.0:
        tau = t - 6.0
        amp = 1.2 * smooth(tau / 2.0) * smooth((16.0 - tau) / 2.0)
        w = 2 * np.pi / 8.0
        p = np.array([amp * np.sin(w * tau), 0.6 * amp * np.sin(2 * w * tau), CRUISE_Z])
    elif t < 32.0:
        tau = t - 22.0
        p = np.array([0.0, 0.0, CRUISE_Z])
        mag = smooth(tau / 1.5) * smooth((10.0 - tau) / 1.5)
        tension = bend_to_tensions(direction=2 * np.pi * tau / 5.0, magnitude=mag)
    else:
        s = smooth((t - 32.0) / 4.0)
        p = np.array([0.0, 0.0, CRUISE_Z + (HOME[2] - CRUISE_Z) * s])
    return p, yaw, tension


def reference_derivs(t, h=1e-3):
    p0, yaw, ten = reference(t)
    pp, _, _ = reference(t + h)
    pm, _, _ = reference(t - h) if t > h else (p0, 0, 0)
    v = (pp - pm) / (2 * h)
    a = (pp - 2 * p0 + pm) / h**2
    return p0, v, a, yaw, ten


class Sim:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(XML)
        self.data = mujoco.MjData(self.model)
        self.ctrl = QuadController(self.model, self.data)
        self.tip_site = self.model.site("tip").id
        self.reset()

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.ctrl.int_err[:] = 0.0

    def step(self):
        d, dt = self.data, self.model.opt.timestep
        p, v, a, yaw, ten = reference_derivs(d.time)
        thrusts = self.ctrl.compute(p, v, a, yaw, dt)
        self.ctrl.apply(thrusts, ten)
        mujoco.mj_step(self.model, d)
        return p, thrusts, ten


# ------------------------------------------------------------------ viewer ---
def add_marker(scn, pos, rgba, size=0.02):
    if scn.ngeom >= scn.maxgeom:
        return
    mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.array([size, 0, 0]), pos, np.eye(3).flatten(), rgba)
    scn.ngeom += 1


def run_viewer():
    import mujoco.viewer
    sim = Sim()
    trail, tip_trail = [], []
    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        viewer.cam.distance, viewer.cam.elevation, viewer.cam.azimuth = 3.5, -15, 135
        viewer.cam.lookat[:] = [0, 0, 1.3]
        print(__doc__)
        print("Viewer controls: double-click to select, Ctrl+drag to perturb, Backspace resets.")
        while viewer.is_running():
            frame_start = time.time()
            for _ in range(int(round((1 / 60) / sim.model.opt.timestep))):
                p_ref, _, _ = sim.step()
            if sim.data.time < 0.02:          # viewer reset (Backspace)
                trail.clear(); tip_trail.clear(); sim.ctrl.int_err[:] = 0
            trail.append(p_ref.copy())
            tip_trail.append(sim.data.site_xpos[sim.tip_site].copy())
            trail, tip_trail = trail[-240:], tip_trail[-240:]

            with viewer.lock():
                viewer.user_scn.ngeom = 0
                for q in trail[::4]:
                    add_marker(viewer.user_scn, q, np.array([0.1, 1.0, 0.3, 0.5]), 0.008)
                for q in tip_trail[::3]:
                    add_marker(viewer.user_scn, q, np.array([1.0, 0.9, 0.1, 0.6]), 0.006)
                add_marker(viewer.user_scn, p_ref, np.array([0.1, 1.0, 0.3, 0.9]), 0.03)
            viewer.sync()
            time.sleep(max(0.0, 1 / 60 - (time.time() - frame_start)))


# ---------------------------------------------------------------- headless ---
def run_headless(duration, out_dir, video=True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    sim = Sim()
    m, d = sim.model, sim.data
    fps, steps_per_frame = 30, int(round(1 / 30 / m.opt.timestep))
    renderer, frames = None, []
    if video:
        renderer = mujoco.Renderer(m, 480, 640)
        cam = mujoco.MjvCamera()
        cam.distance, cam.elevation, cam.azimuth = 2.2, -12, 135
        cam.lookat[:] = [0, 0, 1.2]

    log = {k: [] for k in ("t", "pos", "ref", "tip", "thrust", "tension")}
    n_steps = int(duration / m.opt.timestep)
    for i in range(n_steps):
        p_ref, thrusts, ten = sim.step()
        if i % 10 == 0:
            log["t"].append(d.time); log["pos"].append(d.xpos[sim.ctrl.body].copy())
            log["ref"].append(p_ref); log["tip"].append(d.site_xpos[sim.tip_site].copy())
            log["thrust"].append(thrusts.copy()); log["tension"].append(ten.copy())
        if renderer is not None and i % steps_per_frame == 0:
            cam.lookat[:] = 0.8 * cam.lookat + 0.2 * (d.xpos[sim.ctrl.body] - [0, 0, 0.25])
            renderer.update_scene(d, cam)
            frames.append(renderer.render())
    log = {k: np.array(v) for k, v in log.items()}

    err = np.linalg.norm(log["pos"] - log["ref"], axis=1)
    print(f"Simulated {duration:.1f} s | RMS tracking error {np.sqrt(np.mean(err**2))*100:.1f} cm "
          f"| max {err.max()*100:.1f} cm")

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    for k, lbl in enumerate("xyz"):
        ax[0, 0].plot(log["t"], log["pos"][:, k], label=f"{lbl}")
        ax[0, 0].plot(log["t"], log["ref"][:, k], "--", color=f"C{k}", alpha=0.6)
    ax[0, 0].set(title="Drone position (dashed = reference)", xlabel="t [s]", ylabel="[m]"); ax[0, 0].legend()
    ax[0, 1].plot(log["ref"][:, 0], log["ref"][:, 1], "k--", label="reference")
    ax[0, 1].plot(log["pos"][:, 0], log["pos"][:, 1], label="drone")
    ax[0, 1].plot(log["tip"][:, 0], log["tip"][:, 1], alpha=0.7, label="arm tip")
    ax[0, 1].set(title="Top view", xlabel="x [m]", ylabel="y [m]", aspect="equal"); ax[0, 1].legend()
    ax[1, 0].plot(log["t"], log["thrust"]); ax[1, 0].set(title="Rotor thrusts", xlabel="t [s]", ylabel="[N]")
    ax[1, 0].legend([f"rotor{i+1}" for i in range(4)])
    ax[1, 1].plot(log["t"], log["tension"]); ax[1, 1].set(title="Tendon tensions", xlabel="t [s]", ylabel="[N]")
    ax[1, 1].legend(["tendon a", "tendon b", "tendon c"])
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "flight_log.png"), dpi=120)
    print("Saved", os.path.join(out_dir, "flight_log.png"))

    if frames:
        import imageio
        try:      # mp4 needs the imageio-ffmpeg backend
            path = os.path.join(out_dir, "demo.mp4")
            imageio.mimsave(path, frames, fps=fps, format="FFMPEG")
        except Exception:
            path = os.path.join(out_dir, "demo.gif")
            imageio.mimsave(path, frames[::2], duration=2 / fps)
        print("Saved", path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--headless", action="store_true", help="run without GUI, save plots/video")
    ap.add_argument("--duration", type=float, default=MISSION_T)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    args = ap.parse_args()
    if args.headless:
        run_headless(args.duration, args.out, video=not args.no_video)
    else:
        run_viewer()
