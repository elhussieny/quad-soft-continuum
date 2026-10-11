"""Reproduce every figure and number reported in the paper.

    cd paper && python make_figures.py

Writes figures/*.pdf, figures/snapshots.png and results.json.
"""
import json
import os
import platform
import subprocess
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import demo  # noqa: E402
from controller import QuadController, bend_to_tensions  # noqa: E402

FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.grid": True,
                     "grid.alpha": 0.3, "legend.fontsize": 7, "lines.linewidth": 1.0})
COL_W = 3.5  # IEEE column width [in]
PHASES = [("Hover", 0, 3), ("Climb", 3, 6), ("Figure-eight", 6, 22),
          ("Tendon sweep", 22, 32), ("Return", 32, 36)]
results = {}


def tilt_deg(d, body):
    return np.degrees(np.arccos(np.clip(d.xmat[body][8], -1, 1)))


# --------------------------------------------------------------- E1 mission ---
def mission():
    sim = demo.Sim()
    m, d = sim.model, sim.data
    b, tip = sim.ctrl.body, sim.tip_site
    L = {k: [] for k in ("t", "pos", "ref", "tip", "thr", "ten", "tilt")}
    t0 = time.perf_counter()
    while d.time < demo.MISSION_T - 1e-9:
        p, thr, ten = sim.step()
        L["t"].append(d.time); L["pos"].append(d.xpos[b].copy()); L["ref"].append(p)
        L["tip"].append(d.site_xpos[tip].copy()); L["thr"].append(thr.copy())
        L["ten"].append(ten.copy()); L["tilt"].append(tilt_deg(d, b))
    wall = time.perf_counter() - t0
    L = {k: np.array(v) for k, v in L.items()}
    err = np.linalg.norm(L["pos"] - L["ref"], axis=1)
    rel = L["tip"] - L["pos"]
    ph = {}
    for name, a, c in PHASES:
        k = (L["t"] > a) & (L["t"] <= c)
        ph[name] = dict(rms_cm=float(np.sqrt(np.mean(err[k]**2)) * 100),
                        max_cm=float(err[k].max() * 100),
                        tilt_max_deg=float(L["tilt"][k].max()),
                        tip_lat_max_cm=float(np.linalg.norm(rel[k, :2], axis=1).max() * 100),
                        thrust_min=float(L["thr"][k].min()), thrust_max=float(L["thr"][k].max()))
    results["mission"] = dict(
        rms_cm=float(np.sqrt(np.mean(err**2)) * 100), max_cm=float(err.max() * 100),
        phases=ph, wall_s=wall, sim_s=float(d.time), rtf=float(d.time / wall),
        steps=int(round(d.time / m.opt.timestep)),
        us_per_step=float(wall / (d.time / m.opt.timestep) * 1e6))

    s = slice(None, None, 10)
    fig, ax = plt.subplots(3, 1, figsize=(COL_W, 4.2), sharex=True)
    for k, lbl in enumerate("xyz"):
        ax[0].plot(L["t"][s], L["pos"][s, k], color=f"C{k}", label=f"${lbl}$")
        ax[0].plot(L["t"][s], L["ref"][s, k], "--", color=f"C{k}", alpha=0.6)
    ax[0].set_ylabel("Position [m]"); ax[0].legend(ncol=3, loc="lower left")
    ax[1].semilogy(L["t"][s], np.maximum(err[s], 1e-5) * 100, "k")
    ax[1].set_ylabel(r"$\|e_p\|$ [cm]")
    ax[2].plot(L["t"][s], L["ten"][s]); ax[2].set_ylabel("Tension [N]")
    ax[2].legend(["$T_a$", "$T_b$", "$T_c$"], ncol=3, loc="upper left"); ax[2].set_xlabel("Time [s]")
    for a in ax:
        for _, t0_, _ in PHASES[1:]:
            a.axvline(t0_, color="gray", lw=0.5, ls=":")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "mission.pdf")); plt.close(fig)

    fig, ax = plt.subplots(figsize=(COL_W, 2.3))
    k = (L["t"] > 6) & (L["t"] <= 22)
    ax.plot(L["ref"][k, 0], L["ref"][k, 1], "k--", label="Reference")
    ax.plot(L["pos"][k, 0], L["pos"][k, 1], "C0", label="Quadrotor")
    ax.plot(L["tip"][k, 0], L["tip"][k, 1], "C1", alpha=0.8, label="Arm tip")
    ax.set(xlabel="$x$ [m]", ylabel="$y$ [m]"); ax.set_aspect("equal"); ax.legend(loc="upper center", ncol=3)
    ax.set_ylim(-1.0, 1.25)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "topview.pdf")); plt.close(fig)

    fig, ax = plt.subplots(figsize=(COL_W, 1.8))
    ax.plot(L["t"][s], L["thr"][s]); ax.set(xlabel="Time [s]", ylabel="Thrust [N]")
    ax.legend([f"$f_{i+1}$" for i in range(4)], ncol=4, loc="upper right")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "thrust.pdf")); plt.close(fig)


# ------------------------------------------------------- E2 attitude gains ---
def gain_study():
    sets = [("Soft", (0.35, 0.35, 0.1), (0.06, 0.06, 0.04)),
            ("Medium", (1.0, 1.0, 0.3), (0.15, 0.15, 0.08)),
            ("Stiff", (2.0, 2.0, 0.5), (0.25, 0.25, 0.1)),
            ("Nominal", (4.0, 4.0, 0.8), (0.4, 0.4, 0.15))]
    out = []
    for name, kR, kw in sets:
        sim = demo.Sim()
        sim.ctrl = QuadController(sim.model, sim.data, kR=kR, kw=kw)
        d, e, tilt = sim.data, [], []
        while d.time < demo.MISSION_T - 1e-9:
            p, _, _ = sim.step()
            e.append(np.linalg.norm(d.xpos[sim.ctrl.body] - p)); tilt.append(tilt_deg(d, sim.ctrl.body))
        e, t = np.array(e), np.arange(1, len(e) + 1) * sim.model.opt.timestep
        h = t > 24  # steady part of the tendon-sweep + return
        out.append(dict(name=name, kR=kR[0], kw=kw[0], rms_cm=float(np.sqrt(np.mean(e**2)) * 100),
                        max_cm=float(e.max() * 100), rms_late_cm=float(np.sqrt(np.mean(e[h]**2)) * 100),
                        tilt_max_deg=float(max(tilt))))
    results["gains"] = out


# -------------------------------------------------- E3 bending workspace ---
def hover_with_tension(ten, T=5.0, z=1.6):
    sim = demo.Sim()
    m, d = sim.model, sim.data
    d.qpos[2] = z; mujoco.mj_forward(m, d)
    p = np.array([0, 0, z])
    while d.time < T:
        s = min(d.time / 1.0, 1.0)
        sim.ctrl.apply(sim.ctrl.compute(p), np.asarray(ten) * s)
        mujoco.mj_step(m, d)
    R = d.xmat[sim.ctrl.body].reshape(3, 3)
    rel_body = R.T @ (d.site_xpos[sim.tip_site] - d.xpos[sim.ctrl.body])
    return rel_body, np.linalg.norm(d.xpos[sim.ctrl.body] - p), tilt_deg(d, sim.ctrl.body)


def bending():
    mags = np.linspace(0, 1, 11)
    curve = [hover_with_tension(bend_to_tensions(0.0, a)) for a in mags]
    dirs = np.radians(np.arange(0, 360, 15))
    ring = [hover_with_tension(bend_to_tensions(th, 1.0))[0] for th in dirs]
    ring_half = [hover_with_tension(bend_to_tensions(th, 0.5))[0] for th in dirs]
    tip = np.array([c[0] for c in curve])
    results["bending"] = dict(
        mu=mags.tolist(), tip_x_cm=(tip[:, 0] * 100).tolist(), tip_z_cm=(tip[:, 2] * 100).tolist(),
        hover_offset_cm=[c[1] * 100 for c in curve], tilt_deg=[c[2] for c in curve],
        ring_radius_cm_mean=float(np.mean(np.linalg.norm(np.array(ring)[:, :2], axis=1)) * 100),
        ring_radius_cm_min=float(np.min(np.linalg.norm(np.array(ring)[:, :2], axis=1)) * 100),
        ring_radius_cm_max=float(np.max(np.linalg.norm(np.array(ring)[:, :2], axis=1)) * 100))

    fig, ax = plt.subplots(1, 2, figsize=(COL_W, 1.9))
    ax[0].plot(mags, tip[:, 0] * 100, "o-", ms=2.5, label=r"lateral $x_b$")
    ax[0].plot(mags, -tip[:, 2] * 100, "s-", ms=2.5, label=r"depth $-z_b$")
    ax[0].set(xlabel=r"Bending command $\mu$", ylabel="Tip position [cm]"); ax[0].legend()
    for rr, lab in ((ring_half, r"$\mu$=0.5"), (ring, r"$\mu$=1")):
        rr = np.array(rr + rr[:1]) * 100
        ax[1].plot(rr[:, 0], rr[:, 1], "o-", ms=2, label=lab)
    for a, c in zip((0, 120, 240), ("C3", "C2", "C0")):
        ax[1].annotate("", xy=(14 * np.cos(np.radians(a)), 14 * np.sin(np.radians(a))), xytext=(0, 0),
                       arrowprops=dict(arrowstyle="->", color=c, lw=0.8))
    ax[1].set(xlabel="$x_b$ [cm]", ylabel="$y_b$ [cm]"); ax[1].set_aspect("equal"); ax[1].legend(loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "bending.pdf")); plt.close(fig)


# --------------------------------------------------- E4 tip disturbance ---
def disturbance():
    sim = demo.Sim()
    m, d = sim.model, sim.data
    tipb = m.body("seg12").id
    p = np.array([0, 0, 1.0])
    L = {"t": [], "e": [], "tip": [], "tilt": []}
    while d.time < 8.0:
        d.xfrc_applied[tipb] = 0
        if 2.0 <= d.time < 2.2:
            d.xfrc_applied[tipb, 0] = 2.0  # 2 N lateral push for 0.2 s -> 0.4 N s impulse
        sim.ctrl.apply(sim.ctrl.compute(p))
        mujoco.mj_step(m, d)
        L["t"].append(d.time); L["e"].append(d.xpos[sim.ctrl.body] - p)
        L["tip"].append(d.site_xpos[sim.tip_site] - d.xpos[sim.ctrl.body]); L["tilt"].append(tilt_deg(d, sim.ctrl.body))
    L = {k: np.array(v) for k, v in L.items()}
    en = np.linalg.norm(L["e"], axis=1)
    after = L["t"] > 2.0
    settle = L["t"][after][np.where(en[after] > 0.01)[0][-1]] - 2.0
    results["disturbance"] = dict(impulse_Ns=0.4, max_dev_cm=float(en.max() * 100),
                                  tip_swing_max_cm=float(np.abs(L["tip"][:, 0]).max() * 100),
                                  tilt_max_deg=float(L["tilt"].max()), settle_1cm_s=float(settle))
    fig, ax = plt.subplots(figsize=(COL_W, 1.7))
    ax.plot(L["t"], L["e"][:, 0] * 100, label="Quadrotor $e_x$")
    ax.plot(L["t"], L["tip"][:, 0] * 100, label="Tip lateral $x$ (rel.)")
    ax.axvspan(2.0, 2.2, color="r", alpha=0.2, lw=0)
    ax.set(xlabel="Time [s]", ylabel="[cm]"); ax.legend(loc="upper right")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "disturbance.pdf")); plt.close(fig)


# -------------------------------------------------------------- snapshots ---
def snapshots():
    sim = demo.Sim()
    m, d = sim.model, sim.data
    r = mujoco.Renderer(m, 360, 360)
    cam = mujoco.MjvCamera(); cam.distance, cam.elevation, cam.azimuth = 1.7, -10, 135
    times, imgs = [1.0, 12.3, 24.0, 26.0, 27.5, 29.0], []
    for t in times:
        while d.time < t:
            sim.step()
        cam.lookat[:] = d.xpos[sim.ctrl.body] - [0, 0, 0.25]
        r.update_scene(d, cam); imgs.append(r.render())
    fig, ax = plt.subplots(1, len(imgs), figsize=(7.16, 1.45))
    labels = ["(a) hover", "(b) figure-eight", "(c) bend", "(d) bend", "(e) bend", "(f) bend"]
    for a, im, lab, t in zip(ax, imgs, labels, times):
        a.imshow(im); a.axis("off"); a.set_title(f"{lab}, $t$={t:.1f} s", fontsize=7)
    fig.tight_layout(pad=0.2); fig.savefig(os.path.join(FIG, "snapshots.png"), dpi=300); plt.close(fig)


if __name__ == "__main__":
    for f in (mission, gain_study, bending, disturbance, snapshots):
        print("running", f.__name__, flush=True); f()
    try:
        cpu = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
    except Exception:
        cpu = platform.processor()
    results["env"] = dict(mujoco=mujoco.__version__, python=platform.python_version(), cpu=cpu)
    with open(os.path.join(HERE, "results.json"), "w") as fh:
        json.dump(results, fh, indent=1)
    print(json.dumps(results, indent=1))
