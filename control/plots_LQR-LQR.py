import numpy as np
import matplotlib.pyplot as plt
import os
import pybullet as p
import matplotlib.colors as mcolors
from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.registration import make
from matplotlib.ticker import LogLocator, LogFormatterMathtext
from scipy.stats import mannwhitneyu


DIR_name = "Results_LQR-LQR"

FS = 24

plt.rcParams.update({
    "font.family": "serif",
    "font.size": FS,
    "mathtext.fontset": "cm",
    "axes.labelsize": FS,
    "axes.titlesize": FS,
    "xtick.labelsize": FS,
    "ytick.labelsize": FS,
    "legend.fontsize": FS,
    "figure.titlesize": FS,
})


data = np.load(
    DIR_name + "/experiment_data.npz",
    allow_pickle=True
)

logs_seeds = data["logs_seeds"].tolist()

X_REF = data["X_REF"]
U_REF = data["U_REF"]
x0 = data["x0"]

n_primitives = int(data["n_primitives"])
N_seeds = int(data["N_seeds"])
T = int(data["T"])

colors_tab = ['tab:blue',
              'tab:green',
              'tab:orange',
              'tab:pink',
              'tab:purple',
              'tab:gray',
              ]

FIG_DIR = DIR_name + "/Figures"
os.makedirs(FIG_DIR, exist_ok=True)


# ============================================================
# Aggregate statistics over seeds
# ============================================================
def annotate_medians(ax, bp):
    """
    Writes numeric median values above each box.
    """
    for i, median_line in enumerate(bp['medians']):
        x, y = median_line.get_xydata()[1]  # median position
        ax.text(
            x + 0.05,  # small horizontal shift
            y,
            f"{y:.2f}",
            # fmt.format(y),
            ha='left',
            va='center',
            fontsize=18
        )


labels = (
        ['Ours'] +
        [f"Stabilizing\nprimitive"] +
        [f"Tracking\nprimitive"] +
        ['LQR']
)

method_seed_values = []

# --- Ours
method_seed_values.append(
    np.array([l['mixture']['episode_cost'] for l in logs_seeds])
)

# --- Primitives
for i in range(n_primitives):
    method_seed_values.append(
        np.array([l['primitives'][i]['episode_cost'] for l in logs_seeds])
    )

# --- LQR
method_seed_values.append(
    np.array([l['lqr']['episode_cost'] for l in logs_seeds])
)


# ============================================================
# indices
idx_mixture = 0
idx_lqr = n_primitives + 1

mix_vals = method_seed_values[idx_mixture]
lqr_vals = method_seed_values[idx_lqr]

stat, p_value = mannwhitneyu(
    mix_vals,
    lqr_vals,
    alternative='two-sided'
)
print("\n===== Statistical Test: Ours vs Baseline LQR =====")
print(f"U statistic: {stat:.4f}")
print(f"p-value:     {p_value:.4e}")
print("Significant (p < 0.05)? ->", p_value < 0.05)
print("===========================================\n")


def cliffs_delta(x, y):
    n_x = len(x)
    n_y = len(y)
    greater = 0
    less = 0
    for xi in x:
        greater += np.sum(xi > y)
        less += np.sum(xi < y)
    return (greater - less) / (n_x * n_y)


delta = cliffs_delta(mix_vals, lqr_vals)
print("Cliff's delta:", delta)

prob = np.mean(mix_vals < lqr_vals)
print("P(Ours < LQR):", prob)


# ============================================================
# Total episode cost
# ============================================================
fig, ax = plt.subplots(figsize=(11,8))

x = np.arange(len(labels))

# --- BOX PLOTS ---
bp = ax.boxplot(
    method_seed_values,
    positions=x,
    widths=0.45,
    patch_artist=True,
    showfliers=False,
    zorder=3
)

for i, box in enumerate(bp['boxes']):
    color = colors_tab[i]

    rgba = list(mcolors.to_rgba(color))
    rgba[3] = 0.8  # only fill is transparent

    box.set_facecolor(rgba)
    box.set_edgecolor('k')  # fully opaque edge
    box.set_linewidth(1.8)

# Whiskers (black)
for whisk in bp['whiskers']:
    whisk.set_color('black')
    whisk.set_linewidth(1.5)

# Caps (black)
for cap in bp['caps']:
    cap.set_color('black')
    cap.set_linewidth(1.5)

# Medians (black)
for med in bp['medians']:
    med.set_color('black')
    med.set_linewidth(2.2)


annotate_medians(ax, bp)

ax.set_xticks(x)
ax.set_xticklabels(labels)

ax.set_yscale('log')
ax.set_ylabel(r"Total episode cost")

ax.yaxis.set_major_locator(LogLocator(base=10.0))
ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
ax.set_ylim(10, ax.get_ylim()[1])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig.tight_layout()

fig.savefig(
    os.path.join(FIG_DIR, "total_episode_cost.pdf"),
    dpi=300,
    bbox_inches="tight"
)


# ========= Ours and baseline LQR trajectories plot ============

colors_rgb = [mcolors.to_rgb(c) for c in colors_tab]


def set_camera(env,
               distance=1.2,   # smaller = zoom in
               yaw=-20,
               pitch=-30,
               target=(0, 0, 0.8)
               ):
    p.resetDebugVisualizerCamera(
        cameraDistance=distance,
        cameraYaw=yaw,
        cameraPitch=pitch,
        cameraTargetPosition=target,
        physicsClientId=env.PYB_CLIENT,
    )
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)


def capture_env_image(env, width=4000, height=3000):
    cam = p.getDebugVisualizerCamera(
        physicsClientId=env.PYB_CLIENT
    )
    view = cam[2]
    proj = cam[3]
    # White background
    _, _, px, _, _ = p.getCameraImage(
        width=width,
        height=height,
        viewMatrix=view,
        projectionMatrix=proj,
        renderer=p.ER_TINY_RENDERER,
        lightDirection=[1, 1, 1],
        shadow=0,
        physicsClientId=env.PYB_CLIENT
    )
    img = np.reshape(px, (height, width, 4))[:, :, :3]

    return img, view, proj, width, height


def project_points(points, view, proj, width, height):
    pts = np.c_[points[:, 0],
                np.zeros(len(points)),
                points[:, 2],
                np.ones(len(points))]
    view = np.array(view).reshape(4, 4, order='F')
    proj = np.array(proj).reshape(4, 4, order='F')
    clip = (proj @ view @ pts.T).T
    ndc = clip[:, :3] / clip[:, 3][:, None]
    x_img = (ndc[:, 0] * 0.5 + 0.5) * width
    y_img = (1 - (ndc[:, 1] * 0.5 + 0.5)) * height

    return np.vstack([x_img, y_img]).T


def render_overlay_figure(
        env,
        logs_seeds,
        X_REF,
        colors_rgb,
        FIG_DIR,
        seed_id=0,
        drone_stride=30,
        drone_alpha=0.3
):

    # --------------------------------------------------
    # reset env to clean initial state
    # --------------------------------------------------
    env.reset(seed=seed_id)
    set_camera(env, distance=1.0, yaw=0, pitch=0, target=(0, 0, 1))

    base_img, view, proj, W, H = capture_env_image(env)
    fig, ax = plt.subplots(figsize=(W/300, H/300), dpi=300)
    ax.imshow(base_img)
    ax.axis("off")

    # --------------------------------------------------
    # helper: project trajectory
    # --------------------------------------------------
    def plot_traj(X, color, lw=3.5, linestyle='-', zorder=5, label=None):

        pts = project_points(X, view, proj, W, H)

        if label == "Ref":
            ax.plot(
                pts[:, 0],
                pts[:, 1],
                color=color,
                linewidth=lw,
                linestyle=linestyle,
                alpha=0.9,
                label=label,
                zorder=zorder
            )

        else:
            ax.plot(
                pts[:, 0],
                pts[:, 1],
                color=color,
                linewidth=lw,
                linestyle=linestyle,
                alpha=0.8,
                label=label,
                zorder=zorder
            )

    plot_traj(X_REF,
              colors_rgb[5],
              linestyle='-',
              zorder=Z_REF,
              label="Ref")

    # --------------------------------------------------
    # helper: overlay drones
    # --------------------------------------------------
    def overlay_drones(X, color_rgb):

        for k in range(0, len(X), drone_stride):
            set_drone_state_from_X(env, X[k])
            env._update_and_store_kinematic_information()
            img, _, _, _, _ = capture_env_image(env)
            mask = np.mean(np.abs(img - base_img), axis=2) > 5
            overlay = np.zeros((*img.shape[:2], 4))
            overlay[mask, :3] = color_rgb
            overlay[mask, 3] = drone_alpha
            ax.imshow(overlay, zorder=Z_DRONE)

    # --------------------------------------------------
    # baseline lqr trajectory + drones
    # --------------------------------------------------
    X_lqr = logs_seeds[seed_id]['lqr']['x']
    plot_traj(
        X_lqr,
        colors_rgb[3],
        zorder=Z_LQR,
        label="LQR"
    )
    overlay_drones(X_lqr, colors_rgb[3])

    # --------------------------------------------------
    # ours trajectory + drones
    # --------------------------------------------------
    X_mix = logs_seeds[seed_id]['mixture']['x']
    plot_traj(
        X_mix,
        colors_rgb[0],
        zorder=Z_POLICY,
        label="Ours"
    )
    overlay_drones(X_mix, colors_rgb[0])

    # --------------------------------------------------
    # start marker
    # --------------------------------------------------
    start = project_points(X_mix[:1], view, proj, W, H)[0]

    ax.scatter(
        start[0],
        start[1],
        marker='x',
        s=300,
        linewidth=3,
        color='red',
        zorder=Z_MARKER,
        label='Start'
    )

    path = os.path.join(
        FIG_DIR,
        "traj_ours_lqr.pdf"
    )

    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)

    print("Saved trajectories overlay figure →", path)


def set_drone_state_from_X(env, X):
    x, xdot, z, zdot, theta, thetadot = X
    pos = [x, 0.0, z]
    quat = p.getQuaternionFromEuler([0.0, theta, 0.0])
    vel = [xdot, 0.0, zdot]
    ang_vel = [0.0, thetadot, 0.0]
    p.resetBasePositionAndOrientation(
        env.DRONE_ID,
        pos,
        quat,
        physicsClientId=env.PYB_CLIENT
    )
    p.resetBaseVelocity(
        env.DRONE_ID,
        vel,
        ang_vel,
        physicsClientId=env.PYB_CLIENT
    )


seed_id = min(7, len(logs_seeds) - 1)

config = ConfigFactory().merge([
    "examples/lqr/config_overrides/quadrotor_2D/quadrotor_2D_track.yaml"
])
config.task_config.pop("randomized_init", None)
config.task_config["init_state"] = x0

env_render = make(
        'quadrotor',
        randomized_init=False,
        **config.task_config,
    )

obs, _ = env_render.reset(seed=seed_id)

Z_REF       = 2
Z_POLICY    = 6
Z_LQR       = 5
Z_DRONE     = 8
Z_MARKER    = 10

render_overlay_figure(
    env_render,
    logs_seeds,
    X_REF,
    colors_rgb,
    FIG_DIR,
    seed_id=seed_id,
    drone_stride=22,
    drone_alpha=0.8
)

env_render.close()

if p.isConnected():
    p.disconnect()
