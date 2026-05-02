import matplotlib.pyplot as plt
import matplotlib as mpl
import ternary
import numpy as np
import pickle
from matplotlib.collections import LineCollection
from matplotlib.animation import FFMpegWriter
from tqdm import tqdm
from matplotlib import gridspec
import matplotlib.ticker as ticker


FS = 18
plt.rcParams.update({
    "font.family": "serif",
    "font.size": FS,
    "mathtext.fontset": "cm",        # Computer Modern (LaTeX-like)
    "axes.labelsize": FS,
    "axes.titlesize": FS,
    "xtick.labelsize": FS,
    "ytick.labelsize": FS,
    "legend.fontsize": FS,
    "figure.titlesize": FS,
})

dt = 0.05
max_steps = 440
informed_boids = 0

polarization_history = np.load(f"polarization_history.npy")
#milling_history = np.load(f"milling_history.npy")
weights_history = np.load(f"weights_history.npy")

cmap = plt.get_cmap('viridis')


def plot_weights_simplex(weights_history, agent_indices, dt=0.05, save_path="weights_simplex.pdf"):
    scale = 1.0
    n_agents = len(agent_indices)
    fig, axs = plt.subplots(n_agents, 1, figsize=(5, 5 if n_agents == 1 else 10))  # Adjust height dynamically
    fig.patch.set_facecolor('white')

    # Ensure axs is iterable even for 1 subplot
    if n_agents == 1:
        axs = [axs]

    for i, agent_index in enumerate(agent_indices):
        tax = ternary.TernaryAxesSubplot(ax=axs[i], scale=scale)
        tax.boundary(linewidth=1.2)
        tax.gridlines(color="black", multiple=0.2, linewidth=0.8, alpha=0.6)
        tax.get_axes().axis('off')
        tax.get_axes().set_aspect('equal', adjustable='box')

        tax.set_background_color('white')

        # Axis labels
        tax.left_axis_label(f"separation", fontsize=1.2*FS, offset=0.25)
        tax.right_axis_label(f"alignment", fontsize=1.2*FS, offset=0.25)
        tax.bottom_axis_label(f"cohesion", fontsize=1.2*FS, offset=0.25)

        # Extract weight trajectory
        points = [tuple(weights_history[t][agent_index]) for t in range(max_steps)]

        # Color map for time progression
        norm = mpl.colors.Normalize(vmin=0, vmax=len(points) - 1)
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        points_array = np.array(points)
        t = np.arange(len(points))

        tax.scatter(
            points_array,
            c=t,
            cmap=cmap,
            vmin=0,
            vmax=len(points) - 1,
            s=30
        )

        tax.ticks(axis='lbr', multiple=0.2, offset=0.035, linewidth=1, fontsize=FS, tick_formats="%.1f")
        tax.clear_matplotlib_ticks()

    '''# --- Single shared colorbar (to the right, not overlapping) ---
    cbar_ax = fig.add_axes([0.80, 0.25, 0.02, 0.5])  # [left, bottom, width, height]
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical')

    tick_locs = np.linspace(0, len(points) - 1, 2)
    cbar.set_ticks(tick_locs)
    cbar.set_ticklabels([f"{i * dt:.0f}" for i in tick_locs], fontsize=FS)
    cbar.set_label("$t$ [s]", fontsize=FS)'''

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved as {save_path}")


tracked_index = [0]
plot_weights_simplex(weights_history[:max_steps], agent_indices=tracked_index)


# Load saved data
with open("boids_trajectories.pkl", "rb") as f:
    trajectories = pickle.load(f)
trajectories = [sublist[:max_steps] for sublist in trajectories]

# Set up figure
fig, ax = plt.subplots(figsize=(8, 8))

# Plot trajectories
for idx, traj in enumerate(trajectories):
    if len(traj) >= 2:
        traj = np.array(traj)
        segments = np.array([[traj[i], traj[i + 1]] for i in range(len(traj) - 1)])
        #colors = plt.cm.viridis(np.linspace(0, 1, len(segments)))
        colors = cmap(np.linspace(0, 1, len(segments)))
        lc = LineCollection(segments, colors=colors, linewidths=1.5)
        ax.add_collection(lc)

    # Plot initial positions
    ax.scatter(traj[0, 0], traj[0, 1],
                              marker='x', linewidth=1.5, color='red', s=80)

    color = 'red' if (idx < informed_boids) else 'blue'

    ax.scatter(traj[-1,0], traj[-1,1], color=color, edgecolors='black', s=80, zorder=3)


ax.set_aspect('equal')
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim([-22,5])
ax.set_ylim([-5,22])
plt.tight_layout()
plt.savefig("trajectories.pdf", dpi=300, bbox_inches='tight')

# -------------------------------
# Polarization plot
# Time array
time = np.arange(len(polarization_history[:max_steps])) * dt
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(time, polarization_history[:max_steps], linewidth=2)
ax.set_ylabel(f"$p$", fontsize=FS+4)
ax.set_xlabel(f"Time [s]", fontsize=FS+4)
ax.set_ylim(0, 1.05)
# set tick positions every 0.2
ax.set_yticks(np.arange(0, 1.01, 0.2))
# format tick labels to 1 decimal place
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
ax.set_xlim(0, time[-1])
ax.tick_params(axis='both', labelsize=FS+4)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig("polarization.pdf", dpi=300)
plt.show()


# -------------------------------
# Video parameters
fps = int(2 / dt)   # same speed as before
max_traj_length = 40

# Load trajectories
with open("boids_trajectories.pkl", "rb") as f:
    all_trajectories = pickle.load(f)

num_boids = len(all_trajectories)
#max_steps = max(len(traj) for traj in all_trajectories)

# Figure setup
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim([-22, 5])
ax.set_ylim([-5, 22])
ax.set_xticks([])
ax.set_yticks([])
ax.set_aspect("equal")

# Trajectory lines
trajectory_collections = []
for _ in range(num_boids):
    line = LineCollection([], linewidths=1.5, cmap="viridis", alpha=0.8, zorder=1)
    ax.add_collection(line)
    trajectory_collections.append(line)

# Boid scatter
colors = ["blue"] * num_boids  # adjust if needed
scat = ax.scatter(
    [all_trajectories[i][0][0] for i in range(num_boids)],
    [all_trajectories[i][0][1] for i in range(num_boids)],
    edgecolors="black",
    s=50,
    c=colors
)

# Writer
writer = FFMpegWriter(fps=fps, bitrate=1800)
video_name = "trajectories_video.mov"

print("\nSaving trajectory video...")

with writer.saving(fig, video_name, dpi=200):
    for frame in tqdm(range(max_steps)):

        # Update positions
        positions = [
            all_trajectories[i][frame] if frame < len(all_trajectories[i])
            else all_trajectories[i][-1]
            for i in range(num_boids)
        ]
        scat.set_offsets(np.array(positions))

        # Update tails
        for i in range(num_boids):
            traj = np.array(all_trajectories[i])[max(0, frame - max_traj_length):frame]
            if len(traj) >= 2:
                seg = np.array([[traj[j], traj[j + 1]] for j in range(len(traj) - 1)])
                trajectory_collections[i].set_segments(seg)
                trajectory_collections[i].set_color(
                    plt.cm.viridis(np.linspace(0, 1, len(seg)))
                )

        fig.tight_layout()
        writer.grab_frame()

plt.close(fig)
print(f"Video saved as '{video_name}'")


# -------------------------------
# ALL-BOIDS SIMPLEX GRID PLOT

num_boids = len(weights_history[0])
cols = 5
rows = int(np.ceil(num_boids / cols))

# A4 layout
a4_width_in = 8.27
subplot_width = a4_width_in
fig_height_in = subplot_width / cols * rows

fig = plt.figure(figsize=(a4_width_in, fig_height_in))
fig.patch.set_facecolor('white')

spec = gridspec.GridSpec(rows, cols, wspace=0.5, hspace=0.0)

norm = mpl.colors.Normalize(vmin=0, vmax=len(weights_history) - 1)


def plot_simplex_subplot(ax, weights_history, agent_index, cmap, norm, show_labels=False):
    tax = ternary.TernaryAxesSubplot(ax=ax, scale=1.0)

    tax.clear_matplotlib_ticks()
    tax.gridlines(color="gray", multiple=0.5, linewidth=0.4)
    tax.boundary(linewidth=0.6)
    tax.get_axes().axis('off')
    tax.set_background_color('white')

    points = [tuple(weights_history[t][agent_index]) for t in range(len(weights_history))]
    points_array = np.array(points)
    t_vals = np.arange(len(points))

    tax.scatter(
        points_array,
        c=t_vals,
        cmap=cmap,
        vmin=0,
        vmax=len(points) - 1,
        s=3
    )

    if show_labels:
        tax.left_axis_label("separation", fontsize=9, offset=0.38)
        tax.right_axis_label("alignment", fontsize=9, offset=0.38)
        tax.bottom_axis_label("    cohesion", fontsize=9, offset=0.65)
        tax.ticks(axis='lbr', multiple=0.5, offset=0.065, fontsize=8, linewidth=0.5, tick_formats="%.1f")


for idx in range(num_boids):
    row, col = divmod(idx, cols)

    if row == rows - 1:
        last_row_count = num_boids % cols or cols
        start_col = (cols - last_row_count) // 2
        ax = plt.subplot(spec[row, start_col + col])
    else:
        ax = plt.subplot(spec[row, col])

    ax.set_aspect('equal')

    plot_simplex_subplot(
        ax,
        weights_history[:max_steps],
        idx,
        cmap,
        norm,
        show_labels=(idx == 0)
    )

# Colorbar
cbar_ax = fig.add_axes([0.2, 0.1, 0.6, 0.01])
sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
tick_locs = np.linspace(0, max_steps, 2)
cbar.set_ticks(tick_locs)
cbar.set_ticklabels([f"{int(i * dt)}" for i in tick_locs])
cbar.set_label(f"Time [s]", fontsize=9, labelpad=0)
cbar.ax.tick_params(labelsize=9)

all_weights_name = "all_boids_weights_simplex.pdf"

plt.savefig(all_weights_name, bbox_inches='tight', dpi=300)
plt.close(fig)

print(f"Saved: {all_weights_name}")
