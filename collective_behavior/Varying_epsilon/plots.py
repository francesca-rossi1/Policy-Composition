import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

FS = 26

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

# Define constants
eps_values = [0.1, 1, 10]
dt = 0.05
selected_boids = [0, 39]    #0 is informed, 39 no

# Load saved data
loaded_results = {}
for eps_val in eps_values:
    polarization_array = np.load(f"polarization_eps{eps_val}.npy")
    #milling_array = np.load(f"milling_eps{eps_val}.npy")
    distance_array = np.load(f"distance_eps{eps_val}.npy")
    weights_array = np.load(f"weights_eps{eps_val}.npy")
    loaded_results[eps_val] = (polarization_array, distance_array, weights_array)


# === Plotting functions ===
def plot_metrics_for_multiple_eps(results_dict, dt, filename="metrics_multi_eps.pdf"):
    metrics_names = [f"$p$", f"$d$"]
    colors = ['tab:orange', 'tab:green', 'tab:purple']
    t = np.arange(results_dict[eps_values[0]][0].shape[1]) * dt
    fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    for i, metric in enumerate(metrics_names):
        ax = axs[i]
        for j, eps_val in enumerate(eps_values):
            data = results_dict[eps_val][i]  # polarization=0, milling=1, distance=2
            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0)
            ax.plot(t, mean, label=f"$\\varepsilon$={eps_val}", color=colors[j], linewidth=1.8)
            ax.fill_between(t, mean - std, mean + std, color=colors[j], alpha=0.3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        ax.set_ylabel(metric)
        ax.set_xlim(0, t[-1])
        ax.tick_params(axis='both')
        '''# Add legend to first subplot
        if i == 0:
            ax.legend(fontsize=14)'''

    axs[0].set_ylim(0, 1)
    axs[-1].set_xlabel(f"Time [s]")
    axs[-1].set_ylim(0, None)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()


def plot_weights_multiple_eps(weights_list, epsilons, dt, selected_boids=[0, 39], filename="weights_multiple_eps.pdf"):
    N_eps = len(weights_list)
    N_primitives = weights_list[0].shape[-1]
    t = np.arange(weights_list[0].shape[1]) * dt
    labels = ["separation", "alignment", "cohesion"]
    colors = ["tab:red", "tab:blue"]
    boid_labels = ["Informed", "Uninformed"]
    fig, axes = plt.subplots(N_primitives, N_eps, figsize=(17, 10), sharex='col', sharey='row')

    if N_eps == 1:
        axes = np.expand_dims(axes, axis=1)
    if N_primitives == 1:
        axes = np.expand_dims(axes, axis=0)

    for col, (weights_array, eps) in enumerate(zip(weights_list, epsilons)):
        for row in range(N_primitives):
            ax = axes[row, col]
            for j, boid_idx in enumerate(selected_boids):
                weights = weights_array[:, :, boid_idx, row]
                mean = np.mean(weights, axis=0)
                std = np.std(weights, axis=0)
                ax.plot(t, mean, label=boid_labels[j], color=colors[j], linewidth=2)
                ax.fill_between(t, mean - std, mean + std, color=colors[j], alpha=0.3, edgecolor=None)

            if col == 0:
                ax.set_ylabel(labels[row])
                '''if row == 0:
                    ax.legend(fontsize=13)'''
            if row == N_primitives - 1:
                ax.set_xlabel("Time [s]")
            '''if row == 0:
                ax.set_title(f"$\\varepsilon = {eps}$", fontsize=16)'''
            ax.set_xlim(0, t[-1])
            ax.set_ylim(0, 1)
            ax.tick_params(axis='both')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    #handles, labels = ax.get_legend_handles_labels()
    #fig.legend(handles, labels, loc='lower center', fontsize=12, ncol=len(selected_boids), frameon=False)
    #plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()


# === Call plots ===
plot_metrics_for_multiple_eps(loaded_results, dt)
plot_weights_multiple_eps(
    weights_list=[loaded_results[eps][2] for eps in eps_values],
    epsilons=eps_values,
    dt=dt,
    selected_boids=selected_boids
)