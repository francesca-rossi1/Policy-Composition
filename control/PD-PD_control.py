import numpy as np
from safe_control_gym.utils.registration import make
from scipy.integrate import solve_ivp
from safe_control_gym.envs.benchmark_env import Task
from safe_control_gym.controllers.pid.pid import PID
from safe_control_gym.utils.configuration import ConfigFactory
import sys
import matplotlib.pyplot as plt
from functools import partial
import time
import os
import pybullet as pyb
from safe_control_gym.controllers.lqr.lqr_utils import discretize_linear_system


FS = 16

plt.rcParams.update({
    "font.family": "serif",
    "font.size": FS,
    "mathtext.fontset": "cm",        # Computer Modern
    "axes.labelsize": FS,
    "axes.titlesize": FS,
    "xtick.labelsize": FS,
    "ytick.labelsize": FS,
    "legend.fontsize": FS,
    "figure.titlesize": FS,
})


def set_render_title(seed, controller_name):
    text = f"Seed: {seed} | {controller_name}"
    pyb.addUserDebugText(
        text,
        textPosition=[0.4, 0, 1.6],
        textColorRGB=[0, 0, 0],
        textSize=1.3
    )


def set_camera(env,
               distance=0.8,   # smaller = zoom in
               yaw=0,
               pitch=0,
               target=(0, 0, 1)
               ):
    pyb.resetDebugVisualizerCamera(
        cameraDistance=distance,
        cameraYaw=yaw,
        cameraPitch=pitch,
        cameraTargetPosition=target,
        physicsClientId=env.PYB_CLIENT,
    )
    pyb.configureDebugVisualizer(pyb.COV_ENABLE_GUI, 0)


def draw_reference_trajectory(env, X_REF):
    """
    Draws the reference trajectory in PyBullet.
    X_REF shape: (T, nx)
    """
    pts = np.c_[X_REF[:, 0], np.zeros(len(X_REF)), X_REF[:, 2]]

    pyb.addUserDebugPoints(
        pointPositions=pts.tolist(),
        pointColorsRGB=[[0.2, 0.2, 0.2]] * len(pts),
        pointSize=3.,
        physicsClientId=env.PYB_CLIENT,
    )


# ============================================================
# Environment factories
# ============================================================
def make_env_stabilization(gui=False, **kwargs):
    CONFIG_FACTORY = ConfigFactory()
    config = CONFIG_FACTORY.merge([
        "examples/pid/config_overrides/quadrotor_2D/quadrotor_2D_stab.yaml",
    ])
    env_func = partial(make, "quadrotor", **config.task_config)
    return env_func(gui=gui, **kwargs)


def make_env_tracking(gui=False, **kwargs):
    CONFIG_FACTORY = ConfigFactory()
    config = CONFIG_FACTORY.merge([
        "examples/pid/config_overrides/quadrotor_2D/quadrotor_2D_track.yaml",
    ])

    env_func = partial(make, "quadrotor", **config.task_config)
    env = env_func(gui=gui, **kwargs)
    return env


def make_tracking_baseline(env):
    CONFIG_FACTORY = ConfigFactory()
    config = CONFIG_FACTORY.merge([
        "examples/pid/config_overrides/quadrotor_2D/quadrotor_2D_track.yaml",
    ])

    return make(
        "pid",
        lambda **kwargs: env,   # always return the SAME env
        **config.task_config
    )


# ============================================================
# FULL ACTION SPACE DISCRETIZATION
# ============================================================
def build_full_action_grid(env, N=30):
    low, high = env.physical_action_bounds
    grids = [np.linspace(low[i], high[i], N) for i in range(env.action_dim)]
    mesh = np.meshgrid(*grids)
    U = np.vstack([m.flatten() for m in mesh]).T
    return U


def make_log_template(T, nx, nu, extra_keys=None):
    log = {
        'tracking': np.zeros(T),
        'effort': np.zeros(T),
        'x': np.zeros((T, nx)),
        'u': np.zeros((T, nu)),
        'stage_cost': np.zeros(T),
        'episode_cost': 0.0,
    }

    if extra_keys is not None:
        for k, v in extra_keys.items():
            log[k] = v

    return log


# ============================================================
# PRIMITIVE CONTROLLERS
# ============================================================
class Primitive:
    def __init__(self, pid_ctrl):
        self.ctrl = pid_ctrl
        self.env = pid_ctrl.env

    def reset(self):
        self.ctrl.reset()

    def select_action(self, x, k):
        # --- set environment state
        self.env.state = x.copy()

        # ================================
        # State-holding stabilization
        # ================================
        if self.env.TASK == Task.STABILIZATION:
            x_ref = np.zeros_like(self.ctrl.reference)
            # Stay into current position
            x_ref[0] = x[0]   # x position
            x_ref[2] = x[2]   # z position

            self.ctrl.reference = x_ref

        u = self.ctrl.select_action(x, {'current_step': k})
        low, high = self.env.physical_action_bounds
        u = np.clip(u, low, high)

        return u


# ============================================================
# COST FUNCTIONS
# ============================================================
def tracking_cost(x, u, x_ref, u_ref, Q, R):
    dx = x - x_ref
    du = u - u_ref
    return dx.T @ Q @ dx + du.T @ R @ du


def linear_nominal_step(x, u, x_eq, u_eq, A_d, B_d):
    dx = x - x_eq
    du = u - u_eq
    return x_eq + A_d @ dx + B_d @ du


# ============================================================
# DISTRIBUTIONS
# ============================================================
def mixture_distribution(u_prims, U, w, sigma_act):
    """
    Build discrete mixture distribution over U
    using Gaussian primitives centered at u_prims.
    """
    n_primitives = u_prims.shape[0]
    n_actions = U.shape[0]

    pis = np.zeros((n_primitives, n_actions))
    p = np.zeros(n_actions)

    for i in range(n_primitives):
        d2 = np.sum((U - u_prims[i])**2, axis=1)
        vals = -d2 / (2 * sigma_act ** 2)
        pis[i] = np.exp(vals)
        pis[i] /= np.sum(pis[i])
        p += w[i] * pis[i]

    return p, pis


def generative_distribution(x, k, U, U_REF, X_REF, Q_eval, R_eval, x_eq, u_eq, A_d, B_d):
    n_actions = U.shape[0]
    costs = np.zeros(n_actions)
    u_ref = U_REF[k] if U_REF.ndim == 2 else U_REF
    x_ref_next = X_REF[min(k + 1, len(X_REF) - 1)]

    for m, u in enumerate(U):
        x_next = linear_nominal_step(x, u, x_eq, u_eq, A_d, B_d)
        costs[m] = tracking_cost(
            x_next,
            u,
            x_ref_next,
            u_ref,
            Q_eval,
            R_eval
        )

    q = np.exp(-costs) + 1e-8
    return q / np.sum(q)


# ============================================================
# GRADIENT
# ============================================================
def grad_F(p, q, pis, epsil=1e-8):

    p_safe = np.clip(p, epsil, None)
    q_safe = np.clip(q, epsil, None)

    log_term = np.log(p_safe) - np.log(q_safe)

    n_primitives = len(pis)

    grad = np.zeros(n_primitives)
    for i in range(n_primitives):
        grad[i] = np.sum(pis[i] * log_term)

    return grad


# ============================================================
# WEIGHT DYNAMICS
# ============================================================
def softmax(z):
    z = z - np.max(z)
    return np.exp(z) / np.sum(np.exp(z))


def w_dyn(t, w, gradF, eps, tau):
    return 1/tau * (-w + softmax(-gradF / eps))


def solve_weights(w0, gradF, eps=0.05, tau=1):
    sol = solve_ivp(
        fun=lambda t, w: w_dyn(t, w, gradF, eps, tau),
        method='RK45',
        #method='DOP853',
        t_span=(0.0, 10),
        y0=w0,
        rtol=1e-8,
        atol=1e-10,
    )
    w = sol.y[:, -1]
    w = np.clip(w, 1e-8, None)
    return w / np.sum(w)


# ============================================================
# MIXTURE CONTROL POLICY
# ============================================================
def sample_mixture_action(u_prims, w, u_low, u_high):
    """
    Sample action from mixture
    """
    n_primitives = u_prims.shape[0]
    # --- choose component
    i = np.random.choice(n_primitives, p=w)
    # --- sample
    u = u_prims[i]
    return np.clip(u, u_low, u_high)


def stochastic_primitive_controller(i, primitives):
    def ctrl(obs, k):
        env = primitives[i].env
        u_mean = primitives[i].select_action(obs, k)
        u = u_mean
        return np.clip(u, env.action_space.low, env.action_space.high)
    return ctrl


def make_primitives():
    pd_stab = PID(env_func=make_env_stabilization)
    pd_track = PID(env_func=make_env_tracking)

    pd_stab.reset()
    pd_stab.I_COEFF_FOR[:] = 0.00
    pd_stab.I_COEFF_TOR[:] = 0.00
    pd_stab.P_COEFF_FOR = np.array([0., 0., 0.])
    pd_stab.D_COEFF_FOR = np.array([0.05, 0., 0.1])
    pd_stab.P_COEFF_TOR = np.array([0., 70000., 0.])
    pd_stab.D_COEFF_TOR = np.array([0., 20000., 0.])

    pd_track.reset()
    pd_track.I_COEFF_FOR[:] = 0.00
    pd_track.I_COEFF_TOR[:] = 0.00
    pd_track.P_COEFF_FOR = np.array([0.45, 0., 1.])
    pd_track.D_COEFF_FOR = np.array([0.2, 0., 0.35])
    pd_track.P_COEFF_TOR = np.array([0., 14000., 0.])
    pd_track.D_COEFF_TOR = np.array([0., 10000., 0.])

    return [
        Primitive(pd_stab),    # Primitive 1: stabilizing PD
        Primitive(pd_track),   # Primitive 2: tracking PD
    ]


def rollout_mixture_adaptive(env, sigma_act, nx, nu, X_REF, U_REF, Q_eval, R_eval, primitives, U, T, x_eq, u_eq, A_d, B_d, seed):
    np.random.seed(seed)
    obs, info = env.reset(seed=seed)
    set_render_title(seed, f"Ours")
    set_camera(env)
    draw_reference_trajectory(env, X_REF)

    for pi in primitives:
        if hasattr(pi, "reset"):
            pi.reset()

    print("------- Running ours -------")

    w = np.ones(len(primitives)) / len(primitives)

    log = make_log_template(
        T,
        nx,
        nu,
        extra_keys={
            'w': np.zeros((T, len(primitives)))
        }
    )

    for k in range(T):
        log['x'][k] = obs
        x = obs

        # --- evaluate primitives
        u_prims = []
        for pi in primitives:
            u_prims.append(pi.select_action(x, k))
        u_prims = np.array(u_prims)
        # --- mixture distribution
        p, pis = mixture_distribution(u_prims, U, w, sigma_act)
        q = generative_distribution(x, k, U, U_REF, X_REF, Q_eval, R_eval, x_eq, u_eq, A_d, B_d)
        gradF = grad_F(p, q, pis)
        w = solve_weights(w, gradF)
        # --- sample mixture action
        u_mix = sample_mixture_action(u_prims, w, env.action_space.low, env.action_space.high)
        # --- apply
        obs, _, done, info = env.step(u_mix)
        # --- logging
        u_mix = env.current_noisy_physical_action
        log['u'][k] = u_mix
        dx = obs - X_REF[k+1]  # state error
        log['tracking'][k] = dx.T @ Q_eval @ dx
        du = u_mix - U_REF
        log['effort'][k] = du.T @ R_eval @ du
        log['stage_cost'][k] = dx.T @ Q_eval @ dx + du.T @ R_eval @ du
        log['w'][k] = w

        if env.GUI:
            time.sleep(env.CTRL_TIMESTEP)

        if done:
            log['episode_cost'] = np.sum(log['stage_cost'])
            break
    return log


def rollout_controller(env, nx, nu, Q_eval, R_eval, U_REF, X_REF, controller_fn, T, seed, ind):

    np.random.seed(seed)
    # fixed initial condition
    obs, info = env.reset(seed=seed)
    if env.GUI:
        if ind == 0:
            set_render_title(seed, f"Stabilizing PD primitive")
        else:
            set_render_title(seed, f"Tracking PD primitive")
        set_camera(env)
        draw_reference_trajectory(env, X_REF)

    print(f"------- Running primitive {ind+1} -------")

    log = make_log_template(T, nx, nu)

    for k in range(T):
        log['x'][k] = obs
        u = controller_fn(obs, k)

        obs, _, done, info = env.step(u)
        # Log
        u = env.current_noisy_physical_action
        log['u'][k] = u
        dx = obs - X_REF[k+1]  # state error
        log['tracking'][k] = dx.T @ Q_eval @ dx
        du = u - U_REF
        log['effort'][k] = du.T @ R_eval @ du
        log['stage_cost'][k] = dx.T @ Q_eval @ dx + du.T @ R_eval @ du

        if env.GUI:
            time.sleep(env.CTRL_TIMESTEP)

        if done:
            log['episode_cost'] = np.sum(log['stage_cost'])
            break
    return log


def rollout_baseline(env, nx, nu, Q_eval, R_eval, U_REF, X_REF, T, seed, name="PID"):
    print(f"------- Running Baseline {name} -------")

    np.random.seed(seed)

    obs, info = env.reset(seed=seed)
    ctrl = make_tracking_baseline(env)

    if env.GUI:
        set_render_title(seed, f"Baseline {name}")
        set_camera(env)
        draw_reference_trajectory(env, X_REF)

    log = make_log_template(T, nx, nu)

    for k in range(T):
        log['x'][k] = obs

        u = ctrl.select_action(obs, info)
        u = np.clip(u, env.action_space.low, env.action_space.high)

        obs, _, done, info = env.step(u)
        # Log
        u = env.current_noisy_physical_action
        log['u'][k] = u
        dx = obs - X_REF[k+1]
        log['tracking'][k] = dx.T @ Q_eval @ dx
        du = u - U_REF
        log['effort'][k] = du.T @ R_eval @ du
        log['stage_cost'][k] = dx.T @ Q_eval @ dx + du.T @ R_eval @ du

        if env.GUI:
            time.sleep(env.CTRL_TIMESTEP)

        if done:
            log['episode_cost'] = np.sum(log['stage_cost'])
            break
    return log


def main():
    env_tmp = make_env_tracking(gui=False)
    obs, info = env_tmp.reset()

    X_REF = info["x_reference"]
    U_REF = info["u_reference"]
    symbolic = info["symbolic_model"]

    nx = obs.shape[0]
    nu = env_tmp.action_space.shape[0]
    U = build_full_action_grid(env_tmp, N=30)

    env_tmp.close()
    if pyb.isConnected():
        pyb.disconnect()

    Q_eval = np.diag([1, 0.1, 1, 0.1, 0.1, 0.1])
    R_eval = 0.1 * np.eye(nu)

    sigma_act = 0.01    # actuation noise

    # LINEARIZED DYNAMICS
    df = symbolic.df_func(
        x=symbolic.X_EQ.reshape(-1, 1),
        u=symbolic.U_EQ.reshape(-1, 1)
    )
    A_c = df['dfdx'].toarray()
    B_c = df['dfdu'].toarray()
    A_d, B_d = discretize_linear_system(
        A_c, B_c, symbolic.dt, exact=True
    )
    x_eq = symbolic.X_EQ
    u_eq = symbolic.U_EQ

    # INITIALIZATION
    N_seeds = 6  # number of seeds to test
    T = X_REF.shape[0]-1

    # INITIALIZATION
    logs_seeds = []

    disturbances = {
        'dynamics': [
            {
                'disturbance_func': 'white_noise',
                'std': [0.03, 0.03]
            }
        ],
        'action': [
            {
                'disturbance_func': 'white_noise',
                'std': [sigma_act, sigma_act]
            }]
    }

    config = ConfigFactory().merge([
        "examples/pid/config_overrides/quadrotor_2D/quadrotor_2D_track.yaml"
    ])
    config.task_config.pop("randomized_init", None)
    config.task_config["done_on_out_of_bound"] = False  # Avoid early termination
    config.task_config.pop("seed", None)
    config.task_config["gui"] = False     # Set to True to see rendering
    x0 = np.array([0, 0., 1.2, 0., 0., 0.])     # Initial state
    config.task_config["init_state"] = x0

    for seed in range(N_seeds):

        print(f"\n============== Running seed {seed+1}/{N_seeds} ==============\n")
        seed += 20000

        np.random.seed(seed)

        # Make environment
        env = make(
            'quadrotor',
            randomized_init=False,
            **config.task_config,
            disturbances=disturbances,
            seed=seed
        )
        primitives = make_primitives()
        n_primitives = len(primitives)

        # Mixture rollout
        logs_mixture = rollout_mixture_adaptive(env, sigma_act, nx, nu, X_REF, U_REF, Q_eval, R_eval, primitives, U, T, x_eq, u_eq, A_d, B_d, seed)

        cid = env.PYB_CLIENT
        env.close()
        if pyb.isConnected(cid):
            pyb.disconnect(cid)

        for prim in primitives:
            if hasattr(prim, "ctrl") and hasattr(prim.ctrl, "env"):
                prim.ctrl.env.close()
                if pyb.isConnected(prim.ctrl.env.PYB_CLIENT):
                    pyb.disconnect(prim.ctrl.env.PYB_CLIENT)

        # Individual primitive rollouts
        logs_primitives = []
        for i in range(n_primitives):
            env = make(
                'quadrotor',
                randomized_init=False,
                **config.task_config,
                disturbances=disturbances,
                seed=seed
            )

            primitives = make_primitives()
            logs_primitives.append(
                rollout_controller(env, nx, nu, Q_eval, R_eval, U_REF, X_REF, stochastic_primitive_controller(i, primitives), T, seed, i)
            )

            cid = env.PYB_CLIENT
            env.close()
            if pyb.isConnected(cid):
                pyb.disconnect(cid)

            for prim in primitives:
                if hasattr(prim, "ctrl") and hasattr(prim.ctrl, "env"):
                    prim.ctrl.env.close()
                    if pyb.isConnected(prim.ctrl.env.PYB_CLIENT):
                        pyb.disconnect(prim.ctrl.env.PYB_CLIENT)

        # ---------------------------
        # Baseline PID rollout
        # ---------------------------
        env_pid_baseline = make(
            'quadrotor',
            randomized_init=False,
            **config.task_config,
            disturbances=disturbances,
            seed=seed
        )
        logs_pid_baseline = rollout_baseline(
            env_pid_baseline,
            nx,
            nu,
            Q_eval,
            R_eval,
            U_REF,
            X_REF,
            T,
            seed,
            name="PID"
        )

        cid = env_pid_baseline.PYB_CLIENT
        env_pid_baseline.close()
        if pyb.isConnected(cid):
            pyb.disconnect(cid)

        logs_seeds.append({
            'mixture': logs_mixture,
            'primitives': logs_primitives,
            'pid_baseline': logs_pid_baseline,
        })

    # Save results
    RESULTS_DIR = "Results_PD-PD"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("\nSaving experiment data...")
    np.savez_compressed(
        os.path.join(RESULTS_DIR, "experiment_data.npz"),
        logs_seeds=np.array(logs_seeds, dtype=object),
        X_REF=X_REF,
        U_REF=U_REF,
        x0=x0,
        sigma_act=sigma_act,
        n_primitives=n_primitives,
        N_seeds=N_seeds,
        T=T,
    )
    print("Saved →", os.path.join(RESULTS_DIR, "experiment_data.npz"))


if __name__ == "__main__":
    main()
    sys.exit(0)
