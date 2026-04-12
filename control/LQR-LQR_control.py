import numpy as np
import safe_control_gym
from safe_control_gym.utils.registration import make
from scipy.integrate import solve_ivp
from safe_control_gym.envs.benchmark_env import Task, Cost
from safe_control_gym.controllers.lqr.lqr import LQR
from safe_control_gym.controllers.lqr.lqr_utils import discretize_linear_system
from safe_control_gym.utils.configuration import ConfigFactory
import sys
import matplotlib.pyplot as plt
from functools import partial
import time
import os
import pybullet as pyb
from safe_control_gym.controllers.lqr.lqr_utils import (
    compute_lqr_gain,
    get_cost_weight_matrix
)


def make_log_template(T, nx, nu, extra_keys=None):
    log = {
        'tracking': np.zeros(T),
        'effort': np.zeros(T),
        'u_norm': np.zeros(T),
        'x': np.zeros((T, nx)),
        'u': np.zeros((T, nu)),
        'stage_cost': np.zeros(T),
        'episode_cost': 0.0,
    }

    if extra_keys is not None:
        for k, v in extra_keys.items():
            log[k] = v

    return log


def set_render_title(seed, controller_name, text_id=None):
    text = f"Seed: {seed} | {controller_name}"

    if text_id is None:
        # First time → create text
        text_id = pyb.addUserDebugText(
            text,
            textPosition=[0.38, 0, 1.6],
            textColorRGB=[0, 0, 0],
            textSize=1.3
        )
    else:
        # Update existing text
        text_id = pyb.addUserDebugText(
            text,
            textPosition=[0.38, 0, 1.6],
            textColorRGB=[0, 0, 0],
            textSize=1.3,
            replaceItemUniqueId=text_id
        )
    return


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


'''def draw_reference_trajectory(env, X_REF, color=[0.2, 0.2, 0.2]):
    for k in range(len(X_REF) - 1):
        pyb.addUserDebugLine(
            [X_REF[k, 0], 0, X_REF[k, 2]],
            [X_REF[k + 1, 0], 0, X_REF[k + 1, 2]],
            lineColorRGB=color,
            lineWidth=5,
            lifeTime=0,  # stays forever
            physicsClientId=env.PYB_CLIENT
        )'''


# ============================================================
# Environment factories
# ============================================================
def make_env_tracking(gui=False, **kwargs):
    CONFIG_FACTORY = ConfigFactory()
    config = CONFIG_FACTORY.merge([
        "examples/lqr/config_overrides/quadrotor_2D/quadrotor_2D_track.yaml",
        "examples/lqr/config_overrides/quadrotor_2D/lqr_quadrotor_2D_track.yaml"
    ])
    config.task_config.pop("randomized_init", None)
    config.task_config["done_on_out_of_bound"] = False  # Avoid early termination
    config.task_config.pop("seed", None)
    x0 = np.array([0, 0., 1.2, 0., 0., 0.])
    config.task_config["init_state"] = x0
    config.task_config.pop("init_state_randomization_info", None)

    env_func = partial(make, "quadrotor", **config.task_config)
    env = env_func(gui=gui, **kwargs)
    return env


def make_env_lqr_stabilization(gui=False, **kwargs):
    CONFIG_FACTORY = ConfigFactory()
    config = CONFIG_FACTORY.merge([
        "examples/lqr/config_overrides/quadrotor_2D/quadrotor_2D_stab.yaml",
        "examples/lqr/config_overrides/quadrotor_2D/lqr_quadrotor_2D_stab.yaml",
    ])
    config.task_config.pop("randomized_init", None)
    config.task_config["done_on_out_of_bound"] = False  # Avoid early termination
    config.task_config.pop("seed", None)
    x0 = np.array([0, 0., 1.2, 0., 0., 0.])
    config.task_config["init_state"] = x0
    config.task_config.pop("init_state_randomization_info", None)

    env_func = partial(make, "quadrotor", **config.task_config)
    return env_func(gui=gui, **kwargs)


def make_lqr_stabilization():
    CONFIG_FACTORY = ConfigFactory()

    config = CONFIG_FACTORY.merge([
        "examples/lqr/config_overrides/quadrotor_2D/quadrotor_2D_stab.yaml",
        "examples/lqr/config_overrides/quadrotor_2D/lqr_quadrotor_2D_stab.yaml",
    ])
    config.task_config.pop("randomized_init", None)
    config.task_config["done_on_out_of_bound"] = False  # Avoid early termination
    config.task_config.pop("seed", None)
    x0 = np.array([0, 0., 1.2, 0., 0., 0.])
    config.task_config["init_state"] = x0
    config.task_config.pop("init_state_randomization_info", None)

    return make(
        "lqr",
        make_env_lqr_stabilization,
        **config.algo_config
    )


def make_lqr_track():
    CONFIG_FACTORY = ConfigFactory()

    config = CONFIG_FACTORY.merge([
        "examples/lqr/config_overrides/quadrotor_2D/quadrotor_2D_track.yaml",
        "examples/lqr/config_overrides/quadrotor_2D/lqr_quadrotor_2D_track.yaml",
    ])
    config.task_config.pop("randomized_init", None)
    config.task_config["done_on_out_of_bound"] = False  # Avoid early termination
    config.task_config.pop("seed", None)
    x0 = np.array([0, 0., 1.2, 0., 0., 0.])
    config.task_config["init_state"] = x0
    config.task_config.pop("init_state_randomization_info", None)

    return make(
        "lqr",
        make_env_tracking,
        **config.algo_config
    )


def make_tracking_lqr_default(env):
    CONFIG_FACTORY = ConfigFactory()
    config = CONFIG_FACTORY.merge([
        "examples/lqr/config_overrides/quadrotor_2D/quadrotor_2D_track.yaml",
        "examples/lqr/config_overrides/quadrotor_2D/lqr_quadrotor_2D_track.yaml",
    ])

    return make(
        "lqr",
        lambda **kwargs: env,   # <-- SAME ENV
        **config.algo_config
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


# ============================================================
# PRIMITIVE CONTROLLERS
# ============================================================
class LQRPrimitive:
    def __init__(self, lqr_ctrl):
        self.ctrl = lqr_ctrl
        self.env = lqr_ctrl.env

    def reset(self):
        self.ctrl.reset()

    def select_action(self, x, k, info):
        # Let controller see current state
        self.env.state = x.copy()

        if self.env.TASK == Task.STABILIZATION:
            # x = [px, vx, pz, vz, theta, theta_dot]
            # Build ref equal to current position
            x_ref = np.zeros_like(self.ctrl.env.X_GOAL)
            x_ref[0] = x[0]   # x position
            x_ref[2] = x[2]   # z position
            self.ctrl.env.X_GOAL = x_ref

        # LQR already tracks x_ref internally
        u = self.ctrl.select_action(x, info)

        return np.clip(u, self.env.action_space.low, self.env.action_space.high)


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
    using Gaussian experts centered at u_prims.
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

    return p / np.sum(p), pis


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
    return q / np.sum(q), costs


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
    n_primitives = u_prims.shape[0]
    # --- choose component
    i = np.random.choice(n_primitives, p=w)
    # --- sample
    u = u_prims[i]
    return np.clip(u, u_low, u_high)


def stochastic_primitive_controller(i, primitives):
    def ctrl(obs, k, info):
        env = primitives[i].env
        x = obs

        u_mean = primitives[i].select_action(x, k, info)
        u = u_mean

        return np.clip(u, env.action_space.low, env.action_space.high)
    return ctrl



def make_primitives(seed):
    # --- Stabilizing LQR primitive
    lqr_stab = make_lqr_stabilization()
    lqr_stab.env.reset(seed=seed)
    Q = get_cost_weight_matrix(
        [1e-6, 0.05, 1e-6, 1e-5, 0.5, 0.1], lqr_stab.model.nx)
    R = get_cost_weight_matrix([0.01, 0.01], lqr_stab.model.nu)
    lqr_stab.Q = Q
    lqr_stab.R = R
    lqr_stab.gain = compute_lqr_gain(
        lqr_stab.model,
        lqr_stab.model.X_EQ,
        lqr_stab.model.U_EQ,
        Q,
        R,
        lqr_stab.discrete_dynamics
    )
    # --- Tracking LQR primitive
    lqr_track = make_lqr_track()
    lqr_track.env.reset(seed=seed)
    Q_t = get_cost_weight_matrix(
        [5, 0.1, 5, 0.1, 0.1, 0.1], lqr_track.model.nx)
    R_t = get_cost_weight_matrix([0.1, 0.1], lqr_track.model.nu)
    lqr_track.Q = Q_t
    lqr_track.R = R_t
    lqr_track.gain = compute_lqr_gain(
        lqr_track.model,
        lqr_track.model.X_EQ,
        lqr_track.model.U_EQ,
        Q_t,
        R_t,
        lqr_track.discrete_dynamics
    )

    return [
        LQRPrimitive(lqr_stab),   # Primitive 1: stabilizing LQR
        LQRPrimitive(lqr_track),  # Primitive 2: tracking LQR
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
    w = np.ones(len(primitives)) / len(primitives)
    log = make_log_template(
        T,
        nx,
        nu,
        extra_keys={
            'w': np.zeros((T, len(primitives))),
            'u_primitives': np.zeros((T, len(primitives), nu)),
            'u_mixture': np.zeros((T, nu)),
        }
    )

    print("------- Running ours -------")
    for k in range(T):
        log['x'][k] = obs
        x = obs

        # --- evaluate primitives
        u_prims = []
        for pi in primitives:
            u_prims.append(pi.select_action(x, k, info))
        u_prims = np.array(u_prims)
        # --- mixture distribution from cached actions
        p, pis = mixture_distribution(u_prims, U, w, sigma_act)

        q, costs = generative_distribution(x, k, U, U_REF, X_REF, Q_eval, R_eval, x_eq, u_eq, A_d, B_d)
        gradF = grad_F(p, q, pis)
        w = solve_weights(w, gradF)
        # --- sample mixture action
        u_mix = sample_mixture_action(u_prims, w, env.action_space.low, env.action_space.high)
        # --- apply
        obs, _, done, info = env.step(u_mix)
        u_mix = env.current_noisy_physical_action
        # --- logging
        log['u'][k] = u_mix
        log['w'][k] = w
        dx = obs - X_REF[k+1]  # state error
        log['tracking'][k] = dx.T @ Q_eval @ dx
        du = u_mix - U_REF
        log['effort'][k] = du.T @ R_eval @ du
        log['stage_cost'][k] = dx.T @ Q_eval @ dx + du.T @ R_eval @ du

        if env.GUI:
            time.sleep(env.CTRL_TIMESTEP)
            pyb.stepSimulation()

        if done:
            log['episode_cost'] = np.sum(log['stage_cost'])
            break
    return log


def rollout_controller(env, nx, nu, Q_eval, R_eval, U_REF, X_REF, controller_fn, T, seed, ind):
    np.random.seed(seed)
    obs, info = env.reset(seed=seed)
    if env.GUI:
        if ind == 0:
            set_render_title(seed, f"Stabilizing LQR primitive")
        else:
            set_render_title(seed, f"Tracking LQR primitive")
        set_camera(env)
        draw_reference_trajectory(env, X_REF)

    log = make_log_template(T, nx, nu)

    print(f"------- Running primitive {ind + 1} -------")
    for k in range(T):
        log['x'][k] = obs
        u = controller_fn(obs, k, info)
        obs, _, done, info = env.step(u)
        u = env.current_noisy_physical_action
        # Log
        log['u'][k] = u
        dx = obs - X_REF[k + 1]  # state error
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


def rollout_model_based(env, nx, nu, Q_eval, R_eval, U_REF, X_REF, T, seed, name="LQR"):
    np.random.seed(seed)
    obs, info = env.reset(seed=seed)
    ctrl = make_tracking_lqr_default(env)
    ctrl.env.reset(seed=seed)
    if env.GUI:
        set_render_title(seed, f"Baseline {name}")
        set_camera(env)
        draw_reference_trajectory(env, X_REF)

    log = make_log_template(T, nx, nu)

    print(f"------- Running Baseline {name} -------")
    for k in range(T):
        log['x'][k] = obs
        u = ctrl.select_action(obs, info)
        u = np.clip(u, env.action_space.low, env.action_space.high)
        obs, _, done, info = env.step(u)
        u = env.current_noisy_physical_action
        # Log
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


# ============================================================
# MAIN CONTROL LOOP FOR MULTIPLE SEEDS
# ============================================================
def main():
    # ENVIRONMENT SETUP
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

    sigma_act = 0.01

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
    N_seeds = 1000  # number of seeds to test
    T = X_REF.shape[0] - 1

    # Initialize storage
    logs_seeds = []

    enable_disturbance = 1

    if enable_disturbance:
        disturbances = {
            'dynamics': [
                {
                    'disturbance_func': 'white_noise',
                    'std': [0.03, 0.03]
                },
                {
                    'disturbance_func': 'impulse',
                    'magnitude': 1.
                }],
            'action': [
            {
                'disturbance_func': 'white_noise',
                'std': [sigma_act, sigma_act]
            }]
        }
    else:
        disturbances = None

    config = ConfigFactory().merge(
        ["./examples/lqr/config_overrides/quadrotor_2D/quadrotor_2D_track.yaml",
         "./examples/lqr/config_overrides/quadrotor_2D/lqr_quadrotor_2D_track.yaml"]
    )
    config.task_config.pop("randomized_init", None)
    config.task_config["done_on_out_of_bound"] = False  # Avoid early termination
    config.task_config.pop("seed", None)
    x0 = np.array([0, 0., 1.2, 0., 0., 0.])
    config.task_config["init_state"] = x0
    config.task_config.pop("init_state_randomization_info", None)
    config.task_config["gui"] = False  # Set to True to see rendering

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

        primitives = make_primitives(seed)
        n_primitives = len(primitives)

        # Mixture rollout
        logs_mixture = rollout_mixture_adaptive(env, sigma_act, nx, nu, X_REF, U_REF, Q_eval, R_eval, primitives, U, T, x_eq, u_eq, A_d, B_d, seed)

        cid = env.PYB_CLIENT
        env.close()
        if pyb.isConnected(cid):
            pyb.disconnect(cid)

        for prim in primitives:
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

            logs_primitives.append(
                rollout_controller(env, nx, nu, Q_eval, R_eval, U_REF, X_REF,
                                   stochastic_primitive_controller(i, primitives), T, seed, i)
            )

            cid = env.PYB_CLIENT
            env.close()
            if pyb.isConnected(cid):
                pyb.disconnect(cid)

            for prim in primitives:
                prim.ctrl.env.close()
                if pyb.isConnected(prim.ctrl.env.PYB_CLIENT):
                    pyb.disconnect(prim.ctrl.env.PYB_CLIENT)

        # ---------------------------
        # LQR rollout
        # ---------------------------
        env_lqr_baseline = make(
                'quadrotor',
                randomized_init=False,
                **config.task_config,
                disturbances=disturbances,
                seed=seed
        )

        logs_lqr = rollout_model_based(
            env_lqr_baseline,
            nx,
            nu,
            Q_eval,
            R_eval,
            U_REF,
            X_REF,
            T,
            seed,
            name="LQR"
        )

        cid = env_lqr_baseline.PYB_CLIENT
        env_lqr_baseline.close()
        if pyb.isConnected(cid):
            pyb.disconnect(cid)

        logs_seeds.append({
            'mixture': logs_mixture,
            'primitives': logs_primitives,
            'lqr': logs_lqr
        })

    # Save results
    RESULTS_DIR = "Results_LQR-LQR"
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
