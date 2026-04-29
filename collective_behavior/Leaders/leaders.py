import numpy as np
from scipy import stats
from tqdm import tqdm
import pickle
from scipy.integrate import solve_ivp


class Boid:
    def __init__(self, position, velocity, weights):
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.weights = np.array(weights, dtype=float)
        self.trajectory = [self.position.copy()]  # Track position history

    def get_neighbors(self, boids, perception_radius, fov_degrees=320):
        neighbors = []
        fov_cosine_threshold = np.cos(np.radians(fov_degrees / 2))

        for other_boid in boids:
            if other_boid is not self:
                offset = other_boid.position - self.position
                distance = np.linalg.norm(offset)

                if distance < perception_radius:
                    forward = self.velocity / (np.linalg.norm(self.velocity) + 1e-8)
                    direction_to_other = offset / (distance + 1e-8)
                    dot_product = np.dot(forward, direction_to_other)

                    if dot_product >= fov_cosine_threshold:
                        neighbors.append(other_boid)

        return neighbors


def g_repulsion(d, r_rep):
    if 0 < d < r_rep:
        return 1.0 - d / r_rep
    return 0.0


def g_attraction(d, r_al, r_att):
    if r_al < d < r_att:
        x = (d - r_al) / (r_att - r_al)
        return (1 - x)
    return 0.0


def separation_force(boid, neighbors, r_rep):
    force = np.zeros(2)
    count = 0
    for other in neighbors:
        offset = boid.position - other.position
        d = np.linalg.norm(offset)
        if 0 < d <= r_rep:
            direction = offset / d
            weight = g_repulsion(d, r_rep)
            force += weight * direction
            count += 1
    if count > 0:
        force /= count
    return force


def alignment_force(boid, neighbors, r_rep, r_al):
    avg_heading = np.zeros(2)
    count = 0
    for other in neighbors:
        d = np.linalg.norm(other.position - boid.position)
        if r_rep < d <= r_al:
            velocity_dir = other.velocity / np.linalg.norm(other.velocity)
            avg_heading += velocity_dir
            count += 1
    if count > 0:
        avg_heading /= count
        return avg_heading
    return boid.velocity / np.linalg.norm(boid.velocity)


def cohesion_force(boid, neighbors, r_al, r_att):
    force = np.zeros(2)
    count = 0
    for other in neighbors:
        offset = other.position - boid.position
        d = np.linalg.norm(offset)
        if r_al < d <= r_att:
            direction = offset / d
            weight = g_attraction(d, r_al, r_att)
            force += weight * direction
            count += 1
    if count > 0:
        force /= count
    return force


def softmax(x):
    x = np.array(x)
    x_max = np.max(x)
    ex = np.exp(x - x_max)  # subtract max for numerical stability
    return ex / np.sum(ex)


def compute_primitives(boid, neighbors, radii, primitives_var, N_primitives, u_grid, u_max_norm):
    F = u_max_norm*np.array([
        separation_force(boid, neighbors, radii['sep']),
        alignment_force(boid, neighbors, radii['sep'], radii['ali']),
        cohesion_force(boid, neighbors, radii['ali'], radii['coh'])
    ])

    pi_u = np.zeros((N_primitives, len(u_grid)))

    for j in range(N_primitives):
        # Clip F[j] to max norm to avoid out-of-bound means
        norm_F = np.linalg.norm(F[j])
        mean_vec = F[j]

        if norm_F > u_max_norm:
            mean_vec = (F[j] / norm_F) * u_max_norm

        # Evaluate PDF for all u_grid points relative to mean_vec
        cov = np.diag([primitives_var[j], primitives_var[j]])
        pdf_vals = stats.multivariate_normal.pdf(u_grid, mean=mean_vec, cov=cov) + 1e-10

        pi_u[j] = pdf_vals / np.sum(pdf_vals)

    return pi_u  # shape: (N_primitives, M)


def compute_steering(primitives, weights, u_grid, u_max_norm):
    # primitives shape: (N_primitives, M)
    p_u_steering = np.sum(primitives.T * weights, axis=1) + 1e-10  # shape: (M,)
    p_u_steering /= np.sum(p_u_steering)
    steering_ind = np.random.choice(len(u_grid), p=p_u_steering)
    steering = u_grid[steering_ind]
    # Norm clipping
    norm_steering = np.linalg.norm(steering)
    if norm_steering > u_max_norm:
        steering = (steering / norm_steering) * u_max_norm
    return steering


def update_boid(boid, w, steering, dt, max_speed):
    boid.weights = w
    boid.position += boid.velocity * dt
    boid.trajectory.append(boid.position.copy())
    velocity_update = steering * dt
    boid.velocity += velocity_update
    # Clip velocity
    norm_v = np.linalg.norm(boid.velocity)
    if norm_v > max_speed:
        boid.velocity = (boid.velocity / norm_v) * max_speed


def weight_dynamics(t, w, pi_u_flat, c_tilde, eps):
    sum_terms = w @ pi_u_flat
    log_sum = np.log(np.clip(sum_terms, 1e-10, None))
    exp_terms = np.sum(pi_u_flat * (log_sum + c_tilde), axis=1)
    softmax_result = softmax(-1 / eps * exp_terms)
    tau = 1
    return 1/tau * (-w + softmax_result)


def integrate_weights(w0, pi_u_flat, c_tilde, epsilon, t_span, deltat):
    sol = solve_ivp(weight_dynamics, t_span, w0, method='RK45', t_eval=np.arange(t_span[0], t_span[1], deltat),
                    args=(pi_u_flat, c_tilde, epsilon), vectorized=False, rtol=1e-6, atol=1e-8)
    w = sol.y.T[-1]
    if np.any(w < 0):
        w = np.clip(w, 0, None)
        w /= np.sum(w)
    return w


lambda_z_p = 100.0
lambda_z_v = 100.0
dim = 4
cov_p = np.diag([1 / lambda_z_p, 1 / lambda_z_p, 1 / lambda_z_v, 1 / lambda_z_v])
cov_q = cov_p
inv_cov_q = np.diag([lambda_z_p, lambda_z_p, lambda_z_v, lambda_z_v])


def update_weights(boid, neighbors, pi_u, dt, N_primitives, u_grid, speed_limit, radii, informed, eps):
    p_prev = boid.position
    v_prev = boid.velocity
    w = boid.weights.copy()
    log_det_cov_ratio = np.log(np.linalg.det(cov_q) / np.linalg.det(cov_p))
    # Predict new velocities for each control input
    v_next = v_prev + u_grid * dt
    # Compute norms
    norms = np.linalg.norm(v_next, axis=1)  # shape (M,)
    # Boolean mask for speeds exceeding limit
    too_fast = norms > speed_limit  # shape (M,)
    # Clip velocities by scaling down to speed_limit where necessary
    v_next[too_fast] = v_next[too_fast] / norms[too_fast, np.newaxis] * speed_limit
    # Predict next positions (constant velocity step)
    p_next = p_prev + v_prev * dt  # shape (2,)
    # Tile p_next and v_next for batch calculation
    x_next = np.hstack([np.tile(p_next, (len(v_next), 1)), v_next])  # shape (M, 4)
    center_sum = np.zeros(2)
    center_count = 0
    vel_sum = np.zeros(2)
    vel_count = 0
    boid_pos = boid.position
    for other in neighbors:
        dist = np.linalg.norm(other.position - boid_pos)
        if radii['ali'] < dist < radii['coh']:
            center_sum += other.position
            center_count += 1
        if radii['sep'] < dist < radii['ali']:
            vel_sum += other.velocity
            vel_count += 1
    mean_pos = center_sum / center_count if center_count > 0 else boid_pos
    mean_vel = vel_sum / vel_count if vel_count > 0 else boid.velocity
    mean_target = goal_point if informed else mean_pos
    direction_to_goal = goal_point - boid.position
    target_vel = direction_to_goal / (np.linalg.norm(direction_to_goal) + 1e-8) if informed else mean_vel # normalized vector
    target = np.hstack([np.tile(mean_target, (len(v_next), 1)), np.tile(target_vel, (len(v_next), 1))])
    delta_mu = target - x_next
    mahalanobis = np.sum(delta_mu @ inv_cov_q * delta_mu, axis=1)
    kl_divs = 0.5 * (-dim + np.trace(inv_cov_q @ cov_p) + mahalanobis + log_det_cov_ratio)  # shape (M,)
    # Uniform distribution
    q_u = np.ones(len(u_grid)) / len(u_grid)
    log_q_u = np.log(q_u)
    # Compute softmax update
    c_tilde = kl_divs - log_q_u
    pi_u_flat = pi_u.reshape(N_primitives, -1)  # Shape: (3, n_u_bins^2)
    t_span = (0, 10)
    deltat = 0.01
    w_new = integrate_weights(w, pi_u_flat, c_tilde, eps, t_span, deltat)
    boid.weights = w_new

    return w_new


def run_step(frame_num):
    if step_counter['stop'] or frame_num >= max_steps:
        return []

    step_counter['frame'] = frame_num

    for i, boid in enumerate(flock):
        neighbors = boid.get_neighbors(flock, PERCEPTION_RADIUS)
        primitives = compute_primitives(boid, neighbors, radii, primitives_var, N_primitives, u_grid, u_max_norm)
        if boid in flock[:informed_boids]:
            informed = 1
        else:
            informed = 0
        wk = update_weights(boid, neighbors, primitives, dt, N_primitives, u_grid, speed_limit, radii, informed, eps)
        steering = compute_steering(primitives, wk, u_grid, u_max_norm)
        update_boid(boid, wk, steering, dt, speed_limit)

        '''if boid_out_of_bounds(boid):
            step_counter['stop'] = True
            print(f"Simulation stopped at frame {frame_num} due to boundary exit.")
            return []'''

    # Compute and store polarization and milling
    P = compute_polarization(flock)
    polarization_history.append(P)
    milling = compute_milling_metric(flock)
    milling_history.append(milling)

    frame_weights = [boid.weights for boid in flock]
    weights_history.append(frame_weights)
    velocities_history.append([np.linalg.norm(boid.velocity) for boid in flock])

    # Compute and store mean flock position and its distance from the goal
    mean_pos = np.mean([b.position for b in flock], axis=0)
    distance = np.linalg.norm(mean_pos - goal_point)
    distance_history.append(distance)

    return


# Function to check if any boid is out of bounds
def boid_out_of_bounds(boid):
    return (np.abs(boid.position) > boundary).any()


def compute_polarization(boids):
    normed_vels = [b.velocity / (np.linalg.norm(b.velocity) + 1e-8) for b in boids]
    avg_direction = np.mean(normed_vels, axis=0)
    return np.linalg.norm(avg_direction)


def compute_milling_metric(flock):
    positions = np.array([b.position for b in flock])
    velocities = np.array([b.velocity for b in flock])
    cm = positions.mean(axis=0)
    rel_positions = positions - cm
    normed_rel_pos = rel_positions / (np.linalg.norm(rel_positions, axis=1, keepdims=True) + 1e-8)
    normed_vel = velocities / (np.linalg.norm(velocities, axis=1, keepdims=True) + 1e-8)
    # 2D cross product: r_i x v_i = x_i * v_y - y_i * v_x
    cross_products = normed_rel_pos[:, 0] * normed_vel[:, 1] - normed_rel_pos[:, 1] * normed_vel[:, 0]
    milling_metric = np.abs(np.mean(cross_products))
    return milling_metric


# Simulation parameters
N_primitives = 3
num_boids = 40
print("Total number of agents: ", num_boids)
informed_boids = 4
print("Number of informed agents: ", informed_boids)
speed_limit = 1.0
# Initial conditions
np.random.seed(5)
positions = (np.random.rand(num_boids, 2)-0.5) * 5
# Generate random velocities
velocities = (np.random.rand(num_boids, 2) - 0.5) * 2
# Compute norms
norms = np.linalg.norm(velocities, axis=1)
# Boolean mask for velocities above speed limit
too_fast = norms > speed_limit
# Scale only the ones that are too fast
velocities[too_fast] = velocities[too_fast] / norms[too_fast][:, np.newaxis] * speed_limit
# Uniform weights
weights = np.ones((num_boids, N_primitives))/N_primitives
flock = [Boid(pos, vel, w) for pos, vel, w in zip(positions, velocities, weights)]
radii = {'sep': 1, 'ali': 3, 'coh': 12.0}
dt = 0.05
area_limits = np.array([25, 25])
boundary = area_limits[0]  # limit assumed symmetric
PERCEPTION_RADIUS = radii['coh']
n_u_bins = 30
u_max_norm = 3
u_vals = np.linspace(-u_max_norm, u_max_norm, n_u_bins)
u_axis = [u_vals, u_vals]
u_grid = np.array([
    [ux, uy]
    for ux in u_axis[0]
    for uy in u_axis[1]
    if np.linalg.norm([ux, uy]) <= u_max_norm
])
primitives_var = [0.1, 0.1, 0.1]
# Entropy-regularization gain
eps = 0.5
max_steps = 1220
goal_point = np.array([-15, -15])
step_counter = {'frame': 0, 'stop': False}  # Mutable object to track stopping

polarization_history = []
distance_history = []   # distance form the goal
milling_history = []
weights_history = []
velocities_history = []


# Run simulation
for frame in tqdm(range(max_steps), desc="Running simulation"):
    run_step(frame)
    if step_counter['stop']:
        print(f"\nSimulation stopped at frame {frame}.")
        break


# Save history
np.save(f"weights_history.npy", weights_history)
np.save(f"polarization_history.npy", polarization_history)
np.save(f"milling_history.npy", milling_history)
np.save(f"distance_history.npy", distance_history)

# Save trajectories for all boids
all_trajectories = [boid.trajectory for boid in flock]
with open("boids_trajectories.pkl", "wb") as f:
    pickle.dump(all_trajectories, f)
