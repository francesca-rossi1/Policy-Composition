import numpy as np
from scipy import stats
from tqdm import tqdm
import pickle
from scipy.integrate import solve_ivp
from scipy.spatial import distance


class Boid:
    def __init__(self, position, velocity, weights):
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.weights = np.array(weights, dtype=float)
        self.trajectory = [self.position.copy()]  # Track position history
        self.vel_trajectory = [self.velocity.copy()]  # Track velocity history

    def get_neighbors(self, boids, perception_radius, fov_degrees=320):
        neighbors = []
        fov_cosine_threshold = np.cos(np.radians(fov_degrees / 2))

        for other_boid in boids:
            if other_boid is not self:
                offset = other_boid.position - self.position
                d = np.linalg.norm(offset)

                if d < perception_radius:
                    forward = self.velocity / (np.linalg.norm(self.velocity) + 1e-8)
                    direction_to_other = offset / (d + 1e-8)
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
        x = (d - r_al) / (r_att - r_al)  # normalize to (0,1)
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
    F = u_max_norm * np.array([
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

    # Norm clipping just in case
    norm_steering = np.linalg.norm(steering)
    if norm_steering > u_max_norm:
        steering = (steering / norm_steering) * u_max_norm

    return steering


def update_boid(boid, w, steering, dt, max_speed):
    boid.weights = w
    boid.position += boid.velocity * dt
    velocity_update = steering * dt
    boid.velocity += velocity_update

    # Clip velocity per component or norm
    norm_v = np.linalg.norm(boid.velocity)
    if norm_v > max_speed:
        boid.velocity = (boid.velocity / norm_v) * max_speed

    boid.trajectory.append(boid.position.copy())
    boid.vel_trajectory.append(boid.velocity.copy())


def weight_dynamics(t, w, pi_u_flat, c_tilde, eps):
    sum_terms = w @ pi_u_flat
    log_sum = np.log(np.clip(sum_terms, 1e-10, None))
    exp_terms = np.sum(pi_u_flat * (log_sum + c_tilde), axis=1)
    softmax_result = softmax(-1 / eps * exp_terms)
    tau = 1
    return 1/tau * (-w + softmax_result)


def integrate_weights(w0, pi_u_flat, c_tilde, eps, t_span, deltat):
    sol = solve_ivp(weight_dynamics, t_span, w0, method='RK45', t_eval=np.arange(t_span[0], t_span[1], deltat),
                    args=(pi_u_flat, c_tilde, eps), vectorized=False, rtol=1e-6, atol=1e-8)
    weights_converged = sol.y.T[-1]
    if np.any(weights_converged <= 0):
        weights_converged = np.clip(weights_converged, 0, None)
    return weights_converged


lambda_z_p = 100.0
lambda_z_v = 100.0
dim = 4
cov_p = np.diag([1 / lambda_z_p, 1 / lambda_z_p, 1 / lambda_z_v, 1 / lambda_z_v]) #if informed else np.diag([1 / lambda_z, 1 / lambda_z, 0.1 / lambda_z, 0.1 / lambda_z])
cov_q = cov_p
inv_cov_q = np.diag([lambda_z_p, lambda_z_p, lambda_z_v, lambda_z_v])


def update_weights(boid, neighbors, pi_u, dt, N_primitives, u_grid, speed_limit, radii, eps):
    p_prev = boid.position
    v_prev = boid.velocity
    w = boid.weights.copy()

    log_det_cov_ratio = np.log(np.linalg.det(cov_q) / np.linalg.det(cov_p))

    # Predict new velocities for each control input u_vec
    v_next = v_prev + u_grid * dt
    # Compute norms without keepdims
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

    collision_costs = np.zeros(len(u_grid))
    for other in neighbors:
        dist = np.linalg.norm(other.position - boid_pos)

        if dist <= radii['sep']:
            delta_vec = other.position - boid_pos  # shape (2,)
            delta_norm = np.linalg.norm(delta_vec)
            unit_delta = delta_vec / delta_norm
            # Dot product u · delta direction
            dot_prods = u_grid @ unit_delta  # shape (M,)
            u_norms = np.linalg.norm(u_grid, axis=1) + 1e-8  # avoid 0 division
            # Cosine similarity
            cosines = dot_prods / u_norms
            # Cost: penalize forward motion scaled by action magnitude and proximity
            gain = 2
            penalties = gain * np.maximum(0, cosines) * u_norms / (delta_norm + 1e-6)
            collision_costs += penalties

        elif radii['ali'] < dist <= radii['coh']:
            center_sum += other.position
            center_count += 1

        elif radii['sep'] < dist <= radii['ali']:
            vel_sum += other.velocity
            vel_count += 1

    mean_pos = center_sum / center_count if center_count > 0 else boid_pos
    mean_vel = vel_sum / vel_count if vel_count > 0 else boid.velocity

    mean_target = mean_pos
    target_vel = mean_vel

    target = np.hstack([np.tile(mean_target, (len(v_next), 1)), np.tile(target_vel, (len(v_next), 1))])
    delta_mu = target - x_next

    mahalanobis = np.sum(delta_mu @ inv_cov_q * delta_mu, axis=1)
    kl_divs = 0.5 * (-dim + np.trace(inv_cov_q @ cov_p) + mahalanobis + log_det_cov_ratio)  # shape (M,)

    # Uniform distribution over available controls
    q_u = np.ones(len(u_grid)) / len(u_grid) * np.exp(-collision_costs)
    q_u = q_u / np.sum(q_u)
    log_q_u = np.log(np.clip(q_u, 1e-10, None))

    # Compute softmax update
    c_tilde = kl_divs - log_q_u
    pi_u_flat = pi_u.reshape(N_primitives, -1)  # Shape: (3, n_u_bins^2)
    t_span = (0, 10)
    w_new = integrate_weights(w, pi_u_flat, c_tilde, eps, t_span, 0.01)
    boid.weights = w_new

    return w_new


'''# Function to check if any boid is out of bounds
def boid_out_of_bounds(boid):
    return (np.abs(boid.position) > boundary).any()'''


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


def run_step(frame_num):
    if step_counter['stop'] or frame_num >= max_steps:
        return

    step_counter['frame'] = frame_num

    for i, boid in enumerate(flock):
        neighbors = boid.get_neighbors(flock, PERCEPTION_RADIUS)
        primitives = compute_primitives(boid, neighbors, radii, primitives_var, N_primitives, u_grid, u_max_norm)
        wk = update_weights(boid, neighbors, primitives, dt, N_primitives, u_grid, speed_limit, radii, eps)
        steering = compute_steering(primitives, wk, u_grid, u_max_norm)
        update_boid(boid, wk, steering, dt, speed_limit)

        '''if boid_out_of_bounds(boid):
            step_counter['stop'] = True
            print(f"Simulation stopped at frame {frame_num} due to boundary exit.")
            return'''

    # Compute and store polarization
    P = compute_polarization(flock)
    polarization_history.append(P)
    milling = compute_milling_metric(flock)
    milling_history.append(milling)

    frame_weights = [boid.weights for boid in flock]
    weights_history.append(frame_weights)

    # Compute all pairwise distances (condensed form)
    dists = distance.pdist([boid.position for boid in flock])
    # Find the minimum
    min_dist.append(np.min(dists))

    return


# Simulation parameters
N_primitives = 3
num_boids = 60
np.random.seed(5)
radius = 5
angles = np.linspace(0, 2 * np.pi, num_boids, endpoint=False)

angle_noise = np.random.uniform(-0.2, 0.2, size=num_boids)
radius_noise = np.random.uniform(-1.0, 1.0, size=num_boids)

perturbed_angles = angles + angle_noise*2
perturbed_radii = radius + radius_noise*2

# Compute positions with noise
positions = np.stack((
    perturbed_radii * np.cos(perturbed_angles),
    perturbed_radii * np.sin(perturbed_angles)
), axis=1)

# Compute tangential (perpendicular) velocities
velocities = np.stack((
    -np.sin(perturbed_angles),
    np.cos(perturbed_angles)
), axis=1)

# Add small random noise to velocities
velocity_noise = np.random.normal(scale=0.2, size=velocities.shape)
noisy_velocities = velocities + velocity_noise*2

velocity_magnitude = 1  # control speed

# Normalize and scale the noisy velocities
norms = np.linalg.norm(noisy_velocities, axis=1, keepdims=True)
noisy_velocities = noisy_velocities / norms * velocity_magnitude

velocities = noisy_velocities  # Final velocities with noise

weights = np.ones((num_boids, N_primitives))/N_primitives

flock = [Boid(pos, vel, w) for pos, vel, w in zip(positions, velocities, weights)]
radii = {'sep': 1.0, 'ali': 3, 'coh': 12.0}

dt = 0.05
speed_limit = 1.0
area_limits = np.array([20, 20])    # must be equal

PERCEPTION_RADIUS = radii['coh']

n_u_bins = 30
u_max_norm = 3
u_vals = np.linspace(-u_max_norm , u_max_norm , n_u_bins)
u_axis = [u_vals, u_vals]
u_grid = np.array([
    [ux, uy]
    for ux in u_axis[0]
    for uy in u_axis[1]
    if np.linalg.norm([ux, uy]) <= u_max_norm
])

primitives_var = [0.1, 0.1, 0.1]

eps = 0.5

max_steps = 440

#boundary = area_limits[0]  # positive limit assumed symmetric

polarization_history = []

step_counter = {'frame': 0, 'stop': False}  # Mutable object to track stopping

weights_history = []
milling_history = []
min_dist = []

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
np.save("min_distances.npy", np.array(min_dist))

# Save trajectories for all boids
all_trajectories = [boid.trajectory for boid in flock]
with open("boids_trajectories.pkl", "wb") as f:
    pickle.dump(all_trajectories, f)
all_vel_trajectories = [boid.vel_trajectory for boid in flock]
with open("boids_vel_trajectories.pkl", "wb") as f:
    pickle.dump(all_vel_trajectories, f)


