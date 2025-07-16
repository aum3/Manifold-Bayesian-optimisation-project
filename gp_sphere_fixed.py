import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(33)

def f_vectorised(row_of_points):
    """
    Test function: f(x,y,z) = x (should show gradient along x-axis)
    
    :param row_of_points: Either 2xN (theta,phi) or 3xN (x,y,z) array
    :return: Function values
    """
    # If input is 2D (theta, phi), convert to cartesian
    if row_of_points.shape[0] == 2:
        cartesian_points = []
        for i in range(row_of_points.shape[1]):
            theta_phi = row_of_points[:, i]
            cart_point = double_polar_to_cartesian(theta_phi)
            cartesian_points.append(cart_point.flatten())
        row_of_points = np.array(cartesian_points).T
    
    # Ensure we have the right shape
    if row_of_points.ndim == 1:
        row_of_points = row_of_points.reshape(-1, 1)
    
    # Return x coordinate (first component)
    m = row_of_points.shape[1]
    return row_of_points[0]

def domainbuilder():
    """Build domain grid in (theta, phi) coordinates"""
    # theta: 0 to 359 degrees (azimuthal angle)
    # phi: 0 to 179 degrees (polar angle from north pole)
    theta_vals = np.arange(360)
    phi_vals = np.arange(180)
    
    # Create meshgrid and flatten
    theta_grid, phi_grid = np.meshgrid(theta_vals, phi_vals)
    theta_flat = theta_grid.flatten()
    phi_flat = phi_grid.flatten()
    
    return np.stack([theta_flat, phi_flat], axis=0)

def double_polar_to_cartesian(double_polar):
    """
    Convert (theta, phi) in degrees to (x, y, z) cartesian coordinates
    theta: azimuthal angle (0-360 degrees)
    phi: polar angle from north pole (0-180 degrees)
    """
    double_polar = double_polar.flatten()
    theta_deg = double_polar[0]
    phi_deg = double_polar[1]
    
    # Convert to radians
    theta_rad = theta_deg * np.pi / 180.0
    phi_rad = phi_deg * np.pi / 180.0
    
    # Standard spherical to cartesian conversion
    x = np.sin(phi_rad) * np.cos(theta_rad)
    y = np.sin(phi_rad) * np.sin(theta_rad)
    z = np.cos(phi_rad)
    
    return np.array([x, y, z]).reshape(-1, 1)

def K_scalar(v_1, v_2):
    """Scalar kernel function between two points"""
    vector_1 = double_polar_to_cartesian(v_1).flatten()
    vector_2 = double_polar_to_cartesian(v_2).flatten()
    
    sigma_f = 1
    sigma = 0.6
    distance_sq = np.sum((vector_1 - vector_2)**2)
    return (sigma_f**2) * np.exp(-distance_sq / (2 * sigma**2))

def K_vectorised(left_vector, right_vector):
    """
    Vectorized kernel matrix computation
    
    :param left_vector: (2, M) array of M points in (theta, phi) coordinates
    :param right_vector: (2, N) array of N points in (theta, phi) coordinates
    :return: (M, N) kernel matrix
    """
    if left_vector.ndim == 1:
        left_vector = left_vector.reshape(-1, 1)
    if right_vector.ndim == 1:
        right_vector = right_vector.reshape(-1, 1)
    
    left_size = left_vector.shape[1]
    right_size = right_vector.shape[1]
    
    kernel_matrix = np.zeros((left_size, right_size))
    
    for i in range(left_size):
        for j in range(right_size):
            point_i = left_vector[:, i]
            point_j = right_vector[:, j]
            kernel_matrix[i, j] = K_scalar(point_i, point_j)
    
    return kernel_matrix

def coord_to_index(theta, phi):
    """Convert (theta, phi) coordinates to flattened array index"""
    return int(theta * 180 + phi)

def index_to_coord(index):
    """Convert flattened array index to (theta, phi) coordinates"""
    theta = index // 180
    phi = index % 180
    return theta, phi

def m_builder(row_of_observed_points, prior_mean_vector):
    """
    Build mean vector for observed points
    
    :param row_of_observed_points: (2, N) array of observed (theta, phi) points
    :param prior_mean_vector: Prior mean vector for the entire domain
    :return: Mean vector for observed points
    """
    if row_of_observed_points.ndim == 1:
        row_of_observed_points = row_of_observed_points.reshape(-1, 1)
    
    n_points = row_of_observed_points.shape[1]
    mean_vector = np.zeros(n_points)
    
    for i in range(n_points):
        theta = int(row_of_observed_points[0, i])
        phi = int(row_of_observed_points[1, i])
        idx = coord_to_index(theta, phi)
        mean_vector[i] = prior_mean_vector[idx]
    
    return mean_vector
def GP_K_producer(domain, x_observed, y_observed, prior_K = None):
    '''

    :param domain: (2,n) row of points in (theta, phi) coordinates
    :param x_observed:  (2,n) row of point that we're looking at
    :param y_observed:  (1,n) row of values of observed
    :param prior_K: the previous n_domain * n_domain matrix [massive] that is the covariance matrix of the prior
    :return: posterior covariance matrix
    '''

    


def GP_mew_producer(domain, x_observed, y_observed, prior_mean=None):
    """
    Perform Gaussian Process prediction
    
    :param domain: (2, N_domain) array of all domain points
    :param x_observed: (2, N_obs) array of observed input points
    :param y_observed: (N_obs,) array of observed output values
    :param prior_mean: Prior mean vector (if None, use zeros)
    :return: Posterior mean vector for the entire domain
    """
    n_domain = domain.shape[1]
    n_obs = x_observed.shape[1]
    
    if prior_mean is None:
        prior_mean = np.zeros(n_domain)
    
    # Compute kernel matrices
    K_obs_obs = K_vectorised(x_observed, x_observed)
    K_domain_obs = K_vectorised(domain, x_observed)
    
    # Add noise for numerical stability
    K_obs_obs += 1e-8 * np.eye(n_obs)
    
    # Compute prior mean for observed points
    m_obs = m_builder(x_observed, prior_mean)
    
    # GP prediction
    K_inv = np.linalg.pinv(K_obs_obs)
    posterior_mean = prior_mean + K_domain_obs @ K_inv @ (y_observed - m_obs)
    
    return posterior_mean

def spherical_visualisation(mean_vector, title="GP Prediction on Sphere"):
    """
    Visualize the GP prediction on a sphere
    
    :param mean_vector: Mean vector for the entire domain (360*180 elements)
    :param title: Plot title
    """
    # Reshape mean vector to grid
    mean_grid = mean_vector.reshape((180, 360))
    
    # Create coordinate grids for visualization
    theta_vis = np.linspace(0, 2*np.pi, 360)  # 0 to 2π for full rotation
    phi_vis = np.linspace(0, np.pi, 180)      # 0 to π for full sphere
    
    THETA, PHI = np.meshgrid(theta_vis, phi_vis)
    
    # Convert to cartesian for plotting
    X = np.sin(PHI) * np.cos(THETA)
    Y = np.sin(PHI) * np.sin(THETA)
    Z = np.cos(PHI)
    
    # Create the plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the surface
    surf = ax.plot_surface(X, Y, Z, facecolors=plt.cm.coolwarm(
        (mean_grid - mean_grid.min()) / (mean_grid.max() - mean_grid.min())
    ), alpha=0.8, linewidth=0, antialiased=True)
    
    # Add colorbar
    m = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm)
    m.set_array(mean_vector)
    plt.colorbar(m, ax=ax, shrink=0.5, aspect=10)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # Set equal aspect ratio
    ax.set_box_aspect([1,1,1])
    
    plt.show()

def main():
    """Main execution function"""
    print("Building domain...")
    domain = domainbuilder()
    
    # Generate training data
    print("Generating training data...")
    samplesize = 5
    theta_samples = np.random.randint(0, 360, samplesize)
    phi_samples = np.random.randint(0, 180, samplesize)
    # theta_samples = [90*(1+x) for x in range(samplesize)]
    # phi_samples = [90*(1+x) for x in range(samplesize)]
    x_observed = np.array([theta_samples, phi_samples])
    
    # Compute function values at observed points
    y_observed = f_vectorised(x_observed)
    
    print(f"Training on {samplesize} points...")
    print(f"Sample function values: {y_observed[:5]}")
    
    # Perform GP prediction
    posterior_mean = GP_mew_producer(domain, x_observed, y_observed)
    
    print(f"Posterior mean range: [{posterior_mean.min():.3f}, {posterior_mean.max():.3f}]")
    
    # Visualize results
    spherical_visualisation(posterior_mean, "GP Prediction: f(x,y,z) = x")
    
    # Test a few points
    print("\nTesting predictions:")
    test_points = [(0, 90), (90, 90), (180, 90), (270, 90)]  # Points on equator
    for theta, phi in test_points:
        idx = coord_to_index(theta, phi)
        predicted = posterior_mean[idx]
        # Get actual cartesian coordinates
        cart = double_polar_to_cartesian(np.array([theta, phi])).flatten()
        actual = cart[0]  # x coordinate
        print(f"({theta}°, {phi}°) -> Cartesian: ({cart[0]:.3f}, {cart[1]:.3f}, {cart[2]:.3f})")
        print(f"  Predicted: {predicted:.3f}, Actual: {actual:.3f}")

if __name__ == "__main__":
    main()
