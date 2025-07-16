#######################EXECUTIVE#######################################
# TODO: 1) Variable graduations on sphere fully implemented (90% sure there's no weird errors)
# TODO: 2) The algo as of now is to try and maximise f on the sphere.





#######################################################################




import math

from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(33)
POLAR_GRADUATIONS = 36  #Must be integer factor of 360. Keep this low since it's cubic time complexity with  O(polar_grads*azimuth_grads^3)
AZIMUTH_GRADUATIONS = 18 #Must be integer factor of 180

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
    return row_of_points[0,]

def domainbuilder():
    """Build domain grid in (theta, phi) coordinates"""
    # theta: 0 to 359 degrees (azimuthal angle)
    # phi: 0 to 179 degrees (polar angle from north pole)
    theta_vals = np.arange(360, step = int(360/POLAR_GRADUATIONS))
    phi_vals = np.arange(180, step = int(180/AZIMUTH_GRADUATIONS))
    
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
    sigma = 0.3
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
    return int(int(theta*POLAR_GRADUATIONS/360) * AZIMUTH_GRADUATIONS + int(phi*AZIMUTH_GRADUATIONS/180))

def index_to_coord(index):
    """Convert flattened array index to (theta, phi) coordinates"""
    theta = index // AZIMUTH_GRADUATIONS
    phi = index % AZIMUTH_GRADUATIONS
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
def GP_K_producer(prior_K, C_inv, kappa):
    '''

    :param domain: (2,n) row of points in (theta, phi) coordinates
    :param x_observed:  (2,n) row of point that we're looking at
    :param y_observed:  (1,n) row of values of observed
    :param prior_K: the previous n_domain * n_domain matrix [massive] that is the covariance matrix of the prior
    :return: posterior covariance matrix
    '''

    return prior_K - kappa @ C_inv @ kappa.T


    


def GP_mew_producer(y_observed, x_observed, C_inv, prior_mean=None, k_domain_obs=None, k_obs_obs = None, ):
    """
    Perform Gaussian Process prediction
    
    :param domain: (2, N_domain) array of all domain points
    :param x_observed: (2, N_obs) array of observed input points
    :param y_observed: (N_obs,) array of observed output values
    :param prior_mean: Prior mean vector (if None, use zeros)
    :return: Posterior mean vector for the entire domain
    """

    # Compute prior mean for observed points
    m_obs = m_builder(x_observed, prior_mean)
    
    # GP prediction

    posterior_mean = prior_mean + k_domain_obs @ C_inv @ (y_observed - m_obs)
    
    return posterior_mean

def GP_iterate_defunct(domain, x_observed, y_observed, prior_mean = None, prior_K = None):
    '''

    :param domain: domain
    :param x_observed:
    :param y_observed:
    :param prior_mean:
    :return: mew_vector, K_matrix in that order and format
    '''
    n_domain = domain.shape[1]
    n_obs = x_observed.shape[1]
    first_iteration = None

    if prior_mean is None:
        prior_mean = np.zeros(n_domain)
    if prior_K is None:
        prior_K = K_vectorised(domain, domain)
        first_iteration = prior_K #put this boolean in here since on the first iteration this afunction will first set K to its true value using the K function but then it's processed upon later down immediately



    K_obs_obs = K_vectorised(x_observed, x_observed) #C
    K_domain_obs = K_vectorised(domain, x_observed) #kappa
    K_obs_obs += 1e-6 * np.eye(n_obs)

    # Use proper inverse instead of pseudo-inverse
    try:
        K_inv = np.linalg.inv(K_obs_obs)
    except np.linalg.LinAlgError:
        # If matrix is singular, add more regularization
        K_obs_obs += 1e-4 * np.eye(n_obs)
        K_inv = np.linalg.inv(K_obs_obs)


    mew_new = GP_mew_producer(y_observed, x_observed, K_inv, prior_mean, K_domain_obs)

    if isinstance(first_iteration, np.ndarray):
        K_new = prior_K
    elif first_iteration == None:
        K_new = GP_K_producer(prior_K, K_inv, K_domain_obs)



    return mew_new, K_new

def GP_iterate(x_observed, y_observed, domain):
    """
    Calculate the posterior mean and covariance matrix for y2
    based on the corresponding input X2, the observations (y1, X1),
    and the prior kernel function.
    """
    # Kernel of the observations
    Σ11 = K_vectorised(x_observed, x_observed)
    Σ11 += 1e-6 * np.eye(Σ11.shape[0])
    # Kernel of observations vs to-predict
    Σ12 = K_vectorised(x_observed, domain)
    # Solve
    import scipy
    solved = scipy.linalg.solve(Σ11, Σ12, assume_a='pos').T
    # Compute posterior mean
    μ2 = solved @ y_observed
    # Compute the posterior covariance
    Σ22 = K_vectorised(domain, domain)
    Σ2 = Σ22 - (solved @ Σ12) + 1e-6 * np.eye(domain.shape[1])
    return μ2, Σ2  # mean, covariance


def GP_iterate_defunct(domain, x_observed, y_observed, prior_mean=None, prior_K=None):
    n_domain = domain.shape[1]
    n_obs = x_observed.shape[1]

    if prior_mean is None:
        prior_mean = np.zeros(n_domain)

    # Initialize the *full* covariance matrix if it's the very first iteration
    # Otherwise, prior_K is the posterior from the previous step
    if prior_K is None:
        current_K_for_update = K_vectorised(domain, domain) # This is the K_prior for the *first* K_new calculation
    else:
        current_K_for_update = prior_K # This is the posterior K from the previous iteration, now acting as prior

    K_obs_obs = K_vectorised(x_observed, x_observed)  # C
    K_domain_obs = K_vectorised(domain, x_observed)   # kappa
    K_obs_obs += 1e-6 * np.eye(n_obs)

    try:
        K_inv = np.linalg.inv(K_obs_obs)
    except np.linalg.LinAlgError:
        K_obs_obs += 1e-4 * np.eye(n_obs)
        K_inv = np.linalg.inv(K_obs_obs)

    mew_new = GP_mew_producer(y_observed, x_observed, K_inv, prior_mean, K_domain_obs)

    # Always compute K_new using the 'current_K_for_update'
    K_new = GP_K_producer(current_K_for_update, K_inv, K_domain_obs)

    return mew_new, K_new







def spherical_visualisation(mean_vector, title="GP Prediction on Sphere"):
    """
    Visualize the GP prediction on a sphere
    
    :param mean_vector: Mean vector for the entire domain (POLAR_GRADUATIONS*180 elements)
    :param title: Plot title
    """
    # Reshape mean vector to grid
    mean_grid = mean_vector.reshape((AZIMUTH_GRADUATIONS, POLAR_GRADUATIONS))
    
    # Create coordinate grids for visualization
    theta_vis = np.linspace(0, 2*np.pi, POLAR_GRADUATIONS)  # 0 to 2π for full rotation
    phi_vis = np.linspace(0, np.pi, AZIMUTH_GRADUATIONS)      # 0 to π for full sphere
    
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


def Find_next_point(domain, posterior_mean, posterior_K_matrix, y_observed, x_observed):
    '''

    :param domain: 2 x (domain_size) matrix of the points (ordered by theta then phi is the one that runs around)
    :param posterior_mean: posterior mean vector. must use indexing function to help with this
    :param posterior_K_matrix: posterior K matrix
    :param y_observed: the f_values that have actually been seaen,
    :return: ROW vector that is the next point to sample from according to the acquisition function
    '''

    #doing the simple PI acquisition function:

    x_i = 0.1   #arbitrary parameter.
    f_best = np.max(y_observed)
    sigma = np.sqrt(np.diag(posterior_K_matrix))
    
    pi_vector =  norm.cdf((posterior_mean - f_best - x_i)/sigma)#TODO: gonna have to reverse the 180*a + b thing here
    index_of_best_point = np.argmax(pi_vector)

    best_point = np.array(index_to_coord(index_of_best_point)).reshape(1,-1)*10 #REMOVE THE 10. IT'S EXPERIMENTAL TODO
    return best_point #ensure this is a row vector of shape (1,2)











def main():
    domain = domainbuilder()
    samplesize = 50
    theta_samples = np.random.randint(0, POLAR_GRADUATIONS, samplesize)*int(360/POLAR_GRADUATIONS)
    phi_samples = np.random.randint(0, AZIMUTH_GRADUATIONS, samplesize)*int(180/AZIMUTH_GRADUATIONS)
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
    
#########################TODO: Reminder ##################################
#1) The convention for a vector of points is a column vector where each element is a row vector that is a point.
#2) The domain is a 2 X polar_grad*azimuth_grad column matrix of 2x1 points; from ( [0, 359], [0, 180]). so dom[:, 180a+b] = point(a,b)
#3) all vectors are to be rowvectors and transpoed when it's necessary they are column vecs.
#4) D_seen is a row matrix where each column is a (theta, phi, f(theta, phi)) column vector.

##########################################################################

def new_main(points_to_start_with = None, num_iterations = 1):
    '''

    :param points_to_start_with: row vector containing the points you wanna start from.
    :return:
    '''
    domain = domainbuilder()
    samplesize = num_iterations
    point_to_add = np.array(((1*POLAR_GRADUATIONS,2*AZIMUTH_GRADUATIONS),(3*POLAR_GRADUATIONS,4*AZIMUTH_GRADUATIONS))) #dummy just in case.
    x_observed_unprocessed = []
    domain_size = domain.shape[1]
    mu_prior = np.zeros(domain_size)

    #TODO:for animation later on
    all_posterior_means = []
    all_posterior_covariances = []
    all_sampled_points = []
    #TODO: for animation later on

    x_observed_unprocessed.append(point_to_add.T)
    x_observed = np.hstack(x_observed_unprocessed)
    K_prior = K_vectorised(domain,domain)

    for i in range(num_iterations):
        #Setting up data:

        y_observed = f_vectorised(x_observed)

        #Doing GP
        # posterior_mean = GP_mew_producer(domain, x_observed, y_observed, prior_mean = mu_prior)
        # posterior_K_matrix = GP_K_producer(domain, x_observed, y_observed, prior_K = K_prior)

        posterior_mean, posterior_K_matrix = GP_iterate(x_observed, y_observed, domain) #I've combined both posteriorising functions into one megafunction.

        #storing the data to plot later on
        all_posterior_means.append(posterior_mean)
        all_posterior_covariances.append(posterior_K_matrix)
        all_sampled_points.append(x_observed)


        #before iterating.

        point_to_add  = Find_next_point(domain, posterior_mean, posterior_K_matrix , y_observed, x_observed)

        x_observed = np.hstack((x_observed, point_to_add.T))
        # mu_prior, K_prior = posterior_mean, posterior_K_matrix
        mu_prior = posterior_mean
    spherical_visualisation(all_posterior_means[-1])
    return all_posterior_means, all_posterior_covariances, all_sampled_points


# ... (existing new_main function code) ...
# ... (imports and existing functions) ...












if __name__ == "__main__":

    #new directives TODO:
    #switch from points on sphere abstraction to actual sphere
    #CDF doesnt work on sphere hethinks
    #try proper euclidean bacquisition function
    #the extrinsic bayesian optimisation paper has really nice implementation. There are laods more (nested riemann manifolds is better than the prev. they have github code )
    #Start to code ONLY FROM PAPERS NOT ON YOUR OWN.
    #Start writing on overleaf the motivation/equatiosn beforehand so he knows what you're doing,
    #"manifold gaussian process for regression" paper is reall good and has all the answers.
    #the posterior updating i'm using may not be rigorous enough. Need to make sure that it actaully works.
    #SUMMARY:first do euclidean gp with the poitns. Try EI on this script as it is now.Try Euclidean BO again except code it using a proper infinite real line instead of discretising domain. Try and put this script into manifold instead a bunch fo points. Look at the implememntations from the papers. STarting writing on overleaf as if writing the paper, tarting with BO on sphere implementation
    #Try euclidean BO usiing IPOPT in the package called casadi. You're gonna have to do scipy optimise import nonlinear optimisatojn for optimising on the acquisition function you come up with.
    #Put everything on github so Zhengang can play around with it. Put things into seperate folders. Also write into overleaf "report" the week by week updates.
    means, covs, sampleds = new_main(num_iterations = 5)