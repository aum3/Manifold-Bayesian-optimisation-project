import math

import numpy
import numpy as np
np.random.seed(33)
def f_vectorised(row_of_points): #TODO : THIS IS A DUMMY FUNCTION
    '''

    :param row_of_points: gotta be column vector where each element is a 1x3 ROW VECTOR
    :return:  f(x,y,z) = x^2 - y^2
    '''


    if row_of_points.shape[0] == 2:
        if row_of_points.size == 2:
            row_of_points = row_of_points.reshape(2, -1)
        # row_of_points = np.fromiter([double_polar_to_cartesian(row_of_points[:,i]) for i in range(row_of_points.shape[1])], dtype=float)
        #TODO make sure that this 2xn column of doule_polar coords are turned into a 3xn column of cartesian coords
        cartesianised_row_of_points = []
        for double_polar in row_of_points.T:
            cartesianised_row_of_points.append(double_polar_to_cartesian(double_polar).reshape(-1,1).flatten())
        row_of_points = np.array(cartesianised_row_of_points).T
        #TODO: at this point, the row_of_points is a (20,) nparrayof a bunch of arrays which is wrong.

    return row_of_points[0]
    if row_of_points.shape[0] != 3:
        print("WTFWTF \n\n")


    if row_of_points.ndim == 1:
        row_of_points = row_of_points.reshape(-1, 1)

    #TODO: DUMMY:

    # if row_of_points.size == 3:
    #     return row_of_points[0]**2 - row_of_points[1]**2
    # return row_of_points[: , 0]* row_of_points[:, 0] - row_of_points[:, 1]* row_of_points[:, 1]
def domainbuilder():
    theta = np.repeat(np.arange(360), 180)

    # Creates the phi values (0-179 sequence repeats 360 times)
    phi = np.tile(np.arange(180), 360)

    # Stacks them side-by-side into a (64800, 2) array
    return np.stack([theta, phi], axis=0)
def double_polar_to_cartesian(double_polar):
    double_polar = double_polar.flatten()
    double_polar = ensure_column(double_polar) #TODO might be the other way around
    theta = double_polar[0]*(2*math.pi/360)
    phi = double_polar[1]*(2*math.pi/360)
    return_vector = np.array((np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)))
    return ensure_row(return_vector)
def K_scalar(v_1, v_2):
    vector_1, vector_2 = double_polar_to_cartesian(v_1), double_polar_to_cartesian(v_2)

    sigma_f = 1.4 #random fuckass constants
    sigma = 0.4 #random fuckass constants.
    return (sigma_f**2)*np.exp(-(np.linalg.norm(vector_1 - vector_2))**2 / (2*sigma**2))
def K_vectorised_original(left_vector, right_vector):
    '''
    :param left_vector:  should be a column vector of 1x2 column vectors.
    :return: the covariance matrix between the 2 vectors of points.
    '''

    if left_vector.shape[0] == right_vector.shape[0] and right_vector.shape[0] == 3:
        pass

    if left_vector.ndim == 1:
        left_vector = left_vector.reshape(-1, 1)
    if right_vector.ndim == 1:
        right_vector = right_vector.reshape(-1, 1)


    left_size = left_vector.shape[1]
    right_size = right_vector.shape[1]
    # If a vector is 1D, reshape it into a 2D column vector (2, 1)

    if left_size < right_size:
        left_vector, right_vector = right_vector, left_vector
        left_size, right_size = right_size, left_size


    return_matrix = np.zeros((left_size, right_size))
    right_point = 0
    for left_point in range(0, left_size):

        while right_point <= min(right_size, left_point): #This is to deal with the case rightvector is literally a single point
            cov_value = K_scalar(left_vector[:, left_point], right_vector[:, right_point]) #TODO check if this was done correctly
            try:
                return_matrix[left_point][right_point] = cov_value
            except:

                try:
                    return_matrix[right_point][left_point] = cov_value
                except:
                    print("WTF!!!")
                    break
            right_point += 1

    return return_matrix
def K_vectorised(left_vector, right_vector):
    '''
    :param left_vector: A (D, M) array of M points in D dimensions.
    :param right_vector: A (D, N) array of N points in D dimensions.
    :return: The (M, N) covariance matrix between the two sets of points.
    '''
    # Ensure inputs are 2D arrays
    if left_vector.ndim == 1:
        left_vector = left_vector.reshape(-1, 1)
    if right_vector.ndim == 1:
        right_vector = right_vector.reshape(-1, 1)

    # Get the number of points (columns) in each vector
    left_size = left_vector.shape[1]
    right_size = right_vector.shape[1]

    # Initialize the correct output matrix shape
    return_matrix = np.zeros((left_size, right_size))

    # Use a simple nested loop to iterate through all combinations of points.
    # This replaces the complex and incorrect while loop.
    for i in range(left_size):
        for j in range(right_size):
            # Get the i-th point from the left and j-th point from the right
            point_i = left_vector[:, i]
            point_j = right_vector[:, j]

            # Calculate the scalar covariance
            cov_value = K_scalar(point_i, point_j)

            # Assign the value to the correct position in the matrix
            return_matrix[i, j] = cov_value

    return return_matrix
def ensure_column(vector):
  """Ensures a NumPy array is a 2D column vector of shape (N, 1)."""
  return vector.reshape(-1, 1)
def ensure_row(vector):
    return vector.reshape(1, -1)
def m_builder(row_of_observed_points):
    '''

    :param row_of_points:  takes in the D_seen row_of_points matrix DOUBLE POLAR
    :return: returns the m vector vector which is the current prior mean of what the y value is estimated to be
    at these points. Points are to be taken from the big mew since we're operating only on that.
    '''
    assert row_of_observed_points.shape[0] == 2
    if row_of_observed_points.ndim ==1:
        row_of_observed_points = row_of_observed_points.reshape(-1, 1)
    #Reminder: dom[:, 180a+b] = point(a,b)
    return_vector = np.zeros(row_of_observed_points.shape[1])
    for i,theta_phi in enumerate(row_of_observed_points.T): #TODO: make sure this actually works
        theta_phi = theta_phi.flatten()
        theta = theta_phi[0]
        phi = theta_phi[1]
        return_vector[i] = final_mew[int(180*theta + phi)]
    return return_vector






#IMPERIAL HIGH COMMAND:
#1) The convention for a vector of points is a column vector where each element is a row vector that is a point.
#2) The domain is a 2 X 360*180 column matrix of 2x1 points; from ( [0, 359], [0, 180]). so dom[:, 180a+b] = point(a,b)
#3) all vectors are to be rowvectors and transpoed when it's necessary they are column vecs.
#4) D_seen is a row matrix where each column is a (theta, phi, f(theta, phi)) column vector.

domain_unravelled  = domainbuilder()
final_mew = np.zeros(360*180)
samplesize = 10
x_vector = np.array(list(zip(np.random.randint(0, 361, samplesize), np.random.randint(0, 181, samplesize)))).T
# x_vector = (np.random.rand(2, samplesize) * 360).round(0)  #TODO. This  is creating (360,360) not (360,180). Perhaps multiply by the vector (360,180) or something,
y_vector = f_vectorised(x_vector).reshape(1,-1)
covmatrix = K_vectorised(x_vector, x_vector)


   #stores the (theta, phi, function value) row vector in each of its columns
print_list = []


def iterative_testing_main(samplesize = samplesize):
    '''
    This is for seeing how increasing the data input makes the preditions better (or worse).
    Requires n(n+1)/2 runnings of the algorithm [each iteration also takes more time than te prev]

    '''
    D_seen = np.zeros((3, samplesize))
    for i, sample_point in enumerate(x_vector.T):
        sample_point = sample_point.flatten()
        new_column = np.hstack((sample_point, f_vectorised(sample_point)))

        #TODO: this is the old version pre-sizing in case i mess something up.
        # D_seen[:, i] = new_column
        # y_seen = D_seen[2] #TODO: should the (theta, phi, f(theta,phi)) even be passed into f_vectorised?
        # C = K_vectorised(D_seen[0:2,:], D_seen[0:2,:]) #TODO: I think only C[0:i, o:i] before being fed into the line 137
        # kappa_x = K_vectorised(domain_unravelled, D_seen[0:2,:]) #TODO: make sure this works too
        # mew = m_builder(D_seen[0:2,:])
        # final_mew = final_mew + kappa_x @ C.T @ (y_seen-mew).flatten()

        D_seen[:, i] = new_column
        y_seen = D_seen[2, 0:(i+1)]  # TODO: should the (theta, phi, f(theta,phi)) even be passed into f_vectorised?
        C = K_vectorised(D_seen[0:2, 0:(i+1)],
                         D_seen[0:2, 0:(i+1)])
        C = C + 1e-8 * np.eye(C.shape[0])
        kappa_x = K_vectorised(domain_unravelled, D_seen[0:2, 0:(i+1)])  # TODO: make sure this works too
        mew = m_builder(D_seen[0:2, 0:(i+1)])
        summand = kappa_x @ np.linalg.pinv(C) @ (y_seen - mew).flatten() #TODO: im pretty sure the mu_D = mu_D + whatever messes up since the nunmbers are being added to the wrong index
        print_list.append(summand)
        # Changed this line since
        #I'm quite sure you're not meant to feed back in the data points that have already been seen.

def all_in_one_go_main(samplesize = samplesize):
    '''

    :param samplesize: This doesn't do anything UNTIL THIS SCRIPT IS REFACTORED
    :return:
    '''

    #Data handling
    D_seen = np.vstack((x_vector, y_vector))
    #Data handling


    #producing the output
    y_seen, x_seen = D_seen[2, :], D_seen[0:2, :]  # This is in here so that this script can be refactored easier to take in higher sample sizes
    C = K_vectorised(x_seen, x_seen)
    C = C + 1e-8 * np.eye(C.shape[0])
    kappa_x = K_vectorised(domain_unravelled, x_seen)  # TODO: make sure this works too
    mew = m_builder(x_seen)
    final_output = kappa_x @ np.linalg.pinv(C) @ (y_seen - mew).flatten()
    #producing the output

    return final_output

def fitted_function(mew_vector, row_of_coords):
    '''
    :param row_of_coords: double_polar column coordinate
    :return: what the GP says the mew should be at this point
    '''

    theta, phi = row_of_coords[0], row_of_coords[1]
    return mew_vector[180*theta + phi]

def old_spherical_visualisation(mew_vector):
    '''

    :param mew_vector: the posterior mew vector created by this bumbaclat algorithm
    :return: visualises a sphere which has different colours based on the mu value there.
    '''




    import numpy as np
    theta_phi_output_array = np.vstack((domain_unravelled, mew_vector))


    #someone else's code:
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import pyplot as plt
    import numpy as np
    from matplotlib.cm import get_cmap


    theta = theta_phi_output_array[0, :]
    phi = theta_phi_output_array[1, :]
    func_output = theta_phi_output_array[2, :]

    # Normalize the function output to map to colors
    # Ensure min/max are not identical to avoid division by zero for normalization
    if np.max(func_output) == np.min(func_output):
        normalized_output = np.zeros_like(func_output)  # All same color if output is constant
    else:
        normalized_output = (func_output - np.min(func_output)) / (np.max(func_output) - np.min(func_output))

    # Convert spherical coordinates (with a constant radius for the sphere itself) to Cartesian
    # We'll use a unit sphere for visualization, and map colors based on func_output
    r = 1.0  # Constant radius for the sphere
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)

    # Set up the 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Create a colormap
    # You can choose different colormaps, e.g., 'viridis', 'plasma', 'inferno', 'magma', 'cividis'
    cmap = get_cmap('viridis')
    colors = cmap(normalized_output)

    # Plot the points on the sphere, colored by the function's output
    # Using scatter to plot individual points with their respective colors
    scatter = ax.scatter(x, y, z, c=func_output, cmap=cmap, s=50, alpha=0.8)

    # Add a color bar to show the mapping of colors to function values
    cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label('Function Output Value')

    # Set plot labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Function Output on a Sphere')

    # Set equal aspect ratio
    ax.set_box_aspect([np.ptp(a) for a in [x, y, z]])

    # Show the plot
    plt.show()


def spherical_visualisation(mean_vector, title="GP Prediction on Sphere"):
    """
    Visualize the GP prediction on a sphere

    :param mean_vector: Mean vector for the entire domain (360*180 elements)
    :param title: Plot title
    """
    # Reshape mean vector to grid
    import numpy as np
    import matplotlib.pyplot as plt
    mean_grid = mean_vector.reshape((180, 360))

    # Create coordinate grids for visualization
    theta_vis = np.linspace(0, 2 * np.pi, 360)  # 0 to 2π for full rotation
    phi_vis = np.linspace(0, np.pi, 180)  # 0 to π for full sphere

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
    ax.set_box_aspect([1, 1, 1])

    plt.show()


if __name__ == "__main__":
    j = all_in_one_go_main() #skipped this using caching since the prog was taking too long



    def checking_veracity_mew_vector(theta, phi):
        double_polar = np.array([theta, phi]).T
        return f"the mew_vector gives {j[180 * theta + phi]} while the actual function's value is {f_vectorised(double_polar_to_cartesian(double_polar))}"


    # print(checking_veracity_mew_vector(0,0))
    spherical_visualisation(j)


def checking_veracity_mew_vector(theta, phi):
    double_polar = np.array([theta, phi]).T
    return f"the mew_vector gives {j[180*theta + phi]} while the actual function's value is {f_vectorised(double_polar_to_cartesian(double_polar))}"













