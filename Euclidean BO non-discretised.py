#Prime directive#############################################################################################
#1) Everything is a row of column vectors. .
#2) All points are always column
#3) For now the domain will be from 0 to 3 on the real line inclusive.
#4) start with 1D then go to 2D later on. KEEP THIS IN MIND WHEN IMPLEMENTING
# x is treated as a 1x1 matrix because of the above fact.#
#5) the params vector for now (STRUCTURE): is (constant_value, amplitude, length_scale)

##########################################################################################################



#imports:
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.stats
import math
from sklearn.metrics.pairwise import euclidean_distances




#Setting meta-parameters:
left_endpoint, right_endpoint = 0,3   #So x is to lie on [left_endpoint, right_endpoint]









class Bayesian_optimisation:

    def __init__(self, x_observed, y_observed, initial_guess_for_params):
        '''
        :param x_observed:  is a row matrix of column vector points. dimensionXn.
        :param initial_covariance: initial covariance function
        :param initial_guess_for_params: a numpy COL (vertical) vector of the parameters. For now it's in the order (constant_value, sigma, length_scale)
        :return: the mew_function, the K function
        '''


        #the below vars are storing the "current" value of variables and are changed if a better one is found.

        self.x_observed = x_observed
        self.y_observed = y_observed
        self.current_kernel = self.RBF_Kernel_matrix_producer
        self.mew_function = self.mew
        self.params = initial_guess_for_params
        if not isinstance(initial_guess_for_params, np.ndarray): raise Exception("params is meant to be an nd array (because I'm guessing that's how the scipy optimisation algorithms work)")
        if  isinstance(x_observed.reshape(-1,1)[0][0], np.ndarray) == True: raise Exception("you forgot to convert the x-points into a real number before putting it into the matrix")


    def mew(self, x, new_parameters=None):
        '''
        :param x: the value we wanna work out what the function is. VECTORISE THIS
        :param new_parameters: the new parameter VECTOR that we wanna test. It's like this so that it cna be changed easier.
        :return: for now, we're doing that mu(x) = c where c is a constant
        '''
        if new_parameters is None:
            parameters = self.params

        else:
            parameters = new_parameters

        #Mu function structure!:
        constant_value = parameters[0]    #MEW STRUCTURE
        assert isinstance(x,np.ndarray) == True  # x is to be treated as a 1x1 nd array to make it easier to expand the whole script to 2D and 3D later on.
        return np.repeat(constant_value, x.shape[0], axis=0) #MEW STRUCTURE

    def RBF_Kernel_matrix_producer(self, row_of_points_1, row_of_points_2, new_parameters = None, jitter = 1e-6):
        if not (isinstance(row_of_points_1, np.ndarray) == True and isinstance(row_of_points_2, np.ndarray) == True):
            raise Exception(
                "x and y have to be 2d arrays so that this script can be scaled up")  # x and y are meant to be 1x1 ndarrays to allow this whole script to be Dimension uppped easier.
        row_of_points_1, row_of_points_2 = np.atleast_2d(row_of_points_1), np.atleast_2d(row_of_points_2)
        n = row_of_points_2.shape[1]
        if new_parameters is None:
            parameters = self.params
        else:
            parameters = new_parameters

        constant_value, amplitude, length_scale = parameters  #K STRUCTURE

        col_of_points_1 = row_of_points_1.T
        col_of_points_2 = row_of_points_2.T
        return (amplitude ** 2) * np.exp(-euclidean_distances(col_of_points_1, col_of_points_2, squared=True) / (2 * length_scale ** 2)) + jitter*np.eye(n)
    def regress(self, starting_guess=np.array((3,3,3)).flatten(), method = 'BFGS', epsilon = 1e-2, trying_alternative = False, bounds = [(0,5), (0,5), (0,5)]):
        if trying_alternative == True:
              #bounds for constant_value, amplitude, length_scale respectively
            result = scipy.optimize.differential_evolution(self.log_marginal_likelihood, bounds = bounds)
        else:
            result = scipy.optimize.minimize(self.log_marginal_likelihood, x0 =starting_guess, method = method, options = {"eps": epsilon})

        self.params = result.x #TODO: this might throw.
        return result.x




    def log_marginal_likelihood(self, params):


        y_observed = self.y_observed.reshape(-1,1)
        y_observed = y_observed.T
        n = y_observed.size

        K = self.RBF_Kernel_matrix_producer(self.x_observed, self.x_observed, params) #TODO: why is this outputting a 1x1 matrix when it should be outputting a nxn matrix?
        return ( -1/2) * y_observed @ np.linalg.inv(K) @ y_observed.T - (1 / 2) * np.log(np.linalg.norm(K)) - (n / 2)*np.log(2 * np.pi)

    ############################ visualisation-specific bit ##################################
    def sigma_function(self, x):
        '''
        :param x: number inside domain you wanna find the sigma at, VECTORISED X IS COLUMN OF POINTS
        :return: real number which is the sigma at x
        '''

        x = x.reshape(-1,1)
        if x.shape[0] > 1:
            #dealing with more than one x-value.
            sigma = np.diag(np.sqrt(self.RBF_Kernel_matrix_producer(x.T, x.T)))
        else:
            sigma = np.sqrt(self.RBF_Kernel_matrix_producer(x, x))[0][0]
        return sigma

    def plot_current_GP(self, dimension = 1, graduation_size = 1e-3, confidence_interval = 0.3, regress_first = False):
        '''
        :param dimension:
        :param graduation_size: the difference in the value of the variable as we're sketching the curve
        :param confidence_interval: the confidence interval we want to actually plot.
        :return:
        '''
        if regress_first == True:
            raise Exception("you shouldn't integrate regressing and plot_current into one since you would have to pass through a massive number of args")
        parameters = self.params
        num_of_graduations = int((right_endpoint - left_endpoint)/graduation_size)
        if dimension != 1:
            raise Exception("you haev to code the multidimensional bit")

        mew_function, sigma_function = self.mew_function, self.sigma_function
        x_observed,y_observed  = self.x_observed, self.y_observed
        domain = np.linspace(left_endpoint, right_endpoint, num_of_graduations)

        # plt.scatter(domain, y_observed , label= "Observed data", linestyle="dotted")


        ##########Horeshit copied from online:
        plt.scatter(x_observed, y_observed, label="Observations")
        plt.plot(domain, mew_function(domain), label="Mean prediction")

        Z_threshold = scipy.stats.norm.ppf(1 - confidence_interval)

        plt.fill_between(
            domain,  # <--- Change this from x_observed.ravel() to domain
            mew_function(domain) - Z_threshold * sigma_function(domain),
            mew_function(domain) + Z_threshold * sigma_function(domain),
            alpha=0.5,
            label=f"{confidence_interval*100}% confidence interval",
        )
        plt.legend()
        plt.xlabel("$x$")
        plt.ylabel("$f(x)$")
        _ = plt.title("Gaussian process regression on noise-free dataset")
        plt.show()
























if __name__ == "__main__":

    ###############Trial data#######################################:
    #For now just doing x = y
    Data_x = np.arange(left_endpoint, right_endpoint+1)
    Data_y = np.arange(left_endpoint, right_endpoint+1) +100

    ################################################################
    K = Bayesian_optimisation(Data_x, Data_y, initial_guess_for_params= np.array((99, 0.5, 0.5)))  #parameters is a row vector with (constant_value, amplitude, length_scale)
    bounds = [(98,102), (1e-5, 5), (1e-7, 3)]
    starting_guess = np.array((98,2,2)).flatten()
    K.regress(starting_guess= starting_guess, bounds = bounds )

    # Testing the GP regression##############
    # print(K.regress(trying_alternative=True, bounds =bounds, starting_guess = starting_guess, method = 'ipopt'))

    #Testing the GP plotting
    K.plot_current_GP(confidence_interval = 0.6)