import numpy as np

dimension = 3
dimensions = dimension

# f(x,y,z) = a*x**2 + b*y**2 + c*z**2 + d*x*y + e*x*z + f*y*z
a,b,c = 1,2,0
d,e,g = 0,0,0
#renamed f to g since using f clashed with the def of the function f

def proj(x, v):

    assert x.dtype == "float64"
    '''
    
    :param x:  x is a row-vector (point on the sphere) that is the point on the sphere. v is the gradient vector that we're projeting onto the tangent space of the manifold at x 
    :return: the orthogonal projection of 
    '''

    return ((np.eye(dimension) - np.outer(x,x)) @ v.T).T

def f(x):
    '''
    x is a Rk row vector, same as above. R is on the manifold
    x = (x_1, ... x_d)
    '''


    # f(x,y,z) = a*x**2 + b*y**2 + c*z**2 + d*x*y + e*x*z + f*y*z
    f_quadric_matrix = np.array(
        [[a, d/2, e/2],
        [d/2, b, g/2],
               [e/2, g/2, c]]
    )

    return x @ f_quadric_matrix @ x.T
def gradf(x_pos):
    '''
    x is a Rk row vector, same as above. x is on the manifold
    :param x:     x is a Rk row vector, same as above. x is on the manifold
    :return: a vector in Rd. not necessarily on the manifold. or even parallel to it's tangent space at  the pointx
    '''


    try:
        x,y,z = x_pos[0]
    except:
        x,y,z = x_pos
    return np.array([[2*a*x + d*y + e*z],
                    [2*b*y + d*x + g*z],
                    [2*c*z + e*x + g*y]])

def R_x(v):


    return (v)/np.linalg.norm(v)


def alpha(k):
    return 0.01
from math import sqrt
iteration_list = [np.array([sqrt(0.3), sqrt(0.3), sqrt(0.4)])]
if __name__ == "__main__":
    for k in range (1,50000):
        x_curr = iteration_list.pop(-1) #i.e x_k
        x_next = R_x(x_curr - alpha(k) * proj(x_curr, gradf(x_curr).T).flatten())
        iteration_list.append(x_next)
    print(iteration_list[0])

import numpy as np
from math import sqrt

dimension = 3

# f(x,y,z) = x^2 + 2*y^2
a, b, c = 1, 2, 0
d, e, g = 0, 0, 0


def proj(x, v):
    """Project v onto tangent space at x"""
    x = x.flatten()
    v = v.flatten()
    return v - np.dot(v, x) * x


def f(x):
    """Evaluate f(x,y,z) = x^2 + 2*y^2"""
    x = x.flatten()
    return x[0] ** 2 + 2 * x[1] ** 2


def gradf(x_pos):
    """Gradient of f"""
    x_pos = x_pos.flatten()
    x, y, z = x_pos
    return np.array([2 * x, 4 * y, 0])  # Simplified since c=d=e=g=0


def R_x(v):
    """Project to unit sphere"""
    return v / np.linalg.norm(v)


# def analyze_critical_point(x):
#     """Check if point is a critical point"""
#     grad = gradf(x)
#     proj_grad = proj(x, grad)
#     grad_norm = np.linalg.norm(proj_grad)
#
#     print(f"Point: {x}")
#     print(f"f(x): {f(x):.8f}")
#     print(f"Gradient: {grad}")
#     print(f"Projected gradient: {proj_grad}")
#     print(f"||Projected gradient||: {grad_norm:.10f}")
#     print(f"Is critical point: {grad_norm < 1e-6}")
#     return grad_norm < 1e-6


# print("DIAGNOSTIC: Why is the algorithm stuck?")
# print("=" * 50)
#
# # Your stuck point
# stuck_point = np.array([0.47596315, 0.33655677, 0.81251992])
# print("Analysis of your stuck point:")
# analyze_critical_point(stuck_point)
#
# print("\n" + "=" * 50)
# print("Analysis of true minimum points:")
# print("=" * 50)
#
# print("\nAt (0,0,1):")
# analyze_critical_point(np.array([0, 0, 1]))
#
# print("\nAt (0,0,-1):")
# analyze_critical_point(np.array([0, 0, -1]))
#
# print("\n" + "=" * 50)
# print("TESTING DIFFERENT LEARNING RATES:")
# print("=" * 50)
#
#
# def test_learning_rate(lr, start_point, max_iter=100):
#     """Test gradient descent with given learning rate"""
#     x_curr = start_point.copy()
#
#     for k in range(max_iter):
#         grad = gradf(x_curr)
#         proj_grad = proj(x_curr, grad)
#
#         # Check if we're at critical point
#         if np.linalg.norm(proj_grad) < 1e-10:
#             print(f"  Converged at iteration {k}")
#             break
#
#         # Take step
#         x_curr = R_x(x_curr - lr * proj_grad)
#
#         if k % 20 == 0:
#             print(f"  Iter {k}: f(x) = {f(x_curr):.6f}, ||proj_grad|| = {np.linalg.norm(proj_grad):.8f}")
#
#     return x_curr
#
#
# # Test different learning rates
# learning_rates = [0.01, 0.1, 0.5, 1.0]
# start_point = np.array([sqrt(0.5), sqrt(0.5), 0])
#
# for lr in learning_rates:
#     print(f"\nLearning Rate: {lr}")
#     final_point = test_learning_rate(lr, start_point)
#     print(f"  Final point: {final_point}")
#     print(f"  Final f(x): {f(final_point):.8f}")
#     print(f"  Distance to (0,0,1): {np.linalg.norm(final_point - np.array([0, 0, 1])):.6f}")
#
# print("\n" + "=" * 50)
# print("MANUAL CHECK: What should happen?")
# print("=" * 50)
#
# # Let's manually trace a few steps
# x = np.array([sqrt(0.5), sqrt(0.5), 0])
# print(f"Start: {x}, f = {f(x):.6f}")
#
# for step in range(5):
#     grad = gradf(x)
#     proj_grad = proj(x, grad)
#
#     print(f"\nStep {step + 1}:")
#     print(f"  Gradient: {grad}")
#     print(f"  Projected gradient: {proj_grad}")
#     print(f"  ||Projected gradient||: {np.linalg.norm(proj_grad):.8f}")
#
#     # Take step with lr=0.5
#     x_new = R_x(x - 0.5 * proj_grad)
#     print(f"  Next point: {x_new}")
#     print(f"  f(x_new): {f(x_new):.6f}")
#
#     x = x_new
