import numpy as np

def qp(x):
    f = x[0] **2 + x[1] ** 2 + (x[2] + 1) ** 2
    g = np.array([2 * x[0], 2 * x[1], 2 * (x[2]+1)])
    h = 2 * np.eye(3)
    return f, g, h

def qp_eq_constraints_mat():
    return np.array([[1,1,1]])

def qp_eq_constraints_rhs():
    return np.array([1])

def qp_ineq_constraints():
    return [x_geq_0_constraint, y_geq_0_constraint, z_geq_0_constraint]

def x_geq_0_constraint(x):
    return -x[0], np.array([-1,0,0]), np.zeros((3,3))

def y_geq_0_constraint(x):
    return -x[1], np.array([0,-1,0]), np.zeros((3,3))

def z_geq_0_constraint(x):
    return -x[2], np.array([0,0,-1]), np.zeros((3,3))

def lp(x):
    f = -(x[0] + x[1])
    g = np.array([-1, -1])
    h = np.zeros((2,2))
    return f, g, h

def lp_eq_constraints_mat():
    return None

def lp_eq_constraints_rhs():
    return None

def lp_ineq_constraints():
    return [y_geq_x_minus_1, y_leq_1_constraint, x_leq_2_constraint, y_geq_0_constraint_2d]

def y_geq_x_minus_1(x):
    return (-x[0] - x[1] + 1), np.array([-1, -1]), np.zeros((2,2))

def y_leq_1_constraint(x):
    return x[1] - 1, np.array([0,1]), np.zeros((2,2))

def x_leq_2_constraint(x):
    return x[0] - 2, np.array([1, 0]), np.zeros((2,2))

def y_geq_0_constraint_2d(x):
    return -x[1], np.array([0,-1]), np.zeros((2,2))