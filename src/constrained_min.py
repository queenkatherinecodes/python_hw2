import numpy as np

def interior_pt(func, ineq_constraints, eq_constraints_mat, eq_constraints_rhs, x0):
    t = 1
    mu = 10
    m = len(ineq_constraints)
    epsilon = 1e-10
    x_k = x0
    history = []
    while (m/t) >= epsilon:
        if eq_constraints_mat is not None:
            x_k = kkt(func, eq_constraints_mat, eq_constraints_rhs, x_k, ineq_constraints, t)
        else:
            x_k = unconstrained_minimize(func, x_k, ineq_constraints, t)
        f_x_k, _, _ = func(x_k)
        history.append((x_k, f_x_k))
        t = mu * t

    return x_k, f_x_k, history

def augmented_func(func, x, ineq_constraints, t):
    f_log_barrier, g_log_barrier, h_log_barrier = 0, np.zeros(len(x)), np.zeros((len(x), len(x)))
    for con in ineq_constraints:
        f_con_x, g_con_x, h_con_x = con(x)
        f_log_barrier -= np.log(-f_con_x)
        g_log_barrier -= (1/f_con_x) * g_con_x
        h_log_barrier += (1/(f_con_x ** 2)) * np.outer(g_con_x, g_con_x) - (1/f_con_x) * h_con_x
    f_x, g_x, h_x = func(x)
    return f_x + (1/t) * f_log_barrier, g_x + (1/t) * g_log_barrier, h_x +  (1/t) * h_log_barrier

def kkt(func, eq_constraints_mat, eq_constraints_rhs, x0, ineq_constraints, t, max_iter=100):
    x_k = x0
    p = len(eq_constraints_rhs)
    for i in range(max_iter):
        try:
            x = np.array(x_k, dtype=float)
            f_x, g_x, h_x = augmented_func(func, x, ineq_constraints, t)
            matrix1 = np.concat((h_x, eq_constraints_mat.T), axis = 1)
            matrix2 = np.concat((eq_constraints_mat, np.zeros((p,p))), axis = 1)
            matrix = np.concat((matrix1, matrix2), axis = 0)
            kkt_rhs = np.concat((-g_x, np.zeros(p)), axis = 0)
            n = len(h_x)
            p_k = np.linalg.solve(matrix, kkt_rhs)[:n]
            alpha_k = set_alpha(func, p_k, x, ineq_constraints, t)
            can_terminate_newton = .5 * np.dot(p_k, np.dot(h_x, p_k)) < 1e-12
            x_k = x + (alpha_k * p_k)
            f_x_k, _, _ = augmented_func(func, x_k, ineq_constraints, t)
            if can_terminate(x_k, x, f_x_k, f_x) or can_terminate_newton:
                break
        except RuntimeError:
            return x
    return x_k

def can_terminate(x_k, x, f_x_k, f_x, obj_tol=1e-12, param_tol=1e-8):
    obj_change = abs(f_x_k - f_x)
    param_change = np.linalg.norm(x_k - x)
    return obj_change < obj_tol or param_change < param_tol

def set_alpha(func, p_k, x, ineq_constraints, t, rho=0.5, c=0.01):
    f_x_k, g_x_k, _ = augmented_func(func, x, ineq_constraints, t)
    alpha = 1.0
    
    for con in ineq_constraints:
        f_con_x, _, _ = con(x + alpha * p_k)
        while f_con_x >= 0:  
            alpha *= rho
            if alpha < 1e-10:  
                return alpha
            f_con_x, _, _ = con(x + alpha * p_k)
    
    small, _, _ = augmented_func(func, (x + alpha*p_k), ineq_constraints, t)
    big = f_x_k + c * np.dot(g_x_k, p_k) * alpha
    
    while small > big:
        alpha = rho * alpha
        feasible = True
        for con in ineq_constraints:
            f_con_x, _, _ = con(x + alpha * p_k)
            if f_con_x >= 0:
                feasible = False
                break
        if not feasible:
            alpha /=rho
            continue
        small, _, _ = augmented_func(func, (x + alpha*p_k), ineq_constraints, t)
        big = f_x_k + c * np.dot(g_x_k, p_k) * alpha
    
    return alpha

def unconstrained_minimize(f, x0, ineq_constraints, t, obj_tol=1e-12, param_tol=1e-8, max_iter=100):
    x_k = x0
    x_history = []
    f_history = []
    try:
        for i in range(max_iter):
            x = np.array(x_k, dtype=float)
            f_x, g_x, h_x = augmented_func(f, x, ineq_constraints, t)
            p_k = -g_x
            alpha = set_alpha(f, p_k, x, ineq_constraints, t)
            can_terminate_newton = .5 * np.dot(p_k, np.dot(h_x, p_k)) < obj_tol
            x_history.append(x)
            f_history.append(f_x)
            x_k = x + alpha * p_k
            f_x_k, _, _ = augmented_func(f, x_k, ineq_constraints, t)
            f_x_true, _, _ = f(x_k)
            if can_terminate(x_k, x, f_x_k, f_x, obj_tol, param_tol) or can_terminate_newton:
                x_history.append(x_k)
                f_history.append(f_x_true)
                return x_k
        x_history.append(x_k)
        f_history.append(f_x_true)
        return x_k
    except np.linalg.LinAlgError:
        if x_history:
            return x_history[-1]
        else:
            return x0

