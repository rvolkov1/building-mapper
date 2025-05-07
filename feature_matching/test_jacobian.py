import numpy as np
from rotation_utils import skew, dcm2quat, quat2dcm
from scipy.differentiate import jacobian


# Projection functions
def X_prime(q, t, X):
    if q[0] < 0:
        q = -q
    q = q/np.linalg.norm(q)
    R = quat2dcm(q)
    return (R @ X.reshape((3, 1)) + t.reshape((3, 1))).flatten()


def pi_h(X_prime):
    # WARNING: The negative sign comes from the convention used by Agarwal. May need to change for a different coordinate system.
    X_prime = X_prime.flatten()
    x_p = X_prime[0]
    y_p = X_prime[1]
    z_p = X_prime[2]
    return -np.array([x_p/z_p, y_p/z_p, 1]).flatten()


def pi_k(p_tilde, f):
    K = np.array([[f, 0, 0], [0, f, 0], [0, 0, 0]])
    return K @ p_tilde


def GetJacobian_X_prime_q(camParams, X):
    f_q = lambda q: X_prime(q, camParams[4:], X)
    f_q_vec = lambda q : np.apply_along_axis(f_q, 0, q)
    res = jacobian(f_q_vec, camParams[0:4], initial_step=0.001, maxiter=50, step_factor=2.5)
    return res.df

def GetJacobian_X_prime_t(camParams, X):
    f_t = lambda t: X_prime(camParams[0:4], t, X)
    f_t_vec = lambda t : np.apply_along_axis(f_t, 0, t)
    res = jacobian(f_t_vec, camParams[4:], initial_step=0.1, step_factor=0.5, maxiter=50)
    return res.df


def GetJacobian_X_prime_X(camParams, X):
    f_X = lambda x: X_prime(camParams[0:4], camParams[4:], x)
    f_X_vec = lambda x : np.apply_along_axis(f_X, 0, x)
    res = jacobian(f_X_vec, X)
    return res.df

def GetJacobian_pi_h_X_prime(X_prime):
    f_Xp_vec = lambda x : np.apply_along_axis(pi_h, 0, x)
    res = jacobian(f_Xp_vec, X_prime)
    return res.df

def GetJacobian_full_X(camParams, X, f):
    f_X = lambda x :  pi_k(pi_h(X_prime(camParams[0:4], camParams[4:], x)), f)
    f_X_vec = lambda x :  np.apply_along_axis(f_X, 0, x)
    res = jacobian(f_X_vec, X)
    return res.df

def GetJacobian_full_q(camParams, X, f):
    f_q = lambda q :  pi_k(pi_h(X_prime(q, camParams[4:], X)), f)
    f_q_vec = lambda q :  np.apply_along_axis(f_q, 0, q)
    res = jacobian(f_q_vec, camParams[0:4])
    return res.df

def GetJacobian_full_t(camParams, X, f):
    f_t = lambda t :  pi_k(pi_h(X_prime(camParams[0:4], t, X)), f)
    f_t_vec = lambda t :  np.apply_along_axis(f_t, 0, t)
    res = jacobian(f_t_vec, camParams[4:])
    return res.df
