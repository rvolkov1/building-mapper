import numpy as np
from rotation_utils import skew, dcm2quat, quat2dcm
from scipy.differentiate import jacobian


# PROJECTION FUNCTIONS
def TransformWorldPoint(q, t, X):
    """ 
    Project a point in the world frame to the camera frame for a single world
    point. This is used when computing the Jacobian numerically.
    
    Args:
        q (ndarray): quaternion that rotates from the world frame to the camera
            frame. Shape = (4,)
        t (ndarray): translation vector from the origin of the world frame to
            the origin of the camera frame resolved in the camera frame. 
            Shape = (3,)
        X (ndarray): the 3D point in the world frame. Shape = (3,)
    Returns:
        ndarray: (3,) point in the camera frame
    """
    # Make sure the quaternion is unit norm
    q = q/np.linalg.norm(q)
    R = quat2dcm(q)
    return (R @ X.reshape((3, 1)) + t.reshape((3, 1))).flatten()


def TransformWorldPoint_vec(q, t, X):
    """ 
    Project a point in the world frame to the camera frame for many world
    points.
    
    Args:
        q (ndarray): quaternion that rotates from the world frame to the camera
            frame. Shape = (4,)
        t (ndarray): translation vector from the origin of the world frame to
            the origin of the camera frame resolved in the camera frame. 
            Shape = (3,)
        X (ndarray): the 3D points in the world frame. Shape = (3, num_pts)
            WARNING: Ensure this is the correct shape, even if there is only
            one point.
    Returns:
        ndarray: (3, num_pts) array of points in the camera frame
    """
    # Make sure the quaternion is unit norm
    q = q/np.linalg.norm(q)
    R = quat2dcm(q)
    return R @ X + t.reshape((3, 1))


def Pi_h(X_prime):
    """
    Computes the homogenous coordinates in the camera frame
    
    Args:
        X_prime (ndarray): 3D points in the camera frame. Shape = (3, num_pts)
    Returns:
        ndarray: The homogenous coordinates of each point. Shape = (3, num_pts)
    """
    # WARNING: The negative sign comes from the convention used by Agarwal. May
    # need to change for a different coordinate system.
    X_prime = X_prime.reshape(3, -1)
    p_tilde = -X_prime/X_prime[-1, :]
    return p_tilde


def Pi_k(p_tilde, f):
    """
    Computes the pixel coordinates
    
    Args:
        p_tilde (ndarray): Homogenous points in the camera frame. Shape = (3, num_pts)
        f (float): Camera focal length
    Returns:
        ndarray: Pixel coordinates of each point. Shape = (2, num_pts)
    """
    K = np.array([[f, 0, 0], [0, f, 0], [0, 0, 0]])
    return K @ p_tilde

# NUMERICAL JACOBIANS
def GetJacobian_X_prime_q(camParams, X):
    f_q = lambda q: TransformWorldPoint(q, camParams[4:], X)
    f_q_vec = lambda q : np.apply_along_axis(f_q, 0, q)
    res = jacobian(f_q_vec, camParams[0:4], initial_step=0.001, maxiter=50, step_factor=2.5)
    return res.df


def GetJacobian_X_prime_t(camParams, X):
    f_t = lambda t: TransformWorldPoint(camParams[0:4], t, X)
    f_t_vec = lambda t : np.apply_along_axis(f_t, 0, t)
    res = jacobian(f_t_vec, camParams[4:], initial_step=0.1, step_factor=0.5, maxiter=50)
    return res.df


def GetJacobian_X_prime_X(camParams, X):
    f_X = lambda x: TransformWorldPoint(camParams[0:4], camParams[4:], x)
    f_X_vec = lambda x : np.apply_along_axis(f_X, 0, x)
    res = jacobian(f_X_vec, X)
    return res.df


def GetJacobian_pi_h_X_prime(X_prime):
    f_Xp_vec = lambda x : np.apply_along_axis(Pi_h, 0, x)
    res = jacobian(f_Xp_vec, X_prime)
    return res.df.reshape(3,3)


def GetJacobian_full_X(camParams, X, f):
    f_X = lambda x :  Pi_k(Pi_h(TransformWorldPoint(camParams[0:4], camParams[4:], x)), f)
    f_X_vec = lambda x :  np.apply_along_axis(f_X, 0, x)
    res = jacobian(f_X_vec, X)
    return res.df[0:2].reshape(2,3)


def GetJacobian_full_q(camParams, X, f):
    f_q = lambda q :  Pi_k(Pi_h(TransformWorldPoint(q, camParams[4:], X)), f)
    f_q_vec = lambda q :  np.apply_along_axis(f_q, 0, q)
    res = jacobian(f_q_vec, camParams[0:4])
    return res.df[0:2].reshape(2,4)


def GetJacobian_full_t(camParams, X, f):
    f_t = lambda t :  Pi_k(Pi_h(TransformWorldPoint(camParams[0:4], t, X)), f)
    f_t_vec = lambda t :  np.apply_along_axis(f_t, 0, t)
    res = jacobian(f_t_vec, camParams[4:])
    return res.df[0:2].reshape(2,3)

