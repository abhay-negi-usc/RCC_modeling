import numpy as np 

def z_rotation_matrix(theta): 
    """ Rotation around z-axis """
    return np.array([
        [np.cos(theta), -np.sin(theta), 0, 0],
        [np.sin(theta),  np.cos(theta), 0, 0],
        [0,             0,             1, 0],
        [0,             0,             0, 1]
    ])

def kuka_fk(
        joint_positions, 
        link_lengths = np.array([0.1575, 0.2025, 0.2045, 0.2155, 0.1845, 0.2155, 0.0810, 0.0450]) * 1e3 
    ): 
    if joint_positions.shape[0] == 7:
        joint_positions = np.concatenate((joint_positions, np.array([0])), axis=0) 

    tf_Ji_o = np.empty((8,4,4)) 
    tf_Ji_o[0,:,:] = np.array([
        [ 1.    ,  0.    ,  0.    ,  0.    ],
        [ 0.    ,  1.    ,  0.    ,  0.    ],
        [-0.    ,  0.    ,  1.    ,  link_lengths[0]],
        [ 0.    ,  0.    ,  0.    ,  1.    ]])
    tf_Ji_o[1,:,:] = np.array([
        [-1, 0,  0, 0],
        [ 0, 0,  1, 0],
        [ 0, 1,  0, link_lengths[1]],
        [ 0, 0, 0, 1]])
    tf_Ji_o[2,:,:] = np.array([
        [-1, 0, 0, 0],
        [ 0, 0, 1, link_lengths[2]],
        [ 0, 1, 0, 0],
        [ 0, 0, 0, 1]])
    tf_Ji_o[3,:,:] = np.array([
        [ 1, 0, 0, 0],
        [ 0, 0,-1, 0],
        [ 0, 1, 0, link_lengths[3]],
        [ 0, 0, 0, 1]])
    tf_Ji_o[4,:,:] = np.array([
        [-1, 0, 0, 0],
        [ 0, 0, 1, link_lengths[4]],
        [ 0, 1, 0, 0],
        [ 0, 0, 0, 1]])
    tf_Ji_o[5,:,:] = np.array([
        [ 1, 0,  0,  0],
        [ 0, 0, -1,  0],
        [ 0, 1,  0,  link_lengths[5]],
        [ 0, 0,  0, 1]])
    tf_Ji_o[6,:,:] = np.array([
        [-1, 0, 0, 0],
        [ 0, 0, 1, link_lengths[6]],
        [ 0, 1, 0, 0],
        [ 0, 0, 0, 1]])
    tf_Ji_o[7,:,:] = np.array([
        [ 1,  0,  0,  0.   ],
        [ 0,  1,  0,  0.   ],
        [ 0,  0,  1,  link_lengths[7]],
        [ 0,  0,  0,  1.   ]])
    
    tf = np.eye(4)
    for i in range(8): 
        tf = tf @ tf_Ji_o[i,:,:] @ z_rotation_matrix(joint_positions[i])  
    return tf 

def kuka_fk_batch(
        joint_positions, 
        link_lengths = np.array([0.1575, 0.2025, 0.2045, 0.2155, 0.1845, 0.2155, 0.0810, 0.0450])*1e3
    ): 
    n = joint_positions.shape[0] 
    tf = [] 
    for i in range(n): 
        tf.append(kuka_fk(joint_positions[i,:], link_lengths)) 
    tf = np.array(tf) 
    # tf[:,:3,3] *= 1000 
    tf = tf.reshape((n,4,4))
    return tf 

import numpy as np

def transform_wrench_ref_to_tgt(wrenches_ref: np.ndarray, T_ref_tgt: np.ndarray) -> np.ndarray:
    """
    Convert a batch (N x 6) of wrenches from a reference frame to a target frame.

    Parameters
    ----------
    wrenches_ref : (N, 6) ndarray
        Rows are [fx, fy, fz, tx, ty, tz] expressed in the *reference* frame.
    T_ref_tgt : (4, 4) ndarray
        Homogeneous transform of the *target frame w.r.t. the reference frame*:
        T = [[R(3x3), p(3x1)],
             [  0   ,    1  ]]

    Returns
    -------
    wrenches_tgt : (N, 6) ndarray
        The input wrenches expressed in the *target* frame in the same order
        [fx, fy, fz, tx, ty, tz].

    Notes
    -----
    Uses:
        f_tgt   = R^T @ f_ref
        tau_tgt = R^T @ (tau_ref - p x f_ref)
    """
    w = np.asarray(wrenches_ref)
    if w.ndim != 2 or w.shape[1] != 6:
        raise ValueError("wrenches_ref must be an (N, 6) array of [fx, fy, fz, tx, ty, tz].")

    if T_ref_tgt.shape != (4, 4):
        raise ValueError("T_ref_tgt must be a 4x4 homogeneous transform.")

    R = T_ref_tgt[:3, :3]
    p = T_ref_tgt[:3, 3]
    Rt = R.T

    f_ref = w[:, 0:3]           # (N,3)
    tau_ref = w[:, 3:6]         # (N,3)

    # p x f for all rows
    pxf = np.cross(p[None, :], f_ref, axis=1)  # (N,3)

    f_tgt = (Rt @ f_ref.T).T                    # (N,3)
    tau_tgt = (Rt @ (tau_ref - pxf).T).T       # (N,3)

    return np.hstack([f_tgt, tau_tgt])
