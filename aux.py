import numpy as np

def rotate_vec(omega, phi): # Rotate around omega matrix
    R = np.eye(3)
    omega = omega / np.linalg.norm(omega)

    W = np.zeros((3,3))
    W[0][1] = -omega[2]
    W[0][2] =  omega[1]
    W[1][2] = -omega[0]

    W[1][0] = -W[0][1]
    W[2][0] = -W[0][2]
    W[2][1] = -W[1][2]

    R = R + np.sin(phi)*W + (1.0-np.cos(phi))*np.dot(W,W)
    return R

def get_J_inertion(R, dot_masses):
    n = dot_masses.shape[0]
    J = np.zeros(R.shape[1], R.shape[1])

    for i in xrange(0, n):
       J += (np.dot(R[i],R[i])*np.eye(3) - np.linalg.outer(R[i],R[i])) * dot_masses[i]
    return J

def get_F(B, E, W, Omega, g_abs, dot_masses):
    n  = B.shape[0]
    F = list()
    
    for i in xrange(0, n):
        f = (np.cross(E[i,:], B[i,:]) + np.cross(W[i,:], np.cross(W[i,:], B[i,:])) + g_abs + np.cross(Omega, np.cross(W[i,:], B[i,:]))) * dot_masses[i]
        F.append(f)
    return np.array(F)