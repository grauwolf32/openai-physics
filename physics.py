import numpy as np 

def rotate_vec(omega, phi): # Rotate around omega matrix
    R = np.eye(3)
    
    W = np.zero(3,3)
    W[0][1] = -omega[2]
    W[0][2] =  omega[1]
    W[1][2] = -omega[0]

    W[1][0] = -W[0][1]
    W[2][0] = -W[0][2]
    W[2][1] = -W[1][2]

    R = R + np.sin(phi)*W + (1.0-np.cos(phi))*np.dot(W,W)
    return R

relative_system = np.eye(3)
absolute_system = np.eye(3)

g_abs = absolute_system[2, :] * 9.8

J_ball = np.eye(3)*( 2.0/5.0 * m_ball**2)
J_ball[0][0] += m_ball*r_ball**2 / 4.0
J_ball[1][1] += m_ball*r_ball**2 / 4.0

alpha = 1.0 # Pick right value
a1 = relative_system[3,:] / np.linalg.norm(relative_system[3,:]) * np.sqrt(1.0/8.0)
a2 = np.dot(rotate_vec(relative_system[1,:], alpha), a1)
a3 = np.dot(rotate_vec(relative_system[3,:], 2.0/3.0*np.pi), a2)
a4 = np.dot(rotate_vec(relative_system[3,:], 2.0/3.0*np.pi), a3)

A = np.array([a1, a2, a3, a4])

b1 = np.cross(A[0,:], A[1,:])
b2 = np.cross(A[1,:], A[2,:])
b3 = np.cross(A[2,:], A[3,:])
b4 = np.cross(A[3,:], A[0,:])

b1 = b1 / np.linalg.norm(b1) * tetra_len / np.sqrt(3)
b2 = b2 / np.linalg.norm(b2) * tetra_len / np.sqrt(3)
b3 = b3 / np.linalg.norm(b3) * tetra_len / np.sqrt(3)
b4 = b4 / np.linalg.norm(b4) * tetra_len / np.sqrt(3)

b1 = np.dot(rotate_vec(A[0,:], phi_start[0]), b1)
b2 = np.dot(rotate_vec(A[1,:], phi_start[1]), b2)
b3 = np.dot(rotate_vec(A[2,:], phi_start[2]), b3)
b4 = np.dot(rotate_vec(A[3,:], phi_start[3]), b4)

B = np.array([b1, b2, b3, b4])

r1 = A[0,:] + B[0,:]
r2 = A[1,:] + B[1,:]
r3 = A[2,:] + B[2,:]
r4 = A[3,:] + B[3,:]

J = J_ball + np.linalg.outer(r1,r1) * dot_masses[0] \
           + np.linalg.outer(r2,r2) * dot_masses[1] \ 
           + np.linalg.outer(r3,r3) * dot_masses[2] \
           + np.linalg.outer(r4,r4) * dot_masses[3] 

w1 = A[0,:] / np.linalg.norm(A[0,:]) * w_start[0]
w2 = A[1,:] / np.linalg.norm(A[1,:]) * w_start[1]
w3 = A[2,:] / np.linalg.norm(A[2,:]) * w_start[2]
w4 = A[3,:] / np.linalg.norm(A[3,:]) * w_start[3]
W = np.array([w1, w2, w3, w4])

dr1 = np.cross(B[0,:], W[0,:]) # Give right formulas
dr2 = np.cross(B[1,:], W[1,:]) #
dr3 = np.cross(B[2,:], W[2,:]) #
dr4 = np.cross(B[3,:], W[3,:]) #

dJdt = (np.linalg.outer(dr1,r1) + np.linalg.outer(r1,dr1))* dot_masses[0] +\
       (np.linalg.outer(dr2,r2) + np.linalg.outer(r2,dr2))* dot_masses[1] +\ 
       (np.linalg.outer(dr3,r3) + np.linalg.outer(r3,dr3))* dot_masses[2] +\
       (np.linalg.outer(dr4,r4) + np.linalg.outer(r4,dr4))* dot_masses[3] 



