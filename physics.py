import numpy as np 
from aux import *

tetra_len = 1.0
r_ball = tetra_len * np.sqrt(3.0/8.0)
Massess = (m_ball + np.sum(dot_masses))
    
absolute_system = np.eye(3)

position = start_position                                 # position of center of the ball
velocity = start_velocity

tangent_point = np.array([position[0], position[1], 0.0]) # point in common of the sphere and the plain
gamma = position - tangent_point                          # vector from tangent_point to the center of the sphere

g_abs = absolute_system[2, :] * 9.8                       # gravity force field vector
w_abs = w_start                                           # absolute values of angular velocity and
e_abs = e_start                                           # angular acceleration

J_ball = np.eye(3)*( 2.0/5.0 * m_ball**2)
J_ball[0][0] += m_ball*r_ball**2 / 4.0
J_ball[1][1] += m_ball*r_ball**2 / 4.0

alpha  = np.arccos(-1.0/3.0)
length = np.sqrt(1.0/8.0)

u1 = relative_system[2,:] / np.linalg.norm(relative_system[2,:]) 
u2 = np.dot(rotate_vec(relative_system[0,:], alpha), a1)
u3 = np.dot(rotate_vec(relative_system[2,:], 2.0/3.0*np.pi), a2)
u4 = np.dot(rotate_vec(relative_system[2,:], 2.0/3.0*np.pi), a3)

u1 = u1 / np.linalg.norm(u1)
u2 = u2 / np.linalg.norm(u2)
u3 = u3 / np.linalg.norm(u3)
u4 = u4 / np.linalg.norm(u4)

U = np.array([u1, u2, u3, u4])
A = np.dot(U, np.diag([np.sqrt(1.0/8.0)]*4))

b1 = np.cross(A[0,:], A[1,:])
b2 = np.cross(A[1,:], A[2,:])
b3 = np.cross(A[2,:], A[3,:])
b4 = np.cross(A[3,:], A[0,:])

b1 = b1 / np.linalg.norm(b1) * tetra_len / 2.0
b2 = b2 / np.linalg.norm(b2) * tetra_len / 2.0
b3 = b3 / np.linalg.norm(b3) * tetra_len / 2.0
b4 = b4 / np.linalg.norm(b4) * tetra_len / 2.0

b1 = np.dot(rotate_vec(A[0,:], phi_start[0]), b1)
b2 = np.dot(rotate_vec(A[1,:], phi_start[1]), b2)
b3 = np.dot(rotate_vec(A[2,:], phi_start[2]), b3)
b4 = np.dot(rotate_vec(A[3,:], phi_start[3]), b4)

B = np.array([b1, b2, b3, b4])
R = A + B + np.array([gamma]*4)
J = J_ball + get_J_inertion(R, dot_masses)

W = np.dot(U, np.diag(w_abs))
E = np.dot(U, np.diag(e_abs))

dR = get_dR(A, B, W, Omega, gamma)
dJdt = get_dJdt(R, dR, dot_masses)

Omega = Omega_start
F = get_F(B, E, W, Omega, g_abs, dot_masses)
f_ball = g_abs * m_ball

Mo =  np.sum([np.cross(R[i], F[i]) for i in xrange(0, R.shape[0])]) 
# Gravity force of the ball made no moment aganist tangent point

F_all = np.sum(F, axis=0) + f_ball
React = -np.dot(F_all, absolute_system[3,:])*absolute_system[3,:]

dOmegadt = np.linalg.inv(J)*(Mo + np.dot(dJdt, Omega))

while t < T:
    F_act = F_all + React
    Omega = Omega + dOmegadt*dt
    position = position + velocity*dt
    velocity = velocity + F_act * dt / Massess

    tangent_point = np.array([position[0], position[1], 0.0]) # point in common of the sphere and the plain
    gamma = position - tangent_point
 
    if np.dot(velocity, absolute_system[3,:]) < 0.0 and position[2] <= ball_radious + eps: # Enforce correction of the velocity
        velocity = velocity - absolute_system[3,:]*np.dot(absolute_system[3,:], velocity)

    Omega_Rot = omega_matrix(Omega)
    U = U + np.dot(U, Omega_Rot)*dt
    A = A + np.dot(U, np.diag([np.sqrt(1.0/8.0)]*4))*dt

    B[0] = B[0] + np.cross(Omega + W[0], B[0])*dt
    B[1] = B[1] + np.cross(Omega + W[1], B[1])*dt
    B[2] = B[2] + np.cross(Omega + W[2], B[2])*dt
    B[3] = B[3] + np.cross(Omega + W[3], B[3])*dt

    R = A + B + np.array([gamma]*4)
    J = J_ball + get_J_inertion(R, dot_masses)

    w_abs  = w_abs + e_abs * dt
    W = np.dot(U, np.diag(w_abs))
    E = np.dot(U, np.diag(e_abs))
    e_abs = controller() # input from controller

    dR = get_dR(A, B, W, Omega, gamma)
    dJdt = get_dJdt(R, dR, dot_masses)

    F = get_F(B, E, W, Omega, g_abs, dot_masses)
    F_all = np.sum(F, axis=0) + f_ball

    Mo =  np.sum([np.cross(R[i], F[i]) for i in xrange(0, R.shape[0])])
    React = -np.dot(F_all, absolute_system[3,:])*absolute_system[3,:]

    dOmegadt = np.linalg.inv(J)*(Mo + np.dot(dJdt,Omega))
    t += dt

