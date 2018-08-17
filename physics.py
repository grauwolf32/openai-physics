import numpy as np 
from aux import rotate_vec

# Test
tetra_len = 1.0
r_ball = tetra_len * np.sqrt(3.0/8.0)
    
relative_system = np.eye(3)
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
#P = np.array([position]*4)

R = A + B #+ P
J = J_ball + (np.dot(r1,r1)*np.eye(3) - np.linalg.outer(r1,r1)) * dot_masses[0] \ 
           + (np.dot(r2,r2)*np.eye(3) - np.linalg.outer(r2,r2)) * dot_masses[1] \ 
           + (np.dot(r3,r3)*np.eye(3) - np.linalg.outer(r3,r3)) * dot_masses[2] \
           + (np.dot(r4,r4)*np.eye(3) - np.linalg.outer(r4,r4)) * dot_masses[3]

W = np.dot(U, np.diag(w_abs))
E = np.dot(U, np.diag(e_abs))

dr1 = np.cross(Omega, A[0] + gamma) + np.cross(Omega + W[0,:], B[0,:]) 
dr2 = np.cross(Omega, A[1] + gamma) + np.cross(Omega + W[1,:], B[1,:]) 
dr3 = np.cross(Omega, A[2] + gamma) + np.cross(Omega + W[2,:], B[2,:]) 
dr4 = np.cross(Omega, A[3] + gamma) + np.cross(Omega + W[3,:], B[3,:]) 

dJdt = (np.linalg.outer(dr1,r1) + np.linalg.outer(r1,dr1))* dot_masses[0] +\
       (np.linalg.outer(dr2,r2) + np.linalg.outer(r2,dr2))* dot_masses[1] +\ 
       (np.linalg.outer(dr3,r3) + np.linalg.outer(r3,dr3))* dot_masses[2] +\
       (np.linalg.outer(dr4,r4) + np.linalg.outer(r4,dr4))* dot_masses[3] 

Omega = Omega_start

f1 = (np.cross(E[0,:], B[0,:]) + np.cross(W[0,:],np.cross(W[0,:], B[0,:])) + g_abs + np.cross(Omega, np.cross(W[0,:], B[0,:]))) * dot_masses[0]
f2 = (np.cross(E[1,:], B[1,:]) + np.cross(W[1,:],np.cross(W[1,:], B[1,:])) + g_abs + np.cross(Omega, np.cross(W[1,:], B[1,:]))) * dot_masses[1]
f3 = (np.cross(E[2,:], B[2,:]) + np.cross(W[2,:],np.cross(W[2,:], B[2,:])) + g_abs + np.cross(Omega, np.cross(W[2,:], B[2,:]))) * dot_masses[2]
f4 = (np.cross(E[3,:], B[3,:]) + np.cross(W[3,:],np.cross(W[3,:], B[3,:])) + g_abs + np.cross(Omega, np.cross(W[3,:], B[3,:]))) * dot_masses[3]

f_ball = g_abs * m_ball
F = np.array([f1, f2, f3, f4])

Mo = np.cross(R[0]-tangent_point, F[0]) + \
     np.cross(R[1]-tangent_point, F[1]) + \
     np.cross(R[2]-tangent_point, F[2]) + \
     np.cross(R[3]-tangent_point, F[3]) 
     # Gravity force of the ball made no moment aganist tangent point

F_all = np.sum(F, axis=0) + f_ball
React = -np.dot(F_all, absolute_system[3,:])*absolute_system[3,:]
F_act = F_all + React # Must be parallel to the plain

dOmegadt = np.linalg.inv(J)*(Mo + np.dot(dJdt,Omega))

while t < T:
    Omega = Omega + dOmegadt*dt
    position = position + velocity*dt
    velocity = velocity + F_act * dt / (m_ball + np.sum(dot_masses))

    tangent_point = np.array([position[0], position[1], 0.0]) # point in common of the sphere and the plain
    gamma = position - tangent_point
 
    if np.dot(velocity, absolute_system[3,:]) < 0.0 and position[2] <= ball_radious + eps: # Enforce correction of the velocity
        velocity = velocity - absolute_system[3,:]*np.dot(absolute_system[3,:], velocity)

    Omega_Rot = omega_matrix(Omega)
    P = np.array([position]*4)

    U = U + np.dot(P+U, Omega_Rot)*dt
    A = A + np.dot(P+U, np.diag([np.sqrt(1.0/8.0)]*4))*dt

    B[0] = B[0] + np.dot(np.cross(B[0],W[0])*dt, Omega_Rot)
    B[1] = B[1] + np.dot(np.cross(B[1],W[1])*dt, Omega_Rot)
    B[2] = B[2] + np.dot(np.cross(B[2],W[2])*dt, Omega_Rot)
    B[3] = B[3] + np.dot(np.cross(B[3],W[3])*dt, Omega_Rot)
    R = A + B

    w_abs  = w_abs + e_abs * dt
    W = np.dot(U, np.diag(w_abs))
    E = np.dot(U, np.diag(e_abs))
    e_abs = controller() # input from controller

    f1 = (np.cross(E[0,:], B[0,:]) + np.cross(W[0,:],np.cross(W[0,:], B[0,:])) + g_abs + np.cross(Omega, np.cross(W[0,:], B[0,:]))) * dot_masses[0]
    f2 = (np.cross(E[1,:], B[1,:]) + np.cross(W[1,:],np.cross(W[1,:], B[1,:])) + g_abs + np.cross(Omega, np.cross(W[1,:], B[1,:]))) * dot_masses[1]
    f3 = (np.cross(E[2,:], B[2,:]) + np.cross(W[2,:],np.cross(W[2,:], B[2,:])) + g_abs + np.cross(Omega, np.cross(W[2,:], B[2,:]))) * dot_masses[2]
    f4 = (np.cross(E[3,:], B[3,:]) + np.cross(W[3,:],np.cross(W[3,:], B[3,:])) + g_abs + np.cross(Omega, np.cross(W[3,:], B[3,:]))) * dot_masses[3]

    f_ball = g_abs * m_ball
    F = np.array([f1, f2, f3, f4])

    Mo = np.cross(R[0]-tangent_point, F[0]) + \
         np.cross(R[1]-tangent_point, F[1]) + \
         np.cross(R[2]-tangent_point, F[2]) + \
         np.cross(R[3]-tangent_point, F[3]) 

    F_all = np.sum(F, axis=0) + f_ball
    React = -np.dot(F_all, absolute_system[3,:])*absolute_system[3,:]
    F_act = F_all + React 

    t += dt

