import numpy as np 
from aux import *


class SolidBody(object):
    def __init__(self, mass, angular_velocity, )

class HyroSphere(object):
    def __init__(self, t_len, mass, dot_masses, position, phi_start=np.zeros(3), omega_start=np.zeros(3), ksi_start=np.zeros(3), Omega_start=np.zeros(3)):
        self.radius = t_len * np.sqrt(3.0/8.0)
        self.relative_system = np.eye(3) 
        
        self.mass = mass
        self.dot_masses = dot_masses
        self.position = position # Position of center of the ball
        self.phi = phi_start

        u1 = relative_system[2,:] / np.linalg.norm(relative_system[2,:]) 
        u2 = np.dot(rotate_vec(relative_system[0,:], np.arccos(-1.0/3.0)), u1)
        u3 = np.dot(rotate_vec(relative_system[2,:], 2.0/3.0*np.pi), u2)
        u4 = np.dot(rotate_vec(relative_system[2,:], 2.0/3.0*np.pi), u3)

        u1 = u1 / np.linalg.norm(u1)
        u2 = u2 / np.linalg.norm(u2)
        u3 = u3 / np.linalg.norm(u3)
        u4 = u4 / np.linalg.norm(u4)

        self.U = np.array([u1, u2, u3, u4])

        self.ksi = ksi_start     # absolute values of angular acceleration
        self.omega = omega_start # and angular velocity for dot masses
        self.Omega = Omega_start

        self.J_ball = np.eye(3)*( 2.0/5.0 * self.mass**2)
        self.J_ball[0][0] += self.mass*self.radius**2 / 4.0
        self.J_ball[1][1] += self.mass*self.radius**2 / 4.0

    def move(self, dt, ksi_new, tangent_point):
        gamma = position - tangent_point 
        A = self.U * np.sqrt(1.0/8.0)

        b1 = np.cross(A[0,:], A[1,:])
        b2 = np.cross(A[1,:], A[2,:])
        b3 = np.cross(A[2,:], A[3,:])
        b4 = np.cross(A[3,:], A[0,:])

        b1 = b1 / np.linalg.norm(b1) * t_len / 2.0
        b2 = b2 / np.linalg.norm(b2) * t_len / 2.0
        b3 = b3 / np.linalg.norm(b3) * t_len / 2.0
        b4 = b4 / np.linalg.norm(b4) * t_len / 2.0

        b1 = np.dot(rotate_vec(self.U[0,:], self.phi[0]), b1)
        b2 = np.dot(rotate_vec(self.U[1,:], self.phi[1]), b2)
        b3 = np.dot(rotate_vec(self.U[2,:], self.phi[2]), b3)
        b4 = np.dot(rotate_vec(self.U[3,:], self.phi[3]), b4)

        B = np.array([b1, b2, b3, b4])
        R = self.A + self.B + np.array([gamma]*4)
        J = self.J_ball + get_J_inertion(R, self.dot_masses)

        W = np.dot(self.U, np.diag(self.omega))
        E = np.dot(self.U, np.diag(self.ksi))
        
        dR = get_dR(A, B, W, self.Omega, gamma)
        dJdt = get_dJdt(R, dR, self.dot_masses)

        F = get_F(self.B, E, W, self.Omega, g_abs, self.dot_masses)
        f_ball = g_abs * m_ball

        Mo =  np.sum([np.cross(R[i], F[i]) for i in xrange(0, R.shape[0])]) 
        # Gravity force of the ball made no moment aganist tangent point

        F_all = np.sum(F, axis=0) + f_ball
        React = -np.dot(F_all, absolute_system[3,:])*absolute_system[3,:]

        dOmegadt = np.linalg.inv(J)*(Mo + np.dot(dJdt, Omega))

        self.phi = self.phi + self.omega*dt
        self.ksi = ksi_new

       

#tetra_len = 1.0
#r_ball = tetra_len * np.sqrt(3.0/8.0)
#Massess = (m_ball + np.sum(dot_masses))
    
absolute_system = np.eye(3)
g_abs = absolute_system[2, :] * 9.8                       # gravity force field vector

#position = start_position                                 # position of center of the ball

#!!! velocity = start_velocity - this value could be derived as [Omega, gamma]
# if velocity of tangent point equsals to zero, velocity of the center point and angular velocity can be considered as equal
# but in other case, velocity of tangent point could be calculated as v_center - [Omega, gamma]

tangent_point = np.array([position[0], position[1], 0.0]) # point in common of the sphere and the plain
gamma = position - tangent_point                          # vector from tangent_point to the center of the sphere


w_abs = w_start                                           # absolute values of angular velocity and
e_abs = e_start                                           # angular acceleration

#J_ball = np.eye(3)*( 2.0/5.0 * m_ball**2)
#J_ball[0][0] += m_ball*r_ball**2 / 4.0
#J_ball[1][1] += m_ball*r_ball**2 / 4.0

#alpha  = np.arccos(-1.0/3.0)
#length = np.sqrt(1.0/8.0)

#u1 = relative_system[2,:] / np.linalg.norm(relative_system[2,:]) 
#u2 = np.dot(rotate_vec(relative_system[0,:], alpha), a1)
#u3 = np.dot(rotate_vec(relative_system[2,:], 2.0/3.0*np.pi), a2)
#u4 = np.dot(rotate_vec(relative_system[2,:], 2.0/3.0*np.pi), a3)

#u1 = u1 / np.linalg.norm(u1)
#u2 = u2 / np.linalg.norm(u2)
#u3 = u3 / np.linalg.norm(u3)
#u4 = u4 / np.linalg.norm(u4)

#U = np.array([u1, u2, u3, u4])
#A = np.dot(U, np.diag([np.sqrt(1.0/8.0)]*4))

#b1 = np.cross(A[0,:], A[1,:])
#b2 = np.cross(A[1,:], A[2,:])
#b3 = np.cross(A[2,:], A[3,:])
#b4 = np.cross(A[3,:], A[0,:])

#b1 = b1 / np.linalg.norm(b1) * tetra_len / 2.0
#b2 = b2 / np.linalg.norm(b2) * tetra_len / 2.0
#b3 = b3 / np.linalg.norm(b3) * tetra_len / 2.0
#b4 = b4 / np.linalg.norm(b4) * tetra_len / 2.0

#b1 = np.dot(rotate_vec(A[0,:], phi_start[0]), b1)
#b2 = np.dot(rotate_vec(A[1,:], phi_start[1]), b2)
#b3 = np.dot(rotate_vec(A[2,:], phi_start[2]), b3)
#b4 = np.dot(rotate_vec(A[3,:], phi_start[3]), b4)

#B = np.array([b1, b2, b3, b4])
R = A + B + np.array([gamma]*4)
J = J_ball + get_J_inertion(R, dot_masses)

W = np.dot(U, np.diag(w_abs))
E = np.dot(U, np.diag(e_abs))

dR = get_dR(A, B, W, Omega, gamma)
dJdt = get_dJdt(R, dR, dot_masses)

#Omega = Omega_start
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

