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
        self.velocity = np.zeros(3)
        self.phi = phi_start # absolute values of angular speed
        
        self.ksi = ksi_start     # absolute values of angular acceleration
        self.omega = omega_start # and angular velocity for dot masses
        self.Omega = Omega_start
        self.dOmegadt = np.zeros(3)

        u1 = relative_system[2,:] / np.linalg.norm(relative_system[2,:]) 
        u2 = np.dot(rotate_vec(relative_system[0,:], np.arccos(-1.0/3.0)), u1)
        u3 = np.dot(rotate_vec(relative_system[2,:], 2.0/3.0*np.pi), u2)
        u4 = np.dot(rotate_vec(relative_system[2,:], 2.0/3.0*np.pi), u3)

        u1 = u1 / np.linalg.norm(u1)
        u2 = u2 / np.linalg.norm(u2)
        u3 = u3 / np.linalg.norm(u3)
        u4 = u4 / np.linalg.norm(u4)

        self.U = np.array([u1, u2, u3, u4])

        self.J_ball = np.eye(3)*( 2.0/5.0 * self.mass**2)
        self.J_ball[1][1] += self.mass*self.radius**2  # OS has coordinates (0, 0, -r)

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
        R = self.A + self.B 

        J = self.J_ball + get_J_inertion(R, self.dot_masses)
        W = np.dot(self.U, np.diag(self.omega))
        E = np.dot(self.U, np.diag(self.ksi))
       
        dRdt = get_dRdt(R, U, W, self.Omega)
        dJdt = get_dJdt(R, dR, self.dot_masses)
        
        d2Rdt2 = get_d2Rdt2(R, dRdt, U, self.Omega, self.dOmegadt, E)
        F = get_F(d2Rdt2, 9.8, self.mass, self.dot_masses)

        Ms =  get_Ms(R, F, self.radious)
        # Gravity force of the ball made no moment aganist tangent point

        F_all  = np.sum(F, axis=0)
        Ms_all = np.sum(Ms, axis=0)

        self.dOmegadt = np.linalg.inv(J)*(Ms_all + np.dot(dJdt, self.Omega))

        total_mass  = np.sum(self.dot_masses) + self.mass
        mass_center = np.sum(np.dot(R, np.diag(self.dot_masses)), axis=0) / total_mass

        plane_normal = np.asarray([0, 0, 1.0])
        force_proj = np.dot(F_all, plane_normal)

        if force_proj < 0.0:
            dvcdt = F_all - force_proj*plane_normal
        else:
            dvcdt = F_all
        #### Error! Wrong acceleration: velocity must be increased of ball center accel, not the center mass accel
        
        self.velocity += dvcdt * dt
        vc_proj = np.dot(self.velocity, plane_normal)
        if vc_proj < 0.0 and self.position[-1] <= self.radius + 10e-3:
            self.velocity = self.velocity - vc_proj*plane_normal

        self.position += self.velocity * dt
        self.phi = self.phi + self.Omega * dt
        self.Omega += self.dOmegadt
        self.ksi = ksi_new

def get_dRdt(R, U, W, Omega):
    n = R.shape[0]
    G = []
    for i in range(0, n):
        tmp = np.cross(R[i,:], U[i,:])
        tmp = tmp / np.linalg.norm(tmp)
        G.append(tmp)
    G = np.asarray(G)

    OmegaAbs =  W + np.asarray([Omega]*n)
    dRdt = []

    for i in range(0, n):
        tmp = np.cross(OmegaAbs[i,:], R[i,:])
        proj = np.dot(tmp, G[i,:])*G[i,:]
        dRdt.append(proj)

    dRdt = np.asarray(dRdt)
    return dRdt

def get_dJdt(R, dRdt, Omega, r, masses):
    n = R.shape[0]
    os_vec = np.asarray([0.0, 0.0, -r])
    dosdt  = np.cross(Omega, os_vec)

    dJdt = 2.0*np.diag(masses)*np.diag(os_vec)*np.diag(dosdt)

    for in range(0, n):
        dJdt -= (np.linalg.outer(R[i,:], dRdt[i,:]) + np.linal.gouter(dRdt[i,:], R[i,:])) * masses[i]

    return dJdt

def get_d2Rdt2(R, dRdt, U, Omega, dOmegadt, E):
    n = R.shape[0]
    OmegaAbs =  W + np.asarray([Omega]*n)

    G = []
    for i in range(0, n):
        tmp = np.cross(R[i,:], U[i,:])
        tmp = tmp / np.linalg.norm(tmp)
        G.append(tmp)
    G = np.asarray(G)

    d2Rdt2 = []
    for i in range(0, n):
        tmp = np.cross(OmegaAbs[i,:], R[i,:])
        tmp = np.cross(OmegaAbs[i,:], tmp)
        tmp = tmp + dOmegadt + E[i,:]

        accel  = np.dot(tmp, G[i,:])*G[i,:]

        t_vec = R[i,:] - U[i,:] 
        t_vec = t_vec / np.linalg.norm(t_vec)

        accel += -np.dot(dRdt[i,:], dRdt[i,:]) * t_vec
        d2Rdt2.append(accel)
    return d2Rdt2

def get_F(d2Rdt2, g, ball_mass, masses):
    d2Rdt2 = np.asarray(d2Rdt2)
    n = d2Rdt2.shape[0]

    d2Rdt2 += np.asarray([[0.0, 0.0, -g]] * n)
    d2Rdt2 = np.dot(d2Rdt2, np.diag(masses))

    ball_gravity = ball_mass * [0.0, 0.0, -g]
    F = np.append(d2Rdt2, [ball_gravity], axis=0)
    return F

def get_Ms(R, F, radious, ball_center):
    n = R.shape[0]
    Rs = R + np.asarray([[0, 0, -radious]]*n)

    Ms = []
    for i in range(0, n):
        tmp = np.cross(R[i,:], F[i,:])

    sc_vec = [0, 0, -radious]
    Ms.append(np.cross(sc_vec, F[-1, :])) # Gravity made no torque

    return Ms


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

