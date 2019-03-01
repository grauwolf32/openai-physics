import numpy as np 

class HyroSphere(object):
    def __init__(self, t_len, mass, dot_masses, position, phi_start=np.zeros(4), omega_start=np.zeros(4), ksi_start=np.zeros(4), Omega_start=np.zeros(3)):
        self.radius = t_len * np.sqrt(3.0/8.0)
        self.t_len = t_len
        self.relative_system = np.eye(3) 
        
        self.mass = mass
        self.dot_masses = np.asarray(dot_masses)
        self.position = np.asarray(position) # Position of center of the ball
        self.velocity = np.zeros(3)
        self.phi = np.asarray(phi_start)     # absolute values of angular speed
        
        self.ksi = np.asarray(ksi_start)     # absolute values of angular acceleration
        self.omega = np.asarray(omega_start) # and angular velocity for dot masses
        self.Omega = np.asarray(Omega_start)
        self.dOmegadt = np.zeros(3)

        u1 = self.relative_system[2,:] / np.linalg.norm(self.relative_system[2,:]) 
        u2 = np.dot(rotate_vec(self.relative_system[0,:], np.arccos(-1.0/3.0)), u1)
        u3 = np.dot(rotate_vec(self.relative_system[2,:], 2.0/3.0*np.pi), u2)
        u4 = np.dot(rotate_vec(self.relative_system[2,:], 2.0/3.0*np.pi), u3)

        u1 = u1 / np.linalg.norm(u1)
        u2 = u2 / np.linalg.norm(u2)
        u3 = u3 / np.linalg.norm(u3)
        u4 = u4 / np.linalg.norm(u4)

        self.U = np.array([u1, u2, u3, u4])

        self.J_ball = np.eye(3)*( 2.0/5.0 * self.mass**2)
        self.J_ball[1][1] += self.mass*self.radius**2  # OS has coordinates (0, 0, -r)

    def move(self, dt, ksi_new):
        A = self.U * np.sqrt(1.0/8.0)

        b1 = np.cross(A[0,:], A[1,:])
        b2 = np.cross(A[1,:], A[2,:])
        b3 = np.cross(A[2,:], A[3,:])
        b4 = np.cross(A[3,:], A[0,:])

        b1 = b1 / np.linalg.norm(b1) * self.t_len / 2.0
        b2 = b2 / np.linalg.norm(b2) * self.t_len / 2.0
        b3 = b3 / np.linalg.norm(b3) * self.t_len / 2.0
        b4 = b4 / np.linalg.norm(b4) * self.t_len / 2.0

        b1 = np.dot(rotate_vec(self.U[0,:], self.phi[0]), b1)
        b2 = np.dot(rotate_vec(self.U[1,:], self.phi[1]), b2)
        b3 = np.dot(rotate_vec(self.U[2,:], self.phi[2]), b3)
        b4 = np.dot(rotate_vec(self.U[3,:], self.phi[3]), b4)

        B = np.array([b1, b2, b3, b4])
        R = A + B 

        J = get_Js(self.J_ball, R, self.dot_masses)
        
        dRdt, d2Rdt2 = get_dR(R, self.U, self.omega,self.ksi, self.Omega, self.dOmegadt)
        dJdt = get_dJdt(R, dRdt,self.Omega, self.radius,self.mass, self.dot_masses)

        total_mass  = np.sum(self.dot_masses) + self.mass
        mass_center = np.sum(np.dot(np.diag(self.dot_masses), R), axis=0) / total_mass
        plane_normal = np.asarray([0, 0, 1.0])

        F, F_all = get_F(self.velocity, self.Omega, self.radius, d2Rdt2, 9.8, 0.01, self.mass, self.dot_masses, mass_center, plane_normal)
        Ms =  get_Ms(R, F, self.radius, self.position)
        Ms_all = np.sum(Ms, axis=0)

        self.dOmegadt = np.dot(np.linalg.inv(J), (Ms_all + np.dot(dJdt, self.Omega)))
        
        # Get acceleration of center of the ball from center of mass acceleration
        dvmdt = np.cross(self.dOmegadt, mass_center) 
        dvcdt = F_all/total_mass - np.cross(self.Omega, dvmdt)
        dvcdt_proj = np.dot(dvcdt, plane_normal)

        if dvcdt_proj < 0.0:
            dvcdt = dvcdt - dvcdt_proj*plane_normal

        self.velocity += dvcdt * dt
        vc_proj = np.dot(self.velocity, plane_normal)

        if vc_proj < 0.0 and self.position[-1] <= self.radius + 10e-3:
            self.velocity -= vc_proj*plane_normal

        self.position += self.velocity * dt
        self.omega += self.ksi
        self.ksi = ksi_new

        self.phi = self.phi + self.omega * dt
        for i in range(0, self.phi.shape[0]):
            if self.phi[i] >= 2.0*np.pi:
                self.phi[i] -= 2.0*np.pi
            elif self.phi[i] <= -2.0*np.pi:
                self.phi[i] += 2.0*np.pi

        self.Omega += self.dOmegadt * dt
        self.Omega *= 0.999 # friction loss

        self.U = get_U(self.U, self.Omega, dt)

        return mass_center, F, R, dRdt 

def get_dR(R, U, omega, ksi, Omega,dOmegadt):
    n = R.shape[0]
    dRdt   = []
    d2Rdt2 = []

    for i in range(0, n):
        omega_abs = U[i,:]*omega[i] + Omega
        veloc  = np.cross(omega_abs, R[i,:])
        dRdt.append(veloc)

        accel  = np.cross(omega_abs, veloc) 
        accel += np.cross(dOmegadt + ksi[i]*U[i,:], R[i,:])
        d2Rdt2.append(accel)
    
    d2Rdt2 = np.asarray(d2Rdt2)
    dRdt = np.asarray(dRdt)
    return dRdt, d2Rdt2


def get_Js(J_ball, R, dot_masses):
    dot_masses = np.asarray(dot_masses)

    n = dot_masses.shape[0]
    Js = J_ball

    for i in range(0, n):
       Js += (np.dot(R[i],R[i])*np.eye(3) - np.outer(R[i],R[i])) * dot_masses[i]

    return Js

def get_dJdt(R, dRdt, Omega, r, mass, masses):
    n = R.shape[0]
    m = R.shape[1]

    os_vec = np.asarray([0.0, 0.0, -r])
    dosdt  = np.cross(Omega, os_vec)

    dJdt = 2.0*np.diag([mass]*m)*np.diag(os_vec)*np.diag(dosdt)

    for i in range(0, n):
        dJdt -= (np.outer(R[i,:], dRdt[i,:]) + np.outer(dRdt[i,:], R[i,:])) * masses[i]

    return dJdt

def get_F(velocity, Omega, radius, d2Rdt2, g, mu, ball_mass, masses, mass_center, plane_normal):
    d2Rdt2 = np.asarray(d2Rdt2)
    n = d2Rdt2.shape[0]

    d2Rdt2 += np.asarray([[0.0, 0.0, -g]] * n)
    d2Rdt2 = np.dot(np.diag(masses),d2Rdt2)

    ball_gravity = ball_mass * np.asarray([0.0, 0.0, -g])
    F = np.append(d2Rdt2, [ball_gravity], axis=0)
    F_all = np.sum(F, axis=0)
    
    f_proj = np.dot(F_all, plane_normal)
    if f_proj < 0.0:
        N_abs = -f_proj
    else:
        N_abs = 0.0 

    vs = velocity - np.cross(Omega, np.asarray([0, 0, radius])) # vs = vc - OmegaxSO
    vs_proj = np.dot(vs, plane_normal)
    if vs_proj < 0.0:
        vs -= vs_proj*plane_normal
    vs_norm = np.linalg.norm(vs)
    if vs_norm > 10e-4:
        vs = -vs / vs_norm

    F_fric = mu * N_abs * vs
    F_all += F_fric

    F = np.append(F, [F_fric], axis=0)

    
    return F, F_all

def get_Ms(R, F, radius, ball_center):
    # Gravity force of the ball made no moment aganist tangent point
    # Friction force made no moment aganist tangent point

    n = R.shape[0]
    Rs = R + np.asarray([[0, 0, -radius]]*n)

    Ms = []
    for i in range(0, n):
        tmp = np.cross(Rs[i,:], F[i,:])
        Ms.append(tmp)

    Ms = np.asarray(Ms)

    return Ms

def get_U(U, Omega, dt):
    M = rotate_vec(Omega, np.linalg.norm(Omega)*dt)
    M = np.transpose(M)
    U = np.dot(U, M)

    return U

def rotate_vec(omega, phi): # Rotate around omega matrix
    R = np.eye(3)
    omega_norm = np.linalg.norm(omega)
    if omega_norm < 1e-4:
        return R

    omega = omega / omega_norm

    W = np.zeros((3,3))
    W[0][1] = -omega[2]
    W[0][2] =  omega[1]
    W[1][2] = -omega[0]

    W[1][0] = -W[0][1]
    W[2][0] = -W[0][2]
    W[2][1] = -W[1][2]

    R = R + np.sin(phi)*W + (1.0-np.cos(phi))*np.dot(W,W)
    return R