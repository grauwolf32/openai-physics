import numpy as np 

class HyroSphere(object):
    def __init__(self, t_len, mass, dot_masses, position, phi=np.zeros(4), omega=np.zeros(4), 
                 ksi=np.zeros(4), Omega=np.zeros(3), dOmegadt=np.zeros(3), velocity=np.zeros(3), mu=0.001):
        self.radius = t_len * np.sqrt(3.0/8.0)
        self.t_len = t_len
        self.relative_system = np.eye(3) 
        
        self.mass = mass
        self.dot_masses = np.asarray(dot_masses)
        self.position = np.asarray(position) # Position of center of the ball
        self.velocity = velocity
        self.phi = np.asarray(phi)     # absolute values of angular speed
        
        self.ksi = np.asarray(ksi)     # absolute values of angular acceleration
        self.omega = np.asarray(omega) # and angular velocity for dot masses
        self.Omega = np.asarray(Omega)
        self.dOmegadt = dOmegadt

        self.mu = mu

        u1 = self.relative_system[2,:] / np.linalg.norm(self.relative_system[2,:]) 
        u2 = np.dot(rotate_vec(self.relative_system[0,:], np.arccos(-1.0/3.0)), u1)
        u3 = np.dot(rotate_vec(self.relative_system[2,:], 2.0/3.0*np.pi), u2)
        u4 = np.dot(rotate_vec(self.relative_system[2,:], 2.0/3.0*np.pi), u3)

        u1 = u1 / np.linalg.norm(u1)
        u2 = u2 / np.linalg.norm(u2)
        u3 = u3 / np.linalg.norm(u3)
        u4 = u4 / np.linalg.norm(u4)

        self.U = np.array([u1, u2, u3, u4])

        self.J_ball = np.eye(3)*( 2.0/3.0 * self.mass**2)
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
        
        n = R.shape[0]
        m = R.shape[1]

        plane_normal = np.asarray([0, 0, 1.0])
        os_vec = np.asarray([0.0, 0.0, -self.radius])
        g_vec  = np.asarray([0.0, 0.0, -9.8])

        # Calculate J
        J = self.J_ball

        for i in range(0, n):
            J += (np.dot(R[i],R[i])*np.eye(3) - np.outer(R[i],R[i])) * self.dot_masses[i]

        # ----------------------------------------------
        # Calculate dRdt and d2Rdt2
        dRdt = []
        d2Rdt2 = []

        for i in range(0, n):
            omega_abs = self.U[i,:]*self.omega[i] + self.Omega
            veloc  = np.cross(omega_abs, R[i,:])
            dRdt.append(veloc)

            accel  = np.cross(omega_abs, veloc) 
            accel += np.cross(self.dOmegadt + self.ksi[i]*self.U[i,:], R[i,:])
            d2Rdt2.append(accel)
    
        d2Rdt2 = np.asarray(d2Rdt2)
        dRdt = np.asarray(dRdt)

        # ----------------------------------------------
        # Calculate dJdt
        dosdt  = np.cross(self.Omega, os_vec)
        dJdt = 2.0*np.diag([self.mass]*m)*np.diag(os_vec)*np.diag(dosdt)

        for i in range(0, n):
            dJdt -= (np.outer(R[i,:], dRdt[i,:]) + np.outer(dRdt[i,:], R[i,:])) * self.dot_masses[i]
        # ----------------------------------------------
        Mass = np.diag(self.dot_masses)
        total_mass  = np.sum(self.dot_masses) + self.mass
        mass_center = np.sum(np.dot(Mass, R), axis=0) / total_mass

        # Calculate Forces
        F  = np.dot(Mass, d2Rdt2 + [g_vec] * n)
        F = np.append(F, [self.mass * g_vec], axis=0)
        F_all = np.sum(F, axis=0)
    
        f_proj = np.dot(F_all, plane_normal)
        if f_proj < 0.0:
            N_abs = -f_proj
        else:
            N_abs = 0.0 

        vs = self.velocity - np.cross(self.Omega, -os_vec) # vs = vc - OmegaxSO
        vs_proj = np.dot(vs, plane_normal)
        if vs_proj < 0.0:
            vs -= vs_proj*plane_normal

        vs_norm = np.linalg.norm(vs)
        if vs_norm > 10e-4:
            vs = -vs / vs_norm

        F_fric = self.mu * N_abs * vs
        F_all += F_fric

        F = np.append(F, [F_fric], axis=0)
        # -----------------------------------------------
        # Calculate Ms
        Ms = []
        for i in range(0, n):
            tmp = np.cross(R[i,:] - os_vec, F[i,:])
            Ms.append(tmp)

        Ms = np.asarray(Ms)
        # -----------------------------------------------
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

        M = rotate_vec(self.Omega, np.linalg.norm(self.Omega)*dt)
        M = np.transpose(M)
        self.U = np.dot(self.U, M)

        return mass_center, F, R, dRdt 

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