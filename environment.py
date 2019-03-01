import gym
import physic_fast as physics

from gym.utils import seeding
from gym import error, spaces, utils
from visualization import drawHyrosphere

class HyrospherePhysicsEnv(gym.Env):
    def __init__(self, visualization=False):
        hypersphere = physics.HyroSphere(t_len=1.0, mass=4, dot_masses=np.asarray([1.0]*4),\
                                        position = np.ones(3), phi=np.zeros(4), omega=np.zeros(4), 
                                        ksi=np.zeros(4), Omega=np.zeros(3), dOmegadt=np.zeros(3),
                                        velocity=np.zeros(3), mu=0.001)
        

        