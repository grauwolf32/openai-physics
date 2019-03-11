import gym
import numpy as np

from gym.utils import seeding
from gym import error, spaces, utils

import pygame as pg 

from pygame.locals import *

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

from gym_hyrosphere.envs.visualization import *
from gym_hyrosphere.envs.physics import *

class EnvSpec(object):
    def __init__(self, timestep_limit, id):
        self.timestep_limit = timestep_limit
        self.id = id

class LinearspherePhysicsConstantsClass(object):
    metadata = {'render.modes': ['human']}
    
    def __init__(self):
        self.dt = 0.01
        state_len = 63
        n_objects = 6
        max_ksi   = 10
        max_time  = 10
        max_value = 1000

        self.max_speed = 100.0
        self.friction_loss = 0.001
        self.mu = 0.01

        timestep_limit = int(max_time/self.dt)
        self.spec = EnvSpec(timestep_limit = timestep_limit , id=1)

        ob_low  = np.asarray([-max_value]*state_len)
        ob_high = np.asarray([-max_value]*state_len)
        
        self.ac_space = spaces.Box(low=np.asarray([-max_ksi]*n_objects), high=np.array([max_ksi]*n_objects), dtype=np.float64)
        self.ob_space = spaces.Box(low=ob_low, high=ob_high, dtype=np.float64)

LinearspherePhysicsConstants = LinearspherePhysicsConstantsClass()

class LinearspherePhysicsEnv(gym.Env):
    def __init__(self):
        self.linearsphere = LinearSphere(radius=1.0, mass=8, dot_masses=np.ones(6),shifts=np.zeros(6),
                                    position = np.zeros(3), speeds=np.zeros(6), 
                                    accelerations=np.zeros(6), Omega=np.zeros(3), dOmegadt=np.zeros(3),
                                    velocity=np.zeros(3), mu=LinearspherePhysicsConstants.mu,
                                    max_speed=LinearspherePhysicsConstants.max_speed,
                                    friction_loss=LinearspherePhysicsConstants.friction_loss)
        self.total_time = 0.0
        self.action_space = LinearspherePhysicsConstants.ac_space
        self.observation_space = LinearspherePhysicsConstants.ob_space
        self.spec = LinearspherePhysicsConstants.spec
        self.render_init = False
        self.seed()
        

    def _init_renderer(self):
        pg.init()
            
        self.display = (800, 600)
        self.cam = LookAtCamera(rotation=[90,0,0], distance=1.0)
        self.surface = pg.display.set_mode(self.display, DOUBLEBUF|OPENGL|OPENGLBLIT)
        self.font = pg.font.SysFont("Times New Roman", 12)

        glutInit([])
        gluPerspective(45, self.display[0]/self.display[1], 0.1, 30.0)
        glTranslatef(0.0, 0.0, -2)
        self.render_init = True
        

    def step(self, action):
        accel = action
        dt = LinearspherePhysicsConstants.dt
        R, dRdt = self.linearsphere.move(dt, accel_new=accel)

        ob = np.concatenate((self.linearsphere.velocity, self.linearsphere.speeds,\
              self.linearsphere.accelerations, self.linearsphere.shifts, self.linearsphere.Omega, \
              self.linearsphere.dOmegadt,\
              R[0],R[1],R[2], R[3],R[4],R[5],\
              dRdt[0], dRdt[1], dRdt[2], dRdt[3], dRdt[4], dRdt[5]))
        ob = np.asarray(ob)

        reward =  100*np.dot(self.linearsphere.velocity, np.asarray([0.0,0.0,1.0]))
        reward += np.abs(np.dot(self.linearsphere.velocity, self.linearsphere.velocity)) 
        self.total_time += dt
        
        done = False
        if self.total_time > 10.0:
            done = True

        info = dict()

        return ob, reward, done, info

    def reset(self):
        self.linearsphere = LinearSphere(radius=1.0, mass=8, dot_masses=np.ones(6),shifts=np.zeros(6),
                                    position = np.zeros(3), speeds=np.zeros(6), 
                                    accelerations=np.zeros(6), Omega=np.zeros(3), dOmegadt=np.zeros(3),
                                    velocity=np.zeros(3), mu=LinearspherePhysicsConstants.mu,
                                    max_speed=LinearspherePhysicsConstants.max_speed,
                                    friction_loss=LinearspherePhysicsConstants.friction_loss)
        self.total_time = 0.0

        R = np.dot(np.diag(self.linearsphere.shifts), self.linearsphere.U)
        dRdt = np.zeros((6,3))
        
        ob = np.concatenate((self.linearsphere.velocity, self.linearsphere.speeds,\
              self.linearsphere.accelerations, self.linearsphere.shifts, self.linearsphere.Omega, \
              self.linearsphere.dOmegadt,\
              R[0],R[1],R[2], R[3],R[4],R[5],\
              dRdt[0], dRdt[1], dRdt[2], dRdt[3], dRdt[4], dRdt[5]))
        ob = np.asarray(ob)

        return ob
    
    def render(self, mode='human', close='False'):
        if not self.render_init:
            self._init_renderer()
            
        glClearColor(1.0, 1.0, 1.0, 1.0) 
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        
        self.cam.push()
        drawLinearsphere(self.linearsphere)
        pg.display.flip()

        self.cam.pop()


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def close(self):
        pass

