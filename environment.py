import gym
import physic_fast as physics
import numpy as np

from gym.utils import seeding
from gym import error, spaces, utils
from visualization import drawHyrosphere
from math import sqrt
from physic_fast import rotate_vec

import pygame as pg 
import camera

from pygame.locals import *

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *


class EnvSpec(object):
    def __init__(self, timestep_limit, id):
        self.timestep_limit = timestep_limit
        self.id = id

class HyrospherePhysicsConstantsClass(object):
    def __init__(self):
        self.dt = 0.01
        state_len = 41
        n_objects = 4
        max_value = 1000
        max_ksi   = 10
        max_time  = 10
        timestep_limit = int(max_time/self.dt)

        self.spec = EnvSpec(timestep_limit = timestep_limit , id=1)

        ob_low  = np.asarray([-max_value]*state_len)
        ob_high = np.asarray([-max_value]*state_len)
        
        self.ac_space = spaces.Box(low=np.asarray([-max_ksi]*n_objects), high=np.array([max_ksi]*n_objects))
        self.ob_space = spaces.Box(low=ob_low, high=ob_high)

HyrospherePhysicsConstants = HyrospherePhysicsConstantsClass()

class HyrospherePhysicsEnv(gym.Env):
    def __init__(self, visualization=False):
        self.hyrosphere = physics.HyroSphere(t_len=1.0, mass=4, dot_masses=np.asarray([1.0]*4),\
                                        position = np.zeros(3), phi=np.zeros(4), omega=np.zeros(4), 
                                        ksi=np.zeros(4), Omega=np.zeros(3), dOmegadt=np.zeros(3),
                                        velocity=np.zeros(3), mu=0.001)
        self.total_time = 0.0
        self.action_space = HyrospherePhysicsConstants.ac_space
        self.observation_space = HyrospherePhysicsConstants.ob_space
        self.spec = HyrospherePhysicsConstants.spec
        self.seed()

        if visualization:
            pg.init()
            
            self.display = (800, 600)
            self.cam = camera.LookAtCamera(rotation=[90,0,0], distance=1.0)
            self.surface = pg.display.set_mode(self.display, DOUBLEBUF|OPENGL|OPENGLBLIT)
            self.font = pg.font.SysFont("Times New Roman", 12)

            glutInit([])
            gluPerspective(45, self.display[0]/self.display[1], 0.1, 30.0)
            glTranslatef(0.0, 0.0, -2)
        

    def step(self, action):
        ksi = action
        dt = HyrospherePhysicsConstants.dt
        R, dRdt = self.hyrosphere.move(dt, ksi)

        ob = np.concatenate((self.hyrosphere.velocity, self.hyrosphere.omega,\
              self.hyrosphere.ksi, self.hyrosphere.Omega, \
              self.hyrosphere.dOmegadt,\
              R[0],R[1],R[2], R[3],\
              dRdt[0], dRdt[1], dRdt[2], dRdt[3]))
        ob = np.asarray(ob)

        reward = np.dot(self.hyrosphere.velocity, np.asarray([1.0,0.0,0.0])) 

        self.total_time += dt
        
        done = False
        if self.total_time > 10.0:
            done = True

        info = dict()

        return ob, reward, done, info

    def reset(self):
        self.hyrosphere = physics.HyroSphere(t_len=1.0, mass=4, dot_masses=np.asarray([1.0]*4),\
                                        position = np.zeros(3), phi=np.zeros(4), omega=np.zeros(4), 
                                        ksi=np.zeros(4), Omega=np.zeros(3), dOmegadt=np.zeros(3),
                                        velocity=np.zeros(3), mu=0.001)
        self.total_time = 0.0

        U = self.hyrosphere.U 
        A = U * np.sqrt(1.0/8.0)

        b1 = np.cross(A[0,:], A[1,:])
        b2 = np.cross(A[1,:], A[2,:])
        b3 = np.cross(A[2,:], A[3,:])
        b4 = np.cross(A[3,:], A[0,:])

        b1 = b1 / np.linalg.norm(b1) * self.hyrosphere.t_len / 2.0
        b2 = b2 / np.linalg.norm(b2) * self.hyrosphere.t_len / 2.0
        b3 = b3 / np.linalg.norm(b3) * self.hyrosphere.t_len / 2.0
        b4 = b4 / np.linalg.norm(b4) * self.hyrosphere.t_len / 2.0

        b1 = np.dot(rotate_vec(U[0,:], self.hyrosphere.phi[0]), b1)
        b2 = np.dot(rotate_vec(U[1,:], self.hyrosphere.phi[1]), b2)
        b3 = np.dot(rotate_vec(U[2,:], self.hyrosphere.phi[2]), b3)
        b4 = np.dot(rotate_vec(U[3,:], self.hyrosphere.phi[3]), b4)

        B = np.array([b1, b2, b3, b4])
        R = A + B

        dRdt = np.zeros((4,3))
        
        ob =  np.concatenate((self.hyrosphere.velocity, self.hyrosphere.omega,\
              self.hyrosphere.ksi, self.hyrosphere.Omega, \
              self.hyrosphere.dOmegadt,\
              R[0],R[1],R[2], R[3],\
              dRdt[0], dRdt[1], dRdt[2], dRdt[3]))
        ob = np.asarray(ob)

        return ob
    
    def render(self, mode='human', close='False'):
        glClearColor(1.0, 1.0, 1.0, 1.0) 
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        
        self.cam.push()
        text = "position : {0:.2f} {1:.2f} {2:.2f}".format(self.hyrosphere.position[0], self.hyrosphere.position[1], self.hyrosphere.position[2])
        drawText(position=(-1.0,0.0,1.0), textString=text)

        text = "velocity : {0:.2f} {1:.2f} {2:.2f}".format(self.hyrosphere.velocity[0], self.hyrosphere.velocity[1], self.hyrosphere.velocity[2])
        drawText(position=(-1.0,0.0,0.8), textString=text)

        drawHyrosphere(self.hyrosphere)
        pg.display.flip()

        self.cam.pop()


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def close(self):
        pass

def drawText(position, textString, font_size=32):     
    font = pg.font.Font (None, font_size)
    textSurface = font.render(textString, True, (255,255,255,255), (0,0,0,255))     
    textData = pg.image.tostring(textSurface, "RGBA", True)     
    glRasterPos3d(*position)     
    glDrawPixels(textSurface.get_width(), textSurface.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, textData)

if __name__ == "__main__":
    env = HyrospherePhysicsEnv()
    ob, reward, done, info = env.step(action=np.zeros(4))
    print(ob, ob.shape)

