import sys
import string
import pygame as pg 
import numpy as np

from math import sqrt
from pygame.locals import *

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

from gym_hyrosphere.envs.physics import rotate_vec


def drawText(position, textString, font_size=32):     
    font = pg.font.Font (None, font_size)
    textSurface = font.render(textString, True, (255,255,255,255), (0,0,0,255))     
    textData = pg.image.tostring(textSurface, "RGBA", True)     
    glRasterPos3d(*position)     
    glDrawPixels(textSurface.get_width(), textSurface.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, textData)

def drawSphere(center, radius, colors):
    glPushMatrix()
    glTranslatef(center[0], center[1], center[2])
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glColor4f(colors[0],colors[1],colors[2],colors[3])
    glutSolidSphere(radius,50,50)
    glDisable(GL_BLEND)
    glPopMatrix()

def drawCircle(radius_vec, omega ,center, color):
    glPushMatrix()
    glTranslatef(center[0], center[1], center[2])
    glColor4f(color[0],color[1],color[2],color[3])

    glBegin(GL_LINES)
    c_last = radius_vec
    for i in range(0, 100):
        c = np.dot(rotate_vec(omega, 2.0*np.pi*i/(99)), radius_vec)
        glVertex3fv(c_last)
        glVertex3fv(c)
        c_last = c

    glEnd()
    glPopMatrix()


def drawCylinder(start, end, radius, color, close=True, nseg=20, mseg=40):
    
    if np.linalg.norm(start) > 10e-4:
        d = (end-start)
        p = np.cross(start, d)
    else:
        d = end
        p = np.cross([1.0]*3, d)

    p = p / np.linalg.norm(p)
    p = p * radius

    points = []
    curr_point = p
    alpha = 2.0*np.pi / nseg

    for i in range(0, nseg):
        points.append(curr_point)
        curr_point = np.dot(rotate_vec(d, alpha), curr_point)

    layers = []
    curr_pos = start
    step = d / (mseg - 1) 

    for i in range(0, mseg):
        layer = [point + start + i*step for point in points]
        layers.append(layer)

    quads = []
    for i in range(0, mseg-1):
        for j in range(0, nseg-1):
            quads.append([layers[i][j], layers[i+1][j], layers[i+1][j+1], layers[i][j+1]])
        
        join_layer = [layers[mseg-1][0], layers[mseg-1][nseg-1], layers[0][nseg-1], layers[0][0]]
        quads.append(join_layer)
    
    glPushMatrix()
    glBegin(GL_QUADS)
    glColor4f(color[0],color[1],color[2],color[3])

    for quad in quads:
        glVertex3fv(quad[0])
        glVertex3fv(quad[1])
        glVertex3fv(quad[2])
        glVertex3fv(quad[3])

    glEnd()

    if close:
        glBegin(GL_POLYGON)
        for point in layers[0]:
            glVertex3fv(point)
        glEnd()

        glBegin(GL_POLYGON)
        for point in layers[mseg-1]:
            glVertex3fv(point)
        glEnd()

    glPopMatrix()
        
def drawCylinders(start, end, radius, color, close=True, nseg=10, mseg=10):
    n = np.asarray(start).shape[0]
    for i in range(0, n):
        drawCylinder(start[i], end[i], radius, color, close, nseg, mseg)

def drawCone(start, end, radius, color, close, nseg=20):
    if np.linalg.norm(start) > 10e-4:
        d = (end-start)
        p = np.cross(start, d)
    else:
        d = end
        p = np.cross([1.0]*3, d)

    p = p / np.linalg.norm(p)
    p = p * radius

    points = []
    curr_point = p
    alpha = 2.0*np.pi / nseg

    for i in range(0, nseg):
        points.append(curr_point + start)
        curr_point = np.dot(rotate_vec(d, alpha), curr_point) 

    glPushMatrix()
    glBegin(GL_TRIANGLES)
    glColor4f(color[0],color[1],color[2],color[3])

    for i in range(0, nseg-1):
        glVertex3fv(points[i])
        glVertex3fv(end)
        glVertex3fv(points[i+1])

    glVertex3fv(points[nseg-1])
    glVertex3fv(end)
    glVertex3fv(points[0])

    glEnd()

    if close:
        glBegin(GL_POLYGON)
        for point in points:
            glVertex3fv(point)
        glEnd()

    glPopMatrix()

def drawArrow(start, end, radius, color, nseg=20, mseg=20):
    d = end - start
    h = 0.7*d

    drawCylinder(start, start+h, radius, color,close=True, nseg=nseg, mseg=mseg)
    drawCone(start+h, end, radius=radius*1.4, color=color, close=True, nseg=nseg )

def drawArrows(start, end, radius, color, nseg=20, mseg=20):
    n = np.asarray(start).shape[0]
    for i in range(0, n):
        drawArrow(start[i], end[i], radius, color, nseg, mseg)

def drawLines(start, end, color, position):
    glPushMatrix()
    glTranslatef(position[0], position[1], position[2])
    glColor4f(color[0],color[1],color[2],color[3])
    glBegin(GL_LINES)
    for i in range(0, len(start)):
        glVertex3fv(start[i])
        glVertex3fv(end[i])

    glEnd()
    glPopMatrix()

def drawHyrosphere(hyrosphere):
    #glTranslatef(hyrosphere.position[0],hyrosphere.position[1], hyrosphere.position[2])

    U = hyrosphere.U 
    A = U * np.sqrt(1.0/8.0)

    b1 = np.cross(A[0,:], A[1,:])
    b2 = np.cross(A[1,:], A[2,:])
    b3 = np.cross(A[2,:], A[3,:])
    b4 = np.cross(A[3,:], A[0,:])

    b1 = b1 / np.linalg.norm(b1) * hyrosphere.t_len / 2.0
    b2 = b2 / np.linalg.norm(b2) * hyrosphere.t_len / 2.0
    b3 = b3 / np.linalg.norm(b3) * hyrosphere.t_len / 2.0
    b4 = b4 / np.linalg.norm(b4) * hyrosphere.t_len / 2.0

    b1 = np.dot(rotate_vec(U[0,:], hyrosphere.phi[0]), b1)
    b2 = np.dot(rotate_vec(U[1,:], hyrosphere.phi[1]), b2)
    b3 = np.dot(rotate_vec(U[2,:], hyrosphere.phi[2]), b3)
    b4 = np.dot(rotate_vec(U[3,:], hyrosphere.phi[3]), b4)

    B = np.array([b1, b2, b3, b4])
    R = A + B 

    zeros = np.zeros(3)
    drawLines(start=[zeros,zeros,zeros, zeros], end=A, color=(1,0,0,0.5), position=(0,0,0,0))
    drawLines(start=A, end=R, color=(1,0,0,0.5), position=(0,0,0,0))

    #drawCylinders([zeros,zeros,zeros, zeros],A,radius=0.02, color=(1,0,0,0.5))
    #drawCylinders(A,R,radius=0.02, color=(1,0,0,0.5))
    drawArrow(np.zeros(3), hyrosphere.velocity,radius=0.02, color=(1,0,0,0.5) )

    drawCircle(B[0], A[0], A[0], color=(0,0,0.7,0.3))
    drawCircle(B[1], A[1], A[1], color=(0,0,0.7,0.3))
    drawCircle(B[2], A[2], A[2], color=(0,0,0.7,0.3))
    drawCircle(B[3], A[3], A[3], color=(0,0,0.7,0.3))

    drawSphere(center=(0,0,0), radius=hyrosphere.t_len*np.sqrt(3.0/8.0), colors=(90.0/256, 1.0, 39.0/256, 0.3))
    drawSphere(center=R[0], radius=hyrosphere.t_len*np.sqrt(3.0/8.0)/20, colors=(0,0,0, 0.2))
    drawSphere(center=R[1], radius=hyrosphere.t_len*np.sqrt(3.0/8.0)/20, colors=(0,0,0, 0.2))
    drawSphere(center=R[2], radius=hyrosphere.t_len*np.sqrt(3.0/8.0)/20, colors=(0,0,0, 0.2))
    drawSphere(center=R[3], radius=hyrosphere.t_len*np.sqrt(3.0/8.0)/20, colors=(0,0,0, 0.2))
 
    #os_vec = np.asarray([0.0,0.0,-hyrosphere.radius])
    #drawArrow(os_vec, os_vec + hyrosphere.Omega , radius=0.02, color=(1,0,0,0.5), nseg=10, mseg=10)

    text = "position : {0:.2f} {1:.2f} {2:.2f}".format(hyrosphere.position[0], hyrosphere.position[1], hyrosphere.position[2])
    drawText(position=(-1.0,0.0,1.0), textString=text)

    text = "velocity : {0:.2f} {1:.2f} {2:.2f}".format(hyrosphere.velocity[0], hyrosphere.velocity[1], hyrosphere.velocity[2])
    drawText(position=(-1.0,0.0,0.8), textString=text)

class CameraBase(object):
    """camera.Base camera object all other inherit from..."""
    def __init__(self, pos=[0,0,0], rotation=[0,0,0]):
        """create the camera
           pos = position of the camera
           rotation = rotation of camera"""
        self.posx, self.posy, self.posz = pos
        self.rotx, self.roty, self.rotz = rotation

    def push(self):
        """Activate the camera - anything rendered after this uses the cameras transformations."""
        glPushMatrix()

    def pop(self):
        """Deactivate the camera - must be called after push or will raise an OpenGL error"""
        glPopMatrix()

    def get_pos(self):
        """Return the position of the camera as a tuple"""
        return self.posx, self.posy, self.posz

    def set_pos(self, pos):
        """Set the position of the camera from a tuple"""
        self.posx, self.posy, self.posz = pos

    def get_rotation(self):
        """Return the rotation of the camera as a tuple"""
        return self.rotx, self.roty, self.rotz

    def set_facing_matrix(self):
        """Transforms the matrix so that all objects are facing camera - used in Image3D (billboard sprites)"""
        pass

    def set_skybox_data(self):
        """Transforms the view only for a skybox, ie only rotation is taken into account, not position"""
        pass

class LookFromCamera(CameraBase):
    """camera.LookFromCamera is a FPS camera"""
    def __init__(self, pos=(0,0,0), rotation=(0,0,0)):
        CameraBase.__init__(self, pos, rotation)

    def push(self):
        glPushMatrix()
        glRotatef(self.rotx, 1, 0, 0)
        glRotatef(self.roty, 0, 1, 0)
        glRotatef(self.rotz, 0, 0, 1)
        glTranslatef(-self.posx, -self.posy, self.posz)

    def pop(self):
        glPopMatrix()

    def get_pos(self):
        return self.posx, self.posy, self.posz

    def get_rotation(self):
        return self.rotx, self.roty, self.rotz

    def set_facing_matrix(self):
        glRotatef(-self.rotz, 0, 0, 1)
        glRotatef(-self.roty, 0, 1, 0)
        glRotatef(-self.rotx, 1, 0, 0)

    def set_skybox_data(self):
        glRotatef(self.rotx, 1, 0, 0)
        glRotatef(self.roty, 0, 1, 0)
        glRotatef(self.rotz, 0, 0, 1)

class LookAtCamera(CameraBase):
    """camera.LookAtCamera is a third-person camera"""
    def __init__(self, pos=[0,0,0], rotation=[0,0,0],
                 distance=0):
        """create the camera
           pos is the position the camera is looking at
           rotation is how much we are rotated around the object
           distance is how far back from the object we are"""
        CameraBase.__init__(self, pos, rotation)
        self.distance = distance

    def push(self):
        glPushMatrix()
        glTranslatef(0, 0, -self.distance)
        glRotatef(-self.rotx, 1, 0, 0)
        glRotatef(-self.roty, 0, 1, 0)
        glRotatef(self.rotz, 0, 0, 1)
        glTranslatef(-self.posx, -self.posy, self.posz)

    def set_facing_matrix(self):
        glRotatef(-self.rotz, 0, 0, 1)
        glRotatef(self.roty, 0, 1, 0)
        glRotatef(self.rotx, 1, 0, 0)

    def set_skybox_data(self):
        glRotatef(-self.rotx, 1, 0, 0)
        glRotatef(-self.roty, 0, 1, 0)
        glRotatef(self.rotz, 0, 0, 1)
