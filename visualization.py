import sys
import string
import pygame as pg 
from pygame.locals import *

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

import numpy as np



def rotate_vec(omega, phi): # Rotate around omega matrix
    R = np.eye(3)
    omega = omega / np.linalg.norm(omega)

    W = np.zeros((3,3))
    W[0][1] = -omega[2]
    W[0][2] =  omega[1]
    W[1][2] = -omega[0]

    W[1][0] = -W[0][1]
    W[2][0] = -W[0][2]
    W[2][1] = -W[1][2]

    R = R + np.sin(phi)*W + (1.0-np.cos(phi))*np.dot(W,W)
    return R

phi_start = np.zeros(4)
m_ball = 1.0
tetra_len = 1.0
r_ball = tetra_len * np.sqrt(3.0/8.0)

relative_system = np.eye(3)
absolute_system = np.eye(3)

g_abs = absolute_system[2, :] * 9.8

J_ball = np.eye(3)*( 2.0/5.0 * m_ball**2)
J_ball[0][0] += m_ball*r_ball**2 / 4.0
J_ball[1][1] += m_ball*r_ball**2 / 4.0

alpha = np.arccos(-1.0/3.0)
a1 = relative_system[2,:] / np.linalg.norm(relative_system[2,:]) * np.sqrt(7.0/24.0)
a2 = np.dot(rotate_vec(relative_system[0,:], alpha), a1)
a3 = np.dot(rotate_vec(relative_system[2,:], 2.0/3.0*np.pi), a2)
a4 = np.dot(rotate_vec(relative_system[2,:], 2.0/3.0*np.pi), a3)

A = np.array([a1, a2, a3, a4])

b1 = np.cross(A[0,:], A[1,:])
b2 = np.cross(A[1,:], A[2,:])
b3 = np.cross(A[2,:], A[3,:])
b4 = np.cross(A[3,:], A[0,:])

b1 = b1 / np.linalg.norm(b1) * tetra_len / np.sqrt(3)
b2 = b2 / np.linalg.norm(b2) * tetra_len / np.sqrt(3)
b3 = b3 / np.linalg.norm(b3) * tetra_len / np.sqrt(3)
b4 = b4 / np.linalg.norm(b4) * tetra_len / np.sqrt(3)

b1 = np.dot(rotate_vec(A[0,:], phi_start[0]), b1)
b2 = np.dot(rotate_vec(A[1,:], phi_start[1]), b2)
b3 = np.dot(rotate_vec(A[2,:], phi_start[2]), b3)
b4 = np.dot(rotate_vec(A[3,:], phi_start[3]), b4)


def drawSphere():
    glutSolidSphere(1.0,200,200)

def drawCircle(radius_vec, omega ,center):
    #glPushMatrix()

    glBegin(GL_LINES)
    c_last = radius_vec
    for i in xrange(0, 100):
        c = np.dot(rotate_vec(omega, 2.0*np.pi*i/(99)), radius_vec)
        glVertex3fv(c_last+center)
        glVertex3fv(c+center)
        c_last = c

    glEnd()
    #glPopMatrix()

def drawLines():
    global a1,a2,a3,a4
    global b1,b2,b3,b4

    glBegin(GL_LINES)
    glVertex3fv(np.array([0, 0, 0]))
    glVertex3fv(a1)

    glVertex3fv(np.array([0, 0, 0]))
    glVertex3fv(a2)

    glVertex3fv(np.array([0, 0, 0]))
    glVertex3fv(a3)

    glVertex3fv(np.array([0, 0, 0]))
    glVertex3fv(a4)

    glVertex3fv(a1)
    glVertex3fv(a1 + b1)

    glVertex3fv(a2)
    glVertex3fv(a2 + b2)

    glVertex3fv(a3)
    glVertex3fv(a3 + b3)

    glVertex3fv(a4)
    glVertex3fv(a4 + b4)

    glEnd()

    drawCircle(b1, a1, a1)
    drawCircle(b2, a2, a2)
    drawCircle(b3, a3, a3)
    drawCircle(b4, a4, a4)

def main():
    pg.init()
    glutInit([])
    display = (800, 600)
    pg.display.set_mode(display, DOUBLEBUF|OPENGL)

    gluPerspective(45, display[0]/display[1], 0.1, 50.0)
    glTranslatef(0.0, 0.0, -5)
    glRotatef(0, 0, 0 ,0)

    while True:
        for event in pg.event.get():
            if event == pg.QUIT: 
                pg.quit()
                quit()

            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    pg.quit()
                    quit()

        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        glRotatef(1,3,1,1)
        #glEnable(GL_LIGHTING)
        drawLines()

        pg.display.flip()
        pg.time.wait(30)

if __name__ == "__main__":
    main()