import sys
import string
import pygame as pg 
import camera

from pygame.locals import *

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

import numpy as np

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
a1 = relative_system[2,:] / np.linalg.norm(relative_system[2,:]) * np.sqrt(1.0/8.0)
a2 = np.dot(rotate_vec(relative_system[0,:], alpha), a1)
a3 = np.dot(rotate_vec(relative_system[2,:], 2.0/3.0*np.pi), a2)
a4 = np.dot(rotate_vec(relative_system[2,:], 2.0/3.0*np.pi), a3)

A = np.array([a1, a2, a3, a4])

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


def drawSphere(center, radious, colors):
    glPushMatrix()
    glTranslatef(center[0], center[1], center[2])
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glColor4f(colors[0],colors[1],colors[2],colors[3])
    glutSolidSphere(radious,200,200)
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

def drawShape():
    global a1,a2,a3,a4
    global b1,b2,b3,b4

    zeros = [0,0,0]
    drawLines(start=[zeros,zeros,zeros,zeros], end=[a1,a2,a3,a4], color=(1,0,0,0.5), position=(0,0,0,0))
    drawLines(start=[a1,a2,a3,a4], end=[a1+b1,a2+b2,a3+b3,a4+b4], color=(1,0,0,0.5), position=(0,0,0,0))

    drawCircle(b1, a1, a1, color=(0,0,0.7,0.3))
    drawCircle(b2, a2, a2, color=(0,0,0.7,0.3))
    drawCircle(b3, a3, a3, color=(0,0,0.7,0.3))
    drawCircle(b4, a4, a4, color=(0,0,0.7,0.3))

    drawSphere(center=(0,0,0), radious=tetra_len*np.sqrt(3.0/8.0), colors=(90.0/256, 1.0, 39.0/256, 0.3))
    drawSphere(center=a1 + b1, radious=np.sqrt(3.0/8.0)/20, colors=(0,0,0, 0.2))
    drawSphere(center=a2 + b2, radious=np.sqrt(3.0/8.0)/20, colors=(0,0,0, 0.2))
    drawSphere(center=a3 + b3, radious=np.sqrt(3.0/8.0)/20, colors=(0,0,0, 0.2))
    drawSphere(center=a4 + b4, radious=np.sqrt(3.0/8.0)/20, colors=(0,0,0, 0.2))

def main():
    global a1,a2,a3,a4
    global b1,b2,b3,b4

    pg.init()
    glutInit([])
    display = (800, 600)
    pg.display.set_mode(display, DOUBLEBUF|OPENGL)

    gluPerspective(45, display[0]/display[1], 0.1, 30.0)
    glTranslatef(0.0, 0.0, -5)
    glRotatef(0, 0, 0 ,0)

    cam = camera.LookAtCamera(rotation=[0,0,0], distance=1.0)
    u1 = a1 / np.linalg.norm(a1)

    while True:
        b1 += 0.05*np.cross(a1, b1)
        b1 = (b1 / np.linalg.norm(b1)) *  tetra_len / 2.0

        for event in pg.event.get():
            if event == pg.QUIT: 
                pg.quit()
                quit()

            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    pg.quit()
                    quit()
            
        keys = pg.key.get_pressed()

        if keys[K_UP]:
            cam.distance -= 0.05
        if keys[K_DOWN]:
            cam.distance += 0.05

        if keys[K_LEFT]:
            cam.roty -= 0.5
        if keys[K_RIGHT]:
            cam.roty += 0.5

        if keys[K_w]:
            cam.rotx += 0.5
        if keys[K_s]:
            cam.rotx -= 0.5

        if keys[K_q]:
            cam.rotz += 0.5
        if keys[K_a]:
            cam.rotz -= 0.5
        
        glClearColor(1.0, 1.0, 1.0, 1.0) 
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        
        #glEnable(GL_LIGHTING)
        cam.push()
        drawShape()
        pg.display.flip()
        cam.pop()
        pg.time.wait(20)

if __name__ == "__main__":
    main()