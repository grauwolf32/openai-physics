import sys
import string
import pygame as pg 
import camera

from pygame.locals import *

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

import numpy as np

from aux import rotate_vec
from physics import *

def drawSphere(center, radius, colors):
    glPushMatrix()
    glTranslatef(center[0], center[1], center[2])
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glColor4f(colors[0],colors[1],colors[2],colors[3])
    glutSolidSphere(radius,200,200)
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

def drawHyrosphere(hyrosphere):
    #U = hyrosphere.U 
    relative_system = hyrosphere.relative_system
    u1 = relative_system[2,:] / np.linalg.norm(relative_system[2,:]) 
    u2 = np.dot(rotate_vec(relative_system[0,:], np.arccos(-1.0/3.0)), u1)
    u3 = np.dot(rotate_vec(relative_system[2,:], 2.0/3.0*np.pi), u2)
    u4 = np.dot(rotate_vec(relative_system[2,:], 2.0/3.0*np.pi), u3)

    u1 = u1 / np.linalg.norm(u1)
    u2 = u2 / np.linalg.norm(u2)
    u3 = u3 / np.linalg.norm(u3)
    u4 = u4 / np.linalg.norm(u4)

    U = np.array([u1, u2, u3, u4])
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
    drawLines(start=[zeros,zeros,zeros,zeros], end=A, color=(1,0,0,0.5), position=(0,0,0,0))
    drawLines(start=A, end=R, color=(1,0,0,0.5), position=(0,0,0,0))

    drawCircle(B[0], A[0], A[0], color=(0,0,0.7,0.3))
    drawCircle(B[1], A[1], A[1], color=(0,0,0.7,0.3))
    drawCircle(B[2], A[2], A[2], color=(0,0,0.7,0.3))
    drawCircle(B[3], A[3], A[3], color=(0,0,0.7,0.3))

    drawSphere(center=(0,0,0), radius=hyrosphere.t_len*np.sqrt(3.0/8.0), colors=(90.0/256, 1.0, 39.0/256, 0.3))
    drawSphere(center=R[0], radius=hyrosphere.t_len*np.sqrt(3.0/8.0)/20, colors=(0,0,0, 0.2))
    drawSphere(center=R[1], radius=hyrosphere.t_len*np.sqrt(3.0/8.0)/20, colors=(0,0,0, 0.2))
    drawSphere(center=R[2], radius=hyrosphere.t_len*np.sqrt(3.0/8.0)/20, colors=(0,0,0, 0.2))
    drawSphere(center=R[3], radius=hyrosphere.t_len*np.sqrt(3.0/8.0)/20, colors=(0,0,0, 0.2))

def main():
    pg.init()
    glutInit([])
    display = (800, 600)

    pg.display.set_mode(display, DOUBLEBUF|OPENGL)

    gluPerspective(45, display[0]/display[1], 0.1, 30.0)
    glTranslatef(0.0, 0.0, -5)
    glRotatef(0, 0, 0 ,0)

    hyrosphere = HyroSphere(t_len=1.0, mass=8, dot_masses=[1.0]*4, position=[0.0,0.0,0.0])

    cam = camera.LookAtCamera(rotation=[0,0,0], distance=1.0)

    while True:
       # b1 += 0.05*np.cross(a1, b1)
       # b1 = (b1 / np.linalg.norm(b1)) *  tetra_len / 2.0

        for event in pg.event.get():
            if event == pg.QUIT: 
                pg.quit()
                quit()

            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    pg.quit()
                    quit()
                elif event.key == K_e:
                    hyrosphere.move(dt=0.05, ksi_new=[0,0,0,0.5])
            
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
        drawHyrosphere(hyrosphere)
        hyrosphere.move(dt=0.01,ksi_new=np.zeros(4))
        pg.display.flip()
        cam.pop()
        pg.time.wait(200)

if __name__ == "__main__":
    main()