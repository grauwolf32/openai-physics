from gym_hyrosphere.envs.visualization import *
from gym_hyrosphere.envs.physics import *

def main():
    pg.init()
    glutInit([])
    display = (800, 600)

    pg.display.set_mode(display, DOUBLEBUF|OPENGL)

    gluPerspective(45, display[0]/display[1], 0.1, 30.0)
    glTranslatef(0.0, 0.0, -3)

    hyrosphere = HyroSphere(t_len=1.0, mass=8, dot_masses=[1.0]*4, position=[0.0,0.0,0.0])
    cam = LookAtCamera(rotation=[90,0,0], distance=1.0)

    while True:
        for event in pg.event.get():
            if event == pg.QUIT: 
                pg.quit()
                quit()

            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    pg.quit()
                    quit()
                elif event.key == K_e:
                    R, dRdt = hyrosphere.move(dt=0.01,ksi_new=[0.0,0.0,0.0,0.5])
            
        keys = pg.key.get_pressed()

        if keys[K_UP]:
            cam.distance -= 0.05
        if keys[K_DOWN]:
            cam.distance += 0.05

        if keys[K_LEFT]:
            cam.roty -= 3
        if keys[K_RIGHT]:
            cam.roty += 3

        if keys[K_w]:
            cam.rotx += 3
        if keys[K_s]:
            cam.rotx -= 3

        if keys[K_q]:
            cam.rotz += 3
        if keys[K_a]:
            cam.rotz -= 3
        
        glClearColor(1.0, 1.0, 1.0, 1.0) 
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        
        #glEnable(GL_LIGHTING)
        cam.push()
        R, dRdt = hyrosphere.move(dt=0.01,ksi_new=np.zeros(4))

        K = np.append(R, [np.zeros(3), np.asarray([0,0,-hyrosphere.radius])], axis=0)
        drawHyrosphere(hyrosphere)
        pg.display.flip()


        cam.pop()
        pg.time.wait(20)

if __name__ == "__main__":
    main()