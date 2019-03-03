import time
from gym_hyrosphere.envs import physic_fast as physics

def main():
    hs = physics.HyroSphere(t_len=1.0, mass=8, dot_masses=[1.0]*4, position=[0.0,0.0,0.0])
    n = 10000
    dt = 0.01
    t1 = time.time()
    for i in range(0, n):
        hs.move(dt, [0.0]*4)
    t2 = time.time()

    print("{} trials took {} seconds".format(n, t2-t1))

if __name__ == "__main__":
    main()