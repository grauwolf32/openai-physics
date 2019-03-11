import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as pl

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def main():
    parser = argparse.ArgumentParser(description='Plot progress')
    parser.add_argument('-f', type=str)
    parser.add_argument('-n', type=int, default=None)
    args = parser.parse_args()

    if not os.path.isfile(args.f):
        print("File {} doesn't exists!".format(args.f))
        return

    pl.ion()
    fig = pl.figure()
    ax = fig.add_subplot(111)
    
    while True:
        with open(args.f, 'r') as f:
            data = f.read().split('\n')
    
        labels = data[1].split(',')
        data = data[2:]

        data = [np.array(list(map(lambda x: float(x.strip()), filter(lambda t: t != '', data[i].split(','))))) for i in range(0, len(data))]
        data = np.transpose(np.stack(np.array(data[:-1])))
        n,m = data.shape
    
        if args.n:
            data = [moving_average(data[i], args.n) for i in range(0, n)]
            data = np.stack(np.asarray(data))
            n,m = data.shape

        fig.canvas.flush_events()
        ax.clear()
        timeline = np.linspace(0,1, m)
        for i in range(0, n):
            ax.plot(timeline, data[i], label=labels[i])
        
        ax.legend()

        fig.canvas.draw()
        time.sleep(1.0)
        

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
