import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle

from math import log10
from scipy.constants import G


def get_dirvec(vec1, vec2):
    vector = vec1 - vec2
    return np.abs(vector) / np.linalg.norm(vector)


class Body(object):
    def __init__(self, pos, mass):
        self.mass = mass
        self.rad = log10(mass)

        self.pos = pos
        self.vel = np.zeros(2)
        self.acc = np.zeros(2)

    def get_artist(self, ax):
        self.artist = ax.add_patch(Circle(self.pos, self.rad))
        return self.artist

    def collision_check(self, body):
        vector_len = np.abs(self.pos - body.pos)
        if vector_len <= self.rad + body.rad:
            return True
        return False

    def calc_acc(self, bodies):
        for body in bodies:
            if self is not body:
                dist = np.linalg.norm(self.pos - body.pos) 
                self.acc = -G * (body.mass/dist) * get_dirvec(self.pos, body.pos)

                
    def move(self, dt):
        self.vel += self.acc * dt
        self.pos += self.vel * dt



# Define figure and axes:
fig1, ax1 = plt.subplots()


num_bodies = 10
bodies = []
size = 1000
for i in range(num_bodies):
    pos = np.array((float(np.random.randint(0, size)), 
                    float(np.random.randint(0, size))))
    mass = 10 ** np.random.randint(1, 30)
    bodies.append(Body(pos, mass))


T = 60
FPS = 48
num_frames = T*FPS
dt = (1/FPS)


def init():
    ax1.set_xlim(0, size)
    ax1.set_ylim(0, size)
    ax1.set_aspect(1)
    return []

def animate(frame):
    artists = []
    for body in bodies:
        body.calc_acc(bodies)
        body.move(dt)
        artists.append(body.get_artist(ax1))


    return artists


anim = animation.FuncAnimation(fig1, animate, frames = num_frames,
                                interval = dt*1000,
                                init_func = init, blit = True)
plt.show()