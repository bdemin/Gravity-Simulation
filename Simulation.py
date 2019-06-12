import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle

from math import log10
from scipy.constants import G


class Body(object):
    def __init__(self, pos, mass):
        self.pos = pos
        self.mass = mass
        self.rad = log10(mass)
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
        acc = 0
        for body in bodies:
            acc += G * (self.mass+body.mass) / (self.rad*body.rad)
        self.acc = acc





# Define figure and axes:
fig1, ax1 = plt.subplots()


num_bodies = 10
bodies = []
size = 1000
for i in range(num_bodies):
    pos = np.array((np.random.randint(0, size), np.random.randint(0, size)))
    mass = 10 ** np.random.randint(1, 30)
    bodies.append(Body(pos, mass))


def init():
    ax1.set_xlim(0, size)
    ax1.set_ylim(0, size)
    ax1.set_aspect(1)
    return []

def animate(frame):
    artists = []
    for body in bodies:
        artists.append(body.get_artist(ax1))

    return artists


anim = animation.FuncAnimation(fig1, animate, init_func = init, blit = True)
plt.show()