import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle

from math import log10
from scipy.constants import G


def get_dirvec(vec1, vec2):
    vector = vec2 - vec1
    return np.abs(vector) / np.linalg.norm(vector)


class Body(object):
    def __init__(self, pos, mass, vel):
        self.mass = mass
        self.rad = log10(mass) * 1e5

        self.pos = pos
        self.vel = vel
        self.force = np.zeros(2)
        self.acc = np.zeros(2)

    def get_artist(self, ax):
        self.artist = ax.add_patch(Circle(self.pos, self.rad))
        return self.artist

    def collision_check(self, body):
        vector_len = np.linalg.norm(self.pos - body.pos)
        if vector_len*1.3 <= self.rad + body.rad:
            print('kaboom')
            return True
        return False

    def calc_force(self, body):
        dist = np.linalg.norm(self.pos - body.pos)
        self.force += -(G * self.mass * body.mass)/(dist**2) * get_dirvec(self.pos, body.pos)

    def calc_acc(self, body):
        dist = np.linalg.norm(self.pos - body.pos)
        self.acc += -G * (body.mass/(dist**2)) * get_dirvec(self.pos, body.pos)

    def move(self, dt):
        try:
            self.vel += self.acc * dt
            self.pos += self.vel * dt
        except:
            print('wtf')

    def bounce(self):
        if self.pos[0] <= 0 or self.pos[0] >= size:
            self.vel[0] *= -1
        elif self.pos[1] <= 0 or self.pos[1] >= size:
            self.vel[1] *= -1

# Define figure and axes:
fig1, ax1 = plt.subplots()


num_bodies = 10
bodies = []
size = 150e6 #[Km]
max_vel = 100

star = Body((size/2, size/2), 1.989e30, np.zeros(2))
bodies.append(star)

for i in range(num_bodies):
    pos = np.array((float(np.random.randint(0, size)), 
                    float(np.random.randint(0, size))))
    mass = 10 ** np.random.randint(10, 30)
    vel = np.array((float(np.random.randint(0, max_vel)), 
                    float(np.random.randint(0, max_vel))))
    bodies.append(Body(pos, mass, vel))

# bodies.append(Body((size/2 + 300, size/2 + 300), 1e20, np.zeros(2)))



T = 60
FPS = 60
num_frames = T*FPS
dt = (1/FPS)


def init():
    ax1.set_xlim(0, size)
    ax1.set_ylim(0, size)
    ax1.set_aspect(1)
    return []

def animate(frame):
    artists = []
    for body1 in bodies:
        if body1.mass < 1e30:
            for body2 in bodies:
                if body1 is not body2:
                    body1.calc_force(body2)
                    # body1.calc_acc(body2)
                    

            body1.acc =  body1.force / body1.mass
            body1.move(dt)
            body1.bounce()
        artists.append(body1.get_artist(ax1))
        body1.acc = np.zeros(2)



    return artists


anim = animation.FuncAnimation(fig1, animate, frames = num_frames,
                                interval = dt*1000,
                                init_func = init, blit = True)
plt.show()