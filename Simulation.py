import numpy as np


class Particle(object):
    def __init__(self, pos, mass):
        self.pos = pos
        self.mass = mass
        self.acc = 0



num_particles = 100
particles = []
size = 50
for i in range(num_particles):
    pos = np.array((np.random.randint(0,size), np.random.randint(0, size)))
    mass = np.random.randint(1e5, 1e10)
    particles.append(Particle(pos, mass))