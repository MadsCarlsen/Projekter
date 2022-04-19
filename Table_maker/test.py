import scipy.special as sp
import numpy as np

r = 1.1
theta = 0.8
phi = 0.69

l = 10
m = -9

x = r * np.sin(theta) * np.cos(phi)
y = r * np.sin(theta) * np.sin(phi)
z = r * np.cos(theta)

print(sp.sph_harm(m, l, phi, theta))