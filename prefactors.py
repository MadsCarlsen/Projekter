#%% 
import numpy as np
import matplotlib.pyplot as plt
from OutputInterface import OutputInterface
#from mpmath import hermite 


def hermite(n, x):
    """
    Function that implements the first hermite polynomials. MUCH faster than using mpmath or scipy!

    :param n: Order of the hermite polynomial
    :param x: Value in which to evaluate the polynomial
    """
    if n == 0:
        return 1.0
    elif n == 1:
        return 2*x
    elif n == 2:
        return 4*x**2 -2
    elif n == 3:
        return 8*x**3 -12*x
    else:
        print('This hermite poly is not implemented!')
        return None


def GTO_prefactor_LG(px, py, pz, x0, y0, z0, i, j, k, alpha, N, A, E):
    """
    Prefactor of a single GTO in the length gauge

    :param px: Momentum in x direction
    :param py: Momentum in y direction
    :param pz: Momentum in z direction
    :param x0: x coordinate of GTO center
    :param y0: y coordinate of GTO center
    :param z0: z coordinate of GTO center
    :param i: Exponent of GTO x variable
    :param j: Exponent of GTO y variable
    :param k: Exponent of GTO z variable
    :param alpha: GTO exponent
    :param N: Front factor of GTO - Normalization * contraction coeffcicient * MO-orbital coeffcient
    :param A: Vector field
    :param E: Electric field
    """
    p_t = pz + A  # z component of p_tilde
    val = 1 / (2 * np.sqrt(alpha))  # Just a value used a lot..

    # Evaluate the hermite polynomials
    Hx = hermite(i, px * val)
    Hy = hermite(j, py * val)
    Hz1 = hermite(k, p_t * val)
    Hz2 = hermite(k + 1, p_t * val)

    # Compute the integrals of the three components:
    Ix = np.exp(-1j * px * x0) * np.sqrt(np.pi / alpha) * (-1j * val) ** i \
         * Hx * np.exp(-px ** 2 / (4 * alpha))
    Iy = np.exp(-1j * py * y0) * np.sqrt(np.pi / alpha) * (-1j * val) ** j \
         * Hy * np.exp(-py ** 2 / (4 * alpha))
    Iz = np.exp(-1j * p_t * z0) * np.sqrt(np.pi / alpha) * np.exp(-p_t ** 2 / (4 * alpha)) \
         * (-1j * val) ** k * (Hz1 * z0 - 1j * val * Hz2)

    return N / (2*np.pi)**(3/2) * E * Ix * Iy * Iz



"""

def d_hydrogen_s(px, py, pz, A, E): 
    p = np.sqrt(px*px + py*py +(pz+A)**2)
    theta = np.arccos((pz+A)/p)

    return -1j/np.pi * np.sqrt(2) * E * np.cos(theta) * 8*p/(p**2 + 1)**3


N_points = 10
px_list = np.linspace(0, 1.5, N_points)
pz_list = np.linspace(-1.5, 1.5, N_points*2)
py = 0

A = 0
E = 1

#%% 
# Load the GTO data...
#inter = OutputInterface('CHBrClF.out')
inter = OutputInterface('hydrogen2.out')
GTO_list = inter.output_GTOs()

res_GTO = np.zeros((len(pz_list), len(px_list)), dtype=complex)

for m, pz in enumerate(pz_list):
    print(pz)
   
    for n, px in enumerate(px_list):
        res = 0
        for GTO_params in GTO_list:
            #print(GTO_params)
            N, alpha, i, j, k, x0, y0, z0 = GTO_params
            res += d_GTO_LG(px, py, pz, x0, y0, z0, i, j, k, alpha, N, A, E)
        res_GTO[m,n] = res
    

#%% 
plt.imshow(np.flip((np.abs(res_GTO)**2),0), aspect=1, cmap='inferno', interpolation='bicubic',
     extent = (np.amin(px_list), np.amax(px_list), np.amin(pz_list), np.amax(pz_list)))
plt.colorbar()

#%% 
res_hydrogen = np.zeros((len(pz_list), len(px_list)), dtype=complex)
for i,pz in enumerate(pz_list): 
    print(i)
    for j,px in enumerate(px_list):
        res_hydrogen[i,j] = d_hydrogen_s(px, py, pz, A, E) 

plt.imshow(np.flip((np.abs(res_hydrogen)**2),0), aspect=1, cmap='inferno', interpolation='bicubic',
     extent = (np.amin(px_list), np.amax(px_list), np.amin(pz_list), np.amax(pz_list)))
plt.colorbar()



# %%
plt.imshow(np.flip(np.abs(res_hydrogen)**2 - np.abs(res_GTO)**2,0), aspect=1, cmap='inferno', interpolation='bicubic',
     extent = (np.amin(px_list), np.amax(px_list), np.amin(pz_list), np.amax(pz_list)))


# %% Test in a single point...

px, py, pz = (0.1, 0, 0.4)
res = 0
for GTO_params in GTO_list:
    #print(GTO_params)
    N, alpha, i, j, k, x0, y0, z0 = GTO_params
    res += d_GTO_LG(px, py, pz, x0, y0, z0, i, j, k, alpha, N, A, E)
print(res)
# %%

"""