#%% 
import numpy as np 
import matplotlib.pyplot as plt 
import pyshtools as pysh 
from scipy.special import sph_harm 
from OutputInterface import OutputInterface

#%% 
def eval_GTOs(x,y,z, param_list): 
    res = 0
    for params in param_list: 
        MO, alpha, i, j, k, x0, y0, z0 = params
        xi = x-x0
        yi = y-y0
        zi = z-z0
        res += MO * xi**i * yi**j * zi**k * np.exp(-alpha * (xi**2 + yi**2 + zi**2)) 
    return res 

def spherical_expansion(func, N, plot_coeff=False): 
    '''
    Expands a given function of theta and phi in spherical harmonics. 
    Output is on the form Clm=cilm[0,l,m] and Cl,-m=cilm[1,l,m].
    '''
    if N%2 != 0: 
        print('N should be an even number!')
        N += 1 
    
    phi_list = np.arange(0, 360, 360/N)
    theta_list = np.arange(0, 180, 180/N)
    func_grid = np.zeros((N,N), dtype=complex)

    theta_rad = np.deg2rad(theta_list)
    phi_rad = np.deg2rad(phi_list)

    for i,theta  in enumerate(theta_rad): 
        for j, phi in enumerate(phi_rad): 
            func_grid[i,j] = func(theta, phi)
    grid = pysh.SHGrid.from_array(func_grid,  copy=True)
    #sh_grid = pysh.expand.SHExpandDHC(grid)
    sh_grid = grid.expand(normalization='ortho')

    if plot_coeff: 
        fig, ax = sh_grid.plot_spectrum2d()
        plt.show()

    return sh_grid.to_array()

def eval_sph_from_coeff(theta, phi, coeff_array): 
    max_l = coeff_array.shape[1]
    res = 0 + 0j
    for l in range(max_l):
        if l == 0: 
            res += coeff_array[0,0,0] * sph_harm(0,0, phi, theta)
            continue 
        for m in range(-l, l+1, 1): 
            if m >= 0: 
                sign = 0
            else: 
                sign = 1 
            res +=  (-1)**abs(m) * coeff_array[sign, l, abs(m)] * sph_harm(m,l, phi, theta)
            # WHY DO WE NEED TO MULTIPLY THIS FACTOR ON!? 
            #print(coeff_array[sign, l, abs(m)], sign, l, m)
    return res 

def find_clm(GTO_sph_coeff, r, Ip, Z=1): 
    kappa = np.sqrt(2*Ip)

    clm_list = np.zeros_like(GTO_sph_coeff)
    max_l = clm_list.shape[1]
    radial_part = np.exp(kappa*r) / r**(Z/kappa-1)

    for l in range(max_l):
        if l == 0: 
            clm_list[0,0,0] = GTO_sph_coeff[0,0,0] * radial_part 
            continue 
        for m in range(-l, l+1, 1): 
            if m >= 0: 
                sign = 0
            else: 
                sign = 1 
            clm_list[sign, l, abs(m)] = (-1)**abs(m) * GTO_sph_coeff[sign, l, abs(m)] * radial_part
    return clm_list 

def eval_asymptotic(r, theta, phi, coeff_array, Ip, Z=1):
    max_l = coeff_array.shape[1]
    res = 0 + 0j
    kappa = np.sqrt(2*Ip)
    radial_part = np.exp(-kappa*r) * r**(Z/kappa-1)
    for l in range(max_l):
        if l == 0: 
            res += coeff_array[0,0,0] * sph_harm(0,0, phi, theta) * radial_part
            continue 
        for m in range(-l, l+1, 1): 
            if m >= 0: 
                sign = 0
            else: 
                sign = 1 
            res +=  coeff_array[sign, l, abs(m)] * sph_harm(m,l, phi, theta) * radial_part
    return res 
#%%
def test_func(theta, phi): 
    m, l = (3,3)
    return sph_harm(m, l, phi, theta)

inter = OutputInterface('CHBrClF.out')
GTO_params = inter.output_GTOs()

r = 7

test = spherical_expansion(lambda theta, phi: inter.eval_orbital_spherical(r, theta, phi), 20, True)
#test = spherical_expansion(test_func, 10, True)

#%% PLOT OVER DIFFERENT THETA VALUES
theta_list = np.linspace(0,np.pi, 100)
phi = 2*np.pi

dims_list = [np.real(eval_sph_from_coeff(theta, phi, test)) for theta in theta_list]
inter_list = [inter.eval_orbital_spherical(r, theta, phi) for theta in theta_list]

plt.plot(theta_list, dims_list)
plt.plot(theta_list, inter_list)


#%% Let's find the clm's for the asymptotic form! 

Ip = -inter.saved_orbitals[inter.HOMO][0]
r_list = np.linspace(2,10,20)

clm_r_list = []
for i,r in enumerate(r_list): 
    print(i)
    GTO_sph_coeff = spherical_expansion(lambda theta, phi: inter.eval_orbital_spherical(r, theta, phi), 20)
    clm_r_list.append(find_clm(GTO_sph_coeff, r, Ip))


#%% Plot how the expansion coeffs vary as func of r...
max_index = np.unravel_index(np.argmax(np.abs(clm_r_list[0])),clm_r_list[0].shape)
#max_index = (1,3,1)
max_coeff = []
for clm_i in clm_r_list: 
    max_coeff.append(clm_i[max_index])

plt.plot(r_list, np.real(max_coeff))
plt.plot(r_list, np.imag(max_coeff))


#%% Calculate for about r=7..
r_cal = 7 
GTO_sph_coeffs = spherical_expansion(lambda theta, phi: inter.eval_orbital_spherical(r_cal, theta, phi), 50)
asymptotic_coeffs = find_clm(GTO_sph_coeffs, r_cal, Ip)

#%%
theta = 1
phi = 2 
r = 10
val = eval_asymptotic(r, theta, phi, asymptotic_coeffs, Ip)

print(val)
print(inter.eval_orbital_spherical(r,theta,phi))


#%%
r_plot = np.linspace(3,10, 100)

#%% 


r_plot = np.linspace(3,4,100)
inter_list = [(inter.eval_orbital_spherical(r,0, 0))**2 for r in r_plot]
plt.plot(r_plot, inter_list)
# %%
