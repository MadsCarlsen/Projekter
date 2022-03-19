# %%
import numpy as np
import matplotlib.pyplot as plt
import pyshtools as pysh
from scipy.special import sph_harm
from OutputInterface import OutputInterface
from scipy.signal import find_peaks
# %%


def eval_GTOs(x, y, z, param_list):
    """
    Evaluate gaussian type orbitals from parameter list
    """
    res = 0
    for params in param_list:
        MO, alpha, i, j, k, x0, y0, z0 = params
        xi = x-x0
        yi = y-y0
        zi = z-z0
        res += MO * xi**i * yi**j * zi**k * np.exp(-alpha * (xi**2 + yi**2 + zi**2))
    return res


def spherical_expansion(func, N, plot_coeff=False):
    """
    Expands a given function of theta and phi in spherical harmonics. 
    Output is on the form Clm=cilm[0,l,m] and Cl,-m=cilm[1,l,m].
    """
    if N % 2 != 0:
        print('N should be an even number! Incrementing by one')
        N += 1

    phi_list = np.arange(0, 360, 360/N)
    theta_list = np.arange(0, 180, 180/N)
    func_grid = np.zeros((N, N), dtype=complex)

    theta_rad = np.deg2rad(theta_list)
    phi_rad = np.deg2rad(phi_list)

    for i, theta in enumerate(theta_rad):
        for j, phi in enumerate(phi_rad):
            func_grid[i, j] = func(theta, phi)
    grid = pysh.SHGrid.from_array(func_grid,  copy=True)
    sh_grid = grid.expand(normalization='ortho', csphase=-1)

    if plot_coeff:
        fig, ax = sh_grid.plot_spectrum2d()
        plt.show()

    return sh_grid.to_array()


def eval_sph_from_coeff(theta, phi, coeff_array):
    """
    Evaluate a linear combination of spherical harmonics from the array of coefficients
    at some given angle
    """
    max_l = coeff_array.shape[1]
    res = 0 + 0j
    for l in range(max_l):
        if l == 0:
            res += coeff_array[0, 0, 0] * sph_harm(0, 0, phi, theta)
            continue
        for m in range(-l, l + 1, 1):
            if m >= 0:
                sign = 0
            else:
                sign = 1
            res += coeff_array[sign, l, abs(m)] * sph_harm(m, l, phi, theta)
    return res


def get_flm(GTO_sph_coeff, r, Ip, Z=1):
    """
    Gets the coefficients from the f_lms from the f_lms from the Laplace expansion of the GTOs
    """
    kappa = np.sqrt(2*Ip)

    flm_list = np.zeros_like(GTO_sph_coeff)
    max_l = flm_list.shape[1]
    radial_part = np.exp(kappa*r) / r**(Z/kappa-1)

    for l in range(max_l):
        if l == 0:
            flm_list[0, 0, 0] = GTO_sph_coeff[0, 0, 0] * radial_part
            continue
        for m in range(-l, l+1, 1):
            if m >= 0:
                sign = 0
            else:
                sign = 1
            flm_list[sign, l, abs(m)] = GTO_sph_coeff[sign, l, abs(m)] * radial_part
    return flm_list


def get_asymptotic_coeffs(func, n_pts, n_samp, Ip, Z=1, interval=None, plot=False):
    """
    Gets the coefficients for the asymptotic wave function from the function
    """
    if interval is None:
        interval = [2, 20]

    # First find flms as a function of r
    r_lst = np.linspace(interval[0], interval[-1], n_pts)
    flm_lst = []
    for i, r in enumerate(r_lst):
        print(f'Evaluating at r={r:.4f} \t Nr. {i + 1}/{n_pts}')
        flm_lst.append(spherical_expansion(lambda theta, phi: func(r, theta, phi), n_samp, plot_coeff=False))

    # Loop through all of the coeffiicients and find the constant clm.
    # If it is lower than a threshold, it is set to zero
    flm_lst = np.array(flm_lst)
    ABS_THRESH = 1e-10
    clm_lst = np.zeros_like(flm_lst[0])
    l_max = flm_lst.shape[1]
    kappa = np.sqrt(2*Ip)
    radial = lambda r: r**(Z/kappa - 1) * np.exp(-kappa*r)

    for l in range(l_max):
        if l == 0:
            clm = flm_lst[:, 0, 0, 0]/radial(r_lst)
            idx = find_peaks(np.abs(clm))[0]  # Last radial peak!
            if plot:
                plt.figure(facecolor='white')
                plt.plot(r_lst, np.abs(clm))
                plt.xlabel(f'{int(idx.size > 0)}')
                plt.show()
            if idx.size > 0:
                val = clm[idx[-1]]
                clm_lst[0, 0, 0] = (val if val > ABS_THRESH else 0)
            continue
        for m in range(-l, l + 1):
            sgn = int(m < 0)  # 0 for m >= 0 and 1 for m < 0
            clm = flm_lst[:, sgn, l, abs(m)] / radial(r_lst)
            idx = find_peaks(np.abs(clm))[0]  # Last radial peak!
            if plot:
                plt.figure(facecolor='white')
                plt.plot(r_lst, np.abs(clm))
                plt.xlabel(f'{int(idx.size > 0)}')
                plt.show()
            if idx.size > 0:
                val = clm[idx[-1]]
                clm_lst[sgn, l, abs(m)] = (val if val > ABS_THRESH else 0)

    return clm_lst / np.sum(np.abs(clm_lst)**2)


def eval_asymptotic(r, theta, phi, coeff_array, Ip, Z=1):
    """
    Evaluates the asymptotic wave function in spherical coordinates from the c_lm array of coefficients
    """
    max_l = coeff_array.shape[1]
    res = 0 + 0j
    kappa = np.sqrt(2*Ip)
    radial_part = np.exp(-kappa*r) * r**(Z/kappa-1)
    for l in range(max_l):
        if l == 0:
            res += coeff_array[0, 0, 0] * sph_harm(0, 0, phi, theta) * radial_part
            continue
        for m in range(-l, l + 1, 1):
            if m >= 0:
                sign = 0
            else:
                sign = 1
            res += coeff_array[sign, l, abs(m)] * sph_harm(m, l, phi, theta) * radial_part
    return res
# %%

'''
def test_func(theta, phi): 
    m, l = (3, 3)
    return sph_harm(m, l, phi, theta)

inter = OutputInterface('CHBrClF1.out')
GTO_params = inter.output_GTOs()

r = 7

test = spherical_expansion(lambda theta, phi: inter.eval_orbital_spherical(r, theta, phi), 40, True)
# test = spherical_expansion(test_func, 10, True)

# %% PLOT OVER DIFFERENT THETA VALUES
theta_list = np.linspace(0, np.pi/4, 1000)
phi = np.pi

dims_list = [np.real(eval_sph_from_coeff(theta, phi, test)) for theta in theta_list]
inter_list = [inter.eval_orbital_spherical(r, theta, phi) for theta in theta_list]

plt.plot(theta_list, dims_list)
plt.plot(theta_list, inter_list)
plt.xlabel(r'$\theta$')
plt.show()


# %% Let's find the clm's for the asymptotic form!

Ip = -inter.saved_orbitals[inter.HOMO][0]
r_list = np.linspace(2, 20, 30)

clm_r_list = []
for i,r in enumerate(r_list): 
    print(i)
    GTO_sph_coeff = spherical_expansion(lambda theta, phi: inter.eval_orbital_spherical(r, theta, phi), 20)
    clm_r_list.append(find_clm(GTO_sph_coeff, r, Ip))


# %% Plot how the expansion coeffs vary as func of r...
max_index = np.unravel_index(np.argmax(np.abs(clm_r_list[0])), clm_r_list[0].shape)
# max_index = (1, 3, 1)
max_coeff = []
for clm_i in clm_r_list: 
    max_coeff.append(clm_i[max_index])

plt.plot(r_list, np.real(max_coeff), label=r'Re$[c_{\ell m}]$')
plt.plot(r_list, np.imag(max_coeff), label=r'Im$[c_{\ell m}]$')
plt.plot(r_list, np.abs(max_coeff), label=r'$|c_{\ell m}|$')
inter_list = [(100*inter.eval_orbital_spherical(r, np.pi/2, np.pi/2)) for r in r_list]
plt.plot(r_list, inter_list, label=r'$\psi$')
plt.xlabel(r'$r$ (a.u.)')
plt.ylabel(r'Amplitude')
plt.legend(frameon=False)
plt.show()


# %% Calculate for about r=7..
r_cal = 7 
GTO_sph_coeffs = spherical_expansion(lambda theta, phi: inter.eval_orbital_spherical(r_cal, theta, phi), 50)
asymptotic_coeffs = find_clm(GTO_sph_coeffs, r_cal, Ip)

# %%
theta = 1
phi = 2 
r = 10
val = eval_asymptotic(r, theta, phi, asymptotic_coeffs, Ip)

print(val)
print(inter.eval_orbital_spherical(r, theta, phi))


# %%
r_plot = np.linspace(3, 10, 100)'''

# %%


# %%
