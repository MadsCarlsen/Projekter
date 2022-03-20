# %%
import numpy as np
import matplotlib.pyplot as plt
import pyshtools as pysh
from scipy.special import sph_harm
from scipy.special import lpmv as assoc_legendre
from OutputInterface import OutputInterface
from scipy.signal import find_peaks
import scipy.special as sp
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
    Expands a given function of theta and phi in spherical harmonics (f(theta, phi)).
    Output is on the form C_(l,m) = cilm[0,l,m] and C_(l,-m) = cilm[1,l,m].
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
    at some given direction
    """
    max_l = coeff_array.shape[1]
    res = 0 + 0j
    for l in range(max_l):
        for m in range(-l, l + 1, 1):
            if m >= 0:
                sign = 0
            else:
                sign = 1
            res += coeff_array[sign, l, abs(m)] * sph_harm(m, l, phi, theta)
    return res


def get_asymp_from_sph_coeff(GTO_sph_coeff, r, Ip, Z=1):
    """
    Gets the coeffciencts for the asymptotic expansion, given the coefficients for the sperhical expansion.
    The matching is made at a given value of r.
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


def get_as_coeffs(func, r, n_samp, Ip, Z=1, abs_thresh = 1e-3, normalized=False):
    """
    Get the asymptotic coefficients for a given value of r in a.u.

    :param func: Function to expand, func(r, theta, phi).
    :param r: The r to determine the asymptotic coefficients at.
    :param n_samp: Number of points used in the spherical expansion. Determines the accuracy.
    :param Ip: Ionization potential.
    :param Z: Charge of the leftover core.
    :param abs_thresh: Threshold value of coeffs. in the expansion. All below this is set to 0.
    :param normalized: If True, the coefficients will be normalized.
    """
    flm_lst = spherical_expansion(lambda theta, phi: func(r, theta, phi), n_samp, plot_coeff=False)

    clm_lst = np.zeros_like(flm_lst, dtype=complex)
    l_max = flm_lst.shape[1]
    kappa = np.sqrt(2 * abs(Ip))
    radial = r ** (Z / kappa - 1) * np.exp(-kappa * r)
    for l in range(l_max):
        for m in range(-l, l + 1):
            sgn = 0 if m >= 0 else 1
            clm = flm_lst[sgn, l, abs(m)] / radial
            clm_lst[sgn, l, abs(m)] = clm if abs(clm) > abs_thresh else 0

    if normalized:
        return clm_lst / np.sum(np.abs(clm_lst)**2)
    else:
        return clm_lst


def get_asymptotic_coeffs(func, n_r, n_samp, Ip, Z=1, interval=None, plot=False, normalized=False):
    """
    Gets the coefficients for the asymptotic wave function from the function
    """
    if interval is None:
        interval = [2, 17.5]

    # First find flms as a function of r
    r_lst = np.linspace(interval[0], interval[-1], n_r)
    flm_lst = []
    for i, r in enumerate(r_lst):
        print(f'Evaluating at r={r:.4f} \t Nr. {i + 1}/{n_r}')
        flm_lst.append(spherical_expansion(lambda theta, phi: func(r, theta, phi), n_samp, plot_coeff=False))
    flm_lst = np.array(flm_lst)

    # Loop through all of the coeffiicients and find the constant clm.
    # If it is lower than a threshold, it is set to zero
    ABS_THRESH = 1e-2
    clm_lst = np.zeros_like(flm_lst[0])
    l_max = flm_lst.shape[2]
    kappa = np.sqrt(2*np.abs(Ip))
    radial = lambda r, k: r**(Z/k - 1) * np.exp(-k*r)

    for l in range(l_max):
        for m in range(-l, l + 1):
            sgn = 0 if m >= 0 else 1
            clm = flm_lst[:, sgn, l, abs(m)] / radial(r_lst, kappa)
            idx = find_peaks(np.abs(clm))[0]
            if idx.size > 0:
                print(f'l={l} \t m={m} \t {r_lst[idx]} \t {clm[idx]}')
                val = clm[idx[-1]]
                clm_lst[sgn, l, abs(m)] = (val if np.abs(val) > ABS_THRESH else 0)
                if plot:
                    FUSK_FACTOR = 70
                    plt.figure(facecolor='white')
                    plt.plot(r_lst, np.abs(clm), label=r'$c_{\ell m}$')
                    plt.plot(r_lst, FUSK_FACTOR * np.abs(flm_lst[:, sgn, l, abs(m)]), label=r'$f_{\ell m}$')
                    plt.xlabel(r'Radius $r$')
                    plt.ylabel(r'Absolute amplitude')
                    plt.plot(r_lst[idx], np.abs(clm[idx]), 'o')
                    plt.legend(frameon=False)
                    plt.minorticks_on()
                    plt.show()

    if normalized:
        return clm_lst / np.sum(np.abs(clm_lst)**2)
    else:
        return clm_lst


def eval_asymptotic_cart(x, y, z, coeffs, Ip, Z=1):
    """
    Evaluates the asymptotic wave function in cartesian coordinates
    """
    l_max = coeffs.shape[1]
    kappa = np.sqrt(2 * abs(Ip))
    eta = 2 * Z / kappa + 5
    radial_norm = np.sqrt((2 * kappa) ** eta / sp.gamma(eta))

    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan2(np.sqrt(x**2 + y**2), z)
    phi = np.arctan2(y, x)

    if type(x) is np.ndarray:
        radial_part, angular_sum = np.zeros_like(x, dtype=complex), np.zeros_like(x, dtype=complex)
    else:
        radial_part, angular_sum = 0, 0

    radial_part += radial_norm * r**(Z / kappa - 1) * np.exp(-kappa * r)
    for l in range(l_max):
        for m in range(-l, l + 1):
            sgn = 1 if m >= 0 else 0  # This should be the other way around?
            angular_sum += coeffs[sgn, l, m]*sp.sph_harm(m, l, phi, theta)

    return radial_part * angular_sum


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


def cylindrical_from_spherical(r_par, r_perp, sph_coeffs):
    """
    Calculates the Fourier coefficients of a function given as a spherical expansion.
    Note that if sph_coeffs depend on r, this is valid for this value of r only (r = sqrt(r_par**2 + r_perp**2))!
    """
    r = np.sqrt(r_perp**2 + r_par**2)
    cos_theta = r_par / r
    theta = np.arccos(cos_theta)
    max_m = sph_coeffs.shape[1]-1

    fm_list = []
    for m in range(-max_m, max_m+1):
        sign = 0 if m >= 0 else 1
        fm = 0
        for l, flm in enumerate(sph_coeffs[sign, :, abs(m)]):
            if abs(m) > l or flm == 0 or flm == 0j:
                continue
            #N_lm = np.sqrt((2*l+1)/(4*np.pi) * np.math.factorial(l-m)/np.math.factorial(l+1))  # Should have these precalculated?
            #fm += flm * N_lm * assoc_legendre(m, l, cos_theta)
            fm += flm * sph_harm(m, l, 0, theta)  # This might be better?
        fm_list.append(fm)
    return fm_list


def eval_cylindrical(phi, coeff_list):
    """
    Evaluates a function given as a Fourier series with coefficients in coeff_list
    """
    res = 0
    max_m = int((len(coeff_list) - 1)/2)
    for m, coeff in zip(range(-max_m, max_m+1), coeff_list):
        res += np.exp(1j * m * phi) * coeff
    return res



"""
#%%
def test(theta, phi):
    return 3 * np.exp(1j * 2 * phi)

inter = OutputInterface('output_files/CHBrClF.out')
#GTO_params = inter.output_GTOs()


r_par = 3
r_perp = 3

r = np.sqrt(r_par**2 + r_perp**2)

thetam = np.arccos(r_par / r)
phim = np.pi/2

sph_coeffs = spherical_expansion(lambda theta, phi : inter.eval_orbital_spherical(r, theta, phi), 50, False)
#sph_coeffs = spherical_expansion(test, 50, False)
print(eval_sph_from_coeff(thetam, phim, sph_coeffs))

cylind_coeffs = cylindrical_from_spherical(sph_coeffs, r_par, r_perp)

print(eval_cylindrical(phim, cylind_coeffs))

"""

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
