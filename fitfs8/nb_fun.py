import numpy as np
from numba import njit


@njit(cache=True)
def angle_between(ra_0, dec_0, ra_1, dec_1):
    cos_alpha = np.cos(ra_1-ra_0)*np.cos(dec_0)*np.cos(dec_1) + np.sin(dec_0)*np.sin(dec_1)
    return cos_alpha


@njit(cache=True)
def j0_alt(x):
    return np.sin(x)/x


@njit(cache=True)
def j2_alt(x):
    return (3/x**2-1)*np.sin(x)/x - 3*np.cos(x)/x**2


@njit(cache=True)
def separation(r_0, r_1, cos_alpha):
    return np.sqrt(r_0**2 + r_1**2 - 2*r_0*r_1*cos_alpha)


@njit(cache=True)
def window(k, r_0, r_1, cos_alpha):
    ''' From Johnson et al. 2014 '''
    r = separation(r_0, r_1, cos_alpha)
    sin_alpha_squared = 1-cos_alpha**2
    win = 1/3*np.ones_like(k)
    if r > 0:
        j0kr = j0_alt(k*r)
        j2kr = j2_alt(k*r)
        win = 1/3*(j0kr - 2*j2kr)*cos_alpha
        win = win+(r_0*r_1/r**2*sin_alpha_squared * j2kr)
    return win


@njit(cache=True)
def get_covariance(ra_0, dec_0, r_comov_0, ra_1, dec_1, r_comov_1, k, pk):
    ''' Get cosmological covariance for a given pair of galaxies
        and a given power spectrum (k, pk) in units of h/Mpc and (Mpc/h)^3
    '''
    cos_alpha = angle_between(ra_0, dec_0, ra_1, dec_1)
    win = window(k, r_comov_0, r_comov_1, cos_alpha)
    cova = np.trapz(pk * win, x=k)
    return cova


@njit(cache=True)
def build_covariance_matrix(ra, dec, r_comov, k, pk_nogrid, grid_win=None, n_gals=None):
    ''' Build a 2d array with the theoretical covariance matrix
        based on the positions of galaxies (ra, dec, r_comov)
        and a given power spectrum (k, pk)
    '''
    nh = ra.size
    cov_matrix = np.zeros((nh, nh))
    if grid_win is not None:
        pk = pk_nogrid * grid_win**2
    else:
        pk = pk_nogrid * 1

    for i in range(nh):
        ra_0 = ra[i]
        dec_0 = dec[i]
        r_comov_0 = r_comov[i]
        for j in range(i + 1, nh):
            ra_1 = ra[j]
            dec_1 = dec[j]
            r_comov_1 = r_comov[j]
            cov = get_covariance(ra_0, dec_0, r_comov_0, ra_1, dec_1, r_comov_1, k, pk)
            cov_matrix[i, j] = cov
            cov_matrix[j, i] = cov

    # For diagonal, window = 1/3
    var = np.trapz(pk/3, x=k)

    np.fill_diagonal(cov_matrix, var)

    if grid_win is not None:
        var_nogrid = np.trapz(pk_nogrid / 3, x=k)

        # Eq. 22 of Howlett et al. 2017
        np.fill_diagonal(cov_matrix, var + (var_nogrid - var) / n_gals)

    # Pre-factor H0^2/(2pi^2)
    cov_matrix *= (100)**2 / (2 * np.pi**2)

    return cov_matrix


@njit(cache=True)
def log_likelihood(x, cova):
    ''' Computes log of the likelihood from
        a vector x and a covariance cova
    '''
    nx = x.size
    eigvals = np.linalg.eigvalsh(cova)
    chi2 = x.T @ np.linalg.solve(cova, x)
    log_like = -0.5*(nx*np.log(2*np.pi)
                     + np.sum(np.log(eigvals))
                     + chi2)
    return log_like
