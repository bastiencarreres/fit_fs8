import numpy as np
from numba import njit, prange


@njit(cache=True)
def angle_between(ra_0, dec_0, ra_1, dec_1):
    cos_alpha = np.cos(ra_1 - ra_0) * np.cos(dec_0) * np.cos(dec_1) + np.sin(dec_0) * np.sin(dec_1)
    return cos_alpha


@njit(cache=True)
def j0_alt(x):
    return np.sin(x) / x


@njit(cache=True)
def j2_alt(x):
    return (3 / x**2 - 1) * np.sin(x)/x - 3 * np.cos(x) / x**2


@njit(cache=True)
def separation(r_0, r_1, cos_alpha):
    return np.sqrt(r_0**2 + r_1**2 - 2 * r_0 * r_1 * cos_alpha)


@njit(cache=True)
def window(k, r_0, r_1, cos_alpha):
    ''' From Johnson et al. 2014 '''
    r = separation(r_0, r_1, cos_alpha)
    sin_alpha_squared = 1 - cos_alpha**2
    win = np.outer(1 / 3 * np.ones_like(k), np.ones(r_1.size))
    j0kr = j0_alt(np.outer(k, r))
    j2kr = j2_alt(np.outer(k, r))
    win = 1 / 3 * (j0kr - 2 * j2kr) * cos_alpha
    win += j2kr * r_0 * r_1 / r**2 * sin_alpha_squared
    return win


@njit(cache=True)
def get_covariance(ra_0, dec_0, r_comov_0, ra_1, dec_1, r_comov_1, k, pk):
    ''' Get cosmological covariance for a given pair of galaxies
        and a given power spectrum (k, pk) in units of h/Mpc and (Mpc/h)^3
    '''
    cos_alpha = angle_between(ra_0, dec_0, ra_1, dec_1)
    win = window(k, r_comov_0, r_comov_1, cos_alpha)
    cova = np.trapz(win.T * pk, x=k)
    return cova


@njit(cache=True, parallel=True)
def build_covariance_matrix(ra, dec, r_comov, k, pk_nogrid, grid_win=None, n_gals=None):
    ''' Builds a 2d array with the theoretical covariance matrix
        based on the positions of galaxies (ra, dec, r_comov)
        and a given power spectrum (k, pk)
    '''
    nh = ra.size
    cov_matrix = np.zeros((nh, nh))
    if grid_win is not None:
        print('Apply grid window')
        pk = pk_nogrid * grid_win**2
    else:
        pk = pk_nogrid * 1

    for i in prange(nh):
        cov = get_covariance(ra[i], dec[i], r_comov[i], ra[i+1:], dec[i+1:], r_comov[i+1:], k, pk)
        cov_matrix[i, i+1:] = cov
        cov_matrix[i+1:, i] = cov

    # For diagonal, window = 1/3
    var = np.trapz(pk / 3, x=k)

    np.fill_diagonal(cov_matrix, var)

    if grid_win is not None:
        var_nogrid = np.trapz(pk_nogrid / 3, x=k)
        # Eq. 22 of Howlett et al. 2017
        np.fill_diagonal(cov_matrix, var + (var_nogrid - var) / n_gals)

    # Pre-factor H0^2/(2pi^2)
    cov_matrix *= (100)**2 / (2 * np.pi**2)
    return cov_matrix


@njit(cache=True)
def grid_data(grid_size, ra, dec, r_comov, vpec, vpec_err, use_true_vel):
    x = r_comov * np.cos(ra) * np.cos(dec)
    y = r_comov * np.sin(ra) * np.cos(dec)
    z = r_comov * np.sin(dec)

    pos_min = np.array([np.min(x), np.min(y), np.min(z)])
    pos_max = np.array([np.max(x), np.max(y), np.max(z)])

    position = np.concatenate((x, y, z)).reshape((3, len(x)))

    # Number of grid voxels per axis
    n_grid = np.floor((pos_max - pos_min) / grid_size).astype(np.int64) + 1

    # Total number of voxels
    n_pix = n_grid.prod()

    # Voxel index of each catalog input on each axis
    index = np.floor((position.T - pos_min) / grid_size).astype(np.int64)

    # Voxel index over total number of voxels
    index = (index[:, 0] * n_grid[1] + index[:, 1]) * n_grid[2] + index[:, 2]

    sum_n = np.bincount(index, minlength=n_pix)

    # Consider only voxels with at least one galaxy
    mask = sum_n > 0
    if use_true_vel:
        center_vpec = np.bincount(index,
                                  weights=vpec,
                                  minlength=n_pix)[mask] / sum_n[mask]
        center_vpec_err = np.zeros(np.sum(mask))
    else:
        # Perform averages per voxel
        sum_vpec = np.bincount(index,
                               weights=vpec / vpec_err**2,
                               minlength=n_pix)[mask]
        sum_we = np.bincount(index,
                             weights=1 / vpec_err**2,
                             minlength=n_pix)[mask]
        center_vpec = sum_vpec / sum_we
        center_vpec_err = np.sqrt(1 / sum_we)
    center_ngals = sum_n[mask]

    # Determine the coordinates of the voxel centers
    i_pix = np.arange(n_pix)[mask]
    i_pix_z = i_pix % n_grid[2]
    i_pix_y = ((i_pix - i_pix_z) / n_grid[2]) % n_grid[1]
    i_pix_x = i_pix // (n_grid[1] * n_grid[2])

    cp_x = (i_pix_x + 0.5) * grid_size + pos_min[0]
    cp_y = (i_pix_y + 0.5) * grid_size + pos_min[1]
    cp_z = (i_pix_z + 0.5) * grid_size + pos_min[2]

    # Convert to ra, dec, r_comov
    center_r_comov = np.sqrt(cp_x**2 + cp_y**2 + cp_z**2)
    center_ra = np.arctan2(cp_y, cp_x)
    center_dec = np.arcsin(cp_z / center_r_comov)

    return center_ra, center_dec, center_r_comov, center_vpec, center_vpec_err, center_ngals


@njit(cache=True, parallel=True)
def compute_grid_window(grid_size, k, n):
    window = np.zeros_like(k)
    theta = np.linspace(0, np.pi, n)
    phi = np.linspace(0, 2 * np.pi, n)
    kx = np.outer(np.sin(theta), np.cos(phi))
    ky = np.outer(np.sin(theta), np.sin(phi))
    kz = np.outer(np.cos(theta), np.ones(n))

    # Forgotten in Howlett et al. formula
    # we add spherical coordinate solid angle element
    dthetaphi = np.outer(np.sin(theta), np.ones(phi.size))
    for i in prange(k.size):
        # the factor here has an extra np.pi because of the definition of np.sinc
        fact = (k[i] * grid_size) / (2 * np.pi)
        func = np.sinc(fact * kx) * np.sinc(fact * ky) * np.sinc(fact * kz) * dthetaphi
        win_theta = np.trapz(func, x=phi)
        window[i] = np.trapz(win_theta, x=theta)
    window *= 1 / (4 * np.pi)
    return window


@njit(cache=True)
def log_likelihood(x, cova):
    ''' Computes log of the likelihood from
        a vector x and a covariance cova
    '''
    nx = x.size
    eigvals = np.linalg.eigvalsh(cova)
    chi2 = x.T @ np.linalg.solve(cova, x)
    log_like = -0.5 * (nx * np.log(2*np.pi)
                       + np.sum(np.log(eigvals))
                       + chi2)
    return log_like
