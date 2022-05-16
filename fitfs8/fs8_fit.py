"""Main module."""

import iminuit
import time
import copy
import os
import numpy as np
import pandas as pd
from astropy.table import Table
import astropy.cosmology as acosmo
import matplotlib.pyplot as plt
from . import nb_fun as nbf


def set_cosmo(cosmo):
    """Load an astropy cosmological model.
    Parameters
    ----------
    cosmo : dict
        A dict containing cosmology parameters.
    Returns
    -------
    astropy.cosmology.object
        An astropy cosmological model.
    """
    astropy_mod = list(map(lambda x: x.lower(), acosmo.parameters.available))
    if isinstance(cosmo, str):
        name = cosmo.lower()
        if name in astropy_mod:
            if name == 'planck18':
                return acosmo.Planck18
            elif name == 'planck18_arxiv_v2':
                return acosmo.Planck18_arXiv_v2
            elif name == 'planck15':
                return acosmo.Planck15
            elif name == 'planck13':
                return acosmo.Planck13
            elif name == 'wmap9':
                return acosmo.WMAP9
            elif name == 'wmap7':
                return acosmo.WMAP7
            elif name == 'wmap5':
                return acosmo.WMAP5
        else:
            raise ValueError(f'Available model are {astropy_mod}')
    elif isinstance(cosmo, dict):
        if 'Ode0' not in cosmo.keys():
            cosmo['Ode0'] = 1 - cosmo['Om0']
        return acosmo.w0waCDM(**cosmo)
    else:
        return cosmo


def read_power_spectrum(pow_spec):

    # Read power spectrum from camb
    # units are in h/Mpc and Mpc^3/h^3
    k, pk = np.loadtxt(pow_spec)
    return k, pk


class fs8_fitter:
    def __init__(self, pow_spec, data, cosmo='planck18',
                 kmax=None, kmin=None, key_dic={}, data_mask=None):

        self._pk_nonorm = read_power_spectrum(pow_spec)
        self._kmax = np.inf
        self._kmin = 0.
        if kmax is None:
            self.kmax = np.max(self._pk_nonorm[0])
        else:
            self.kmax = kmax
        if kmin is None:
            self.kmin = np.min(self._pk_nonorm[0])
        else:
            self.kmin = kmin
        self.data_mask = data_mask
        self._cosmo = set_cosmo(cosmo)

        self._init_data(data, key_dic)

        self.data_grid = None
        self.cov_cosmo = None
        self._grid_size = None
        self._grid_window = None

    def _init_data(self, data, key_dic):
        if isinstance(data, str):
            ext = os.path.splitext(data)[-1]
            if ext == '.fits':
                self._data = Table.read(data).to_pandas()
            elif ext == '.csv':
                self._data = pd.read_csv(data)
            elif ext == '.parquet':
                self._data = pd.read_parquet(data)
            else:
                raise ValueError('Support .csv and .fits file')

        elif isinstance(data, pd.core.frame.DataFrame):
            self._data = data

        self._data.rename(columns=key_dic, inplace=True)
        self._data['r_comov'] = self._cosmo.comoving_distance(self._data['zobs']).value
        self._data['r_comov'] *= self._cosmo.h

    @property
    def kmax(self):
        return self._kmax

    @kmax.setter
    def kmax(self, kmax):
        if kmax > np.min(self._pk_nonorm[0]) and kmax > self.kmin:
            self._kmax = kmax
        else:
            raise ValueError('kmax must be > kmin')

    @property
    def kmin(self):
        return self._kmin

    @kmin.setter
    def kmin(self, kmin):
        if kmin < np.max(self._pk_nonorm[0]) and kmin < self.kmax:
            self._kmin = kmin
        else:
            raise ValueError('kmin must be < kmax')

    @property
    def pk(self):
        k_cut = self._pk_nonorm[0] <= self.kmax
        k_cut &= self._pk_nonorm[0] >= self.kmin
        k = self._pk_nonorm[0][k_cut]
        pk = self._pk_nonorm[1][k_cut]
        return k, pk

    @property
    def sigma8(self):
        return self._sigma8

    @sigma8.setter
    def sigma8(self, s8):
        if s8 > 0:
            self._sigma8 = s8
        else:
            raise ValueError('Sigma8 must be positive')

    @property
    def sigma_u(self):
        return self._sigma_u

    @property
    def grid_size(self):
        return self._grid_size

    @property
    def data(self):
        return self._data

    def grid_data(self, grid_size, use_true=False, mean='howlett'):
        self._grid_size = grid_size
        t0 = time.time()
        self._grid_vel_mu(use_true, mean=mean)
        t1 = time.time()
        print(f'{t1 - t0:.2f} s to grid data')
        self._compute_grid_window()
        print(f'{time.time() - t1:.2f} s to compute grid window')

    def _grid_vel_mu(self, use_true, mean='howlett'):
        data = copy.copy(self.data)

        if self.data_mask is not None:
            data.query(self.data_mask, inplace=True)

        if use_true:
            print('Use True val')
            key = 'vpec_true'
            nanerrmask = np.ones(len(data[key]), dtype='bool')
        else:
            key = 'vpec_est'
            nanerrmask = ~np.isnan(data['vpec_err'])
            nanerrmask &= data['vpec_err'] > 0

        nanmask = ~np.isnan(data[key])
        nanmask &= nanerrmask

        data = data[nanmask]

        print(f"Apply mask : {self.data_mask if self.data_mask is not None else 'No'}")
        print(f'N sn = {len(data)}')

        if use_true:
            err = np.zeros(len(data))
        else:
            err = data['vpec_err'].to_numpy()

        vel = data[key].to_numpy()

        if self.grid_size == 0:
            print('No grid')
            self.data_grid = {'ra': data['ra'].to_numpy(),
                              'dec': data['dec'].to_numpy(),
                              'r_comov': data['r_comov'].to_numpy(),
                              'z': data['zobs'].to_numpy(),
                              'vel': vel,
                              'err': err}
            return

        print('Create velocities grid')
        if mean == 'howlett':
            print('Use normal mean')
            mean = 0
        elif mean == 'weight':
            print('Use weighted mean')
            mean = 1
        else:
            raise ValueError("Mean should be 'howlett' or 'weight'")

        grid = nbf.grid_data(self.grid_size,
                             data['ra'].to_numpy(),
                             data['dec'].to_numpy(),
                             data['r_comov'].to_numpy(),
                             vel,
                             err,
                             use_true,
                             mean)

        z = np.linspace(0, self._data['zobs'].max() + 0.5, 1000)

        rcomo = self._cosmo.comoving_distance(z).value
        rcomo *= self._cosmo.h

        self.data_grid = {'ra': grid[0],
                          'dec': grid[1],
                          'r_comov': grid[2],
                          'z': np.interp(grid[2], rcomo, z),
                          'vel': grid[3],
                          'err': grid[4],
                          'nobj': grid[5]}

    def _compute_grid_window(self, n=1000):
        if self.grid_size == 0:
            self._grid_window = None
            return

        self._grid_window = nbf.compute_grid_window(self.grid_size,
                                                    self.pk[0],
                                                    n)

    def _compute_cov(self, kbin=None):
        if kbin is None:
            kbin = np.array([self.pk[0].min() * 0.999,
                             self.pk[0].max() * 1.1])
        if 'nobj' not in self.data_grid.keys():
            n_gals = None
        else:
            n_gals = self.data_grid['nobj']
        self.cov_cosmo = nbf.build_covariance_matrix(self.data_grid['ra'],
                                                     self.data_grid['dec'],
                                                     self.data_grid['r_comov'],
                                                     self.pk[0],
                                                     self.pk[1],
                                                     kbin,
                                                     grid_win=self._grid_window,
                                                     n_gals=n_gals)

    def get_log_like(self, x):
        fs8 = x[:-1]
        sig_v = x[-1]

        # Build cov matrix
        cov_matrix = np.zeros(self.cov_cosmo[0].shape)
        for i, f in enumerate(fs8):
            cov_matrix += self.cov_cosmo[i] * f**2
        if 'nobj' not in self.data_grid:
            nobj = 1.
        else:
            nobj = self.data_grid['nobj']

        diag_add = sig_v**2 / nobj
        diag_add += self.data_grid['err']**2
        cov_matrix += np.diag(diag_add)
        log_like = nbf.log_likelihood(self.data_grid['vel'], cov_matrix)
        return log_like

    def neg_log_like(self, x):
        return -self.get_log_like(x)

    def fit_iminuit(self, grid_size, use_true=False,
                    minos=False, fs8_lim=(0., None),
                    sigv_lim=(0., None), kbin=None,
                    mean='howlett'):

        if kbin is None:
            kbin = np.array([self.pk[0].min() * 0.999,
                             self.pk[0].max() * 1.1])

        print(f'Grid size = {grid_size}')
        print(f'kmin = {self.kmin}, kmax = {self.kmax}')
        # Run all neccessary function
        t0 = time.time()
        self.grid_data(grid_size, use_true=use_true, mean=mean)
        t1 = time.time()
        print(f'Grid data : {t1 - t0:.2f} seconds')
        self._compute_cov(kbin=kbin)
        t2 = time.time()
        print(f'Compute cosmo covariance : {t2 - t1:.2f} seconds')
        if len(kbin) - 1 > 1:
            name = [f"fs8_{i}" for i in range(len(kbin) - 1)]
            init = [0.5] * (len(kbin) - 1)
            limits = [fs8_lim] * (len(kbin) - 1)
        else:
            name = ["fs8"]
            init = [0.5]
            limits = [fs8_lim]

        name.append("sig_v")
        init.append(200.)
        limits.append(sigv_lim)

        m = iminuit.Minuit(self.neg_log_like, init, name=name)

        m.errordef = iminuit.Minuit.LIKELIHOOD
        m.limits = limits

        t3 = time.time()
        print(f'Init iminuit : {t3 - t2:.2f} seconds')
        print('Begin fit')
        m.migrad()
        t4 = time.time()
        print(f'Fit iminuit : {(t4 - t3) / 60:.2f} minutes')
        if minos:
            print('Compute minos error')
            m.minos()
            t5 = time.time()
            print(f'Minos error : {(t5 - t4) / 60:.2f} minutes')
        return m

    def plot_pk(self, **kwargs):
        plt.figure()
        plt.title("Power Spectrum")
        plt.plot(self.pk[0], self.pk[1], **kwargs)
        plt.xlabel('k [h Mpc$^-1$]')
        plt.ylabel('P(k)')
        plt.show()

    def plot_grid(self, **kwargs):
        plt.subplot(projection='mollweide')
        plt.title(f"Data grid, grid size = {self._grid_size}")
        ra = self.data_grid['ra']
        ra -= 2 * np.pi * (ra > np.pi)
        f = plt.scatter(ra, self.data_grid['dec'], c=self.data_grid['vpec'],
                        vmin=-1500, vmax=1500, **kwargs)
        plt.colorbar(f)
        plt.show()

    def plot_corr(self, **kwargs):
        sqrt_diag = np.sqrt(np.diag(self.cov_cosmo))
        corr_cosmo = self.cov_cosmo / sqrt_diag
        corr_cosmo = corr_cosmo.T / sqrt_diag
        plt.matshow(corr_cosmo, **kwargs)
        plt.title('Cosmo correlation matrix')
        plt.show()
