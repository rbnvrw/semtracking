from __future__ import (division, unicode_literals, print_function,
                        absolute_import)
import six
import numpy as np
import pandas as pd
from pims import plots_to_frame, Frame
from pims_nd2 import ND2_Reader

from scipy.optimize import curve_fit
import matplotlib as mpl
import matplotlib.pyplot as plt
from warnings import warn
from .algebraic import gauss
from .plot import (plot_h_histogram, plot_sphere_density,
                   plot3D_vesicle_features, scatter, plot_traj, annotate_max,
                   pair_slice, annotate_mip)
from .utils import (gen_nd2_paths, fancy_indexing, bin_column)
from .tracking import (SUFFIX)
from rdf import erdf_series, plot_rdf, plot_energy
from rdf.algebraic import dist_sphere
import clustertracking as ct
from mpl_toolkits.mplot3d import Axes3D
import os
import yaml
import pickle
import logging
logger = logging.getLogger(__name__)


def extract_pairs(f, separation, R, extra_columns=None):
    pos_columns = ['th', 'phi']
    if extra_columns is not None:
        pos_columns = ['th', 'phi'] + list(extra_columns)
    pairs = ct.find(f, separation, pos_columns=['x', 'y', 'z'])
    pairs = pairs[pairs.cluster_size == 2]

    result_cols = [p + '0' for p in pos_columns] + \
                  [p + '1' for p in pos_columns] + \
                  ['p0', 'p1', 'frame', 't', 'cluster']

    result = np.empty((pairs.cluster.nunique(), len(result_cols)))

    for i, (pair_id, pair) in enumerate(pairs.groupby('cluster')):
        p0, p1 = pair.particle.unique()
        result[i] = list(pair[pos_columns].values.ravel()) + \
                    [p0, p1, pair.frame.min(), pair['t'].mean(), pair_id]

    result = pd.DataFrame(result, columns=result_cols).set_index('cluster')

    result['s'] = dist_sphere(result[['th0', 'phi0']],
                              result[['th1', 'phi1']], R)

    for (p0, p1), pair in result.groupby(['p0', 'p1']):
        mask = (result['p0'] == p0) & (result['p1'] == p1)
        result.loc[mask, 'ds'] = pair['s'].diff()
       # result.loc[mask, 'v_inter'] = pair['s'].diff() / pair['t'].diff()
        result.loc[mask, 's'] = pd.rolling_mean(pair['s'], 2)

    return result


class VesicleFeatures(object):
    """ Class to combine feature and vesicle csv, analyze tracking quality,
    link features, determine on/off, calculate rdf."""
    def __init__(self, filenames, show=False):
        # initialize vesicle and feature dataframes
        v_columns = ['frame', 'xc', 'yc', 'zc', 'xr', 'yr', 'zr', 'file',
                     'frame_file']
        f_columns = ['frame', 'particle', 'x', 'y', 'z', 't', 'file',
                     'frame_file', 'x_px', 'y_px', 'z_px']

        # make sure filenames is an iterable
        if isinstance(filenames, six.string_types):
            if os.path.isfile(filenames):
                filenames = [filenames]
            elif os.path.isdir(filenames):
                filenames = [os.path.join(a, b)
                             for (a, b) in gen_nd2_paths(filenames)]

        next_first_frame = 0
        next_first_particle = 0
        f_collect = []
        v_collect = []
        first_frames = []
        mpps = []
        for i, filename in enumerate(filenames):
            # open files, v = vesicles, f = features
            with open(filename + '_params.yml') as yml:
                meta = yaml.load(yml)
                mpp = meta['mpp']
                length = meta['length']
                if 'ignore' in meta:
                    ignore = meta['ignore']
                else:
                    ignore = []
            v = pd.DataFrame.from_csv(filename + SUFFIX['v'])
            v['file'] = i
            # load features
            f = pd.DataFrame.from_csv(filename + SUFFIX['r'])
    
            if len(ignore) > 0:
                v.drop(ignore, inplace=True)
                f.drop(f['frame'].isin(ignore), inplace=True)
            v.reset_index(drop=False, inplace=True)
            f.reset_index(drop=True, inplace=True)
            f['file'] = i

            # define new frame index, but keep old one belonging to the file
            v.rename(columns={'frame': 'frame_file'}, inplace=True)
            v['frame'] = v['frame_file'] + next_first_frame
            f.rename(columns={'frame': 'frame_file'}, inplace=True)
            f['frame'] = f['frame_file'] + next_first_frame

            # define new particle index
            f['particle'] += next_first_particle

            # transform to um, rest of analysis is in real units
            v['xr'] *= mpp[2]
            v['yr'] *= mpp[1]
            v['zr'] *= mpp[0]
            v['xc'] *= mpp[2]
            v['yc'] *= mpp[1]
            v['zc'] *= mpp[0]
            
            # keep feature locations in pixels, for annotating convenience
            f.rename(columns={'x': 'x_px', 'y': 'y_px', 'z': 'z_px'},
                     inplace=True)
            f['z'] = f['z_px']*mpp[0]
            f['y'] = f['y_px']*mpp[1]
            f['x'] = f['x_px']*mpp[2]

            # append the features and vesicle data
            v_collect.append(v[v_columns])
            f_collect.append(f[f_columns])
            first_frames.append(next_first_frame)
            mpps.append(mpp)

            # increase frame and particle counters
            next_first_frame += length
            next_first_particle = f['particle'].max() + 1

        self.v = pd.concat(v_collect)
        self.f = pd.concat(f_collect)

        # make sure that v indexes with frame and f indexes uniquely
        self.v.set_index('frame', inplace=True, drop=False)
        self.f.reset_index(inplace=True, drop=True)
        self.f['frame'] = self.f['frame'].astype(int)
        self.f['particle'] = self.f['particle'].astype(int)

        #TODO drop vesicles that are not tracked properly
        #self.v.drop(~np.isfinite(f).all(1), inplace=True)

        self._len = next_first_frame
        self.show = show
        self.filenames = filenames
        self.first_frame = first_frames
        self.calibration = mpps
        self.mips = {}

        self.pair_separation = None

    def __len__(self):
        return self._len

    def recalibrate(self):
        if self.show:
            (self.v['xr'] / self.v['yr']).hist(bins=20)
            (self.v['zr'] / self.v['yr']).hist(bins=20)

        xy = (self.v['xr'] / self.v['yr']).mean()
        zy = (self.v['zr'] / self.v['yr']).mean()
        self.v['xr'] /= xy
        self.v['zr'] /= zy
        self.v['xc'] /= xy
        self.v['zc'] /= zy
        self.f['z'] /= zy
        self.f['x'] /= xy
        
        self.calibration = [[c[0] / zy, c[1], c[2] / xy]
                            for c in self.calibration]    
        return xy, zy

    def calc_R(self, window=10, max_ves_aspect=0.1, max_r_std=0.1):
        assert self.v.xr.std() < max_r_std
        assert self.v.yr.std() < max_r_std
        assert self.v.zr.std() < max_r_std
        self.v['Rxy'] = pd.rolling_mean(self.v.yr, window,
                                        min_periods=window/2, center=True)
        self.v['Rz'] = pd.rolling_mean(self.v.zr, window,
                                       min_periods=window/2, center=True)
        if self.show:
            self.v.Rxy.plot()
            self.v.Rz.plot()

        self.R = self.v['Rxy'].mean()
        np.testing.assert_allclose(self.v.Rxy, self.v.Rz, rtol=max_ves_aspect)

    def calc_relative_coords(self):
        f = self.f.join(self.v[['xc', 'yc', 'zc', 'Rxy', 'Rz']], on='frame')
        f['x_rel'] = f['x'] - f['xc']
        f['y_rel'] = f['y'] - f['yc']
        f['z_rel'] = f['z'] - f['zc']
        f['r'] = np.sqrt((f[['x_rel', 'y_rel', 'z_rel']]**2).sum(1))
        f['h'] = f['r'] - f['Rxy']
        f['phi'] = np.arctan2(f['y_rel'], f['x_rel'])
        f['th'] = np.pi/2 - np.arcsin(f['z_rel'] / f['r'])
        self.f = f

    def drop_off(self, min_length=10, max_h_var=0.1):
        grouped = self.f.groupby('particle')

        # first filter short trajectories
        count = grouped['particle'].count()
        p_drop = set(count.index[count < min_length])

        # then filter on h value fluctuation
        var = grouped['h'].var()
        p_drop = p_drop.union(set(var.index[var > max_h_var]))

        self.f = self.f[~self.f['particle'].isin(p_drop)].copy()

    def calc_displacements(self):
        particles = self.f.groupby(['particle'])
        self.f['dx'] = particles['x'].diff()
        self.f['dy'] = particles['y'].diff()
        self.f['dz'] = particles['z'].diff()
        self.f['dh'] = particles['h'].diff()
        self.f['dt'] = particles['t'].diff()

        # calculate total displacement X and tangent displacement.
        self.f['dX'] = np.sqrt((self.f[['dx', 'dy', 'dz']]**2).sum(1))
        self.f['dX_tangent'] = np.sqrt(self.f['dX']**2 - self.f['dh']**2)
        
        self.f['v_x'] = self.f['dx'] / self.f['dt']
        self.f['v_y'] = self.f['dy'] / self.f['dt']
        self.f['v_z'] = self.f['dz'] / self.f['dt']
        self.f['v_h'] = self.f['dh'] / self.f['dt']
        self.f['v_t'] = self.f['dX_tangent'] / self.f['dt']
        self.f['v_abs'] = self.f['dX'] / self.f['dt']
        self.f['D'] = self.f['dX_tangent'] / self.f['dt'] / 4

    def run(self, max_ves_displ_um, R_av_window, max_ves_aspect, max_R_std,
            lower_bound, upper_bound, persistence):
        self.calc_R(R_av_window, max_ves_aspect, max_R_std)
        self.calc_relative_coords()
        self.drop_on_h(lower_bound, upper_bound)
        self.calc_displacements()
        self.determine_on(persistence)

    def _get_file_index(self, t):
        for i in range(1, len(self.first_frame)):
            if self.first_frame[i] > t:
                return i - 1
            elif i == len(self.first_frame) - 1:
                return i

    def _get_file_name(self, t):
        i = self._get_file_index(t)
        return self.filenames[i]

    def _get_file_frame(self, t):
        i = self._get_file_index(t)
        return t - self.first_frame[i]

    def _get_file_mpp(self, t):
        i = self._get_file_index(t)
        return self.mpp[i]

    def frame(self, t):
        with ND2_Reader(self._get_file_name(t) + '.nd2') as im:
            im.bundle_axes = 'czyx'
            frame = im[self._get_file_frame(t)]
        return frame

    def bead_distance(self, bin_edges=None, angle=90):
        if bin_edges is None:
            bin_edges = np.arange(-2, 2, 0.02)

        assert angle > 0 and angle <= 90

        if angle < 90:
            f = self.f[abs(self.f['z_rel']) < self.f['Rxy']*np.sin(angle / 180 * np.pi)]
        else:
            f = self.f

        hist, _ = np.histogram(f['h'].values, bins=bin_edges)
        h = (bin_edges[1:] + bin_edges[:-1]) / 2

        p0 = [max(hist), h[np.argmax(hist)], 0.2]
        fit, _ = curve_fit(gauss, h, hist, p0=p0)
        _, mu, sigma = fit

        if self.show:
            plot_h_histogram(hist, bin_edges, fit)

        self.h_mean = mu
        self.h_std = sigma
        return mu, sigma

#    def refine_z(self, cutoff_distance=None):
#        if cutoff_distance is None:
#            cutoff_distance = 4*self.h_std
#        Rp = self.R + self.h_mean
#        xy2 = self.f.xrel.values**2 + self.f.yrel.values**2
#        xy2[xy2 > (Rp + cutoff_distance)**2] = np.nan  # kick out outliers
#        xy2[xy2 > Rp**2] = Rp**2  # make sure that the rest is inside xy
#        self.f['zrel_old'] = self.f['zrel']
#        self.f['zrel'] = np.sign(self.f.zrel_old) * np.sqrt(Rp**2 - xy2)
#        self.f['th_old'] = self.f['th']
#        self.f['th'] = np.arccos(self.f.zrel / self.f.r)

    def binaverage(self, data, ignore_nan=True, perarea=False, **plot_kwargs):
        if ignore_nan:
            mask = np.isfinite(data)
            hs, _, _ = np.histogram2d(self.f['phi'][mask],
                                      self.f['th'][mask],
                                      weights=data[mask], bins=self.bins)
        else:
            hs, _, _ = np.histogram2d(self.f['phi'], self.f['th'],
                                      weights=data, bins=self.bins)
        result = hs/self.bincount
        if self.show:
            try:
                label = {'h': r'$h [\mu m]$',
                         'r': r'$r [\mu m]$',
                         'v_x': r'$v_x [\mu m s^{-1}]$',
                         'v_y': r'$v_y [\mu m s^{-1}]$',
                         'v_z': r'$v_z [\mu m s^{-1}]$',
                         'v_h': r'$v_\bot [\mu m s^{-1}]$',
                         'v_t': r'$v_\parallel [\mu m s^{-1}]$',
                         'v_abs': r'$|v| [\mu m s^{-1}]$',
                         'D': r'$D [\mu m^2 s^{-1}]$'}[data.name]
            except KeyError:
                label = data.name
            plot_sphere_density(result, label=label, **plot_kwargs)
        return result
        
    def plot_velocity_distribution(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)

        stdev = self.f[['v_x', 'v_y', 'v_z']].std().mean()
        edges = np.linspace(-5*stdev, 5*stdev, max(int(len(self.f)/100), 10))
        for col in ['v_x', 'v_y', 'v_z']:
            hist, _ = np.histogram(self.f[col], bins=edges, normed=True)
            ax.step(edges[1:], hist, label=col)
        ax.set_xlim(-4*stdev, 4*stdev)
        ax.grid()
        ax.legend()
        ax.set_xlabel('$v [\mu m s^{-1}]$')

    def binarea(self, R=None):
        if R is None:
            R = self.R
        th_area = np.cos(self.bins_th[:-1]) - np.cos(self.bins_th[1:])
        phi_area = np.diff(self.bins_phi)
        return R**2 * th_area[np.newaxis, :] * phi_area[:, np.newaxis]
        
    @fancy_indexing
    def pairs(self, t=None, separation=None):
        if separation is None:
            raise ValueError('Please provide separation')
        if t is not None:
            f = self.f[self.f.frame.isin(t)]
        else:
            f = self.f
        return extract_pairs(f, separation, self.R)

    @fancy_indexing
    def annotate(self, t, mask=None, **kwargs):
        pos_columns = ['z_px', 'y_px', 'x_px']
        _kwargs = dict(width=512, size=6, fontsize=18, width_mpp=60,
                       outline='yellow', particle_labels=True)
        _kwargs.update(kwargs)
        if mask is None:
            mask = [True] * len(self.f)
        if len(t) > 1:
            result = []
            for i in t:
                result.append(self.annotate(i, mask, **_kwargs))
            return Frame(result)
        else:
            i = self._get_file_index(t[0])
            if i not in self.mips:
                with open(self._get_file_name(t[0]) + '_mip.p', 'rb') as fl:
                    self.mips[i] = pickle.load(fl)
            mask_file = mask & (self.f['file'] == i)
            return annotate_mip(self.f.loc[mask_file],
                                self.mips[i][self._get_file_frame(t[0])],
                                self.v, pos_columns=pos_columns,
                                frame_column='frame_file', **_kwargs)

    def annotate_mpl(self, t, proj='xy', **kwargs):
        return annotate_max(self.f_px[self.f.frame == t], self.frame(t), proj)

    def plot3D(self, features, t, tracks=True, ax=None, display='NIS'):
        pos_columns = ['xrel', 'yrel', 'zrel']
        t_column = 'frame'
        if isinstance(t, slice):
            t = np.arange(t.stop)[t]

        if not hasattr(t, '__iter__'):
            return plot3D_vesicle_features(features, self.R, tracks, t, ax,
                                           t_column, pos_columns, display)
        else:
            f = features[features.frame.isin(t)]
            plots = [None] * len(t)
            for i, ti in enumerate(t):
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                plots[i] = plot3D_vesicle_features(f, self.R, tracks, ti,
                                                   ax, t_column,
                                                   pos_columns, display).figure
            return plots_to_frame(plots, close_fig=True)


    def plot_traj(self, f, delay=0.03, tracks=10, output=False):
        delay = 0.01
        if output:
            plots = [None] * f['frame'].nunique()

        for i, (j, f_part) in enumerate(f.groupby('frame')):
            if output:
                fig = plt.figure()
                ax = fig.add_subplot(111)
            else:
                plt.cla()
                ax = plt.gca()
            ax = scatter(f_part, ax=ax)
            if tracks > 0:
                plot_traj(f[(f.frame <= j) & (f.frame > j - tracks)],
                           ax=ax, label=True)
            ax.set_title('t={0:.2f} s'.format(f_part['t'].mean()))
            ax.grid()
            if output:
                plots[i] = ax.figure
            else:
                plt.show()
                plt.pause(delay)
        if output:
            return plots_to_frame(plots, close_fig=True)
        else:
            plt.close()

    def pair_movie(self, pairs):
        result = []
        for (p0, p1), pair in pairs.groupby(['p0', 'p1']):
            for i, plot in pair.iterrows():
                t = int(plot[['frame']].values[0])
                center = self.v.loc[self.v.frame == t, ['zc', 'yc', 'xc']].values
                if len(center) > 0:
                    result.append(pair_slice(plot, center[0], self.R*2.2,
                                             self.mpp, self.frame(t)))

        if result[0].ndim == 3:
            result = np.swapaxes(result, 0, 1)
        else:
            result = np.array(result)
        return Frame(result)

    def fit_maxwell(self, f, bw=0.2):
        def maxwell(x, A, c):
            return A*x**2*np.exp(-c*x**2)
        bins = np.arange(0, f.v_tangent.max(), bw)
        y, edges = np.histogram(f.v_tangent, bins=bins, normed=True)
        x = (edges[1:] + edges[:-1])/2

        params, _ = curve_fit(maxwell, x, y)

        plt.bar(edges[:-1], y, width=edges[1]-edges[0])
        plt.plot(x, maxwell(x, *params), color='r', linewidth=2)
        return params

    def rdf(self, hist_mode='hist', bw=0.2, min_dist=0.1, max_dist=None):
        R = self.R + self.h_mean
        if max_dist is None:
            max_dist = R*np.pi
        bins, hist = erdf_series(self.f, hist_mode, 'sphere', min_dist,
                                 max_dist, bw, R=R)
        if self.show:
            plot_rdf(bins, hist)
            plot_energy(bins, hist)
        return bins, hist
       
    @fancy_indexing
    def __getitem__(self, key):
        return self.f[self.f.frame.isin(key)]
        
    def extract_pairs(self, separation=None, bins=None, minN=5):
        extra_cols = ['dt', 'dh', 'dX']
        if separation is None:
            separation = self.pair_separation
        if not hasattr(self, 'pairs') or separation != self.pair_separation:
            self.pairs = extract_pairs(self.f, separation, self.R, extra_cols)
        self.pair_separation = separation

        if bins is None:
            bins = np.linspace(0, separation, 50)

        bin_column(self.pairs, bins, 's', 's_binned')

        to_drop = []
        [to_drop.extend([col + '0', col + '1']) for col in extra_cols]
        mean_cols = ['s', 'ds'] + to_drop

        means = self.pairs.groupby('s_binned')[mean_cols].mean()
        stdevs = self.pairs.groupby('s_binned')[mean_cols].std()
        N = self.pairs.groupby('s_binned')['s'].count()


        to_drop = []
        [to_drop.extend([col + '0', col + '1']) for col in extra_cols]

        for col in extra_cols:
            means[col] = (means[col + '0'] + means[col + '1']) / 2
        
        means.drop(to_drop, axis=1, inplace=True)
        means['N'] = N
        for col in ['s', 'ds']:
            means[col + '_std'] = stdevs[col] / np.sqrt(N - 1)
        for col in extra_cols:
            means[col + '_std'] = np.sqrt(stdevs[col + '0']**2 + stdevs[col + '1']**2) / np.sqrt(2*N - 1)
            
        means = means[means['N'] > minN]
        return means


    def plot_v_inter(self, pairs, bins):
        def map_bin(x, bins):
            bin_indices = np.digitize([x], bins, right=True)[0]
            centers = np.concatenate([[np.nan], (bins[1:] + bins[:-1])/2,
                                      [np.nan]])
            return centers[bin_indices]
        
        pairs['binned'] = pairs['s'].apply(map_bin, bins=np.arange(0, 5, 0.1))
        pairs.groupby('binned')['v_inter'].mean().plot()


#    @property
#    def f_appear(self):
#        return self.f.loc[((self.f.frames_before == 0) &
#                           (self.f.h < self.h_max) &
#                           (self.f.frame != self.first_frame) &
#                           self.f.on)]
#
#    @property
#    def f_lost(self):
#        return self.f.loc[((self.f.frames_after == 0) &
#                           (self.f.h < self.h_max) &
#                           (self.f.frame != self.last_frame) &
#                           self.f.on)]
#
#    @property
#    def f_attach(self):
#        return self.f.loc[((self.f.frames_before == 0) &
#                           (self.f.h >= self.h_max) &
#                           self.f.on)]
#
#    @property
#    def f_escape(self):
#        return self.f.loc[((self.f.frames_after == 0) &
#                           (self.f.h >= self.h_max) &
#                           self.f.on)]

    @property
    def grid(self):
        return (len(self.bins_phi) - 1, len(self.bins_th) - 1)
    @grid.setter
    def grid(self, value):
        """Number of bins in (phi, th) directions."""
        self.bins_phi = np.linspace(-np.pi, np.pi, value[0] + 1)
        self.bins_th = np.linspace(0, np.pi, value[1] + 1)
        self.bincount, _, _ = np.histogram2d(self.f['phi'], self.f['th'],
                                             bins=self.bins)
        if self.show:
            plot_sphere_density(self.bincount / self.binarea(self.R),
                                label=r'$n [\mu m^{-2}]$')

    @property
    def bins(self):
        return (self.bins_phi, self.bins_th)
