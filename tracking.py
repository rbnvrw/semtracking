from __future__ import (division, unicode_literals, print_function,
                        absolute_import)
import six
import trackpy as tp
import numpy as np
import pandas as pd
import itertools as it
from pims import Frame, to_rgb
from pims_nd2 import ND2_Reader

from scipy.ndimage.filters import correlate1d
import matplotlib as mpl
import matplotlib.pyplot as plt
from warnings import warn
from .algebraic import gauss, ellipse_perimeter
from .plot import (annotate3d_max, create_mip, annotate_mip, annotate_ellipse)
from .radialprofile import locate_ellipsoid
from .utils import (image_md, to_odd_int, update_meta,
                    gen_nd2_paths, export, crop_pad, play_file,
                    fancy_indexing, guess_pos_columns, exec_cluster,
                    FramesFunction)
from rdf import erdf_series, plot_rdf, auto_box_3d
from rdf.algebraic import dist_sphere
import clustertracking as ct
from mpl_toolkits.mplot3d import Axes3D
import os
import yaml
from scipy.spatial import cKDTree
import pickle
import logging
logger = logging.getLogger(__name__)


SUFFIX = dict(f='_f.txt', r='_r.txt', v='_v.txt', mip='_mip.p',
              contour='_contour.p', params='_params.yml', mipavi='_mip.avi',
              vesavi='_vesicle.avi')

def lowpass(image, noise_size, threshold=0):
    noise_size = tp.utils.validate_tuple(noise_size, image.ndim)
    result = np.array(image, dtype=np.float)
    for (axis, size) in enumerate(noise_size):
        correlate1d(result, tp.masks.gaussian_kernel(size, 4), axis,
                    output=result, mode='constant', cval=0.0)
    return np.where(result > threshold, result, 0)

def locate(frame, v_channel, v_n_xy, v_n_xz, v_rad_range,
           v_maxfit_size, v_spline_order, v_threshold,
           f_crop_margin, f_channel, f_lowpass_size, f_threshold,
           f_diameter, f_minmass, f_separation, mip_shape=None,
           mip_spacing=5):
    # Locate vesicle
    vesicle, contour = locate_ellipsoid(frame[v_channel], v_n_xy, v_n_xz,
                                        v_rad_range, v_maxfit_size,
                                        v_spline_order, v_threshold)
    v_success = not np.any(np.isnan(vesicle))
    vesicle['frame'] = frame.frame_no

    # Crop image, only when vesicle was found
    if v_success:
        center = vesicle[['zc', 'yc', 'xc']].values
        radius = vesicle[['zr', 'yr', 'xr']].values

        origin = list(np.round(center - radius - f_crop_margin).astype(int))
        shape = list(np.round(2*(f_crop_margin + radius)).astype(int))
        image_shape = frame.shape[1:]
        for i in range(3):
            if origin[i] < 0:
                shape[i] = shape[i] + origin[i]
                origin[i] = 0
            elif origin[i] > image_shape[i]:
                shape[i] = 0
                origin[i] = image_shape[i]
            elif shape[i] > image_shape[i] - origin[i]:
                shape[i] = image_shape[i] - origin[i]

        # for safety: only crop when the area is larger than 2x diameter
        if np.all([s > (d * 2) for (s, d) in zip(shape, f_diameter)]):
            slicer = [slice(o, o+s) for (o, s) in zip(origin, shape)]
            frame_crop = frame[f_channel, slicer[0], slicer[1], slicer[2]]
        else:
            frame_crop = frame[f_channel]
            origin = [0.0, 0.0, 0.0]
            center = None
    else:
        frame_crop = frame[f_channel]
        origin = [0.0, 0.0, 0.0]
        center = None

    # Lowpass filter
    frame_f = lowpass(frame_crop, f_lowpass_size, f_threshold)

    # Locate features
    features = tp.locate(frame_f, f_diameter, minmass=f_minmass, maxsize=None,
                         separation=f_separation,
                         preprocess=False, filter_before=False, engine='numba')
    features['z'] += origin[0]
    features['y'] += origin[1]
    features['x'] += origin[2]
    features['frame'] = frame.frame_no

    # Add 't' column. Trackpy uses 0.5 as origin so add 0.5 to index z.
    z_indexes = np.round(features['z'] + 0.5).astype(int)
    ts = frame.metadata['t_ms'][v_channel] / 1000.
    features['t'] = np.nan
    features['t'] = ts[np.clip(z_indexes, 0, len(ts) - 1)]

    # Create MIP for quick annotation
    if v_success and (mip_shape is not None):
        mip = create_mip(frame, center, mip_shape, mip_spacing)
    else:
        mip = None

    logger.info("Frame %d: %d features; vesicle success: %s",
                frame.frame_no, len(features), v_success)
    return vesicle, features, contour, mip


def identify_duplicates(f, separation, pos_columns):
    mass = f['mass'] 
    duplicates = cKDTree(f[pos_columns]/separation, 30).query_pairs(1)
    to_drop = []
    for p0, p1 in duplicates:
        if p0 in to_drop or p1 in to_drop:
            continue  # pair does not exist anymore; skip
        # Drop the dimmer one.
        m0, m1 = mass.iloc[p0], mass.iloc[p1]
        if m0 < m1:
            to_drop.append(p0)
        else:
            to_drop.append(p1)
    return f.index[to_drop]


def eliminate_duplicates(f, separation, pos_columns, inplace=False):
    to_drop = []
    for i, f_frame in f.groupby('frame'):
        to_drop.extend(list(identify_duplicates(f_frame, separation, pos_columns)))
    return f.drop(to_drop, inplace=inplace)


def refine_single(f, frames, diameter, fit_function=None,
                  pos_columns=None, size_columns=None):
    """ This function refines all single features in f using a fit function."""
    if pos_columns is None:
        pos_columns = guess_pos_columns(f)
    f = f[f.cluster_size == 1].copy()

    if fit_function is not None:
        pass
    elif len(pos_columns) == 3 and 'size' in f:
        fit_function = 'gauss3D_varInt'
    elif len(pos_columns) == 3:
        fit_function = 'gauss3D_a_varInt'
    elif len(pos_columns) == 2 and 'size' in f:
        fit_function = 'gauss2D_varInt'
    elif len(pos_columns) == 2:
        fit_function = 'gauss2D_a_varInt'
    else:
        raise ValueError('Unable to guess fit_function')

    result = []
    for frame_no, f_frame in f.groupby('frame'):
        result.append(ct.refine(f_frame, frames[frame_no], diameter,
                                fit_function, None, pos_columns,
                                size_columns, False, False))
    return pd.concat(result)


def locate_cluster(f_initial, f_background, frame, diameter, separation,
                   search_range, noise_size, threshold, fit_function,
                   fit_params, pos_columns, size_columns):
    radius = [d // 2 for d in diameter]
    f_new = ct.fit(f_initial, frame, diameter, separation, fit_function,
                   fit_params, pos_columns, size_columns, False, False)

    f_new['fail'] = False
    # check if fit succeeded
    for i, (cluster_id, f_cluster) in enumerate(f_new.groupby('cluster')):
        if not np.all(f_cluster['gaussian']):
            # fit failed, slice region around cluster
            im, origin = ct.slice_image(f_cluster[pos_columns].values, frame,
                                        search_range)
            # accumulate coords that are in the background
            ignore_coords = (f_new[(f_new['cluster'] != cluster_id) &
                             f_new['gaussian']])
            if f_background is not None:
                ignore_coords = pd.concat([f_background, ignore_coords])
            if ignore_coords.shape[0] > 0:
                im = ct.mask_image(ignore_coords[pos_columns].values, im,
                                   radius, origin, True)

            im = lowpass(im, noise_size, threshold)
            im = (im / np.max(im) * 255).astype(np.uint8)
            dilation_radius = [max(d - 1, 1) for d in radius]
            dilation_radius_list = list(it.product(*[list(range(n, 0, -1)) for n in dilation_radius]))
            dilation_radius_sorting = np.argsort(np.sum(dilation_radius_list,
                                                        axis=1))
            fail_coords = None
            for di in reversed(dilation_radius_sorting):
                dilation_radius = dilation_radius_list[di]
                coords = tp.local_maxima(im, tuple(dilation_radius))
                if len(coords) == len(f_cluster):
                    break
                if ((len(f_cluster) > 1) and (fail_coords is None) and
                    (len(coords) == len(f_cluster) - 1)):
                    new_coord = coords[-1].copy()
                    new_coord[2] += 1
                    fail_coords = np.concatenate([coords, [new_coord]],
                                                 axis=0)
            if len(coords) == len(f_cluster):
                f_cluster_new = f_cluster.copy()
                f_cluster_new[pos_columns] = coords.astype(np.float) + origin
                f_new.update(f_cluster_new)
            elif fail_coords is not None:
                f_cluster_new = f_cluster.copy()
                f_cluster_new[pos_columns] = fail_coords.astype(np.float) + origin
                f_new.update(f_cluster_new)
            else:
                f_new.loc[f_new['cluster'] == cluster_id, 'fail'] = True

    return f_new


def zip_locate(f_initial, frames, diameter, separation, search_range,
               noise_size, threshold, fit_function, fit_params=None,
               pos_columns=None, size_columns=None, refine=True):
    """Finds features, each time using the previous feature locations as
    initial fit parameters. When refine==True, each frame is passed through
    fit_cluster again to fit gaussian in a better fitregion, and to identify
    clusters properly.
    You can put refine==False if you want to pass it through ct.fit later.
    """
    if pos_columns is None:
        pos_columns = guess_pos_columns(f_initial)
    f = [f_initial[list(pos_columns) + ['particle']]]
    for frame in frames:
        # try to find particle locations based on previous one
        f_new = locate_cluster(f[-1], None, frame, diameter, separation,
                               search_range, noise_size, threshold,
                               fit_function, fit_params, pos_columns,
                               size_columns)

        if refine:
            ct.fit(f_new, frame, diameter, separation, fit_function,
                   fit_params, pos_columns, size_columns, True, False)
            # Add 't' column. ct.fit uses 0.5 as origin so add 0.5 to index z.
            z_indexes = np.round(f_new['z_px'] + 0.5).astype(int)
            ts = frame.metadata['t_ms'] / 1000.
            f_new['t'] = np.nan
            f_new['t'] = ts[np.clip(z_indexes, 0, len(ts) - 1)]
        
        # Add frame column.
        f_new['frame'] = frame.frame_no

        f.append(f_new)

    result = pd.concat(f[1:]).reset_index(drop=True)
    result = result[~result['fail'].astype(bool)]
    result.drop('fail', axis=1, inplace=True)
    return result


def combine_f(f_old, f_new, separation, pos_columns=None):
    """Overwrites features in f_old with features in f_new. Features in f_old
    that will be deleted are specified by separation: when features in f_old
    are closer than separation to a feature in f_new, the feature in f_old is
    dropped."""
    if pos_columns is None:
        pos_columns = guess_pos_columns(f_old)

    new_coords = f_new[pos_columns] / separation

    to_drop = []
    for i, f_frame in f_old.groupby('frame'):
        kdt = cKDTree(f_frame[pos_columns] / separation)
        for i in f_new[f_new.frame == i].index:
            kdt_index = kdt.query_ball_point(new_coords.loc[i], 1)
            to_drop.extend(f_frame.index[kdt_index])

    result = pd.concat([f_old.drop(to_drop), f_new])
    result.reset_index(drop=True, inplace=True)
    return result


def track_multicore(path, skip=True):
    def locate_multicore(fn):
        sufs = ['f', 'v', 'params', 'mip']
        if not np.all([os.path.isfile(fn + SUFFIX[suf]) for suf in sufs]):
            Track(fn).run(save=True)
        sufs = ['mipavi', 'vesavi']    
        if not np.all([os.path.isfile(fn + SUFFIX[suf]) for suf in sufs]):
            f = Features(fn)
            f.ignore = None
            f.identify_drift()
            f.export_annotate()   
            f.export_vesicle()

    exec_cluster(locate_multicore, gen_nd2_paths(path))

def repeat_zip(filename):
    if os.path.isfile(filename + SUFFIX['r']):
        os.rename(filename + SUFFIX['r'], filename + SUFFIX['r'] + '.bak')
    f = Features(filename)
    f.identify_drift()
    f.get_fit_params()
    _zip_kwargs = dict(inplace=True, refine=True)
    for zip_kwargs in f._cluster_zipped:
        _zip_kwargs.update(zip_kwargs)
        f.zip_locate(**_zip_kwargs)
    return f
    
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

class Track(object):
    def __init__(self, filename):
        self.filename = filename
        self._len, self.mpp, self.fps = image_md(filename)

        # Load previous params if they exist
        if os.path.isfile(self.filename + SUFFIX['params']):
            with open(self.filename + '_params.yml') as yml:
                meta = yaml.load(yml)
            self.f_kwargs = {key[2:]: meta[key] for key in meta if key[:2] == 'f_'}
            self.v_kwargs = {key[2:]: meta[key] for key in meta if key[:2] == 'v_'}
            if 'refine' in self.v_kwargs:
                del self.v_kwargs['refine']
            if 'tol' in self.v_kwargs:
                del self.v_kwargs['tol']
            if 'n' in self.v_kwargs:
                del self.v_kwargs['n']
            self.mip_shape = meta['mip_shape']
            self.mip_margin = 0.2
            self.mip_spacing = meta['mip_spacing']
            self.link_kwargs = dict(memory=meta['memory'],
                                    search_range=meta['search_range'])
        else:
            # Load some default values
            self.v_kwargs = dict(channel=1, n_xy=None, n_xz=None,
                                 rad_range=None, maxfit_size=2, spline_order=3,
                                 threshold=0.1)
            self.f_kwargs = dict(channel=0, lowpass_size=0.5,
                                 minmass=100000, threshold=100)
            self.diameter_um = (5.5, 1.9, 2.1)
            self.separation_um = (0.5, 0.5, 0.5)
    
            self.link_kwargs = dict(memory=1)
            self._D_sigmas = 5
            self.expected_D = 0.5
    
            self.mip_shape = None
            self.mip_margin = 0.2
            self.mip_spacing = 5

    def __len__(self):
        return self._len

    @property
    def diameter_um(self):
        return tuple([d * mpp for (d, mpp) in zip(self.f_kwargs['diameter'],
                                                  self.mpp)])
    @diameter_um.setter
    def diameter_um(self, value):
        diameter = list([to_odd_int(d / mpp)
                         for (d, mpp) in zip(value, self.mpp)])
        self.f_kwargs['diameter'] = diameter
        self.f_kwargs['crop_margin'] = [d * 2 for d in diameter]

    @property
    def separation_um(self):
        return tuple([s * mpp for (s, mpp) in zip(self.f_kwargs['separation'],
                                                  self.mpp)])
    @separation_um.setter
    def separation_um(self, value):
        separation = list([max(s / mpp, 1)
                           for (s, mpp) in zip(value, self.mpp)])
        self.f_kwargs['separation'] = separation

    @property
    def minmass(self):
        return self.f_kwargs['minmass']
    @minmass.setter
    def minmass(self, value):
        self.f_kwargs['minmass'] = int(value)

    @property
    def expected_D(self):
        search_range = self.link_kwargs['search_range']
        return search_range**2 * self.fps / (2 * self._D_sigmas)
    @expected_D.setter
    def expected_D(self, value):
        search_range = float(np.sqrt(2*self._D_sigmas*value/self.fps))
        self.link_kwargs['search_range'] = search_range
        
    @property
    def threshold(self):
        return self.f_kwargs['threshold']
    @threshold.setter
    def threshold(self, value):
        if value is None:
            value = 0
        self.f_kwargs['threshold'] = value 

    def run(self, n=None, save=False):
        if save:
            meta = self.filename + SUFFIX['params']
        else:
            meta = None
        self.prime()
        self.batch_locate(n, meta)
        self.link()

        if save:
            update_meta(meta, self.link_kwargs)
            self.save_f()
            self.save_v()
            self.save_contours()
            if (hasattr(self, 'mips') and 
                self.mips is not None and 
                len(self.mips) > 1):
                self.save_mip()

    @fancy_indexing
    def run_mip(self, n=None, save=False):
        margin = self.mip_margin
        mip_spacing = self.mip_spacing
        mip_shape = self.mip_shape
        v = pd.DataFrame.from_csv(self.filename + '_v.txt')
        
        if mip_shape is None:
            v_guess = v.loc[:4]
            if np.any(np.isnan(v_guess[['zc', 'yc', 'xc', 'zr', 'yr', 'xr']])):
                raise ValueError('Unable to calculate crop area size')
            else:
                radius = v[['zr', 'yr', 'yr']].mean().values.astype(np.float64)
                mip_shape = [int(round(r * 2 * (1 + margin))) for r in radius]  

        mips = {}
        with ND2_Reader(self.filename + '.nd2') as frames:
            frames.bundle_axes = 'czyx'
            if n is None:
                n = slice(len(frames))
            for frame in frames[n]:
                vesicle = v.loc[frame.frame_no]
                if len(vesicle) > 0:
                    center = vesicle[['zc', 'yc', 'xc']].values.astype(np.float64)
                    mips[frame.frame_no] = create_mip(frame, center, mip_shape,
                                                      mip_spacing)

        self.mips = mips
        if save:
            update_meta(self.filename + SUFFIX['params'],
                        dict(mip_spacing=mip_spacing, mip_margin=margin,
                             mip_shape=mip_shape))
            self.save_mip()

    @fancy_indexing
    def prime(self, n=None, margin=0.2):
        """Primes the tracker by finding out the size of vesicle to expect"""
        with ND2_Reader(self.filename + '.nd2',
                        channel=self.v_kwargs['channel']) as frames:
            frames.bundle_axes = 'zyx'
            if n is None:
                n = slice(min(5, len(frames)))
            kwargs = self.v_kwargs.copy()
            del kwargs['channel']
            v = []
            for frame in frames[n]:
                vesicle, _ = locate_ellipsoid(frame, **kwargs)
                v.append(vesicle)

        v = pd.DataFrame(v)
        if np.any(np.isnan(v[['zc', 'yc', 'xc', 'zr', 'yr', 'xr']])):
            raise ValueError('Unable to prime tracker.')
            self.mip_shape = None  # don't do mip
            self.v_kwargs['n_xy'] = None  # let radialprofile decide n_xy
            self.v_kwargs['n_xz'] = None  # let radialprofile decide n_xz
        else:
            radius = v[['zr', 'yr', 'xr']].mean().values
            self.mip_shape = [int(round(r * 2 * (1 + margin))) for r in radius]
            self.mip_shape[2] = self.mip_shape[1]  # make it square
            self.v_kwargs['n_xy'] = int(ellipse_perimeter(radius[1], radius[2]))
            self.v_kwargs['n_xz'] = int(ellipse_perimeter(radius[0], radius[2]))
        return self.mip_shape, self.v_kwargs['n_xy'], self.v_kwargs['n_xz']

    @fancy_indexing
    def batch_locate(self, n, meta=None):
        locate_kwargs = dict(mip_shape=self.mip_shape,
                             mip_spacing=self.mip_spacing)
        locate_kwargs.update({'f_' + key: self.f_kwargs[key] for key in self.f_kwargs})
        locate_kwargs.update({'v_' + key: self.v_kwargs[key] for key in self.v_kwargs})
        if meta is not None:
            timestamp = pd.datetime.utcnow().strftime('%Y-%m-%d-%H%M%S')
            metadata = dict(timestamp=timestamp,
                            source=os.path.abspath(self.filename),
                            mpp=self.mpp,
                            fps=self.fps,
                            length=self._len)
            metadata.update(locate_kwargs)
            with open(meta, "w") as yml:
                yaml.dump(metadata, yml)      

        v = []
        mips = {}
        f = []
        self.contours = {}
        with ND2_Reader(self.filename + '.nd2') as frames:
            frames.bundle_axes = 'czyx'
            if n is None:
                n = slice(len(frames))
            for frame in frames[n]:
                vesicle, features, r, mip = locate(frame, **locate_kwargs)
                f.append(features)
                v.append(vesicle)
                self.contours[frame.frame_no] = r
                if locate_kwargs['mip_shape'] is not None:
                    mips[frame.frame_no] = mip

        self.f = pd.concat(f).reset_index(drop=True)
        self.v = pd.DataFrame(v)
        self.v['frame'] = self.v['frame'].astype(int)
        self.v.set_index('frame', drop=True, inplace=True)
        if locate_kwargs['mip_shape'] is not None:
            self.mips = mips

    def link(self, meta=None):
        pos_columns = ['zum', 'yum', 'xum']
        self.f['zum'] = self.f['z'] * self.mpp[0]
        self.f['yum'] = self.f['y'] * self.mpp[1]
        self.f['xum'] = self.f['x'] * self.mpp[2]

        tp.link_df(self.f, pos_columns=pos_columns, **self.link_kwargs)
        self.f['particle'] = self.f['particle'].astype(int)

    def save_f(self):
        columns = ['frame', 'particle', 'x', 'y', 'z', 't', 'mass']
        if 'size' in self.f:
            columns += ['size']
        if 'size_x' in self.f:
            columns += ['size_x', 'size_y', 'size_z']
        self.f[columns].to_csv(self.filename + SUFFIX['f'])

    def save_v(self):
        self.v.to_csv(self.filename + SUFFIX['v'])

    def save_mip(self):
        if self.mips is None:
            raise AttributeError('mips have not been generated')
        if os.path.isfile(self.filename + SUFFIX['mip']):
            os.remove(self.filename + SUFFIX['mip'])
        with open(self.filename + SUFFIX['mip'], 'wb') as mipdump:
            pickle.dump(self.mips, mipdump)

    def save_contours(self):
        if os.path.isfile(self.filename + SUFFIX['contour']):
            os.remove(self.filename + SUFFIX['contour'])
        with open(self.filename + SUFFIX['contour'], 'wb') as mipdump:
            pickle.dump(self.contours, mipdump)        

    def frame(self, t):
        with ND2_Reader(self.filename + '.nd2') as im:
            im.bundle_axes = 'czyx'
            frame = im[t]
        return frame

    def annotate(self, t):
        return annotate3d_max(self.f[self.f.frame == t], self.frame(t))


class Features(object):
    """Class to open feature csv and analyze tracking quality.
    Example:
    f = Features(filename)
    f.play_annotate()
    f.identify_drift()
    f.get_fit_params()
    f.zip_locate(start, stop, particles, inplace=True, refine=False)
    f.refine_clusters(inplace=True)
    f.link(search_range, memory)
    f.category = ''
    f.save()
    f.export_annotate(suffix='_refined.avi')    
    """
    def __init__(self, filename, show=False, load_refined=False):
        # load features
        if load_refined:
            self.f = pd.DataFrame.from_csv(filename + SUFFIX['r'])
        else:
            self.f = pd.DataFrame.from_csv(filename + SUFFIX['f'])

        if len(self.f) == 0:
            raise ValueError('No particles found')

        self.f.reset_index(drop=True, inplace=True)

        # only do frame here, particle, cluster, cluster_size may contain NaN        
        self.f['frame'] = self.f['frame'].astype(int)

        if 'size' in self.f:
            self.f['size_x'] = self.f['size']
            self.f['size_y'] = self.f['size']
            self.f['size_z'] = self.f['size']

        if 'gaussian' not in self.f:
            self.f['gaussian'] = False

        # load vesicles
        self.v = pd.DataFrame.from_csv(filename + SUFFIX['v'])

        if len(self.v) == 0:
            raise ValueError('No vesicles found')
        if 'comments' in self.v:
            del self.v['comments']
            
        # load metadata
        with open(filename + SUFFIX['params']) as yml:
            self.meta = yaml.load(yml)
        self.fps = self.meta['fps']
        self.mpp = self.meta['mpp']
        self._len = self.meta['length']
        self.filename = filename
        self.show = show
        
        if 'category' in self.meta:
            self.category = self.meta['category']
        else:
            self.category = None

        if 'ignore' in self.meta:
            self.ignore = set(self.meta['ignore'])
        else:
            self.ignore_reset()

        if 'cluster' in self.meta:
            self._cluster = self.meta['cluster']
            del self.meta['cluster']
        else:
            self._cluster = set()

        if 'clustering' in self.meta:
            del self.meta['clustering']

        if 'cluster_zipped' in self.meta:
            self._cluster_zipped = self.meta['cluster_zipped']
        else:
            self._cluster_zipped = []

        if 'cluster_refined' in self.meta:
            self._cluster_refined = self.meta['cluster_refined']
        else:
            self._cluster_refined = None

        with open(self.filename + '_mip.p', 'rb') as fl:
            self.mips = pickle.load(fl)

        # make um columns
        self.f['z_um'] = self.f['z']*self.mpp[0]
        self.f['y_um'] = self.f['y']*self.mpp[1]
        self.f['x_um'] = self.f['x']*self.mpp[2]
        self.f.rename(columns={'x': 'x_px', 'y': 'y_px', 'z': 'z_px'},
                      inplace=True)

        self._calc_D()

    def __len__(self):
        return self._len

    def _calc_D(self, minD=0.05):
        pos_columns = ['z_um', 'y_um', 'x_um']
        ndim = len(pos_columns)
        grouped = self.f.groupby('particle')
        self.f['dt'] = grouped['t'].diff()
        self.f['dX2'] = (grouped[pos_columns].diff()**2).sum(1)
        for i, particle in grouped:
            D = (particle['dX2'] / particle['dt']).mean() / (2 * ndim)
            self.f.loc[self.f.particle == i, 'D'] = D

        #stuck = self.f.D.index[self.f.D < minD]
        #self.f.drop(stuck, inplace=True)
        self.mean_D = self.f.D[self.f.D > minD].mean()
        del self.f['dt']
        del self.f['dX2']
        self.f['D']
        return self.mean_D

    def identify_drift(self, maxD=None, factor=5):
        if maxD is None:
            maxD = self.mean_D / factor
        v = self.v
        v['t'] = (v.index / self.fps)
        v['dt'] = v['t'].diff()
        v['dX2'] = (v[['yc', 'xc']].diff()**2).sum(1)
        v['D'] = (v['dX2'] * self.mpp[1]**2 / v['dt']) / (2 * ndim)
        del v['t']
        del v['dt']
        del v['dX2']
        shifts = v.index[v['D'] > maxD]
        self.ignore_add(set(shifts).union(set(shifts - 1)))

    def subpx_bias(self):
        pos_columns = ['z_px', 'y_px', 'x_px']
        subpx = self.f[pos_columns].applymap(lambda x: x % 1)
        if self.show:
            subpx.hist()
        bias = []
        for col in pos_columns:
            hist, _ = np.histogram(subpx[col], bins=np.arange(0, 1.1, 0.1))
            bias.append(float((hist[0] + hist[9]) / (hist[4] + hist[5])))
        return bias

    @fancy_indexing
    def annotate(self, t, **kwargs):
        pos_columns = ['z_px', 'y_px', 'x_px']
        _kwargs = dict(width=512, size=6, fontsize=18, width_mpp=60,
                       outline='yellow', particle_labels=True)
        _kwargs.update(kwargs)               
        if len(t) > 1:
            result = []
            for i in t:
                if i in self._ignore or i not in self.mips:
                    continue   # skip when i is not in self.mips
                result.append(self.annotate(i, **_kwargs))
            return Frame(result)
        else:
            return annotate_mip(self.f, self.mips[t[0]], self.v,
                                pos_columns=pos_columns, **_kwargs)

    @fancy_indexing
    def export_annotate(self, indices=None, suffix=None, **kwargs):
        if indices is None:
            indices = list(self.mips.keys())
        if suffix is None:
            suffix = SUFFIX['mipavi']
        indices = [i for i in indices if (i not in self._ignore and
                                          i in self.mips and
                                          self.mips[i] is not None)]

        sequence = FramesFunction(lambda i: self.annotate(indices[i], **kwargs),
                                  len(indices))
        export(sequence, self.filename + suffix)

    @fancy_indexing
    def vesicle(self, t, **kwargs):
        t = [i for i in t if (i not in self._ignore and
                              i in self.mips and i in self.v.index)]

        with ND2_Reader(self.filename + '.nd2', channel=1) as frames:
            frames.bundle_axes = 'yx'
            result = []
            for i in t:
                vesicle = self.v.loc[i]
                origin = self.mips[i].metadata['mip_origin'][1:]
                shape = self.mips[i].metadata['mip_shape'][1:]
                frames.default_coords['z'] = int(round(vesicle['zc'] + 0.5))
                frame = frames[i]
                metadata = frame.metadata
                frame = to_rgb(crop_pad(frame, origin, shape))
                frame = Frame(frame, frame_no=i, metadata=metadata)
                center = (vesicle['yc'] - origin[0], vesicle['xc'] - origin[1])
                radius = vesicle[['yr', 'xr']].values
                result.append(annotate_ellipse(frame, center, radius,
                                               t_label=True, mpp=self.mpp[1],
                                               **kwargs))
        if len(result) == 1:
            result = result[0]
        return Frame(result)

    @fancy_indexing
    def export_vesicle(self, indices=None, suffix=None, **kwargs):
        if indices is None:
            indices = list(self.v.index)
        if suffix is None:
            suffix = SUFFIX['vesavi']
        _kwargs = dict(width=512, fontsize=18, width_mpp=60)
        _kwargs.update(kwargs)

        indices = [i for i in indices if (i not in self._ignore and
                                          i in self.mips and
                                          i in self.v.index)]

        sequence = FramesFunction(lambda i: self.vesicle(indices[i], **kwargs),
                                  len(indices))

        export(sequence, self.filename + suffix)

    @fancy_indexing
    def get_fit_params(self, good_frames=None, diameter=None,
                       separation=None):
        if diameter is None:
            diameter = self.meta['f_diameter']
        if separation is None:
            separation = [d*1.5 for d in diameter]
        if good_frames is None:
            good_frames = [i for i in range(len(self)) if i not in self.cluster][:100]
        features = ct.find(self.f[self.f['frame'].isin(good_frames)],
                           separation, ['z_px', 'y_px', 'x_px'], False)
        with ND2_Reader(self.filename + '.nd2', channel=0) as frames:
            result = refine_single(features, frames, diameter,
                                   'gauss3D_a_varInt',
                                   ['z_px', 'y_px', 'x_px'],
                                   ['size_z', 'size_y', 'size_x'])
        self.av_mass = result.mass.median()
        self.av_size = tuple(result[['size_z', 'size_y', 'size_x']].median())
        return self.av_mass, self.av_size

    def zip_locate(self, start, stop, particles, diameter=None,
                   separation=None, mass=None, size=None, search_range=None,
                   inplace=False, refine=True):
        if diameter is None:
            diameter = self.meta['f_diameter']
        if separation is None:
            separation = [d*1.5 for d in diameter]
        if search_range is None:
            search_range = [int(round(2*self.meta['search_range'])/m)
                            for m in self.mpp]
        if mass is None:
            mass = self.av_mass
        if size is None:
            size = self.av_size

        f_initial = self.f[(self.f['frame'] == start) &
                           (self.f['particle'].isin(particles))]
        if len(f_initial) != len(particles):
            raise ValueError('Particles {} not present '.format(particles) +
                             'in frame {}'.format(start))

        if stop is None or start > stop:
            step = -1
        else:
            step = 1
        if stop is not None and stop < 0:
            stop = None

        with ND2_Reader(self.filename + '.nd2', channel=0) as frames:
            f_new = zip_locate(f_initial, frames[start+step:stop:step],
                               diameter, separation, search_range,
                               self.meta['f_lowpass_size'],
                               self.meta['f_threshold'], 
                               'gauss3D_a', dict(mass=mass, size=size),
                               ['z_px', 'y_px', 'x_px'], None, True)
        f_new['z_um'] = f_new['z_px']*self.mpp[0]
        f_new['y_um'] = f_new['y_px']*self.mpp[1]
        f_new['x_um'] = f_new['x_px']*self.mpp[2]

        result = combine_f(self.f, f_new, diameter,
                           pos_columns=['z_px', 'y_px', 'x_px'])
        if inplace:
            if stop is not None:
                stop = int(stop)
            args = dict(start=int(start), stop=stop,
                        particles=[int(p) for p in particles],
                        diameter=[int(d) for d in diameter],
                        separation=[float(s) for s in separation],
                        mass=float(mass), size=[float(s) for s in size],
                        search_range=[int(s) for s in search_range])
            self._cluster_zipped.append(args)
            self.f = result
        return f_new

    def refine_clusters(self, diameter=None, separation=None,
                        mass=None, size=None, inplace=False):
        if diameter is None:
            diameter = self.meta['f_diameter']
        if separation is None:
            separation = [d*1.5 for d in diameter]
        if mass is None:
            mass = self.av_mass
        if size is None:
            size = self.av_size
        pos_columns = ['z_px', 'y_px', 'x_px']

        with ND2_Reader(self.filename + '.nd2', channel=0) as frames:
            f = self.f.copy()
            ct.find(f, separation, pos_columns, inplace=True)
            if inplace:
                self.f = f.copy()
            for i, f_frame in f.groupby('frame'):
                if len(f_frame[f_frame['cluster_size'] > 1]) == 0:
                    continue
                frame = frames[i]
                f_new = ct.refine(f_frame, frame, diameter, 'gauss3D_a',
                                  dict(size=size, mass=mass), pos_columns,
                                  None, False, True)
                f_new['mass', 'size_x', 'size_y', 'size_z'] = np.nan
                f_new['z_um'] = f_new['z_px']*self.mpp[0]
                f_new['y_um'] = f_new['y_px']*self.mpp[1]
                f_new['x_um'] = f_new['x_px']*self.mpp[2]

                # Add 't' column. ct.fit uses 0.5 as origin so add 0.5 to index z.
                z_indexes = np.round(f_new['z_px'] + 0.5).astype(int)
                ts = frame.metadata['t_ms'] / 1000.
                f_new['t'] = np.nan
                f_new['t'] = ts[np.clip(z_indexes, 0, len(ts) - 1)]

                f.update(f_new)

        if inplace:
            self._cluster_refined = dict(diameter=[int(d) for d in diameter],
                        separation=[float(s) for s in separation],
                        mass=float(mass), size=[float(s) for s in size])
            self.f.update(f)
        return f
        
    def link(self, search_range=None, memory=None):
        if search_range is not None:
            self.meta['search_range'] = float(search_range)            
        if memory is not None:
            self.meta['memory'] = int(memory)

        self.f['z_um'] = self.f['z_px']*self.mpp[0]
        self.f['y_um'] = self.f['y_px']*self.mpp[1]
        self.f['x_um'] = self.f['x_px']*self.mpp[2]

        tp.link_df(self.f, self.meta['search_range'], self.meta['memory'],
                   pos_columns=['z_um', 'y_um', 'x_um'])

    def play_annotate(self, suffix=None, **kwargs):
        if suffix is None:
            suffix = SUFFIX['mipavi']
        _kwargs = {'rate': 0.5, 'input-repeat': 10}
        _kwargs.update(kwargs)
        play_file(self.filename + suffix, **_kwargs)

    def play_vesicle(self, suffix=None, **kwargs):
        if suffix is None:
            suffix = SUFFIX['vesavi']
        _kwargs = {'rate': 1.0, 'input-repeat': 10}
        _kwargs.update(kwargs)
        play_file(self.filename + suffix, **_kwargs)        

    def mip(self, t):
        return self.mips[t]

    def frame(self, t):
        with ND2_Reader(self.filename + '.nd2') as im:
            im.bundle_axes = 'czyx'
            frame = im[t]
        return frame

    @property
    def ignore(self):
        return self._ignore
    @ignore.setter
    def ignore(self, value):
        if value is None:
            self._ignore = set()
        elif not hasattr(value, '__iter__'):
            self._ignore = set(int(value))
        else:
            self._ignore = set([int(i) for i in value])
    def ignore_add(self, indices):
        if isinstance(indices, slice):
            indices = np.arange(indices.stop)[indices]
        self.ignore = self._ignore.union(set(indices))
    def ignore_reset(self):
        indices = range(self._len)
        self.ignore = set([i for i in indices if i not in self.v.index])

    @property
    def cluster(self):
        return self._cluster

    @fancy_indexing
    def __getitem__(self, key):
        return self.f[self.f.frame.isin(key)]

    def rdf(self, hist_mode='hist', bw=0.2, min_dist=0.1, max_dist=10,
            offset=1.0):
        f, box = auto_box_3d(self.f, offset)
        bins, hist = erdf_series(f, hist_mode, '3d_bounded', min_dist,
                                 max_dist, bw, box=box)
        if self.show:
            plot_rdf(bins, hist)
        return bins, hist

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.save()

    def save(self):
        bias = self.subpx_bias()
        self.meta.update(dict(category=str(self.category),
                              ignore=list(sorted(self.ignore)),
                              cluster_zipped=self._cluster_zipped),
                              cluster_refined=self._cluster_refined,
                              subpx_bias=[float(b) for b in bias])
        with open(self.filename + SUFFIX['params'], "w") as yml:
            yaml.dump(self.meta, yml)
        columns = ['frame', 'particle', 'x_px', 'y_px', 'z_px', 't',
                   'mass', 'size_x', 'size_y', 'size_z',
                   'cluster', 'cluster_size', 'gaussian']
        ft = self.f[columns].copy()
        ft.rename(columns={'x_px': 'x', 'y_px': 'y', 'z_px': 'z'},
                 inplace=True)
        ft.to_csv(self.filename + SUFFIX['r'])
