from __future__ import (division, unicode_literals)

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import pandas as pd
from pandas import DataFrame
from .algebraic import gauss, drawEllipse
from .utils import crop_pad, fancy_index_to_int_list, export
from pims import to_rgb, pipeline, Frame, plots_to_frame
from trackpy import annotate as annotate_tp
import os

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from functools import wraps

from scipy.ndimage.interpolation import map_coordinates

def _normalize_kwargs(kwargs, kind='patch'):
    """Convert matplotlib keywords from short to long form."""
    # Source:
    # github.com/tritemio/FRETBursts/blob/fit_experim/fretbursts/burst_plot.py
    if kind == 'line2d':
        long_names = dict(c='color', ls='linestyle', lw='linewidth',
                          mec='markeredgecolor', mew='markeredgewidth',
                          mfc='markerfacecolor', ms='markersize',)
    elif kind == 'patch':
        long_names = dict(c='color', ls='linestyle', lw='linewidth',
                          ec='edgecolor', fc='facecolor',)
    for short_name in long_names:
        if short_name in kwargs:
            kwargs[long_names[short_name]] = kwargs.pop(short_name)
    return kwargs

def make_axes(func):
    """
    A decorator for plotting functions.
    NORMALLY: Direct the plotting function to the current axes, gca().
              When it's done, make the legend and show that plot.
              (Instant gratificaiton!)
    BUT:      If the uses passes axes to plotting function, write on those axes
              and return them. The user has the option to draw a more complex
              plot in multiple steps.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        if kwargs.get('ax') is None:
            kwargs['ax'] = plt.gca()
            # Delete legend keyword so remaining ones can be passed to plot().
            try:
                legend = kwargs['legend']
            except KeyError:
                legend = None
            else:
                del kwargs['legend']
            result = func(*args, **kwargs)
            if not (kwargs['ax'].get_legend_handles_labels() == ([], []) or \
                    legend is False):
                plt.legend(loc='best')
            plt.show()
            return result
        else:
            return func(*args, **kwargs)
    return wrapper


def movie_plotting(func):
    def function(*args, **kwargs):
        if len(args) > 1:
            indices = fancy_index_to_int_list(args[1])
        else:
            indices = None

        if 'output' in kwargs:
            output = kwargs['output']
            del kwargs['output']
        else:
            output = None

        if indices is None or len(indices) <= 1:
            if indices is not None:
                args[1] = indices[0]
            if kwargs.get('ax') is None:
                kwargs['ax'] = plt.gca()
                result = func(*args, **kwargs)
                plt.show()
                return result
            else:
                return func(*args, **kwargs)
        else:
            if 'ax' in kwargs:
                raise ValueError('When plotting movies, do not specify axis.')
            if mpl.get_backend()[:2] == 'Qt':
                kwargs['ax'] = plt.gca()
                plt.waitforbuttonpress()
                for i in indices:
                    plt.cla()
                    func(args[0], i, *args[2:], **kwargs)
                    plt.show()
                    plt.pause(0.03)
                plt.close()
            elif mpl.get_backend()[-6:] == 'inline':                
                plots = [None] * len(indices)
                for j, i in enumerate(indices):
                    fig = plt.figure()
                    ax = fig.add_subplot(111)
                    kwargs['ax'] = ax
                    ax = func(args[0], i, *args[2:], **kwargs)
                    plots[j] = ax.figure
                result = plots_to_frame(plots, close_fig=True)
                if output is not None:
                    export(lambda i: result[i, :, :, :3], len(indices), output)
                else:
                    return result
            else:
                raise NotImplemented('Use inline or qt matplotlib backends.')

    return function

def add_scalebar(ax, x, y, mpp, width=20, pad=2, hide_axes=True):
    import matplotlib as mpl
    microns = int((width * mpp) // 1)  # round to whole microns
    width = microns / mpp  # recalculate width
    fontsize = ax.figure.get_size_inches()[0] * width / 140 * 12
    if mpl.rcParams['text.usetex']:
        label = r'{} \textmu m'.format(microns)
    else:
        label = '{} \xb5m'.format(microns)
    ax.add_patch(mpl.patches.Rectangle((x - pad - width, y - pad - 2),
                                       width, 2, color='w'))
    ax.text(x - pad - width / 2, y - pad - 4, label,
            fontdict=dict(color='w', size=fontsize, ha='center'))
    if hide_axes:
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)

@make_axes
def scatter(centroids, ax=None, plot_style={}):
    """Scatter plot of all points of each particle."""
    pos_columns = ['phi', 'th']
    if len(centroids) == 0:
        raise ValueError("DataFrame of centroids is empty.")
        
    _plot_style = dict(marker='o', linestyle='none')
    _plot_style.update(**_normalize_kwargs(plot_style, 'line2d'))

    # Axes labels
    ax.set_xlabel(r'$\phi$')
    ax.set_ylabel(r'$\theta$')
    ax.set_xlim([-np.pi, np.pi])
    ax.set_ylim([0, np.pi])
    ax.plot(centroids[pos_columns[0]], centroids[pos_columns[1]], **_plot_style)
    return ax

@make_axes
def plot_traj(traj, ax=None, t_column=None, label=False, plot_style={}):
    if t_column is None:
        t_column = 'frame'
    pos_columns = ['phi', 'th']
    if len(traj) == 0:
        raise ValueError("DataFrame of trajectories is empty.")
    _plot_style = dict(linewidth=1)
    _plot_style.update(**_normalize_kwargs(plot_style, 'line2d'))

    # Axes labels
    ax.set_xlabel(r'$\phi$')
    ax.set_ylabel(r'$\theta$')
    ax.set_xlim([-np.pi, np.pi])
    ax.set_ylim([0, np.pi])

    # Trajectories
    # Unstack particles into columns.
    unstacked = traj.set_index(['particle', t_column])[pos_columns].unstack()
    for i, trajectory in unstacked.iterrows():
        ax.plot(trajectory[pos_columns[0]],
                trajectory[pos_columns[1]], **_plot_style)

    if label:
        unstacked = traj.set_index([t_column, 'particle'])[pos_columns].unstack()
        first_frame = int(traj[t_column].min())
        coords = unstacked.fillna(method='backfill').stack().loc[first_frame]
        for particle_id, coord in coords.iterrows():
            ax.text(*coord.tolist(), s="%d" % particle_id,
                    horizontalalignment='center',
                    verticalalignment='center')
    return ax

@movie_plotting
def plot_traj_movie(features, t, tracks=10, ax=None):
    f = features[features.frame == t]
    ax = scatter(f, ax=ax)
    if tracks > 0:
        plot_traj(features[(features.frame <= t) & (features.frame > t - tracks)],
                  ax=ax, label=True)
    ax.set_title('t={0:.2f} s'.format(f['t'].mean()))
    ax.grid()
    return ax

@make_axes
def plot_fluctuation_spectrum(powersp, ax=None, **plot_kwargs):
    modes = np.arange(1, len(powersp) + 1)
    ax.plot(modes, powersp, **plot_kwargs)
    ax.grid()
    ax.set_xlabel('mode')
    ax.set_ylabel('DFT')
    return ax

@make_axes
def plot_h_histogram(hist, bin_edges, fit, ax=None, **plot_kwargs):
    _, mu, sigma = fit
    h = (bin_edges[1:] + bin_edges[-1]) / 2
    max_value = max(hist) * 1.1
    ax.set_xlim(min(mu - 5*sigma, -0.5), mu + 5*sigma)
    ax.set_ylim(0, max_value)
    ax.set_xlabel('$h [\mu m]$')
    ax.set_ylabel('$N$')
    ax.bar(bin_edges[:-1], hist, width=bin_edges[1:] - bin_edges[:-1],
           color='g')
    ax.plot(h, gauss(h, *fit), **plot_kwargs)
    ax.fill_between([-0.1, 0.1], [max_value, max_value], color='m')
    ax.text(0, max_value / 2, '$membrane$', rotation='vertical',
            ha='center', va='center', size='larger')
    return ax

@make_axes
def plot_sphere_density(dens, ax=None, label='', **plot_kwargs):
    im = ax.imshow(dens.T, extent=(-np.pi, np.pi, 0, np.pi), **plot_kwargs)
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(0, np.pi)
    ax.set_xlabel(r'$\phi$')
    ax.set_ylabel(r'$\theta$')
    plt.colorbar(im, label=label)
    return ax

@make_axes
def annotate_max(f, frame, proj='xy', ax=None, **kwargs):
    pos_columns = ['z', 'y', 'x']

    d0, d1 = list(proj)
    for d2 in pos_columns:
        if d2 not in [d0, d1]:
            break
    axis = pos_columns.index(d2)

    f_plot = f[pos_columns].copy()
    f_plot.rename(columns={d2: 'z', d1: 'y', d0: 'x'}, inplace=True)

    frame = frames_to_rgb(frame.max(axis=axis-3))
    if d0 > d1:
        frame = np.transpose(frame, axes=(1, 0, 2))

    return annotate_tp(f_plot, frame, **kwargs)


def pair_slice(pair, center, size, mpp, frame):
    has_channels = (frame.ndim == 4)

    th = pair[['th0', 'th1']].values.astype(np.float)
    phi = pair[['phi0', 'phi1']].values.astype(np.float)
    x0, x1 = np.array([np.cos(th),
                       np.sin(th)*np.sin(phi),
                       np.sin(th)*np.cos(phi)]).T

    normal = np.cross(x0, x1)
    base1 = np.cross(x0, normal)
    base1 /= np.linalg.norm(base1)
    base2 = np.cross(base1, normal)
    base2 /= np.linalg.norm(base2)

    grid_1d = np.arange(0, size/2, mpp[1])
    grid_1d = np.concatenate([-grid_1d[:1:-1], grid_1d])

    coords = ((grid_1d * base1[:, np.newaxis])[:, np.newaxis, :] +
              (grid_1d * base2[:, np.newaxis])[:, :, np.newaxis])
    coords += center[:, np.newaxis, np.newaxis]
    coords = coords / np.array(mpp)[:, np.newaxis, np.newaxis]

    if has_channels:
        result = [map_coordinates(c, coords) for c in frame]
    else:
        result = map_coordinates(frame, coords)
        
#    if False:
#        import matplotlib.pyplot as plt
#        from mpl_toolkits.mplot3d import Axes3D
#        
#        fig = plt.figure()
#        ax = fig.add_subplot(111, projection='3d')
#        
#        ax.plot(coords[2].ravel(), coords[1].ravel(), zs=coords[0].ravel(), marker='.', linestyle='None')
#        ax.plot(np.array([x0[0]+center[0], x1[0]+center[0], center[0]])/vf.mpp[2],
#                np.array([x0[1]+center[1], x1[1]+center[1], center[1]])/vf.mpp[1],
#                zs=np.array([x0[2]+center[2], x1[2]+center[2], center[2]])/vf.mpp[0], color='r', marker='o')
#        plt.show()
    return Frame(result)


def annotate_ellipsoid(v, frame, proj='xy', n=100, **kwargs):
    """ Plots an ellipse on top of a center slice of the 3d frame.

    Parameters
    ----------
    v : pandas.DataFrame or pandas.Series
        when it is a multiline DataFrame, frame_no of the frame is searched in
        the frame column of v
    frame : pims.Frame (3d)
    proj : {xy, zx, zy, yx, xz, yz}
        determines the horizontal (first) and vertical (second) axes
    n : int
        number of points calculated to plot ellipse
    kwargs :
        keyword arguments passed to trackpy.annotate

    Returns
    -------
    matplotlib.axes object, having image and ellipse plotted
    """
    assert frame.ndim == 3

    # interpret parameters
    if isinstance(v, DataFrame) and len(v) > 1:
        p = v[v['frame'] == frame.frame_no].loc[0]
    elif len(v) == 1:
        p = v.loc[0]
    else:
        p = v

    d0, d1 = list(proj)
    for d2 in ['z', 'y', 'x']:
        if d2 not in [d0, d1]:
            break

    # because xr does not exist, do this:
    radius = np.empty((2,))
    for i, d in enumerate([d1, d0]):
        if d == 'x':
            radius[i] = p['yr']
        else:
            radius[i] = p[d + 'r']

    center = [p[d1 + 'c'], p[d0 + 'c']]
    middle = p[d2 + 'c']

    image0 = np.take(frame, int(middle), axis=['z', 'y', 'x'].index(d2))
    image1 = np.take(frame, int(middle) + 1, axis=['z', 'y', 'x'].index(d2))

    middle_slice = image0 * (1 - middle % 1) + image1 * middle % 1

    if d0 > d1:
        middle_slice = middle_slice.T
    ax = annotate_ellipse_mpl((radius[0], radius[1], center[0], center[1]),
                              middle_slice, n, **kwargs)
    ax.set_xlabel(d0)
    ax.set_ylabel(d1)
    return ax


def annotate_ellipse_mpl(p, image, n=100, **kwargs):
    assert image.ndim == 2
    if 'plot_style' not in kwargs:
        kwargs['plot_style'] = {}
    _plot_style = {'marker': None, 'ls': '--', 'c': 'r'}
    _plot_style.update(kwargs['plot_style'])
    kwargs['plot_style'] = _plot_style

    f = DataFrame(drawEllipse(p, 0, n), columns=['y', 'x'])
    ax = annotate_tp(f, image, **kwargs)

    # crop image around ellipse
    ax.set_xlim(max(round(p[3] - 1.2 * p[1]), 0),
                min(round(p[3] + 1.2 * p[1]), image.shape[1] - 1))
    ax.set_ylim(max(round(p[2] - 1.2 * p[0]), 0),
                min(round(p[2] + 1.2 * p[0]), image.shape[0] - 1))
    ax.invert_yaxis()
    return ax


def drawEllipsoid(v, proj='xy', nmax=100):
    if isinstance(v, DataFrame):
        if 'frame' in v and hasattr(image, 'frame_no'):
            p = v[v['frame'] == image.frame_no]
        else:
            p = v
    else:
        p = DataFrame(v, columns=['zc', 'yc', 'xc', 'zr', 'yr'])    
    if proj == 'yx' or proj == 'xy':
        radius = np.array([p.yr[0], p.yr[0]])
        center = (p.yc[0], p.xc[0])
        d2r = p.zr[0]
        d2c = p.zc[0]
        ds = np.arange(int(d2c - d2r), int(d2c + d2r), 1)
    f=[]
    for d in ds:
        size = np.cos((d - d2c) / d2r)
        if size > 0:
            f1 = DataFrame(drawEllipse(radius * size, center, 
                                       n=int(nmax/size)), columns = ['y', 'x'])
            f1['z'] = d
            f.append(f1)
    f = pd.concat(f).reset_index(drop=True)
    return f


def plot3D_vesicle_features(features, R, tracks = True, frame_no = 0, ax = None, 
                            t_column = None, pos_columns = None, display = 'NIS'):
    if ax == None:
        fig = plt.figure(figsize=[8,6])
        ax = fig.add_subplot(111, projection='3d')

    if t_column is None:
        t_column = 'frame'
    if pos_columns is None:
        pos_columns = ['x','y','z']    
    
    artists = []
    
    radiusXY = radiusZ = R
    axisX = radiusXY*1.5
    axisZ = radiusZ*1.2

    #ax.set_aspect(axisZ/axisX)
    ax.auto_scale_xyz([-1*axisX, axisX], [-1*axisX, axisX], [-1*axisZ, axisZ])        
    ax.set_xlim3d([-1*axisX, axisX])
    ax.set_ylim3d([-1*axisX, axisX])
    ax.set_zlim3d([-1*axisZ, axisZ])

    u = np.linspace(0, 2 * np.pi, 200)
    v = np.linspace(0, np.pi, 200)

    xv = radiusXY * np.outer(np.cos(u), np.sin(v))
    yv = radiusXY * np.outer(np.sin(u), np.sin(v))
    zv = radiusZ * np.outer(np.ones(np.size(u)), np.cos(v))

    #ax.plot_wireframe(xv, yv, zv,  rstride=6, cstride=8, color = 'r')
    artists.append(ax.plot_wireframe(xv, yv, zv, alpha = 0.2, linewidth=0.5, rstride = 5, cstride = 5, color = 'r'))
               
    if (features is not None) and ('particle' in features) and tracks:
        unstacked = features[features[t_column] <= frame_no].set_index([t_column, 'particle']).unstack()  
        X = unstacked[pos_columns[0]]
        Y = unstacked[pos_columns[1]]
        Z = unstacked[pos_columns[2]] 
        for i in X.columns:
            artists.append(ax.plot(X[i], Y[i], zs=Z[i], linewidth=1, color='black')[0])

    if features is not None:
        featuresT = features[features[t_column] == frame_no]
        if 'on' in featuresT:
            featon = featuresT[featuresT['on']]
            featoff = featuresT[~featuresT['on']] 
            artists.append(ax.scatter(featon[pos_columns[0]], featon[pos_columns[1]], 
                                      featon[pos_columns[2]], marker='o', color = 'b'))
        else:    
            featoff = featuresT
    
        artists.append(ax.scatter(featoff[pos_columns[0]], featoff[pos_columns[1]], 
                                  featoff[pos_columns[2]], marker='o', color = 'b'))
    
    if plt.rcParams['text.usetex']:
        ax.set_xlabel(r'x [\textmu m]')
        ax.set_ylabel(r'y [\textmu m]')
        ax.set_zlabel(r'z [\textmu m]')
    else:
        ax.set_xlabel('x [\xb5m]')
        ax.set_ylabel('y [\xb5m]')
        ax.set_zlabel('z [\xb5m]')
        
    if hasattr(display, '__iter__'):
        ax.view_init(display[0],display[1])
    elif display == 'NIS':
        ax.view_init(225, 315)
    else:
        ax.view_init(45, 45)
    
    return ax

def annotate(f, frame, size=7, outline='red', labels=None, font="arial.ttf",
             fontsize=10, mpp=None, width_mpp=30, pad=5, t_label=False,
             pos_columns=None, width=None, max_height=None):
    """Annotates an image using PIL Image and ImageDraw. PIL has the convention
    with the origin at the center of the topleft pixel, the same as trackpy.
    Still leaving out +0.5 gives shifted pictures... ??"""
    if pos_columns is None:
        pos_columns = ['y', 'x']
    im = Image.fromarray(frame)
    if width is not None:
        w = width
        h = (im.size[1] * w) // im.size[0]
        if max_height is not None and h > max_height:
            h = max_height
            width = (im.size[0] * h) // im.size[1]
        factor = w / im.size[0]
        im = im.resize((w, h), Image.BILINEAR)
    else:
        factor = 1

    draw = ImageDraw.Draw(im)
    font = ImageFont.truetype(font, fontsize)
    for i in f.index:
        x = f[pos_columns[1]][i] * factor
        y = f[pos_columns[0]][i] * factor
        draw.ellipse((x-size+0.5, y-size+0.5, x+size+0.5, y+size+0.5),
                     outline=outline)
        if labels:
            try:
                label_string = str(int(f[labels][i]))
                draw.text((x+size, y+size), label_string,
                          font=font, fill=outline)
            except:
                pass
    if mpp:  # add scalebar
        microns = int((width_mpp * mpp) // 1)  # round to whole microns
        width_mpp = microns / mpp  # recalculate width
        label = '{} \xb5m'.format(microns)
        draw.rectangle((im.size[0] - pad - width_mpp, im.size[1] - pad - 2,
                        im.size[0] - pad, im.size[1] - pad), fill='white')
        text_width, text_height = draw.textsize(label, font=font)
        draw.text((im.size[0] - pad - width_mpp / 2 - text_width / 2,
                   im.size[1] - pad - 4 - text_height), label,
                  font=font, fill='white')
    if t_label:
        if hasattr(frame, 'frame_no'):
            label = 'index = {}'.format(frame.frame_no)
            text_width, text_height = draw.textsize(label, font=font)
            draw.text((im.size[0] - pad - text_width,
                       im.size[1] - pad - 6 - text_height*2),
                      label, font=font, fill='white')
        if hasattr(frame, 'metadata'):
            if 't_ms' in frame.metadata:
                label = 't = {0:.2f}s'.format(np.mean(frame.metadata['t_ms']) / 1000)
                text_width, text_height = draw.textsize(label, font=font)
                draw.text((im.size[0] - pad - text_width,
                          im.size[1] - pad - 6 - text_height*3),
                          label, font=font, fill='white')
    del draw
    return Frame(im)


def annotate_ellipse(frame, center, radius, outline='yellow',
                     labels=None, font="arial.ttf", fontsize=10, mpp=None,
                     width_mpp=30, pad=5, t_label=False,
                     width=None, max_height=None):
    im = Image.fromarray(frame)
    if width is not None:
        w = width
        h = (im.size[1] * w) // im.size[0]
        if max_height is not None and h > max_height:
            h = max_height
            width = (im.size[0] * h) // im.size[1]
        factor = w / im.size[0]
        im = im.resize((w, h), Image.BILINEAR)
    else:
        factor = 1

    draw = ImageDraw.Draw(im)
    font = ImageFont.truetype(font, fontsize)
    
    if not hasattr(radius, '__iter__'):
        radius = (radius,) * 2
    draw.ellipse(((center[1]-radius[1]+0.5) * factor,
                  (center[0]-radius[0]+0.5) * factor,
                  (center[1]+radius[1]+0.5) * factor,
                  (center[0]+radius[0]+0.5) * factor),
                 outline=outline)

    if mpp:  # add scalebar
        microns = int((width_mpp * mpp) // 1)  # round to whole microns
        width_mpp = microns / mpp  # recalculate width
        label = '{} \xb5m'.format(microns)
        draw.rectangle((im.size[0] - pad - width_mpp, im.size[1] - pad - 2,
                        im.size[0] - pad, im.size[1] - pad), fill='white')
        text_width, text_height = draw.textsize(label, font=font)
        draw.text((im.size[0] - pad - width_mpp / 2 - text_width / 2,
                   im.size[1] - pad - 4 - text_height), label,
                  font=font, fill='white')
    if t_label:
        if hasattr(frame, 'frame_no'):
            label = 'index = {}'.format(frame.frame_no)
            text_width, text_height = draw.textsize(label, font=font)
            draw.text((im.size[0] - pad - text_width,
                       im.size[1] - pad - 6 - text_height*2),
                      label, font=font, fill='white')
        if hasattr(frame, 'metadata'):
            if 't_ms' in frame.metadata:
                label = 't = {0:.2f}s'.format(np.mean(frame.metadata['t_ms']) / 1000)
                text_width, text_height = draw.textsize(label, font=font)
                draw.text((im.size[0] - pad - text_width,
                          im.size[1] - pad - 6 - text_height*3),
                          label, font=font, fill='white')
    del draw
    return Frame(im)


def create_mip(image, center, shape, spacing=5):
    """ Returns a max intensity projections in all three directions, of defined
    shape and center.
    """
    if image.ndim == 3:
        image = image[np.newaxis, :, :, :]
    assert image.ndim == 4

    if image.shape[-1] in (3, 4):
        raise ValueError("RGB(A) not supported")

    if shape is None or center is None:
        cropped = image
        origin = [0., 0., 0.]
        shape = cropped.shape
    else:
        shape_a = np.array(shape, dtype=np.float64)
        origin = list(np.round(center - shape_a/2).astype(int))
        cropped = crop_pad(image, origin, list(shape))

    c, z, y, x = cropped.shape
    result = np.zeros((c, y + z + spacing, x + z + spacing),
                      dtype=cropped.dtype)

    for i in range(c):
        result[i, :y, :x] = cropped[i].max(0)
        result[i, -z:, :x] = cropped[i].max(1)[::-1]
        result[i, :y, -z:] = np.swapaxes(cropped[i].max(2)[::-1], 0, 1)

    try:
        colors = image.metadata['colors']
    except AttributeError or KeyError:
        colors = None

    metadata = dict(mip_origin=origin, mip_shape=shape, mip_spacing=spacing)
    if hasattr(image, 'metadata'):
        metadata.update(image.metadata)

    if hasattr(image, 'frame_no'):
        frame_no = image.frame_no
    else:
        frame_no = None

    return Frame(to_rgb(result, colors), metadata=metadata, frame_no=frame_no)


def annotate_mip(centroids, mip, vesicle=None, pos_columns=None,
                 frame_column='frame', particle_labels=False, **kwargs):
    """An extension of annotate that annotates a 3D image and returns a max
    intensity projections in all three directions."""
    if mip.ndim > 3:
        raise ValueError("Annotate_mip only takes 2D images (MIP)")
    if mip.ndim == 3 and (mip.shape[2] not in [3, 4]):
        raise ValueError("Annotate_mip only takes RGB(A) images (MIP)")
    if pos_columns is None:
        pos_columns = ['z', 'y', 'x']

    spacing = mip.metadata['mip_spacing']
    z, y, x = mip.metadata['mip_origin']
    d, h, w = mip.metadata['mip_shape']

    f = centroids[(centroids[pos_columns[2]] >= z) &
                  (centroids[pos_columns[0]] < z + d) &
                  (centroids[pos_columns[1]] >= y) &
                  (centroids[pos_columns[1]] < y + h) &
                  (centroids[pos_columns[2]] >= x) &
                  (centroids[pos_columns[2]] < x + w) &
                  (centroids[frame_column] == mip.frame_no)].copy()

    if vesicle is not None:
        if frame_column in vesicle and mip.frame_no in vesicle[frame_column]:
            f_v = vesicle[vesicle[frame_column] == mip.frame_no].iloc[0]
        elif mip.frame_no in vesicle.index:
            f_v = vesicle.loc[mip.frame_no]
        else:
            f_v = None
        if f_v is not None:
            f_v = DataFrame([f_v[['zc', 'yc', 'xc']].values],
                            columns=pos_columns)
            f_v['particle'] = -1
        f = pd.concat([f, f_v], ignore_index=True)

    f[pos_columns[2]] -= x
    f[pos_columns[1]] -= y
    f[pos_columns[0]] -= z
    f_xz = f.copy()
    f_yz = f.copy()
    f_xz[pos_columns[2]] = w + d - f_xz[pos_columns[0]] + spacing - 1
    f_yz[pos_columns[1]] = h + d - f_yz[pos_columns[0]] + spacing - 1
    f_plot = pd.concat([f, f_xz, f_yz], ignore_index=True)

    if particle_labels and 'particle' in f_plot:
        labels_column = 'particle'
    else:
        labels_column = None

    return annotate(f_plot, mip, labels=labels_column, t_label=True,
                    mpp=mip.metadata['mpp'], pos_columns=pos_columns[1:],
                    **kwargs)

def annotate3d_max(centroids, image, mpp=None, crop=None,
                   particle_labels=False, **kwargs):
    """
    An extension of annotate that annotates a 3D image and returns a max
    intensity projections in all three directions.
    """
    if image.ndim == 3:
        image = image[np.newaxis, :, :, :]
    assert image.ndim == 4

    try:
        frame_no = image.frame_no
    except AttributeError:
        frame_no = None
    try:
        metadata = image.metadata
    except AttributeError:
        metadata = None

    if image.shape[-1] in (3, 4):
        raise ValueError("RGB(A) not supported")

    SPACING = 5
    c, z, y, x = image.shape

    if crop:
        Cz, Cy, Cx, Cd, Ch, Cw = crop
        image = crop_pad(image, [Cz, Cy, Cx], [Cd, Ch, Cw])
        _, z, y, x = image.shape
    else:
        Cz = Cy = Cx = 0
        Cd, Ch, Cw = z, y, x

    frame = np.zeros((c, y + z + SPACING, x + z + SPACING),
                     dtype=image.dtype)

    for i in range(c):
        frame[i, :y, :x] = image[i].max(0)
        frame[i, -z:, :x] = image[i].max(1)[::-1]
        frame[i, :y, -z:] = np.swapaxes(image[i].max(2)[::-1], 0, 1)

    frame = Frame(to_rgb(frame), frame_no=frame_no, metadata=metadata)

    f = centroids[(centroids.z >= Cz) & (centroids.z < Cz + Cd) &
                  (centroids.y >= Cy) & (centroids.y < Cy + Ch) &
                  (centroids.x >= Cx) & (centroids.x < Cx + Cw)]

    f.x -= Cx
    f.y -= Cy
    f.z -= Cz
    f_xz = f.copy()
    f_yz = f.copy()
    f_xz['x'] = x + z - f_xz.z + SPACING - 1
    f_yz['y'] = y + z - f_yz.z + SPACING - 1
    f_plot = pd.concat([f, f_xz, f_yz], ignore_index=True)

    if particle_labels and 'particle' in f_plot:
        labels_column = 'particle'
    else:
        labels_column = None

    return annotate(f_plot, frame, labels=labels_column, t_label=True,
                    mpp=mpp, **kwargs)
          
def closeup(f, frame, diameter, pos_columns):
    if pos_columns is None:
        pos_columns = ['z', 'y', 'x']
    f_closeup = f[f.frame == frame.frame_no]

    shape = [d + 3 for d in diameter]
    cols = 3
    rows = (len(f_closeup) + cols - 1) // cols
    image = np.zeros((shape[0], shape[1] * rows, shape[2] * cols), dtype=frame.dtype)

    x = 0
    y = 0
    for i in f_closeup.index:
        center = f_closeup.loc[i, pos_columns]
        edges = [c - s // 2 for (c, s) in zip(center, shape)]
        rect = [slice(e, e + s) for (e, s) in zip(edges, shape)]
        subimage = frame[rect]
        image[0:subimage.shape[0],
              y:y+subimage.shape[1],
              x:x+subimage.shape[2]] = subimage
        if x == (cols - 1) * shape[2]:
            x = 0
            y += shape[1]
        else:
            x += shape[2]

    return Frame((image / (np.max(image) / 255)).astype(np.uint8))
