from __future__ import (division, unicode_literals, print_function,
                        absolute_import)
import six
import warnings
import numpy as np
import pandas as pd
from pims_nd2 import ND2_Reader
import os
import yaml
from pims import Frame, pipeline, to_rgb, normalize, FramesSequence
from subprocess import Popen
try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from moviepy.editor import VideoClip
except ImportError:
    VideoClip = None


class FramesFunction(FramesSequence):
    def __init__(self, func, length, **kwargs):
        self._func = func
        self._len = length
        self._kwargs = kwargs
        
        first_frame = self.get_frame(0)
        self._shape = first_frame.shape
        self._dtype = first_frame.dtype

    def __len__(self):
        return self._len

    def get_frame(self, i):
        return self._func(i, **self.kwargs)

    @property
    def frame_shape(self):
        return self._shape

    @property
    def pixel_type(self):
        return self._dtype
    

def fancy_index_to_int_list(indexer):
    if indexer is None:
        return None
    if isinstance(indexer, slice):
        indexer = range(indexer.stop)[indexer]
    if not hasattr(indexer, '__iter__'):
        indexer = [indexer]
    return [int(i) for i in indexer]


def fancy_indexing(func):
    def function(*args, **kwargs):
        if len(args) < 2:
            return func(*args, **kwargs)
        else:
            n = fancy_index_to_int_list(args[1])
            return func(args[0], n, *args[2:], **kwargs)
    return function

def guess_pos_columns(f):
    if 'z' in f:
        pos_columns = ['z', 'y', 'x']
    else:
        pos_columns = ['y', 'x']
    if 'x_px' in f:
        pos_columns = [p + '_px' for p in pos_columns]
    return pos_columns

def copy_to_pixels(df, mpp):
    result = df.copy()
    for col in ['z', 'zrel', 'zc']:
        if col in result:
            result[col] /= mpp[0]
    for col in ['y', 'yrel', 'yc']:
        if col in result:
            result[col] /= mpp[1]
    for col in ['x', 'xrel', 'xc']:
        if col in result:
            result[col] /= mpp[2]
    return result

def gen_nd2_paths(filepath):
    for subdir, dirs, files in os.walk(filepath):
        for filepath in files:
            filename, extension = os.path.splitext(filepath)
            if extension == '.nd2':
                yield subdir, filename

def image_md(filename):
    with ND2_Reader(filename + '.nd2') as im:
        if (('c' not in im.axes) or ('z' not in im.axes) or
            ('t' not in im.axes)):
            raise IOError('Incorrect image dimensionality')
        if im.sizes['c'] != 2:
            raise IOError('Incorrect channel count')
        if im.sizes['t'] == 1:
            raise IOError('Image has only 1 timepoint')

        mpp = [im.calibrationZ, im.calibration, im.calibration]
        length = len(im)
        t_first = im.get_frame_2D(z=0, t=0).metadata['t_ms']
        t_last = im.get_frame_2D(z=0, t=length-1).metadata['t_ms']
        fps = 1000 * (length - 1) / (t_last - t_first)
    return length, mpp, fps

def image_loc(filename):
    with ND2_Reader(filename + '.nd2') as frames:
        frames.bundle_axes = 'yx'
        md = frames[0].metadata
        x, y, z = round(md['x_um']), round(md['y_um']), round(md['z_um'])
        return z, y, x

def list_loc(path):
    for (directory, fn) in gen_nd2_paths(path):
        fullpath = os.path.join(directory, fn)
        print(fn, image_loc(fullpath))

def to_odd_int(value):
    diameter = int(round(value))
    if diameter % 2 == 0:
        diameter += 1
    return diameter

def crop_pad(image, corner, shape):
    ndim = len(corner)
    corner = [int(round(c)) for c in corner]
    shape = [int(round(s)) for s in shape]
    original = image.shape[-ndim:]
    zipped = zip(corner, shape, original)

    if np.any(c < 0 or c + s > o for (c, s, o) in zipped):
        no_padding = [(0, 0)] * (image.ndim - ndim)
        padding = [(max(-c, 0), max(c + s - o, 0)) for (c, s, o) in zipped]
        corner = [c + max(-c, 0) for c in corner]
        image_temp = np.pad(image, no_padding + padding, mode='constant')
    else:
        image_temp = image

    no_crop = [slice(o+1) for o in image.shape[:-ndim]]
    crop = [slice(c, c+s) for (c, s) in zip(corner, shape)]
    return image_temp[no_crop + crop]

def inliers(x, absdev, window):
    """Returns a boolean mask. Values are False for points that have a
    deviation larger than maxdev, compared to the points in the range
    +-window."""
    maxsqrdev = absdev**2
    runningmedian = [np.median(x[max(0, n - window): n + window])
                     for n in range(len(x))]
    tokeep = (x - runningmedian)**2 < maxsqrdev
    return tokeep
    
def open_meta(fn):
    with open(fn, "r") as yml:
        meta = yaml.load(yml)
    if meta is None:
        meta = dict()
    return meta


def update_meta(fn, meta_new):
    meta = open_meta(fn)
    meta.update(meta_new)
    try:
        with open(fn + '_temp_update_yml', "w") as yml:
            yaml.dump(meta, yml)
    except:
        raise IOError('Invalid metadata field datatypes')
    else:
        with open(fn, "w") as yml:
            yaml.dump(meta, yml)
    finally:
        os.remove(fn + '_temp_update_yml')
    return meta

@pipeline
def frames_to_rgb(image):
    if image.ndim > 2 and image.shape[-1] in [3, 4]:
        return image
    else:
        return to_rgb(image)

def export_imageio(sequence, filename, preset='mp4', **kwargs):
    from imageio import mimwrite
    _kwargs = {'mp4': dict(format='mp4'),
               'ppt': dict(format='wmv2', bitrate=400000)}[preset]
    _kwargs.update(kwargs)
    mimwrite(filename, frames_to_rgb(sequence), **_kwargs)

def _to_rgb_uint8(image, autoscale):
    if autoscale:
        image = (normalize(image) * 255).astype(np.uint8)
    elif image.dtype is not np.uint8:
        if np.issubdtype(image.dtype, np.integer):
            max_value = np.iinfo(image.dtype).max
            # sometimes 12-bit images are stored as unsigned 16-bit
            if max_value == 2**16 - 1 and image.max() < 2**12:
                max_value = 2**12 - 1
            image = (image / max_value * 255).astype(np.uint8)
        else:
            image = (image * 255).astype(np.uint8)

    ndim = image.ndim
    shape = image.shape
    if ndim == 3 and shape.count(3) == 1:
        # This is a color image. Ensure that the color axis is axis 2.
        color_axis = shape.index(3)
        image = np.rollaxis(image, color_axis, 3)
    elif image.ndim == 3 and shape.count(4) == 1:
        # This is an RGBA image. Drop the A values.
        color_axis = shape.index(4)
        image = np.rollaxis(image, color_axis, 4)[:, :, :3]
    elif ndim == 2:
        # Expand into color to satisfy moviepy's expectation
        image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
    else:
        raise ValueError("Images have the wrong shape.")

    return np.asarray(image)

def _estimate_bitrate(shape, frame_rate):
    "Return a bitrate that will guarantee lossless video."
    # Total Pixels x 8 bits x 3 channels x FPS
    return shape[0] * shape[1] * 8 * 3 * frame_rate

def export_moviepy(sequence, filename, rate=30, bitrate=None, width=None,
                   height=None, codec='mpeg4', format='yuv420p', 
                   autoscale=True, quality=None, verbose=True,
                   ffmpeg_params=None):
    """Export a sequence of images as a standard video file.

    N.B. If the quality and detail are insufficient, increase the
    bitrate.

    Parameters
    ----------
    sequence : any iterator or array of array-like images
        The images should have two dimensions plus an
        optional third dimensions representing color.
    filename : string
        name of output file
    rate : integer, optional
        frame rate of output file, 30 by default
    bitrate : integer or string, optional
        Video bitrate is crudely guessed if None is given.
    width : integer, optional
        By default, set the width of the images.
    height : integer, optional
        By default, set the  height of the images. If width is specified
        and height is not, the height is autoscaled to maintain the aspect
        ratio.
    codec : string, optional
        a valid video encoding, 'mpeg4' by default
    format: string, optional
        Video stream format, 'yuv420p' by default.
    quality: integer or string, optional
        Use this for variable bitrates. 1 = high quality, 5 = default.
    autoscale : boolean, optional
        Linearly rescale the brightness to use the full gamut of black to
        white values. True by default.
    verbose : boolean, optional
        Determines whether MoviePy will print progress. True by default.
    ffmpeg_params : dictionary, optional
        Dictionary of parameters that will be passed to ffmpeg. By default
        {'pixel_format': str(format), 'qscale:v': str(quality)}

    See Also
    --------
    http://zulko.github.io/moviepy/ref/VideoClip/VideoClip.html#moviepy.video.VideoClip.VideoClip.write_videofile
    """
    if VideoClip is None:
        raise ImportError('The MoviePy exporter requires moviepy to work.')

    if ffmpeg_params is None:
        ffmpeg_params = dict()
    if quality is not None:
        ffmpeg_params['qscale:v'] = str(quality)
    if format is not None:
        ffmpeg_params['pixel_format'] = str(format)
    if bitrate is None:
        bitrate = _estimate_bitrate(sequence[0].shape, rate)

    _ffmpeg_params = []
    [_ffmpeg_params.extend(['-' + key, ffmpeg_params[key]])
     for key in ffmpeg_params]

    if rate < 10:
        warnings.warn('Framerates lower than 10 may give playback issues.')

    clip = VideoClip(lambda t: _to_rgb_uint8(sequence[int(round(t*rate))],
                                             autoscale))
    clip.duration = (len(sequence) - 1) / rate
    if not (height is None and width is None):
        clip = clip.resize(height=height, width=width)
    clip.write_videofile(filename, rate, codec, str(bitrate), audio=False,
                         verbose=verbose, ffmpeg_params=_ffmpeg_params)


export = export_moviepy

def play_file(fn, **kwargs):
    _kwargs = {'rate': 1.0, 'input-repeat': 10}
    _kwargs.update(kwargs)
    if not os.path.isfile(fn):
        raise IOError('File {} does not exist'.format(fn))
    switch = ['--{0}={1}'.format(k, _kwargs[k]) for k in _kwargs]
    Popen([r"C:\Program Files (x86)\VideoLAN\VLC\vlc.exe", fn] + switch)

def legacy_play(sequence, **kwargs):
    from IPython.display import display
    from pims.display import repr_video
    from tempfile import NamedTemporaryFile
    temp = NamedTemporaryFile(suffix='.mp4')
    temp.close()
    export(sequence, temp.name, 'mp4', **kwargs)
    display(repr_video(temp.name, 'x-webm'))
    os.remove(temp.name)

def exec_cluster(func, iterable):
    from IPython.parallel import Client
    client = Client()
    view = client.load_balanced_view()
    amr = view.map_async(func, iterable)
    amr.wait_interactive()

def bin_column(df, bins, column, binned_column='binned'):
    centers = np.concatenate([[np.nan], (bins[1:] + bins[:-1])/2, [np.nan]])
    mapping = lambda x: centers[np.digitize([x], bins, right=True)[0]]
    df[binned_column] = df[column].apply(mapping)
