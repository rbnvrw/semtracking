from __future__ import (division, unicode_literals)

import numpy as np
from guvtracking import *
import unittest
import nose
from numpy.testing import (assert_almost_equal, assert_allclose)


def gen_artificial_ellipsoid(shape, center, R, FWHM, ar=1, noise=0):
    assert center[0] - R / ar >= 0
    assert center[1] - R >= 0
    assert center[2] - R >= 0
    assert center[0] + R / ar < shape[0]
    assert center[1] + R < shape[1]
    assert center[2] + R < shape[2]
    d, h, w = shape
    zc, yc, xc = center
    zc = zc * ar
    sigma = FWHM / 2.35482
    z, y, x = np.meshgrid(np.arange(0, d) * ar, np.arange(0, h),
                          np.arange(0, w), indexing='ij')
    r = np.sqrt((z - zc)**2+(y - yc)**2+(x - xc)**2) - R
    im = np.random.random(shape) * noise
    mask = np.abs(r) < 2 * FWHM
    im[mask] = im[mask] + \
               np.exp((r[mask] / sigma)**2/-2)/(sigma*np.sqrt(2*np.pi))
    im = im / im.max()  # use full range
    return (im * 255).astype(np.uint8)


def gen_fluct_vesicle(r, sigma, kappa, n_points, n_frames=1):
    modes = np.arange(1, n_points // 2 + 1)
    theta = np.linspace(-np.pi, np.pi, n_points, endpoint=False)
    powersp = powerspectrum(modes / r, sigma, kappa)
    fft_magn = np.sqrt(powersp / (r * 2 * np.pi))
    fft_magn[0] = 0  # first order = displacement. turn this off.
    coords = np.empty((n_frames, n_points, 2))
    for i in range(n_frames):
        phases = np.random.random(len(powersp))*2*np.pi-np.pi
        fft = fft_magn*np.exp(-1j*phases)
        ifft = np.sum(fft[:, np.newaxis] *
                      np.exp(1j * modes[:, np.newaxis] * theta[np.newaxis, :]),
                      axis=0) * 2
        r_dev = np.real(ifft)
        coords[i] = np.array(algebraic.to_cartesian(r+r_dev, theta)).T
    return np.squeeze(coords)


class TestEllipsoid(unittest.TestCase):
    def setUp(self):
        self.shape = (250, 300, 300)
        self.center = (125, 150, 150)
        self.R = 100
        self.ar = 2.5
        self.FWHM = 5
        self.N = 1
        self.noise = 0.02

    def test_locate_noisy(self):
        noise = 0.2
        self.im = gen_artificial_ellipsoid(self.shape, self.center,
                                           self.R, self.FWHM, self.ar,
                                           noise)
        result, _ = locate_ellipsoid(self.im)
        assert_almost_equal(result['zc'], self.center[0], 0)
        assert_almost_equal(result['yc'], self.center[1], 0)
        assert_almost_equal(result['xc'], self.center[2], 0)
        assert_almost_equal(result['xr'], self.R, 0)
        assert_almost_equal(result['yr'], self.R, 0)
        assert_almost_equal(result['zr'], self.R / self.ar, 0)

    def test_locate_ar(self):
        for i in range(self.N):
            ar = np.random.random() * 4 + 1
            self.im = gen_artificial_ellipsoid(self.shape, self.center,
                                               self.R, self.FWHM, ar,
                                               self.noise)
            result, _ = locate_ellipsoid(self.im)
            assert_almost_equal(result['zc'], self.center[0], 1)
            assert_almost_equal(result['yc'], self.center[1], 1)
            assert_almost_equal(result['xc'], self.center[2], 1)
            assert_almost_equal(result['xr'], self.R, 0)            
            assert_almost_equal(result['yr'], self.R, 0)
            assert_almost_equal(result['zr'], self.R / ar, 0)

    def test_change_center(self):
        for i in range(self.N):
            padding = [self.R / self.ar + self.FWHM * 3,
                       self.R + self.FWHM * 3,
                       self.R + self.FWHM * 3]

            center = tuple([np.random.random() * (s - 2 * p) + p
                            for (s, p) in zip(self.shape, padding)])
            self.im = gen_artificial_ellipsoid(self.shape, center,
                                               self.R, self.FWHM, self.ar,
                                               self.noise)
            result, _ = locate_ellipsoid(self.im)
            assert_almost_equal(result['zc'], center[0], 1)
            assert_almost_equal(result['yc'], center[1], 1)
            assert_almost_equal(result['xc'], center[2], 1)
            assert_almost_equal(result['xr'], self.R, 0)
            assert_almost_equal(result['yr'], self.R, 0)
            assert_almost_equal(result['zr'], self.R / self.ar, 0)

    def test_change_radius(self):
        for i in range(self.N):
            radius = np.random.random() * 50 + 50
            self.im = gen_artificial_ellipsoid(self.shape, self.center,
                                               radius, self.FWHM, self.ar,
                                               self.noise)
            result, _ = locate_ellipsoid(self.im)
            assert_almost_equal(result['zc'], self.center[0], 1)
            assert_almost_equal(result['yc'], self.center[1], 1)
            assert_almost_equal(result['xc'], self.center[2], 1)
            assert_almost_equal(result['xr'], radius, 0)            
            assert_almost_equal(result['yr'], radius, 0)
            assert_almost_equal(result['zr'], radius / self.ar, 0)

    def test_center_no_refine(self):
        for i in range(self.N):
            padding = [self.R / self.ar + self.FWHM * 3,
                       self.R + self.FWHM * 3,
                       self.R + self.FWHM * 3]

            center = tuple([np.random.random() * (s - 2 * p) + p
                            for (s, p) in zip(self.shape, padding)])
            self.im = gen_artificial_ellipsoid(self.shape, center, 
                                               self.R, self.FWHM, self.ar,
                                               self.noise)
            result = find_ellipsoid(self.im)
            assert_almost_equal(result[3], center[0], 0)
            assert_almost_equal(result[4], center[1], 0)
            assert_almost_equal(result[5], center[2], 0)   


class TestFluctuation(unittest.TestCase):
    def setUp(self):
        self.max_mode = 100
        self.modes = np.arange(1, self.max_mode + 1)
        self.n_points = 512
        self.N = 10

#    def test_fft(self):
#        theta = np.linspace(0, 2*np.pi, 512)
#        base_freqs = np.array([5, 15, 25, 35])
#        max_int = 20
#        modes = np.arange(1, 40 + 1)
#        for i in range(self.N):
#            freqs = base_freqs + np.random.random(4)*3
#            ints = np.random.random(4) * max_int + 10
#            phases = np.random.random(4) * 2*np.pi - np.pi
#            a = np.sum(ints[:, np.newaxis] *
#                       np.sin(freqs[:, np.newaxis]*theta[np.newaxis, :] +
#                              phases[:, np.newaxis]), axis=0)
#            fft_np = (np.abs(np.fft.rfft(a, n=512)[1:40+1])/512*2*np.pi)**2
#            fft_trapz = fluctuation_spectrum(theta, a, r=2*np.pi, modes=modes)
#            assert_allclose(np.sort(fft_np)[-4:], np.sort(fft_trapz)[-4:],
#                            rtol=.5)

    def test_fit_powerspectrum(self):
        for i in range(self.N):
            r = np.random.random() * 90 + 10
            q = self.modes / r
            kappa = np.random.random() * 99 + 1  # kT
            sigma = 10**(np.random.random() * 3 - 2) 
            powersp = powerspectrum(q, sigma, kappa)
            actual = fit_fluctuation_2param(q, powersp)
            assert_allclose(actual, (sigma, kappa), rtol=0.01)

    def test_fluctuationspectrum(self):
        for i in range(self.N):
            r = np.random.random() * 90 + 10
            kappa = np.random.random() * 99 + 1  # kT
            sigma = 10**(np.random.random() * 3 - 2)
            q = np.arange(1, self.n_points // 2 + 1) / r
            expected = powerspectrum(q, sigma, kappa)
            expected[0] = 0  # first mode = 0

            coords = gen_fluct_vesicle(r, sigma, kappa, self.n_points)
            center, r, theta, r_dev = circle_deviation(coords)
            actual = fluctuation_spectrum(theta, r_dev, r, self.modes)
            assert_allclose(actual[6:], expected[6:self.max_mode],
                            rtol=0.1, atol=0.1)

    def test_efluctuationspectrum(self):
        n_frames = 100
        for i in range(self.N):
            r = np.random.random() * 90 + 10
            kappa = np.random.random() * 99 + 1  # kT
            sigma = 10**(np.random.random() * 3 - 2)
            q = np.arange(1, self.n_points // 2 + 1) / r
            expected = powerspectrum(q, sigma, kappa)
            expected[0] = 0  # first mode = 0

            coords = gen_fluct_vesicle(r, sigma, kappa, self.n_points,
                                       n_frames)
            _, actual = efluctuation_spectrum(coords, self.max_mode)
            assert_allclose(actual[6:], expected[6:self.max_mode],
                            rtol=0.01, atol=0.01)

    def test_determine_fit_spectrum(self):
        n_frames = 100
        for i in range(self.N):
            r = np.random.random() * 90 + 10
            kappa = np.random.random() * 99 + 1  # kT
            sigma = 10**(np.random.random() * 3 - 2)
            q = np.arange(1, self.n_points // 2 + 1) / r
            expected = powerspectrum(q, sigma, kappa)
            expected[0] = 0  # first mode = 0

            coords = gen_fluct_vesicle(r, sigma, kappa, self.n_points,
                                       n_frames)
            _, actual_sp = efluctuation_spectrum(coords, self.max_mode)
            actual = fit_fluctuation_2param(q[:self.max_mode], actual_sp)
            assert_allclose(actual, (sigma, kappa), rtol=0.05)

    def test_efluctuationspectrum_part(self):
        n_frames = 100
        for i in range(self.N):
            r = np.random.random() * 90 + 10
            kappa = np.random.random() * 99 + 1  # kT
            sigma = 10**(np.random.random() * 3 - 2)
            coords = gen_fluct_vesicle(r, sigma, kappa, self.n_points,
                                       n_frames)

            for part in range(6, 11):
                max_mode = self.max_mode // part
                modes = np.arange(1, max_mode + 1)
                expected = powerspectrum_part(modes, sigma, kappa, r,
                                              self.n_points, part)
                _, actual = efluctuation_spectrum(coords, max_mode, part=part)
#                print('relative error', actual[6//part:]/expected[6//part:] - 1)
#                print('absolute error', actual[6//part:] - expected[6//part:]) 
                assert_allclose(actual[6//part:], expected[6//part:],
                                rtol=0.5, atol=0.1)  # HIGH TOLERANCE

if __name__ == '__main__':
    nose.runmodule(argv=[__file__, '-vvs', '-x'], exit=False)
