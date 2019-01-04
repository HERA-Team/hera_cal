# -*- coding: utf-8 -*-
# Copyright 2018 the HERA Project
# Licensed under the MIT License

"""Unit tests for the hera_cal.apply_cal module."""

from __future__ import absolute_import, division, print_function

import unittest
import numpy as np
import os
import sys


from hera_cal import io
from hera_cal import noise
from hera_cal.data import DATA_PATH
from hera_cal.utils import split_pol
from hera_cal.apply_cal import apply_cal


class Test_Noise(unittest.TestCase):

    def test_interleaved_noise_variance_estimate(self):
        const_test = noise.interleaved_noise_variance_estimate(np.ones((10, 10)))
        np.testing.assert_array_equal(const_test, 0)

        np.random.seed(21)
        gauss_test = np.mean(noise.interleaved_noise_variance_estimate(np.random.randn(1000, 1000)))
        np.testing.assert_almost_equal(gauss_test, 1, decimal=2)

        kernels = [('2x2 diff', [[1, -1], [-1, 1]]),
                   ('2D plus', [[0, 1, 0], [1, -4, 1], [0, 1, 0]]),
                   ('2D box', [[1, 1, 1], [1, -8, 1], [1, 1, 1]]),
                   ('2D hybrid', [[1, -2, 1], [-2, 4, -2], [1, -2, 1]]),
                   ('1D 2-term', [[1, -1]]),
                   ('1D 3-term', [[1, -2, 1]]),
                   ('1D 5-term', [[-1, 4, -6, 4, -1]]),
                   ('1D 7-term', [[2, -9, 18, -22, 18, -9, 2]])]
        for kname, kernel in kernels:
            gauss_test = np.mean(noise.interleaved_noise_variance_estimate(np.random.randn(1000, 1000), kernel=kernel))
            np.testing.assert_almost_equal(gauss_test, 1, decimal=2)

        with self.assertRaises(AssertionError):
            noise.interleaved_noise_variance_estimate(np.random.randn(10, 10), kernel=[[.5, 1.0, .5]])
        with self.assertRaises(AssertionError):
            noise.interleaved_noise_variance_estimate(np.random.randn(10, 10), kernel=[-1, 1])

    def test_predict_noise_variance_from_autos(self):
        hd = io.HERAData(os.path.join(DATA_PATH, 'zen.2458098.43124.subband.uvh5'))
        data, flags, nsamples = hd.read()
        for k in data.keys():
            if k[0] != k[1]:
                sigmasq = noise.predict_noise_variance_from_autos(k, data)
                noise_var = noise.interleaved_noise_variance_estimate(data[k])
                np.testing.assert_array_equal(np.abs(np.mean(np.mean(noise_var, axis=0) / np.mean(sigmasq, axis=0)) - 1) <= .1, True)
                times = hd.times_by_bl[k[:2]]
                sigmasq2 = noise.predict_noise_variance_from_autos(k, data, df=(hd.freqs[1] - hd.freqs[0]), dt=((times[1] - times[0]) * 24. * 3600.))
                np.testing.assert_array_equal(sigmasq, sigmasq2)

    def test_per_antenna_noise_std(self):
        infile = os.path.join(DATA_PATH, 'zen.2458098.43124.downsample.uvh5')
        hd = io.HERAData(infile)
        data, _, _ = hd.read()
        n = noise.per_antenna_noise_std(data)
        for bl in data.keys():
            if (bl[0] == bl[1]) and (split_pol(bl[2])[0] == split_pol(bl[2])[1]):
                self.assertTrue(bl in n)
                self.assertEqual(n[bl].shape, data[bl].shape)
                np.testing.assert_array_equal(n[bl].imag, 0.0)
            else:
                self.assertFalse(bl in n)

    def test_write_per_antenna_noise_std_from_autos(self):
        infile = os.path.join(DATA_PATH, 'zen.2458098.43124.downsample.uvh5')
        calfile = os.path.join(DATA_PATH, 'test_input/zen.2458098.43124.downsample.omni.calfits')
        outfile = os.path.join(DATA_PATH, 'test_output/noise.uvh5')
        noise.write_per_antenna_noise_std_from_autos(infile, outfile, calfile=calfile, add_to_history='testing', clobber=True)

        hd = io.HERAData(outfile)
        self.assertTrue('testing' in hd.history.replace('\n', '').replace(' ', ''))
        self.assertTrue('Thisfilewasproducedbythefunction' in hd.history.replace('\n', '').replace(' ', ''))
        n, f, _ = hd.read()
        hc = io.HERACal(calfile)
        g, gf, _, _ = hc.read()
        for bl in n.keys():
            self.assertEqual(bl[0], bl[1])
            self.assertEqual(split_pol(bl[2])[0], split_pol(bl[2])[1])
            np.testing.assert_array_equal(n[bl].imag, 0.0)
            np.testing.assert_array_equal(f[bl], gf[bl[0], split_pol(bl[2])[0]])
        os.remove(outfile)

    def test_noise_std_argparser(self):
        sys.argv = [sys.argv[0], 'a', 'b', '--calfile', 'd']
        a = noise.noise_std_argparser()
        args = a.parse_args()
        self.assertEqual(args.infile, 'a')
        self.assertEqual(args.outfile, 'b')
        self.assertEqual(args.calfile, ['d'])


if __name__ == '__main__':
    unittest.main()
