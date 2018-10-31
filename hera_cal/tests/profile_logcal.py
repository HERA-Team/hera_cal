from __future__ import print_function
import hera_cal.redcal as om
import hera_cal.omni
import omnical.calib
import numpy as np
import unittest
import time
from copy import deepcopy
import pstats
import cProfile

np.random.seed(0)
# SHAPE = (1,1024)
SHAPE = (60, 1024)
# SHAPE = (2,1024)

NANTS = 18


def build_linear_array(nants, sep=14.7):
    antpos = {i: np.array([sep * i, 0, 0]) for i in range(nants)}
    return antpos


def build_hex_array(hexNum, sep=14.7):
    antpos, i = {}, 0
    for row in range(hexNum - 1, -(hexNum), -1):
        for col in range(2 * hexNum - abs(row) - 1):
            xPos = ((-(2 * hexNum - abs(row)) + 2) / 2.0 + col) * sep
            yPos = row * sep * 3**.5 / 2
            antpos[i] = np.array([xPos, yPos, 0])
            i += 1
    return antpos


antpos = build_linear_array(NANTS)
reds = om.get_reds(antpos, pols=['xx'], pol_mode='1pol')
info = om.RedundantCalibrator(reds)
gains, true_vis, d = om.sim_red_data(reds, shape=SHAPE, gain_scatter=.0099999)
d = {key: value + 1e-3 * om.noise(value.shape) for key, value in d.items()}
d = {key: value.astype(np.complex64) for key, value in d.items()}
w = dict([(k, 1.) for k in d.keys()])


class TestRedundantCalibrator(unittest.TestCase):
    def setUp(self):
        self.pr = cProfile.Profile()
        self.pr.enable()

    def tearDown(self):
        p = pstats.Stats(self.pr)
        p.strip_dirs()
        p.sort_stats('cumtime')
        p.print_stats(20)

    def test_logcal(self):
        NANTS = 18
        antpos = build_linear_array(NANTS)
        reds = om.get_reds(antpos, pols=['xx'], pol_mode='1pol')
        info = om.RedundantCalibrator(reds)
        gains, true_vis, d = om.sim_red_data(reds, shape=SHAPE, gain_scatter=.05)
        d = {key: value.astype(np.complex64) for key, value in d.items()}
        w = dict([(k, 1.) for k in d.keys()])
        t0 = time.time()
        for i in xrange(1):
            sol = info.logcal(d)
        # print('logcal', time.time() - t0)
        for i in xrange(NANTS):
            self.assertEqual(sol[(i, 'x')].shape, SHAPE)
        for bls in reds:
            ubl = sol[bls[0]]
            self.assertEqual(ubl.shape, SHAPE)
            for bl in bls:
                d_bl = d[bl]
                mdl = sol[(bl[0], 'x')] * sol[(bl[1], 'x')].conj() * ubl
                # np.testing.assert_almost_equal(np.abs(d_bl), np.abs(mdl), 10)
                # np.testing.assert_almost_equal(np.angle(d_bl * mdl.conj()), 0, 10)
                np.testing.assert_almost_equal(np.abs(d_bl), np.abs(mdl), 5)
                np.testing.assert_almost_equal(np.angle(d_bl * mdl.conj()), 0, 5)

    def test_omnilogcal(self):
        NANTS = 18
        antpos = build_linear_array(NANTS)
        hcreds = om.get_reds(antpos, pols=['xx'], pol_mode='1pol')
        pols = ['x']
        antpos_ideal = np.array(antpos.values())
        xs, ys, zs = antpos_ideal.T
        layout = np.arange(len(xs))
        antpos = -np.ones((NANTS * len(pols), 3))
        for ant, x, y in zip(layout.flatten(), xs.flatten(), ys.flatten()):
            for z, pol in enumerate(pols):
                z = 2**z  # exponential ensures diff xpols aren't redundant w/ each other
                i = hera_cal.omni.Antpol(ant, pol, NANTS)
                antpos[int(i), 0], antpos[int(i), 1], antpos[int(i), 2] = x, y, z
        reds = hera_cal.omni.compute_reds(NANTS, pols, antpos[:NANTS], tol=.01)
        # reds = hera_cal.omni.filter_reds(reds, **kwargs)
        info = hera_cal.omni.RedundantInfo(NANTS)
        info.init_from_reds(reds, antpos_ideal)
        # info = om.RedundantCalibrator(reds)
        # gains, true_vis, d = om.sim_red_data(hcreds, shape=SHAPE, gain_scatter=.05)
        data = {}
        for key in d.keys():
            if not data.has_key(key[:2]):
                data[key[:2]] = {}
            data[key[:2]][key[-1]] = d[key].astype(np.complex64)
        t0 = time.time()
        for i in xrange(1):
            m1, g1, v1 = omnical.calib.logcal(data, info)
        # print('omnilogcal', time.time() - t0)
        for i in xrange(NANTS):
            self.assertEqual(g1['x'][i].shape, SHAPE)
        for bls in reds:
            ubl = v1['xx'][(int(bls[0][0]), int(bls[0][1]))]
            self.assertEqual(ubl.shape, SHAPE)
            for bl in bls:
                d_bl = data[(int(bl[0]), int(bl[1]))]['xx']
                mdl = g1['x'][bl[0]] * g1['x'][bl[1]].conj() * ubl
                np.testing.assert_almost_equal(np.abs(d_bl), np.abs(mdl), 5)
                np.testing.assert_almost_equal(np.angle(d_bl * mdl.conj()), 0, 5)

    def test_lincal(self):
        NANTS = 18
        antpos = build_linear_array(NANTS)
        reds = om.get_reds(antpos, pols=['xx'], pol_mode='1pol')
        info = om.RedundantCalibrator(reds)
        gains, true_vis, d = om.sim_red_data(reds, shape=SHAPE, gain_scatter=.0099999)
        d = {key: value.astype(np.complex64) for key, value in d.items()}
        w = dict([(k, 1.) for k in d.keys()])
        sol0 = dict([(k, np.ones_like(v)) for k, v in gains.items()])
        sol0.update(info.compute_ubls(d, sol0))
        sol0 = {k: v.astype(np.complex64) for k, v in sol0.items()}
        # sol0 = info.logcal(d)
        # for k in sol0: sol0[k] += .01*capo.oqe.noise(sol0[k].shape)
        meta, sol = info.lincal(d, sol0)
        for i in xrange(NANTS):
            self.assertEqual(sol[(i, 'x')].shape, SHAPE)
        for bls in reds:
            ubl = sol[bls[0]]
            self.assertEqual(ubl.shape, SHAPE)
            for bl in bls:
                d_bl = d[bl]
                mdl = sol[(bl[0], 'x')] * sol[(bl[1], 'x')].conj() * ubl
                np.testing.assert_almost_equal(np.abs(d_bl), np.abs(mdl), 5)
                np.testing.assert_almost_equal(np.angle(d_bl * mdl.conj()), 0, 5)

    def test_omnical(self):
        NANTS = 18
        antpos = build_linear_array(NANTS)
        reds = om.get_reds(antpos, pols=['xx'], pol_mode='1pol')
        info = om.RedundantCalibrator(reds)
        # gains, true_vis, d = om.sim_red_data(reds, shape=SHAPE, gain_scatter=.0099999)
        # d = {key:value.astype(np.complex64) for key,value in d.items()}
        w = dict([(k, 1.) for k in d.keys()])
        sol0 = dict([(k, np.ones_like(v)) for k, v in gains.items()])
        sol0.update(info.compute_ubls(d, sol0))
        sol0 = {k: v.astype(np.complex64) for k, v in sol0.items()}
        meta, sol = info.omnical(d, sol0, gain=.5, maxiter=500, check_after=30, check_every=6)
        # meta, sol = info.omnical(d, sol0, gain=.5, maxiter=50, check_after=1, check_every=1)
        # print(meta)
        for i in xrange(NANTS):
            self.assertEqual(sol[(i, 'x')].shape, SHAPE)
        for bls in reds:
            ubl = sol[bls[0]]
            self.assertEqual(ubl.shape, SHAPE)
            for bl in bls:
                d_bl = d[bl]
                mdl = sol[(bl[0], 'x')] * sol[(bl[1], 'x')].conj() * ubl
                np.testing.assert_almost_equal(np.abs(d_bl), np.abs(mdl), 5)
                np.testing.assert_almost_equal(np.angle(d_bl * mdl.conj()), 0, 5)

    def test_omnical_original(self):
        NANTS = 18
        antpos = build_linear_array(NANTS)
        hcreds = om.get_reds(antpos, pols=['xx'], pol_mode='1pol')
        pols = ['x']
        antpos_ideal = np.array(antpos.values())
        xs, ys, zs = antpos_ideal.T
        layout = np.arange(len(xs))
        antpos = -np.ones((NANTS * len(pols), 3))
        for ant, x, y in zip(layout.flatten(), xs.flatten(), ys.flatten()):
            for z, pol in enumerate(pols):
                z = 2**z  # exponential ensures diff xpols aren't redundant w/ each other
                i = hera_cal.omni.Antpol(ant, pol, NANTS)
                antpos[int(i), 0], antpos[int(i), 1], antpos[int(i), 2] = x, y, z
        reds = hera_cal.omni.compute_reds(NANTS, pols, antpos[:NANTS], tol=.01)
        # reds = hera_cal.omni.filter_reds(reds, **kwargs)
        info = hera_cal.omni.RedundantInfo(NANTS)
        info.init_from_reds(reds, antpos_ideal)
        # info = om.RedundantCalibrator(reds)
        # gains, true_vis, d = om.sim_red_data(hcreds, shape=SHAPE, gain_scatter=.0099999)
        data = {}
        for key in d.keys():
            if not data.has_key(key[:2]):
                data[key[:2]] = {}
            data[key[:2]][key[-1]] = d[key].astype(np.complex64)
        t0 = time.time()
        for i in xrange(1):
            m1, g1, v1 = omnical.calib.lincal(data, info, maxiter=50)
        # print('omnilincal', time.time() - t0)
        for i in xrange(NANTS):
            self.assertEqual(g1['x'][i].shape, SHAPE)
        for bls in reds:
            ubl = v1['xx'][(int(bls[0][0]), int(bls[0][1]))]
            self.assertEqual(ubl.shape, SHAPE)
            for bl in bls:
                d_bl = data[(int(bl[0]), int(bl[1]))]['xx']
                mdl = g1['x'][bl[0]] * g1['x'][bl[1]].conj() * ubl
                np.testing.assert_almost_equal(np.abs(d_bl), np.abs(mdl), 5)
                np.testing.assert_almost_equal(np.angle(d_bl * mdl.conj()), 0, 5)


'''

    def test_lincal_hex_end_to_end_1pol_with_remove_degen_and_firstcal(self):
        antpos = build_hex_array(3)
        reds = om.get_reds(antpos, pols=['xx'], pol_mode='1pol')
        rc = om.RedundantCalibrator(reds)
        freqs = np.linspace(.1, .2, 10)
        gains, true_vis, d = om.sim_red_data(reds, gain_scatter=.1, shape=(1, len(freqs)))
        fc_delays = {ant: 100 * np.random.randn() for ant in gains.keys()}  # in ns
        fc_gains = {ant: np.reshape(np.exp(-2.0j * np.pi * freqs * delay), (1, len(freqs))) for ant, delay in fc_delays.items()}
        for ant1, ant2, pol in d.keys():
            d[(ant1, ant2, pol)] *= fc_gains[(ant1, pol[0])] * np.conj(fc_gains[(ant2, pol[1])])
        for ant in gains.keys():
            gains[ant] *= fc_gains[ant]

        w = dict([(k, 1.) for k in d.keys()])
        sol0 = rc.logcal(d, sol0=fc_gains, wgts=w)
        meta, sol = rc.lincal(d, sol0, wgts=w)

        np.testing.assert_array_less(meta['iter'], 50 * np.ones_like(meta['iter']))
        np.testing.assert_almost_equal(meta['chisq'], np.zeros_like(meta['chisq']), decimal=10)

        np.testing.assert_almost_equal(meta['chisq'], 0, 10)
        for i in xrange(len(antpos)):
            self.assertEqual(sol[(i, 'x')].shape, (1, len(freqs)))
        for bls in reds:
            ubl = sol[bls[0]]
            self.assertEqual(ubl.shape, (1, len(freqs)))
            for bl in bls:
                d_bl = d[bl]
                mdl = sol[(bl[0], 'x')] * sol[(bl[1], 'x')].conj() * ubl
                np.testing.assert_almost_equal(np.abs(d_bl), np.abs(mdl), 10)
                np.testing.assert_almost_equal(np.angle(d_bl * mdl.conj()), 0, 10)

        sol_rd = rc.remove_degen(antpos, sol)
        g, v = om.get_gains_and_vis_from_sol(sol_rd)
        ants = [key for key in sol_rd.keys() if len(key) == 2]
        gainSols = np.array([sol_rd[ant] for ant in ants])
        meanSqAmplitude = np.mean([np.abs(g[key1] * g[key2]) for key1 in g.keys()
                                   for key2 in g.keys() if key1[1] == 'x' and key2[1] == 'x' and key1[0] != key2[0]], axis=0)
        np.testing.assert_almost_equal(meanSqAmplitude, 1, 10)
        #np.testing.assert_almost_equal(np.mean(np.angle(gainSols), axis=0), 0, 10)

        for bls in reds:
            ubl = sol_rd[bls[0]]
            self.assertEqual(ubl.shape, (1, len(freqs)))
            for bl in bls:
                d_bl = d[bl]
                mdl = sol_rd[(bl[0], 'x')] * sol_rd[(bl[1], 'x')].conj() * ubl
                np.testing.assert_almost_equal(np.abs(d_bl), np.abs(mdl), 10)
                np.testing.assert_almost_equal(np.angle(d_bl * mdl.conj()), 0, 10)

        sol_rd = rc.remove_degen(antpos, sol, degen_sol=gains)
        g, v = om.get_gains_and_vis_from_sol(sol_rd)
        meanSqAmplitude = np.mean([np.abs(g[key1] * g[key2]) for key1 in g.keys()
                                   for key2 in g.keys() if key1[1] == 'x' and key2[1] == 'x' and key1[0] != key2[0]], axis=0)
        degenMeanSqAmplitude = np.mean([np.abs(gains[key1] * gains[key2]) for key1 in g.keys()
                                        for key2 in g.keys() if key1[1] == 'x' and key2[1] == 'x' and key1[0] != key2[0]], axis=0)
        np.testing.assert_almost_equal(meanSqAmplitude, degenMeanSqAmplitude, 10)
        #np.testing.assert_almost_equal(np.mean(np.angle(gainSols), axis=0), 0, 10)

        for key, val in sol_rd.items():
            if len(key) == 2:
                np.testing.assert_almost_equal(val, gains[key], 10)
            if len(key) == 3:
                np.testing.assert_almost_equal(val, true_vis[key], 10)

        rc.pol_mode = 'unrecognized_pol_mode'
        with self.assertRaises(ValueError):
            sol_rd = rc.remove_degen(antpos, sol)
'''


if __name__ == '__main__':
    unittest.main()
