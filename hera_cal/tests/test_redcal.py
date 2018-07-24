import hera_cal.redcal as om
import numpy as np
import unittest
from copy import deepcopy
from hera_cal.utils import split_pol, conj_pol

np.random.seed(0)


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


class TestMethods(unittest.TestCase):

    def test_noise(self):
        n = om.noise((1024, 1024))
        self.assertEqual(n.shape, (1024, 1024))
        self.assertAlmostEqual(np.var(n), 1, 2)

    def test_sim_red_data(self):
        antpos = build_linear_array(10)
        reds = om.get_reds(antpos, pols=['XX'], pol_mode='1pol')
        gains, true_vis, data = om.sim_red_data(reds)
        self.assertEqual(len(gains), 10)
        self.assertEqual(len(data), 45)
        for bls in reds:
            bl0 = bls[0]
            ai, aj, pol = bl0
            ans0 = data[bl0] / (gains[(ai, 'jxx')] * gains[(aj, 'jxx')].conj())
            for bl in bls[1:]:
                ai, aj, pol = bl
                ans = data[bl] / (gains[(ai, 'jxx')] * gains[(aj, 'jxx')].conj())
                np.testing.assert_almost_equal(ans0, ans, 7)

        reds = om.get_reds(antpos, pols=['XX', 'YY', 'XY', 'YX'], pol_mode='4pol')
        gains, true_vis, data = om.sim_red_data(reds)
        self.assertEqual(len(gains), 20)
        self.assertEqual(len(data), 4 * (45))
        for bls in reds:
            bl0 = bls[0]
            ai, aj, pol = bl0
            ans0xx = data[(ai, aj, 'XX',)] / (gains[(ai, 'jxx')] * gains[(aj, 'jxx')].conj())
            ans0xy = data[(ai, aj, 'XY',)] / (gains[(ai, 'jxx')] * gains[(aj, 'jyy')].conj())
            ans0yx = data[(ai, aj, 'YX',)] / (gains[(ai, 'jyy')] * gains[(aj, 'jxx')].conj())
            ans0yy = data[(ai, aj, 'YY',)] / (gains[(ai, 'jyy')] * gains[(aj, 'jyy')].conj())
            for bl in bls[1:]:
                ai, aj, pol = bl
                ans_xx = data[(ai, aj, 'XX',)] / (gains[(ai, 'jxx')] * gains[(aj, 'jxx')].conj())
                ans_xy = data[(ai, aj, 'XY',)] / (gains[(ai, 'jxx')] * gains[(aj, 'jyy')].conj())
                ans_yx = data[(ai, aj, 'YX',)] / (gains[(ai, 'jyy')] * gains[(aj, 'jxx')].conj())
                ans_yy = data[(ai, aj, 'YY',)] / (gains[(ai, 'jyy')] * gains[(aj, 'jyy')].conj())
                np.testing.assert_almost_equal(ans0xx, ans_xx, 7)
                np.testing.assert_almost_equal(ans0xy, ans_xy, 7)
                np.testing.assert_almost_equal(ans0yx, ans_yx, 7)
                np.testing.assert_almost_equal(ans0yy, ans_yy, 7)

        reds = om.get_reds(antpos, pols=['XX', 'YY', 'XY', 'yX'], pol_mode='4pol_minV')
        gains, true_vis, data = om.sim_red_data(reds)
        self.assertEqual(len(gains), 20)
        self.assertEqual(len(data), 4 * (45))
        for bls in reds:
            bl0 = bls[0]
            ai, aj, pol = bl0
            ans0xx = data[(ai, aj, 'XX',)] / (gains[(ai, 'jxx')] * gains[(aj, 'jxx')].conj())
            ans0xy = data[(ai, aj, 'XY',)] / (gains[(ai, 'jxx')] * gains[(aj, 'jyy')].conj())
            ans0yx = data[(ai, aj, 'YX',)] / (gains[(ai, 'jyy')] * gains[(aj, 'jxx')].conj())
            ans0yy = data[(ai, aj, 'YY',)] / (gains[(ai, 'jyy')] * gains[(aj, 'jyy')].conj())
            np.testing.assert_almost_equal(ans0xy, ans0yx, 7)
            for bl in bls[1:]:
                ai, aj, pol = bl
                ans_xx = data[(ai, aj, 'XX',)] / (gains[(ai, 'jxx')] * gains[(aj, 'jxx')].conj())
                ans_xy = data[(ai, aj, 'XY',)] / (gains[(ai, 'jxx')] * gains[(aj, 'jyy')].conj())
                ans_yx = data[(ai, aj, 'YX',)] / (gains[(ai, 'jyy')] * gains[(aj, 'jxx')].conj())
                ans_yy = data[(ai, aj, 'YY',)] / (gains[(ai, 'jyy')] * gains[(aj, 'jyy')].conj())
                np.testing.assert_almost_equal(ans0xx, ans_xx, 7)
                np.testing.assert_almost_equal(ans0xy, ans_xy, 7)
                np.testing.assert_almost_equal(ans0yx, ans_yx, 7)
                np.testing.assert_almost_equal(ans0yy, ans_yy, 7)

    def test_check_polLists_minV(self):
        polLists = [['XY']]
        self.assertFalse(om.check_polLists_minV(polLists))
        polLists = [['XX', 'XY']]
        self.assertFalse(om.check_polLists_minV(polLists))
        polLists = [['XX', 'XY', 'YX']]
        self.assertFalse(om.check_polLists_minV(polLists))
        polLists = [['XY', 'YX'], ['XX'], ['YY'], ['XX'], ['YX', 'XY'], ['YY']]
        self.assertTrue(om.check_polLists_minV(polLists))

    def test_parse_pol_mode(self):
        reds = [[(0, 1, 'XX')]]
        self.assertEqual(om.parse_pol_mode(reds), '1pol')
        reds = [[(0, 1, 'XX')], [(0, 1, 'YY')]]
        self.assertEqual(om.parse_pol_mode(reds), '2pol')
        reds = [[(0, 1, 'XX')], [(0, 1, 'XY')], [(0, 1, 'YX')], [(0, 1, 'YY')]]
        self.assertEqual(om.parse_pol_mode(reds), '4pol')
        reds = [[(0, 1, 'XX')], [(0, 1, 'XY'), (0, 1, 'YX')], [(0, 1, 'YY')]]
        self.assertEqual(om.parse_pol_mode(reds), '4pol_minV')

        reds = [[(0, 1, 'XX')], [(0, 1, 'XY'), (0, 1, 'YX')], [(0, 1, 'LR')]]
        self.assertEqual(om.parse_pol_mode(reds), 'unrecognized_pol_mode')
        reds = [[(0, 1, 'XX')], [(0, 1, 'XY')]]
        self.assertEqual(om.parse_pol_mode(reds), 'unrecognized_pol_mode')
        reds = [[(0, 1, 'XY')]]
        self.assertEqual(om.parse_pol_mode(reds), 'unrecognized_pol_mode')
        reds = [[(0, 1, 'XX')], [(0, 1, 'XY'), (0, 1, 'YY')], [(0, 1, 'YX')]]
        self.assertEqual(om.parse_pol_mode(reds), 'unrecognized_pol_mode')

    def test_get_pos_red(self):

        pos = build_hex_array(3, sep=14.7)
        self.assertEqual(len(om.get_pos_reds(pos)), 30)

        pos = build_hex_array(7, sep=14.7)
        self.assertEqual(len(om.get_pos_reds(pos)), 234)
        for ant, r in pos.items():
            pos[ant] += [0, 0, 1 * r[0] - .5 * r[1]]
        self.assertEqual(len(om.get_pos_reds(pos)), 234)

        pos = build_hex_array(7, sep=1)
        self.assertLess(len(om.get_pos_reds(pos)), 234)
        self.assertEqual(len(om.get_pos_reds(pos, bl_error_tol=.1)), 234)

        pos = build_hex_array(7, sep=14.7)
        blerror = 1.0 - 1e-12
        error = blerror / 4
        for key, val in pos.items():
            th = np.random.choice([0, np.pi / 2, np.pi])
            phi = np.random.choice([0, np.pi / 2, np.pi, 3 * np.pi / 2])
            pos[key] = val + error * np.array([np.sin(th) * np.cos(phi), np.sin(th) * np.sin(phi), np.cos(th)])
        self.assertEqual(len(om.get_pos_reds(pos, bl_error_tol=1.0)), 234)
        self.assertGreater(len(om.get_pos_reds(pos, bl_error_tol=.99)), 234)

        pos = {0: np.array([0, 0, 0]), 1: np.array([20, 0, 0]), 2: np.array([10, 0, 0])}
        self.assertEqual(om.get_pos_reds(pos), [[(0, 2), (2, 1)], [(0, 1)]])
        self.assertEqual(om.get_pos_reds(pos, low_hi=True), [[(0, 2), (1, 2)], [(0, 1)]])

    def test_add_pol_reds(self):
        reds = [[(1, 2)]]
        polReds = om.add_pol_reds(reds, pols=['XX'], pol_mode='1pol')
        self.assertEqual(polReds, [[(1, 2, 'XX')]])
        polReds = om.add_pol_reds(reds, pols=['XX', 'YY'], pol_mode='2pol')
        self.assertEqual(polReds, [[(1, 2, 'XX')], [(1, 2, 'YY')]])
        polReds = om.add_pol_reds(reds, pols=['XX', 'XY', 'YX', 'YY'], pol_mode='4pol')
        self.assertEqual(polReds, [[(1, 2, 'XX')], [(1, 2, 'XY')], [(1, 2, 'YX')], [(1, 2, 'YY')]])
        polReds = om.add_pol_reds(reds, pols=['XX', 'XY', 'YX', 'YY'], pol_mode='4pol_minV')
        self.assertEqual(polReds, [[(1, 2, 'XX')], [(1, 2, 'XY'), (1, 2, 'YX')], [(1, 2, 'YY')]])

        polReds = om.add_pol_reds(reds, pols=['XX', 'YY'], pol_mode='2pol', ex_ants=[(2, 'jyy')])
        self.assertEqual(polReds, [[(1, 2, 'XX')], []])
        polReds = om.add_pol_reds(reds, pols=['XX', 'XY', 'YX', 'YY'], pol_mode='4pol', ex_ants=[(2, 'jyy')])
        self.assertEqual(polReds, [[(1, 2, 'XX')], [], [(1, 2, 'YX')], []])
        polReds = om.add_pol_reds(reds, pols=['XX', 'XY', 'YX', 'YY'], pol_mode='4pol_minV', ex_ants=[(2, 'jyy')])
        self.assertEqual(polReds, [[(1, 2, 'XX')], [(1, 2, 'YX')], []])

    def test_multiply_by_gains(self):
        vis_in = {(1, 2, 'XX'): 1.6 + 2.3j}
        gains = {(1, 'jxx'): .3 + 2.6j, (2, 'jxx'): -1.2 - 7.3j}
        vis_out = om.multiply_by_gains(vis_in, gains, target_type='vis')
        self.assertAlmostEqual(1.6 + 2.3j, vis_in[(1, 2, 'XX')], 10)
        self.assertAlmostEqual(-28.805 - 45.97j, vis_out[(1, 2, 'XX')], 10)

        gains_out = om.multiply_by_gains(gains, gains, target_type='gain')
        self.assertAlmostEqual(.3 + 2.6j, gains[(1, 'jxx')], 10)
        self.assertAlmostEqual(-6.67 + 1.56j, gains_out[(1, 'jxx')], 10)

    def test_divide_by_gains(self):
        vis_in = {(1, 2, 'XX'): 1.6 + 2.3j}
        gains = {(1, 'jxx'): .3 + 2.6j, (2, 'jxx'): -1.2 - 7.3j}
        vis_out = om.divide_by_gains(vis_in, gains, target_type='vis')
        self.assertAlmostEqual(1.6 + 2.3j, vis_in[(1, 2, 'XX')], 10)
        self.assertAlmostEqual(-0.088244747606364887 - 0.11468109538397521j, vis_out[(1, 2, 'XX')], 10)

        gains_out = om.divide_by_gains(gains, gains, target_type='gain')
        self.assertAlmostEqual(.3 + 2.6j, gains[(1, 'jxx')], 10)
        self.assertAlmostEqual(1.0, gains_out[(1, 'jxx')], 10)


class TestRedundantCalibrator(unittest.TestCase):

    def test_build_eq(self):
        antpos = build_linear_array(3)
        reds = om.get_reds(antpos, pols=['XX'], pol_mode='1pol')
        gains, true_vis, data = om.sim_red_data(reds)
        info = om.RedundantCalibrator(reds)
        eqs = info.build_eqs(data.keys())
        self.assertEqual(len(eqs), 3)
        self.assertEqual(eqs['g_0_jxx * g_1_jxx_ * u_0_XX'], (0, 1, 'XX'))
        self.assertEqual(eqs['g_1_jxx * g_2_jxx_ * u_0_XX'], (1, 2, 'XX'))
        self.assertEqual(eqs['g_0_jxx * g_2_jxx_ * u_1_XX'], (0, 2, 'XX'))

        reds = om.get_reds(antpos, pols=['XX', 'YY', 'XY', 'YX'], pol_mode='4pol')
        gains, true_vis, data = om.sim_red_data(reds)
        info = om.RedundantCalibrator(reds)
        eqs = info.build_eqs(data.keys())
        self.assertEqual(len(eqs), 3 * 4)
        self.assertEqual(eqs['g_0_jxx * g_1_jyy_ * u_4_XY'], (0, 1, 'XY'))
        self.assertEqual(eqs['g_1_jxx * g_2_jyy_ * u_4_XY'], (1, 2, 'XY'))
        self.assertEqual(eqs['g_0_jxx * g_2_jyy_ * u_5_XY'], (0, 2, 'XY'))
        self.assertEqual(eqs['g_0_jyy * g_1_jxx_ * u_6_YX'], (0, 1, 'YX'))
        self.assertEqual(eqs['g_1_jyy * g_2_jxx_ * u_6_YX'], (1, 2, 'YX'))
        self.assertEqual(eqs['g_0_jyy * g_2_jxx_ * u_7_YX'], (0, 2, 'YX'))

        reds = om.get_reds(antpos, pols=['XX', 'YY', 'XY', 'YX'], pol_mode='4pol_minV')
        gains, true_vis, data = om.sim_red_data(reds)
        info = om.RedundantCalibrator(reds)
        eqs = info.build_eqs(data.keys())
        self.assertEqual(len(eqs), 3 * 4)
        self.assertEqual(eqs['g_0_jxx * g_1_jyy_ * u_4_XY'], (0, 1, 'XY'))
        self.assertEqual(eqs['g_1_jxx * g_2_jyy_ * u_4_XY'], (1, 2, 'XY'))
        self.assertEqual(eqs['g_0_jxx * g_2_jyy_ * u_5_XY'], (0, 2, 'XY'))
        self.assertEqual(eqs['g_0_jyy * g_1_jxx_ * u_4_XY'], (0, 1, 'YX'))
        self.assertEqual(eqs['g_1_jyy * g_2_jxx_ * u_4_XY'], (1, 2, 'YX'))
        self.assertEqual(eqs['g_0_jyy * g_2_jxx_ * u_5_XY'], (0, 2, 'YX'))

    def test_solver(self):
        antpos = build_linear_array(3)
        reds = om.get_reds(antpos, pols=['XX'], pol_mode='1pol')
        info = om.RedundantCalibrator(reds)
        gains, true_vis, d = om.sim_red_data(reds)
        w = {}
        w = dict([(k, 1.) for k in d.keys()])

        def solver(data, wgts, sparse, **kwargs):
            np.testing.assert_equal(data['g_0_jxx * g_1_jxx_ * u_0_XX'], d[0, 1, 'XX'])
            np.testing.assert_equal(data['g_1_jxx * g_2_jxx_ * u_0_XX'], d[1, 2, 'XX'])
            np.testing.assert_equal(data['g_0_jxx * g_2_jxx_ * u_1_XX'], d[0, 2, 'XX'])
            if len(wgts) == 0:
                return
            np.testing.assert_equal(wgts['g_0_jxx * g_1_jxx_ * u_0_XX'], w[0, 1, 'XX'])
            np.testing.assert_equal(wgts['g_1_jxx * g_2_jxx_ * u_0_XX'], w[1, 2, 'XX'])
            np.testing.assert_equal(wgts['g_0_jxx * g_2_jxx_ * u_1_XX'], w[0, 2, 'XX'])
            return
        info._solver(solver, d)
        info._solver(solver, d, w)

    def test_logcal(self):
        NANTS = 18
        antpos = build_linear_array(NANTS)
        reds = om.get_reds(antpos, pols=['XX'], pol_mode='1pol')
        info = om.RedundantCalibrator(reds)
        gains, true_vis, d = om.sim_red_data(reds, gain_scatter=.05)
        w = dict([(k, 1.) for k in d.keys()])
        sol = info.logcal(d)
        for i in xrange(NANTS):
            self.assertEqual(sol[(i, 'jxx')].shape, (10, 10))
        for bls in reds:
            ubl = sol[bls[0]]
            self.assertEqual(ubl.shape, (10, 10))
            for bl in bls:
                d_bl = d[bl]
                mdl = sol[(bl[0], 'jxx')] * sol[(bl[1], 'jxx')].conj() * ubl
                np.testing.assert_almost_equal(np.abs(d_bl), np.abs(mdl), 10)
                np.testing.assert_almost_equal(np.angle(d_bl * mdl.conj()), 0, 10)

    def test_lincal(self):
        NANTS = 18
        antpos = build_linear_array(NANTS)
        reds = om.get_reds(antpos, pols=['XX'], pol_mode='1pol')
        info = om.RedundantCalibrator(reds)
        gains, true_vis, d = om.sim_red_data(reds, gain_scatter=.0099999)
        w = dict([(k, 1.) for k in d.keys()])
        sol0 = dict([(k, np.ones_like(v)) for k, v in gains.items()])
        sol0.update(info.compute_ubls(d, sol0))
        #sol0 = info.logcal(d)
        #for k in sol0: sol0[k] += .01*capo.oqe.noise(sol0[k].shape)
        meta, sol = info.lincal(d, sol0)
        for i in xrange(NANTS):
            self.assertEqual(sol[(i, 'jxx')].shape, (10, 10))
        for bls in reds:
            ubl = sol[bls[0]]
            self.assertEqual(ubl.shape, (10, 10))
            for bl in bls:
                d_bl = d[bl]
                mdl = sol[(bl[0], 'jxx')] * sol[(bl[1], 'jxx')].conj() * ubl
                np.testing.assert_almost_equal(np.abs(d_bl), np.abs(mdl), 10)
                np.testing.assert_almost_equal(np.angle(d_bl * mdl.conj()), 0, 10)

    def test_lincal_hex_end_to_end_1pol_with_remove_degen_and_firstcal(self):

        antpos = build_hex_array(3)
        reds = om.get_reds(antpos, pols=['XX'], pol_mode='1pol')
        rc = om.RedundantCalibrator(reds)
        freqs = np.linspace(.1, .2, 10)
        gains, true_vis, d = om.sim_red_data(reds, gain_scatter=.1, shape=(1, len(freqs)))
        fc_delays = {ant: 100 * np.random.randn() for ant in gains.keys()}  # in ns
        fc_gains = {ant: np.reshape(np.exp(-2.0j * np.pi * freqs * delay), (1, len(freqs))) for ant, delay in fc_delays.items()}
        for ant1, ant2, pol in d.keys():
            d[(ant1, ant2, pol)] *= fc_gains[(ant1, split_pol(pol)[0])] * np.conj(fc_gains[(ant2, split_pol(pol)[1])])
        for ant in gains.keys():
            gains[ant] *= fc_gains[ant]

        w = dict([(k, 1.) for k in d.keys()])
        sol0 = rc.logcal(d, sol0=fc_gains, wgts=w)
        meta, sol = rc.lincal(d, sol0, wgts=w)

        np.testing.assert_array_less(meta['iter'], 50 * np.ones_like(meta['iter']))
        np.testing.assert_almost_equal(meta['chisq'], np.zeros_like(meta['chisq']), decimal=10)

        np.testing.assert_almost_equal(meta['chisq'], 0, 10)
        for i in xrange(len(antpos)):
            self.assertEqual(sol[(i, 'jxx')].shape, (1, len(freqs)))
        for bls in reds:
            ubl = sol[bls[0]]
            self.assertEqual(ubl.shape, (1, len(freqs)))
            for bl in bls:
                d_bl = d[bl]
                mdl = sol[(bl[0], 'jxx')] * sol[(bl[1], 'jxx')].conj() * ubl
                np.testing.assert_almost_equal(np.abs(d_bl), np.abs(mdl), 10)
                np.testing.assert_almost_equal(np.angle(d_bl * mdl.conj()), 0, 10)

        sol_rd = rc.remove_degen(antpos, sol)
        g, v = om.get_gains_and_vis_from_sol(sol_rd)
        ants = [key for key in sol_rd.keys() if len(key) == 2]
        gainSols = np.array([sol_rd[ant] for ant in ants])
        meanSqAmplitude = np.mean([np.abs(g[key1] * g[key2]) for key1 in g.keys()
                                   for key2 in g.keys() if key1[1] == 'jxx' and key2[1] == 'jxx' and key1[0] != key2[0]], axis=0)
        np.testing.assert_almost_equal(meanSqAmplitude, 1, 10)
        #np.testing.assert_almost_equal(np.mean(np.angle(gainSols), axis=0), 0, 10)

        for bls in reds:
            ubl = sol_rd[bls[0]]
            self.assertEqual(ubl.shape, (1, len(freqs)))
            for bl in bls:
                d_bl = d[bl]
                mdl = sol_rd[(bl[0], 'jxx')] * sol_rd[(bl[1], 'jxx')].conj() * ubl
                np.testing.assert_almost_equal(np.abs(d_bl), np.abs(mdl), 10)
                np.testing.assert_almost_equal(np.angle(d_bl * mdl.conj()), 0, 10)

        sol_rd = rc.remove_degen(antpos, sol, degen_sol=gains)
        g, v = om.get_gains_and_vis_from_sol(sol_rd)
        meanSqAmplitude = np.mean([np.abs(g[key1] * g[key2]) for key1 in g.keys()
                                   for key2 in g.keys() if key1[1] == 'jxx' and key2[1] == 'jxx' and key1[0] != key2[0]], axis=0)
        degenMeanSqAmplitude = np.mean([np.abs(gains[key1] * gains[key2]) for key1 in g.keys()
                                        for key2 in g.keys() if key1[1] == 'jxx' and key2[1] == 'jxx' and key1[0] != key2[0]], axis=0)
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

    def test_lincal_hex_end_to_end_4pol_with_remove_degen_and_firstcal(self):
        antpos = build_hex_array(3)
        reds = om.get_reds(antpos, pols=['XX', 'XY', 'YX', 'YY'], pol_mode='4pol')
        rc = om.RedundantCalibrator(reds)
        freqs = np.linspace(.1, .2, 10)
        gains, true_vis, d = om.sim_red_data(reds, gain_scatter=.09, shape=(1, len(freqs)))
        fc_delays = {ant: 100 * np.random.randn() for ant in gains.keys()}  # in ns
        fc_gains = {ant: np.reshape(np.exp(-2.0j * np.pi * freqs * delay), (1, len(freqs))) for ant, delay in fc_delays.items()}
        for ant1, ant2, pol in d.keys():
            d[(ant1, ant2, pol)] *= fc_gains[(ant1, split_pol(pol)[0])] * np.conj(fc_gains[(ant2, split_pol(pol)[1])])
        for ant in gains.keys():
            gains[ant] *= fc_gains[ant]

        w = dict([(k, 1.) for k in d.keys()])
        sol0 = rc.logcal(d, sol0=fc_gains, wgts=w)
        meta, sol = rc.lincal(d, sol0, wgts=w)

        np.testing.assert_array_less(meta['iter'], 50 * np.ones_like(meta['iter']))
        np.testing.assert_almost_equal(meta['chisq'], np.zeros_like(meta['chisq']), decimal=10)

        np.testing.assert_almost_equal(meta['chisq'], 0, 10)
        for i in xrange(len(antpos)):
            self.assertEqual(sol[(i, 'jxx')].shape, (1, len(freqs)))
            self.assertEqual(sol[(i, 'jyy')].shape, (1, len(freqs)))
        for bls in reds:
            for bl in bls:
                ubl = sol[bls[0]]
                self.assertEqual(ubl.shape, (1, len(freqs)))
                d_bl = d[bl]
                mdl = sol[(bl[0], split_pol(bl[2])[0])] * sol[(bl[1], split_pol(bl[2])[1])].conj() * ubl
                np.testing.assert_almost_equal(np.abs(d_bl), np.abs(mdl), 10)
                np.testing.assert_almost_equal(np.angle(d_bl * mdl.conj()), 0, 10)

        sol_rd = rc.remove_degen(antpos, sol)

        ants = [key for key in sol_rd.keys() if len(key) == 2]
        gainPols = np.array([ant[1] for ant in ants])
        bl_pairs = [key for key in sol.keys() if len(key) == 3]
        visPols = np.array([[bl[2][0], bl[2][1]] for bl in bl_pairs])
        bl_vecs = np.array([antpos[bl_pair[0]] - antpos[bl_pair[1]] for bl_pair in bl_pairs])
        gainSols = np.array([sol_rd[ant] for ant in ants])
        g, v = om.get_gains_and_vis_from_sol(sol_rd)
        meanSqAmplitude = np.mean([np.abs(g[key1] * g[key2]) for key1 in g.keys()
                                   for key2 in g.keys() if key1[1] == 'jxx' and key2[1] == 'jxx' and key1[0] != key2[0]], axis=0)
        np.testing.assert_almost_equal(meanSqAmplitude, 1, 10)
        meanSqAmplitude = np.mean([np.abs(g[key1] * g[key2]) for key1 in g.keys()
                                   for key2 in g.keys() if key1[1] == 'jyy' and key2[1] == 'jyy' and key1[0] != key2[0]], axis=0)
        np.testing.assert_almost_equal(meanSqAmplitude, 1, 10)
        #np.testing.assert_almost_equal(np.mean(np.angle(gainSols[gainPols=='jxx']), axis=0), 0, 10)
        #np.testing.assert_almost_equal(np.mean(np.angle(gainSols[gainPols=='jyy']), axis=0), 0, 10)

        for bls in reds:
            for bl in bls:
                ubl = sol_rd[bls[0]]
                self.assertEqual(ubl.shape, (1, len(freqs)))
                d_bl = d[bl]
                mdl = sol_rd[(bl[0], split_pol(bl[2])[0])] * sol_rd[(bl[1], split_pol(bl[2])[1])].conj() * ubl
                np.testing.assert_almost_equal(np.abs(d_bl), np.abs(mdl), 10)
                np.testing.assert_almost_equal(np.angle(d_bl * mdl.conj()), 0, 10)

        sol_rd = rc.remove_degen(antpos, sol, degen_sol=gains)
        g, v = om.get_gains_and_vis_from_sol(sol_rd)
        meanSqAmplitude = np.mean([np.abs(g[key1] * g[key2]) for key1 in g.keys()
                                   for key2 in g.keys() if key1[1] == 'jxx' and key2[1] == 'jxx' and key1[0] != key2[0]], axis=0)
        degenMeanSqAmplitude = np.mean([np.abs(gains[key1] * gains[key2]) for key1 in g.keys()
                                        for key2 in g.keys() if key1[1] == 'jxx' and key2[1] == 'jxx' and key1[0] != key2[0]], axis=0)
        np.testing.assert_almost_equal(meanSqAmplitude, degenMeanSqAmplitude, 10)
        meanSqAmplitude = np.mean([np.abs(g[key1] * g[key2]) for key1 in g.keys()
                                   for key2 in g.keys() if key1[1] == 'jyy' and key2[1] == 'jyy' and key1[0] != key2[0]], axis=0)
        degenMeanSqAmplitude = np.mean([np.abs(gains[key1] * gains[key2]) for key1 in g.keys()
                                        for key2 in g.keys() if key1[1] == 'jyy' and key2[1] == 'jyy' and key1[0] != key2[0]], axis=0)
        np.testing.assert_almost_equal(meanSqAmplitude, degenMeanSqAmplitude, 10)

        gainSols = np.array([sol_rd[ant] for ant in ants])
        degenGains = np.array([gains[ant] for ant in ants])
        np.testing.assert_almost_equal(np.mean(np.angle(gainSols[gainPols == 'jxx']), axis=0),
                                       np.mean(np.angle(degenGains[gainPols == 'jxx']), axis=0), 10)
        np.testing.assert_almost_equal(np.mean(np.angle(gainSols[gainPols == 'jyy']), axis=0),
                                       np.mean(np.angle(degenGains[gainPols == 'jyy']), axis=0), 10)

        for key, val in sol_rd.items():
            if len(key) == 2:
                np.testing.assert_almost_equal(val, gains[key], 10)
            if len(key) == 3:
                np.testing.assert_almost_equal(val, true_vis[key], 10)

    def test_lincal_hex_end_to_end_4pol_minV_with_remove_degen_and_firstcal(self):

        antpos = build_hex_array(3)
        reds = om.get_reds(antpos, pols=['XX', 'XY', 'YX', 'YY'], pol_mode='4pol_minV')
        rc = om.RedundantCalibrator(reds)
        freqs = np.linspace(.1, .2, 10)
        gains, true_vis, d = om.sim_red_data(reds, gain_scatter=.1, shape=(1, len(freqs)))
        fc_delays = {ant: 100 * np.random.randn() for ant in gains.keys()}  # in ns
        fc_gains = {ant: np.reshape(np.exp(-2.0j * np.pi * freqs * delay), (1, len(freqs))) for ant, delay in fc_delays.items()}
        for ant1, ant2, pol in d.keys():
            d[(ant1, ant2, pol)] *= fc_gains[(ant1, split_pol(pol)[0])] * np.conj(fc_gains[(ant2, split_pol(pol)[1])])
        for ant in gains.keys():
            gains[ant] *= fc_gains[ant]

        w = dict([(k, 1.) for k in d.keys()])
        sol0 = rc.logcal(d, sol0=fc_gains, wgts=w)
        meta, sol = rc.lincal(d, sol0, wgts=w)

        np.testing.assert_array_less(meta['iter'], 50 * np.ones_like(meta['iter']))
        np.testing.assert_almost_equal(meta['chisq'], np.zeros_like(meta['chisq']), decimal=10)

        np.testing.assert_almost_equal(meta['chisq'], 0, 10)
        for i in xrange(len(antpos)):
            self.assertEqual(sol[(i, 'jxx')].shape, (1, len(freqs)))
            self.assertEqual(sol[(i, 'jyy')].shape, (1, len(freqs)))
        for bls in reds:
            ubl = sol[bls[0]]
            for bl in bls:
                self.assertEqual(ubl.shape, (1, len(freqs)))
                d_bl = d[bl]
                mdl = sol[(bl[0], split_pol(bl[2])[0])] * sol[(bl[1], split_pol(bl[2])[1])].conj() * ubl
                np.testing.assert_almost_equal(np.abs(d_bl), np.abs(mdl), 10)
                np.testing.assert_almost_equal(np.angle(d_bl * mdl.conj()), 0, 10)

        sol_rd = rc.remove_degen(antpos, sol)
        g, v = om.get_gains_and_vis_from_sol(sol_rd)
        ants = [key for key in sol_rd.keys() if len(key) == 2]
        gainPols = np.array([ant[1] for ant in ants])
        bl_pairs = [key for key in sol.keys() if len(key) == 3]
        visPols = np.array([[bl[2][0], bl[2][1]] for bl in bl_pairs])
        visPolsStr = np.array([bl[2] for bl in bl_pairs])
        bl_vecs = np.array([antpos[bl_pair[0]] - antpos[bl_pair[1]] for bl_pair in bl_pairs])
        gainSols = np.array([sol_rd[ant] for ant in ants])
        meanSqAmplitude = np.mean([np.abs(g[key1] * g[key2]) for key1 in g.keys()
                                   for key2 in g.keys() if key1[1] == 'jxx' and key2[1] == 'jxx' and key1[0] != key2[0]], axis=0)
        np.testing.assert_almost_equal(meanSqAmplitude, 1, 10)
        meanSqAmplitude = np.mean([np.abs(g[key1] * g[key2]) for key1 in g.keys()
                                   for key2 in g.keys() if key1[1] == 'jyy' and key2[1] == 'jyy' and key1[0] != key2[0]], axis=0)
        np.testing.assert_almost_equal(meanSqAmplitude, 1, 10)
        #np.testing.assert_almost_equal(np.mean(np.angle(gainSols), axis=0), 0, 10)

        for bls in reds:
            ubl = sol_rd[bls[0]]
            for bl in bls:
                self.assertEqual(ubl.shape, (1, len(freqs)))
                d_bl = d[bl]
                mdl = sol_rd[(bl[0], split_pol(bl[2])[0])] * sol_rd[(bl[1], split_pol(bl[2])[1])].conj() * ubl
                np.testing.assert_almost_equal(np.abs(d_bl), np.abs(mdl), 10)
                np.testing.assert_almost_equal(np.angle(d_bl * mdl.conj()), 0, 10)

        sol_rd = rc.remove_degen(antpos, sol, degen_sol=gains)
        g, v = om.get_gains_and_vis_from_sol(sol_rd)

        for bls in reds:
            ubl = sol_rd[bls[0]]
            for bl in bls:
                self.assertEqual(ubl.shape, (1, len(freqs)))
                d_bl = d[bl]
                mdl = sol_rd[(bl[0], split_pol(bl[2])[0])] * sol_rd[(bl[1], split_pol(bl[2])[1])].conj() * ubl
                np.testing.assert_almost_equal(np.abs(d_bl), np.abs(mdl), 10)
                np.testing.assert_almost_equal(np.angle(d_bl * mdl.conj()), 0, 10)

        gainSols = np.array([sol_rd[ant] for ant in ants])
        degenGains = np.array([gains[ant] for ant in ants])
        np.testing.assert_almost_equal(np.mean(np.angle(gainSols), axis=0),
                                       np.mean(np.angle(degenGains), axis=0), 10)

        meanSqAmplitude = np.mean([np.abs(g[key1] * g[key2]) for key1 in g.keys()
                                   for key2 in g.keys() if key1[1] == 'jxx' and key2[1] == 'jxx' and key1[0] != key2[0]], axis=0)
        degenMeanSqAmplitude = np.mean([np.abs(gains[key1] * gains[key2]) for key1 in g.keys()
                                        for key2 in g.keys() if key1[1] == 'jxx' and key2[1] == 'jxx' and key1[0] != key2[0]], axis=0)
        np.testing.assert_almost_equal(meanSqAmplitude, degenMeanSqAmplitude, 10)
        meanSqAmplitude = np.mean([np.abs(g[key1] * g[key2]) for key1 in g.keys()
                                   for key2 in g.keys() if key1[1] == 'jyy' and key2[1] == 'jyy' and key1[0] != key2[0]], axis=0)
        degenMeanSqAmplitude = np.mean([np.abs(gains[key1] * gains[key2]) for key1 in g.keys()
                                        for key2 in g.keys() if key1[1] == 'jyy' and key2[1] == 'jyy' and key1[0] != key2[0]], axis=0)
        np.testing.assert_almost_equal(meanSqAmplitude, degenMeanSqAmplitude, 10)

        visSols = np.array([sol_rd[bl] for bl in bl_pairs])
        degenVis = np.array([true_vis[bl] for bl in bl_pairs])
        np.testing.assert_almost_equal(np.mean(np.angle(visSols), axis=0),
                                       np.mean(np.angle(degenVis), axis=0), 10)

        for key, val in sol_rd.items():
            if len(key) == 2:
                np.testing.assert_almost_equal(val, gains[key], 10)
            if len(key) == 3:
                np.testing.assert_almost_equal(val, true_vis[key], 10)

    def test_lincal_hex_end_to_end_2pol_with_remove_degen_and_firstcal(self):

        antpos = build_hex_array(3)
        reds = om.get_reds(antpos, pols=['XX', 'YY'], pol_mode='2pol')
        rc = om.RedundantCalibrator(reds)
        freqs = np.linspace(.1, .2, 10)
        gains, true_vis, d = om.sim_red_data(reds, gain_scatter=.1, shape=(1, len(freqs)))
        fc_delays = {ant: 100 * np.random.randn() for ant in gains.keys()}  # in ns
        fc_gains = {ant: np.reshape(np.exp(-2.0j * np.pi * freqs * delay), (1, len(freqs))) for ant, delay in fc_delays.items()}
        for ant1, ant2, pol in d.keys():
            d[(ant1, ant2, pol)] *= fc_gains[(ant1, split_pol(pol)[0])] * np.conj(fc_gains[(ant2, split_pol(pol)[1])])
        for ant in gains.keys():
            gains[ant] *= fc_gains[ant]

        w = dict([(k, 1.) for k in d.keys()])
        sol0 = rc.logcal(d, sol0=fc_gains, wgts=w)
        meta, sol = rc.lincal(d, sol0, wgts=w)

        np.testing.assert_array_less(meta['iter'], 50 * np.ones_like(meta['iter']))
        np.testing.assert_almost_equal(meta['chisq'], np.zeros_like(meta['chisq']), decimal=10)

        np.testing.assert_almost_equal(meta['chisq'], 0, 10)
        for i in xrange(len(antpos)):
            self.assertEqual(sol[(i, 'jxx')].shape, (1, len(freqs)))
            self.assertEqual(sol[(i, 'jyy')].shape, (1, len(freqs)))
        for bls in reds:
            for bl in bls:
                ubl = sol[bls[0]]
                self.assertEqual(ubl.shape, (1, len(freqs)))
                d_bl = d[bl]
                mdl = sol[(bl[0], split_pol(bl[2])[0])] * sol[(bl[1], split_pol(bl[2])[1])].conj() * ubl
                np.testing.assert_almost_equal(np.abs(d_bl), np.abs(mdl), 10)
                np.testing.assert_almost_equal(np.angle(d_bl * mdl.conj()), 0, 10)

        sol_rd = rc.remove_degen(antpos, sol)

        ants = [key for key in sol_rd.keys() if len(key) == 2]
        gainPols = np.array([ant[1] for ant in ants])
        bl_pairs = [key for key in sol.keys() if len(key) == 3]
        visPols = np.array([[bl[2][0], bl[2][1]] for bl in bl_pairs])
        bl_vecs = np.array([antpos[bl_pair[0]] - antpos[bl_pair[1]] for bl_pair in bl_pairs])
        gainSols = np.array([sol_rd[ant] for ant in ants])
        g, v = om.get_gains_and_vis_from_sol(sol_rd)

        meanSqAmplitude = np.mean([np.abs(g[key1] * g[key2]) for key1 in g.keys()
                                   for key2 in g.keys() if key1[1] == 'jxx' and key2[1] == 'jxx' and key1[0] != key2[0]], axis=0)
        np.testing.assert_almost_equal(meanSqAmplitude, 1, 10)
        meanSqAmplitude = np.mean([np.abs(g[key1] * g[key2]) for key1 in g.keys()
                                   for key2 in g.keys() if key1[1] == 'jyy' and key2[1] == 'jyy' and key1[0] != key2[0]], axis=0)
        np.testing.assert_almost_equal(meanSqAmplitude, 1, 10)
        #np.testing.assert_almost_equal(np.mean(np.angle(gainSols[gainPols=='jxx']), axis=0), 0, 10)
        #np.testing.assert_almost_equal(np.mean(np.angle(gainSols[gainPols=='jyy']), axis=0), 0, 10)

        for bls in reds:
            for bl in bls:
                ubl = sol_rd[bls[0]]
                self.assertEqual(ubl.shape, (1, len(freqs)))
                d_bl = d[bl]
                mdl = sol_rd[(bl[0], split_pol(bl[2])[0])] * sol_rd[(bl[1], split_pol(bl[2])[1])].conj() * ubl
                np.testing.assert_almost_equal(np.abs(d_bl), np.abs(mdl), 10)
                np.testing.assert_almost_equal(np.angle(d_bl * mdl.conj()), 0, 10)

        sol_rd = rc.remove_degen(antpos, sol, degen_sol=gains)
        g, v = om.get_gains_and_vis_from_sol(sol_rd)
        gainSols = np.array([sol_rd[ant] for ant in ants])
        degenGains = np.array([gains[ant] for ant in ants])

        meanSqAmplitude = np.mean([np.abs(g[key1] * g[key2]) for key1 in g.keys()
                                   for key2 in g.keys() if key1[1] == 'jxx' and key2[1] == 'jxx' and key1[0] != key2[0]], axis=0)
        degenMeanSqAmplitude = np.mean([np.abs(gains[key1] * gains[key2]) for key1 in g.keys()
                                        for key2 in g.keys() if key1[1] == 'jxx' and key2[1] == 'jxx' and key1[0] != key2[0]], axis=0)
        np.testing.assert_almost_equal(meanSqAmplitude, degenMeanSqAmplitude, 10)
        meanSqAmplitude = np.mean([np.abs(g[key1] * g[key2]) for key1 in g.keys()
                                   for key2 in g.keys() if key1[1] == 'jyy' and key2[1] == 'jyy' and key1[0] != key2[0]], axis=0)
        degenMeanSqAmplitude = np.mean([np.abs(gains[key1] * gains[key2]) for key1 in g.keys()
                                        for key2 in g.keys() if key1[1] == 'jyy' and key2[1] == 'jyy' and key1[0] != key2[0]], axis=0)
        np.testing.assert_almost_equal(meanSqAmplitude, degenMeanSqAmplitude, 10)

        np.testing.assert_almost_equal(np.mean(np.angle(gainSols[gainPols == 'jxx']), axis=0),
                                       np.mean(np.angle(degenGains[gainPols == 'jxx']), axis=0), 10)
        np.testing.assert_almost_equal(np.mean(np.angle(gainSols[gainPols == 'jyy']), axis=0),
                                       np.mean(np.angle(degenGains[gainPols == 'jyy']), axis=0), 10)

        for key, val in sol_rd.items():
            if len(key) == 2:
                np.testing.assert_almost_equal(val, gains[key], 10)
            if len(key) == 3:
                np.testing.assert_almost_equal(val, true_vis[key], 10)

    def test_count_redcal_degeneracies(self):
        pos = build_hex_array(3)
        self.assertEqual(om.count_redcal_degeneracies(pos, bl_error_tol=1), 4)
        pos[0] += [.5, 0, 0]
        self.assertEqual(om.count_redcal_degeneracies(pos, bl_error_tol=.1), 6)
        self.assertEqual(om.count_redcal_degeneracies(pos, bl_error_tol=1), 4)

    def test_is_redundantly_calibratable(self):
        pos = build_hex_array(3)
        self.assertTrue(om.is_redundantly_calibratable(pos, bl_error_tol=1))
        pos[0] += [.5, 0, 0]
        self.assertFalse(om.is_redundantly_calibratable(pos, bl_error_tol=.1))
        self.assertTrue(om.is_redundantly_calibratable(pos, bl_error_tol=1))


if __name__ == '__main__':
    unittest.main()
