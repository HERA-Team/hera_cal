# -*- coding: utf-8 -*-
# Copyright 2019 the HERA Project
# Licensed under the MIT License

import pytest
import numpy as np
from copy import deepcopy
import warnings
import os
import sys
import shutil
from hera_sim.antpos import linear_array, hex_array
from hera_sim.vis import sim_red_data
from hera_sim.sigchain import gen_gains

from .. import redcal as om
from .. import io, abscal
from ..utils import split_pol, conj_pol, split_bl
from ..apply_cal import calibrate_in_place
from ..data import DATA_PATH
from ..datacontainer import DataContainer

np.random.seed(0)


class TestMethods(object):

    def test_check_polLists_minV(self):
        polLists = [['xy']]
        assert not om._check_polLists_minV(polLists)
        polLists = [['xx', 'xy']]
        assert not om._check_polLists_minV(polLists)
        polLists = [['xx', 'xy', 'yx']]
        assert not om._check_polLists_minV(polLists)
        polLists = [['xy', 'yx'], ['xx'], ['yy'], ['xx'], ['yx', 'xy'], ['yy']]
        assert om._check_polLists_minV(polLists)

    def test_parse_pol_mode(self):
        reds = [[(0, 1, 'xx')]]
        assert om.parse_pol_mode(reds) == '1pol'
        reds = [[(0, 1, 'xx')], [(0, 1, 'yy')]]
        assert om.parse_pol_mode(reds) == '2pol'
        reds = [[(0, 1, 'xx')], [(0, 1, 'xy')], [(0, 1, 'yx')], [(0, 1, 'yy')]]
        assert om.parse_pol_mode(reds) == '4pol'
        reds = [[(0, 1, 'xx')], [(0, 1, 'xy'), (0, 1, 'yx')], [(0, 1, 'yy')]]
        assert om.parse_pol_mode(reds) == '4pol_minV'

        reds = [[(0, 1, 'xx')], [(0, 1, 'xy'), (0, 1, 'yx')], [(0, 1, 'LR')]]
        assert om.parse_pol_mode(reds) == 'unrecognized_pol_mode'
        reds = [[(0, 1, 'xx')], [(0, 1, 'xy')]]
        assert om.parse_pol_mode(reds) == 'unrecognized_pol_mode'
        reds = [[(0, 1, 'xy')]]
        assert om.parse_pol_mode(reds) == 'unrecognized_pol_mode'
        reds = [[(0, 1, 'xx')], [(0, 1, 'xy'), (0, 1, 'yy')], [(0, 1, 'yx')]]
        assert om.parse_pol_mode(reds) == 'unrecognized_pol_mode'

    def test_get_pos_red(self):
        pos = hex_array(3, sep=14.6, split_core=False, outriggers=0)
        assert len(om.get_pos_reds(pos)) == 30

        pos = hex_array(7, sep=14.6, split_core=False, outriggers=0)
        assert len(om.get_pos_reds(pos)) == 234
        for ant, r in pos.items():
            pos[ant] += [0, 0, 1 * r[0] - .5 * r[1]]
        assert len(om.get_pos_reds(pos)) == 234

        pos = hex_array(7, sep=1, split_core=False, outriggers=0)
        assert len(om.get_pos_reds(pos)) < 234
        assert len(om.get_pos_reds(pos, bl_error_tol=.1)) == 234

        pos = hex_array(7, sep=14.6, split_core=False, outriggers=0)
        blerror = 1.0 - 1e-12
        error = blerror / 4
        for key, val in pos.items():
            th = np.random.choice([0, np.pi / 2, np.pi])
            phi = np.random.choice([0, np.pi / 2, np.pi, 3 * np.pi / 2])
            pos[key] = val + error * np.array([np.sin(th) * np.cos(phi), np.sin(th) * np.sin(phi), np.cos(th)])
        assert len(om.get_pos_reds(pos, bl_error_tol=1.0)) == 234
        assert len(om.get_pos_reds(pos, bl_error_tol=.99)) > 234

        pos = {0: np.array([0, 0, 0]), 1: np.array([20, 0, 0]), 2: np.array([10, 0, 0])}
        assert om.get_pos_reds(pos) == [[(0, 2), (2, 1)], [(0, 1)]]

        # test branch cut
        pos = {0: np.array([-.03, 1., 0.]),
               1: np.array([1., 1., 0.]),
               2: np.array([0.03, 0.0, 0.]),
               3: np.array([1., 0., 0.])}
        assert len(om.get_pos_reds(pos, bl_error_tol=.1)) == 4

    def test_filter_reds(self):
        antpos = linear_array(7)
        reds = om.get_reds(antpos, pols=['xx'], pol_mode='1pol')
        # exclude ants
        red = om.filter_reds(reds, ex_ants=[0, 4])
        assert red == [[(1, 2, 'xx'), (2, 3, 'xx'), (5, 6, 'xx')], [(1, 3, 'xx'), (3, 5, 'xx')], [(2, 5, 'xx'), (3, 6, 'xx')],
                       [(1, 5, 'xx'), (2, 6, 'xx')], [(1, 6, 'xx')]]
        # include ants
        red = om.filter_reds(reds, ants=[0, 1, 4, 5, 6])
        assert red == [[(0, 1, 'xx'), (4, 5, 'xx'), (5, 6, 'xx')], [(4, 6, 'xx')], [(1, 4, 'xx')], [(0, 4, 'xx'), (1, 5, 'xx')],
                       [(0, 5, 'xx'), (1, 6, 'xx')], [(0, 6, 'xx')]]
        # exclued bls
        red = om.filter_reds(reds, ex_bls=[(0, 2), (1, 2), (0, 6)])
        assert red == [[(0, 1, 'xx'), (2, 3, 'xx'), (3, 4, 'xx'), (4, 5, 'xx'), (5, 6, 'xx')],
                       [(1, 3, 'xx'), (2, 4, 'xx'), (3, 5, 'xx'), (4, 6, 'xx')], [(0, 3, 'xx'), (1, 4, 'xx'), (2, 5, 'xx'), (3, 6, 'xx')],
                       [(0, 4, 'xx'), (1, 5, 'xx'), (2, 6, 'xx')], [(0, 5, 'xx'), (1, 6, 'xx')]]
        # include bls
        red = om.filter_reds(reds, bls=[(0, 1), (1, 2)])
        assert red == [[(0, 1, 'xx'), (1, 2, 'xx')]]
        # include ubls
        red = om.filter_reds(reds, ubls=[(0, 2), (1, 4)])
        assert red == [[(0, 2, 'xx'), (1, 3, 'xx'), (2, 4, 'xx'), (3, 5, 'xx'), (4, 6, 'xx')],
                       [(0, 3, 'xx'), (1, 4, 'xx'), (2, 5, 'xx'), (3, 6, 'xx')]]
        # exclude ubls
        red = om.filter_reds(reds, ex_ubls=[(0, 2), (1, 4), (4, 5), (0, 5), (2, 3), (0, 6)])
        assert red == [[(0, 4, 'xx'), (1, 5, 'xx'), (2, 6, 'xx')]]
        # exclude crosspols
        # reds = omni.filter_reds(self.info.get_reds(), ex_crosspols=()

    def test_filter_reds_2pol(self):
        antpos = linear_array(4)
        reds = om.get_reds(antpos, pols=['xx', 'yy'], pol_mode='1pol')
        # include pols
        red = om.filter_reds(reds, pols=['xx'])
        assert red == [[(0, 1, 'xx'), (1, 2, 'xx'), (2, 3, 'xx')], [(0, 2, 'xx'), (1, 3, 'xx')], [(0, 3, 'xx')]]
        # exclude pols
        red = om.filter_reds(reds, ex_pols=['yy'])
        assert red == [[(0, 1, 'xx'), (1, 2, 'xx'), (2, 3, 'xx')], [(0, 2, 'xx'), (1, 3, 'xx')], [(0, 3, 'xx')]]
        # exclude ants
        red = om.filter_reds(reds, ex_ants=[0])
        assert red == [[(1, 2, 'xx'), (2, 3, 'xx')], [(1, 3, 'xx')], [(1, 2, 'yy'), (2, 3, 'yy')], [(1, 3, 'yy')]]
        # include ants
        red = om.filter_reds(reds, ants=[1, 2, 3])
        red = om.filter_reds(reds, ex_ants=[0])
        # exclued bls
        red = om.filter_reds(reds, ex_bls=[(1, 2), (0, 3)])
        assert red == [[(0, 1, 'xx'), (2, 3, 'xx')], [(0, 2, 'xx'), (1, 3, 'xx')], [(0, 1, 'yy'), (2, 3, 'yy')], [(0, 2, 'yy'), (1, 3, 'yy')]]
        # include bls
        red = om.filter_reds(reds, bls=[(0, 1), (1, 2)])
        assert red == [[(0, 1, 'xx'), (1, 2, 'xx')], [(0, 1, 'yy'), (1, 2, 'yy')]]
        # include ubls
        red = om.filter_reds(reds, ubls=[(0, 2)])
        assert red == [[(0, 2, 'xx'), (1, 3, 'xx')], [(0, 2, 'yy'), (1, 3, 'yy')]]
        # exclude ubls
        red = om.filter_reds(reds, ex_ubls=[(2, 3), (0, 3)])
        assert red == [[(0, 2, 'xx'), (1, 3, 'xx')], [(0, 2, 'yy'), (1, 3, 'yy')]]
        # test baseline length min and max cutoffs
        antpos = hex_array(4, sep=14.6, split_core=False, outriggers=0)
        reds = om.get_reds(antpos, pols=['xx'], pol_mode='1pol')
        assert om.filter_reds(reds, antpos=antpos, min_bl_cut=85) == reds[-3:]
        assert om.filter_reds(reds, antpos=antpos, max_bl_cut=15) == reds[:3]

    def test_filter_reds_max_dim(self):
        # build hex array with 4 on a side and 7 total rows
        antpos = hex_array(4, split_core=False, outriggers=0)
        antpos[37] = np.array([np.pi, np.pi, 0])  # add one off-grid antenna
        reds = om.get_reds(antpos)
        # remove third, fourth, fifth, and sixth rows
        reds = om.filter_reds(reds, ex_ants=list(range(9, 33)))

        # Max 1 dimension means largest 1D array
        new_reds = om.filter_reds(reds, max_dims=1)
        ant_inds = set([ant[0] for red in new_reds for bl in red for ant in split_bl(bl)])
        assert ant_inds == set(range(4, 9))

        # Max 2 dimensions means only rows 1 and 2
        new_reds = om.filter_reds(reds, max_dims=2)
        ant_inds = set([ant[0] for red in new_reds for bl in red for ant in split_bl(bl)])
        assert ant_inds == set(range(0, 9))

        # Max 3 dimensions means all 3 good rows, but keeps out the off-grid antenna
        new_reds = om.filter_reds(reds, max_dims=3)
        ant_inds = set([ant[0] for red in new_reds for bl in red for ant in split_bl(bl)])
        assert ant_inds == (set(range(0, 9)) | set(range(33, 37)))

    def test_add_pol_reds(self):
        reds = [[(1, 2)]]
        polReds = om.add_pol_reds(reds, pols=['xx'], pol_mode='1pol')
        assert polReds == [[(1, 2, 'xx')]]
        polReds = om.add_pol_reds(reds, pols=['xx', 'yy'], pol_mode='2pol')
        assert polReds == [[(1, 2, 'xx')], [(1, 2, 'yy')]]
        polReds = om.add_pol_reds(reds, pols=['xx', 'xy', 'yx', 'yy'], pol_mode='4pol')
        assert polReds == [[(1, 2, 'xx')], [(1, 2, 'xy')], [(1, 2, 'yx')], [(1, 2, 'yy')]]
        polReds = om.add_pol_reds(reds, pols=['xx', 'xy', 'yx', 'yy'], pol_mode='4pol_minV')
        assert polReds == [[(1, 2, 'xx')], [(1, 2, 'xy'), (1, 2, 'yx')], [(1, 2, 'yy')]]

    def test_reds_to_antpos(self):
        # Test 1D
        true_antpos = linear_array(10)
        reds = om.get_reds(true_antpos, pols=['xx', 'yy'], pol_mode='2pol', bl_error_tol=1e-10)
        inferred_antpos = om.reds_to_antpos(reds,)
        for pos in inferred_antpos.values():
            assert len(pos) == 1
        new_reds = om.get_reds(inferred_antpos, pols=['xx', 'yy'], pol_mode='2pol', bl_error_tol=1e-10)
        for nred in new_reds:
            for red in reds:
                if nred[0] in red:
                    found_match = True
                    assert len(set(nred).difference(set(red))) == 0
            assert found_match
            found_match = False

        # Test 2D
        true_antpos = hex_array(5, split_core=False, outriggers=0)
        reds = om.get_reds(true_antpos, pols=['xx'], pol_mode='1pol', bl_error_tol=1e-10)
        inferred_antpos = om.reds_to_antpos(reds)
        for pos in inferred_antpos.values():
            assert len(pos) == 2
        new_reds = om.get_reds(inferred_antpos, pols=['xx'], pol_mode='1pol', bl_error_tol=1e-10)
        for nred in new_reds:
            for red in reds:
                if nred[0] in red:
                    found_match = True
                    assert len(set(nred).difference(set(red))) == 0
            assert found_match
            found_match = False

        # Test 2D with split
        true_antpos = hex_array(5, split_core=True, outriggers=0)
        reds = om.get_pos_reds(true_antpos, bl_error_tol=1e-10)
        inferred_antpos = om.reds_to_antpos(reds)
        for pos in inferred_antpos.values():
            assert len(pos) == 2
        new_reds = om.get_pos_reds(inferred_antpos, bl_error_tol=1e-10)
        for nred in new_reds:
            for red in reds:
                if nred[0] in red:
                    found_match = True
                    assert len(set(nred).difference(set(red))) == 0
            assert found_match
            found_match = False

        # Test 2D with additional degeneracy
        true_antpos = {0: [0, 0], 1: [1, 0], 2: [0, 1], 3: [1, 1],
                       4: [100, 100], 5: [101, 100], 6: [100, 101], 7: [101, 101]}
        reds = om.get_pos_reds(true_antpos, bl_error_tol=1e-10)
        inferred_antpos = om.reds_to_antpos(reds)
        for pos in inferred_antpos.values():
            assert len(pos) == 3
        new_reds = om.get_pos_reds(inferred_antpos, bl_error_tol=1e-10)
        for nred in new_reds:
            for red in reds:
                if nred[0] in red:
                    found_match = True
                    assert len(set(nred).difference(set(red))) == 0
            assert found_match
            found_match = False

    def test_find_polarity_flipped_ants(self):
        # test normal operation
        antpos = hex_array(3, split_core=False, outriggers=0)
        reds = om.get_reds(antpos, pols=['ee'], pol_mode='1pol')
        rc = om.RedundantCalibrator(reds)
        freqs = np.linspace(.1, .2, 100)
        ants = [(ant, 'Jee') for ant in antpos]
        gains = gen_gains(freqs, ants)
        for ant in [3, 10, 11]:
            gains[ant, 'Jee'] *= -1
        _, true_vis, data = sim_red_data(reds, gains=gains, shape=(2, len(freqs)))
        meta, g_fc = rc.firstcal(data, freqs)
        for ant in antpos:
            if ant in [3, 10, 11]:
                assert np.all(meta['polarity_flips'][ant, 'Jee'])
            else:
                assert not np.any(meta['polarity_flips'][ant, 'Jee'])

        # test operation where no good answer is possible, so we expect it to fail
        data[(0, 1, 'ee')] *= -1
        meta, g_fc = rc.firstcal(data, freqs)
        for ant in meta['polarity_flips']:
            assert np.all([m is None for m in meta['polarity_flips'][ant]])

        # test errors
        with pytest.raises(ValueError):
            om._build_polarity_baseline_groups(data, reds, edge_cut=100)
        with pytest.raises(ValueError):
            om._build_polarity_baseline_groups(data, reds, max_rel_angle=np.pi)


class TestRedundantCalibrator(object):

    def test_init(self):
        # test a very small array
        pos = hex_array(3, split_core=False, outriggers=0)
        pos = {ant: pos[ant] for ant in range(4)}
        reds = om.get_reds(pos)
        rc = om.RedundantCalibrator(reds)
        with pytest.raises(ValueError):
            rc = om.RedundantCalibrator(reds, check_redundancy=True)

        # test disconnected redundant array
        pos = hex_array(5, split_core=False, outriggers=0)
        pos = {ant: pos[ant] for ant in pos if ant in [0, 1, 5, 6, 54, 55, 59, 60]}
        reds = om.get_reds(pos)
        try:
            rc = om.RedundantCalibrator(reds, check_redundancy=True)
        except ValueError:
            assert False, 'This array is actually redundant, so check_redundancy should not raise a ValueError.'

    def test_build_eq(self):
        antpos = linear_array(3)
        reds = om.get_reds(antpos, pols=['xx'], pol_mode='1pol')
        gains, true_vis, data = sim_red_data(reds)
        info = om.RedundantCalibrator(reds)
        eqs = info.build_eqs(data)
        assert len(eqs) == 3
        assert eqs['g_0_Jxx * g_1_Jxx_ * u_0_xx'] == (0, 1, 'xx')
        assert eqs['g_1_Jxx * g_2_Jxx_ * u_0_xx'] == (1, 2, 'xx')
        assert eqs['g_0_Jxx * g_2_Jxx_ * u_1_xx'] == (0, 2, 'xx')

        reds = om.get_reds(antpos, pols=['xx', 'yy', 'xy', 'yx'], pol_mode='4pol')
        gains, true_vis, data = sim_red_data(reds)
        info = om.RedundantCalibrator(reds)
        eqs = info.build_eqs(data)
        assert len(eqs) == 3 * 4
        assert eqs['g_0_Jxx * g_1_Jyy_ * u_4_xy'] == (0, 1, 'xy')
        assert eqs['g_1_Jxx * g_2_Jyy_ * u_4_xy'] == (1, 2, 'xy')
        assert eqs['g_0_Jxx * g_2_Jyy_ * u_5_xy'] == (0, 2, 'xy')
        assert eqs['g_0_Jyy * g_1_Jxx_ * u_6_yx'] == (0, 1, 'yx')
        assert eqs['g_1_Jyy * g_2_Jxx_ * u_6_yx'] == (1, 2, 'yx')
        assert eqs['g_0_Jyy * g_2_Jxx_ * u_7_yx'] == (0, 2, 'yx')

        reds = om.get_reds(antpos, pols=['xx', 'yy', 'xy', 'yx'], pol_mode='4pol_minV')
        gains, true_vis, data = sim_red_data(reds)
        info = om.RedundantCalibrator(reds)
        eqs = info.build_eqs(data)
        assert len(eqs) == 3 * 4
        assert eqs['g_0_Jxx * g_1_Jyy_ * u_4_xy'] == (0, 1, 'xy')
        assert eqs['g_1_Jxx * g_2_Jyy_ * u_4_xy'] == (1, 2, 'xy')
        assert eqs['g_0_Jxx * g_2_Jyy_ * u_5_xy'] == (0, 2, 'xy')
        assert eqs['g_0_Jyy * g_1_Jxx_ * u_4_xy'] == (0, 1, 'yx')
        assert eqs['g_1_Jyy * g_2_Jxx_ * u_4_xy'] == (1, 2, 'yx')
        assert eqs['g_0_Jyy * g_2_Jxx_ * u_5_xy'] == (0, 2, 'yx')

        with pytest.raises(KeyError):
            info.build_eqs({})

    def test_solver(self):
        antpos = linear_array(3)
        reds = om.get_reds(antpos, pols=['xx'], pol_mode='1pol')
        info = om.RedundantCalibrator(reds)
        gains, true_vis, d = sim_red_data(reds)
        w = {}
        w = dict([(k, 1.) for k in d.keys()])

        def solver(data, wgts, **kwargs):
            np.testing.assert_equal(data['g_0_Jxx * g_1_Jxx_ * u_0_xx'], d[0, 1, 'xx'])
            np.testing.assert_equal(data['g_1_Jxx * g_2_Jxx_ * u_0_xx'], d[1, 2, 'xx'])
            np.testing.assert_equal(data['g_0_Jxx * g_2_Jxx_ * u_1_xx'], d[0, 2, 'xx'])
            if len(wgts) == 0:
                return
            np.testing.assert_equal(wgts['g_0_Jxx * g_1_Jxx_ * u_0_xx'], w[0, 1, 'xx'])
            np.testing.assert_equal(wgts['g_1_Jxx * g_2_Jxx_ * u_0_xx'], w[1, 2, 'xx'])
            np.testing.assert_equal(wgts['g_0_Jxx * g_2_Jxx_ * u_1_xx'], w[0, 2, 'xx'])
            return
        info._solver(solver, d)
        info._solver(solver, d, w)

    def test_firstcal_iteration(self):
        NANTS = 18
        NFREQ = 64
        antpos = linear_array(NANTS)
        reds = om.get_reds(antpos, pols=['xx'], pol_mode='1pol')
        info = om.RedundantCalibrator(reds)
        fqs = np.linspace(.1, .2, NFREQ)
        g, true_vis, d = sim_red_data(reds, shape=(1, NFREQ), gain_scatter=0)
        delays = {k: np.random.randn() * 30 for k in g.keys()}  # in ns
        fc_gains = {k: np.exp(2j * np.pi * v * fqs) for k, v in delays.items()}
        delays = {k: np.array([[v]]) for k, v in delays.items()}
        fc_gains = {i: v.reshape(1, NFREQ) for i, v in fc_gains.items()}
        gains = {k: v * fc_gains[k] for k, v in g.items()}
        gains = {k: v.astype(np.complex64) for k, v in gains.items()}
        calibrate_in_place(d, gains, old_gains=g, gain_convention='multiply')
        d = {k: v.astype(np.complex64) for k, v in d.items()}
        dly_sol, off_sol = info._firstcal_iteration(d, df=fqs[1] - fqs[0], f0=fqs[0], medfilt=False)
        sol_degen = info.remove_degen_gains(dly_sol, degen_gains=delays, mode='phase')
        for i in range(NANTS):
            assert dly_sol[(i, 'Jxx')].dtype == np.float64
            assert dly_sol[(i, 'Jxx')].shape == (1, 1)
            assert np.allclose(np.round(sol_degen[(i, 'Jxx')] - delays[(i, 'Jxx')], 0), 0)

    def test_firstcal(self):
        np.random.seed(21)
        antpos = hex_array(2, split_core=False, outriggers=0)
        reds = om.get_reds(antpos, pols=['xx'], pol_mode='1pol')
        rc = om.RedundantCalibrator(reds)
        freqs = np.linspace(1e8, 2e8, 1024)

        # test firstcal where the degeneracies of the phases and delays have already been removed so no abscal is necessary
        gains, true_vis, d = sim_red_data(reds, gain_scatter=0, shape=(2, len(freqs)))
        fc_delays = {ant: [[100e-9 * np.random.randn()]] for ant in gains.keys()}  # in s
        fc_delays = rc.remove_degen_gains(fc_delays)
        fc_offsets = {ant: [[.49 * np.pi * (np.random.rand() > .90)]] for ant in gains.keys()}  # the .49 removes the possibly of phase wraps that need abscal
        fc_offsets = rc.remove_degen_gains(fc_offsets)
        fc_gains = {ant: np.reshape(np.exp(-2.0j * np.pi * freqs * delay - 1.0j * fc_offsets[ant]), (1, len(freqs)))
                    for ant, delay in fc_delays.items()}
        for ant1, ant2, pol in d.keys():
            d[(ant1, ant2, pol)] *= fc_gains[(ant1, split_pol(pol)[0])] * np.conj(fc_gains[(ant2, split_pol(pol)[1])])
        for ant in gains.keys():
            gains[ant] *= fc_gains[ant]
        meta, g_fc = rc.firstcal(d, freqs, conv_crit=0)
        np.testing.assert_array_almost_equal(np.linalg.norm([g_fc[ant] - gains[ant] for ant in g_fc]), 0, decimal=3)

        # test firstcal with only phases (no delays)
        gains, true_vis, d = sim_red_data(reds, gain_scatter=0, shape=(2, len(freqs)))
        fc_delays = {ant: [[0 * np.random.randn()]] for ant in gains.keys()}  # in s
        fc_offsets = {ant: [[.49 * np.pi * (np.random.rand() > .90)]] for ant in gains.keys()}  # the .49 removes the possibly of phase wraps that need abscal
        fc_offsets = rc.remove_degen_gains(fc_offsets)
        fc_gains = {ant: np.reshape(np.exp(-2.0j * np.pi * freqs * delay - 1.0j * fc_offsets[ant]), (1, len(freqs)))
                    for ant, delay in fc_delays.items()}
        for ant1, ant2, pol in d.keys():
            d[(ant1, ant2, pol)] *= fc_gains[(ant1, split_pol(pol)[0])] * np.conj(fc_gains[(ant2, split_pol(pol)[1])])
        for ant in gains.keys():
            gains[ant] *= fc_gains[ant]
        meta, g_fc = rc.firstcal(d, freqs, conv_crit=0)
        np.testing.assert_array_almost_equal(np.linalg.norm([g_fc[ant] - gains[ant] for ant in g_fc]), 0, decimal=10)  # much higher precision

    def test_logcal(self):
        NANTS = 18
        antpos = linear_array(NANTS)
        reds = om.get_reds(antpos, pols=['xx'], pol_mode='1pol')
        info = om.RedundantCalibrator(reds)
        gains, true_vis, d = sim_red_data(reds, gain_scatter=.05)
        w = dict([(k, 1.) for k in d.keys()])
        meta, sol = info.logcal(d)
        for i in range(NANTS):
            assert sol[(i, 'Jxx')].shape == (10, 10)
        for bls in reds:
            ubl = sol[bls[0]]
            assert ubl.shape == (10, 10)
            for bl in bls:
                d_bl = d[bl]
                mdl = sol[(bl[0], 'Jxx')] * sol[(bl[1], 'Jxx')].conj() * ubl
                np.testing.assert_almost_equal(np.abs(d_bl), np.abs(mdl), decimal=10)
                np.testing.assert_almost_equal(np.angle(d_bl * mdl.conj()), 0, decimal=10)

        for k in d.keys():
            d[k] = np.zeros_like(d[k])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            meta, sol = info.logcal(d)
        om.make_sol_finite(sol)
        for red in reds:
            np.testing.assert_array_equal(sol[red[0]], 0.0)
        for ant in gains.keys():
            np.testing.assert_array_equal(sol[ant], 1.0)

    def test_omnical(self):
        NANTS = 18
        antpos = linear_array(NANTS)
        reds = om.get_reds(antpos, pols=['xx'], pol_mode='1pol')
        info = om.RedundantCalibrator(reds)
        gains, true_vis, d = sim_red_data(reds, gain_scatter=.0099999)
        w = dict([(k, 1.) for k in d.keys()])
        sol0 = dict([(k, np.ones_like(v)) for k, v in gains.items()])
        sol0.update(info.compute_ubls(d, sol0))

        def wgt_func1(abs2):
            return 1.

        def wgt_func2(abs2):
            return np.where(abs2 > 0, 5 * np.tanh(abs2 / 5) / abs2, 1)

        for wgt_func in [wgt_func1, wgt_func2]:
            meta, sol = info.omnical(d, sol0, conv_crit=1e-12, gain=.5, maxiter=500, check_after=30, check_every=6, wgt_func=wgt_func)
            for i in range(NANTS):
                assert sol[(i, 'Jxx')].shape == (10, 10)
            for bls in reds:
                ubl = sol[bls[0]]
                assert ubl.shape == (10, 10)
                for bl in bls:
                    d_bl = d[bl]
                    mdl = sol[(bl[0], 'Jxx')] * sol[(bl[1], 'Jxx')].conj() * ubl
                    np.testing.assert_almost_equal(np.abs(d_bl), np.abs(mdl), decimal=10)
                    np.testing.assert_almost_equal(np.angle(d_bl * mdl.conj()), 0, decimal=10)

    def test_omnical64(self):
        NANTS = 18
        antpos = linear_array(NANTS)
        reds = om.get_reds(antpos, pols=['xx'], pol_mode='1pol')
        info = om.RedundantCalibrator(reds)
        gains, true_vis, d = sim_red_data(reds, shape=(2, 1), gain_scatter=.0099999)
        w = dict([(k, 1.) for k in d.keys()])
        sol0 = dict([(k, np.ones_like(v)) for k, v in gains.items()])
        sol0.update(info.compute_ubls(d, sol0))
        d = {k: v.astype(np.complex64) for k, v in d.items()}
        sol0 = {k: v.astype(np.complex64) for k, v in sol0.items()}

        def wgt_func1(abs2):
            return 1.

        def wgt_func2(abs2):
            return np.where(abs2 > 0, 5 * np.tanh(abs2 / 5) / abs2, 1)

        for wgt_func in [wgt_func1, wgt_func2]:
            meta, sol = info.omnical(d, sol0, gain=.5, maxiter=500, check_after=30, check_every=6, wgt_func=wgt_func)
            for bls in reds:
                ubl = sol[bls[0]]
                assert ubl.dtype == np.complex64
                for bl in bls:
                    d_bl = d[bl]
                    mdl = sol[(bl[0], 'Jxx')] * sol[(bl[1], 'Jxx')].conj() * ubl
                    np.testing.assert_almost_equal(np.abs(d_bl), np.abs(mdl), decimal=6)

    def test_omnical128(self):
        NANTS = 18
        antpos = linear_array(NANTS)
        reds = om.get_reds(antpos, pols=['xx'], pol_mode='1pol')
        info = om.RedundantCalibrator(reds)
        gains, true_vis, d = sim_red_data(reds, shape=(2, 1), gain_scatter=.0099999)
        w = dict([(k, 1.) for k in d.keys()])
        sol0 = dict([(k, np.ones_like(v)) for k, v in gains.items()])
        sol0.update(info.compute_ubls(d, sol0))
        d = {k: v.astype(np.complex128) for k, v in d.items()}
        sol0 = {k: v.astype(np.complex128) for k, v in sol0.items()}

        def wgt_func1(abs2):
            return 1.

        def wgt_func2(abs2):
            return np.where(abs2 > 0, 5 * np.tanh(abs2 / 5) / abs2, 1)

        for wgt_func in [wgt_func1, wgt_func2]:
            meta, sol = info.omnical(d, sol0, conv_crit=1e-12, gain=.5, maxiter=500, check_after=30, check_every=6, wgt_func=wgt_func)
            for bls in reds:
                ubl = sol[bls[0]]
                assert ubl.dtype == np.complex128
                for bl in bls:
                    d_bl = d[bl]
                    mdl = sol[(bl[0], 'Jxx')] * sol[(bl[1], 'Jxx')].conj() * ubl
                    np.testing.assert_almost_equal(np.abs(d_bl), np.abs(mdl), decimal=10)
                    np.testing.assert_almost_equal(np.angle(d_bl * mdl.conj()), 0, decimal=10)

    def test_omnical_outliers(self):
        NANTS = 18 * 5  # need a largish array to retain 2 dec accuracy
        antpos = linear_array(NANTS)
        reds = om.get_reds(antpos, pols=['xx'], pol_mode='1pol')
        info = om.RedundantCalibrator(reds)
        gains, true_vis, d = sim_red_data(reds, gain_scatter=.0099999)
        bad_bl = (4, 5, 'xx')
        d[bad_bl] *= 30  # corrupt a measurement
        w = dict([(k, 1.) for k in d.keys()])
        sol0 = dict([(k, np.ones_like(v)) for k, v in gains.items()])
        sol0.update(info.compute_ubls(d, sol0))

        def wgt_func(abs2, thresh=3):
            # return 1.  # this fails the test, proving the below is effective
            return np.where(abs2 > 0, thresh * np.tanh(abs2 / thresh) / abs2, 1)

        meta, sol = info.omnical(d, sol0, conv_crit=1e-12, gain=.5, maxiter=500, check_after=30, check_every=6, wgt_func=wgt_func)
        for i in range(NANTS):
            assert sol[(i, 'Jxx')].shape == (10, 10)
        for bls in reds:
            ubl = sol[bls[0]]
            assert ubl.shape == (10, 10)
            for bl in bls:
                if bl == bad_bl:
                    continue
                d_bl = d[bl]
                mdl = sol[(bl[0], 'Jxx')] * sol[(bl[1], 'Jxx')].conj() * ubl
                np.testing.assert_almost_equal(np.abs(d_bl), np.abs(mdl), decimal=2)
                np.testing.assert_almost_equal(np.angle(d_bl * mdl.conj()), 0, decimal=2)

    def test_lincal(self):
        NANTS = 18
        antpos = linear_array(NANTS)
        reds = om.get_reds(antpos, pols=['xx'], pol_mode='1pol')
        info = om.RedundantCalibrator(reds)
        gains, true_vis, d = sim_red_data(reds, gain_scatter=.0099999)
        w = dict([(k, 1.) for k in d.keys()])
        sol0 = dict([(k, np.ones_like(v)) for k, v in gains.items()])
        sol0.update(info.compute_ubls(d, sol0))
        meta, sol = info.lincal(d, sol0)
        for i in range(NANTS):
            assert sol[(i, 'Jxx')].shape == (10, 10)
        for bls in reds:
            ubl = sol[bls[0]]
            assert ubl.shape == (10, 10)
            for bl in bls:
                d_bl = d[bl]
                mdl = sol[(bl[0], 'Jxx')] * sol[(bl[1], 'Jxx')].conj() * ubl
                assert np.allclose(np.abs(d_bl), np.abs(mdl), atol=1e-10)
                assert np.allclose(np.angle(d_bl * mdl.conj()), 0, atol=1e-10)

    def test_lincal64(self):
        NANTS = 18
        antpos = linear_array(NANTS)
        reds = om.get_reds(antpos, pols=['xx'], pol_mode='1pol')
        info = om.RedundantCalibrator(reds)
        gains, true_vis, d = sim_red_data(reds, shape=(2, 1), gain_scatter=.0099999)
        w = dict([(k, 1.) for k in d.keys()])
        sol0 = dict([(k, np.ones_like(v)) for k, v in gains.items()])
        sol0.update(info.compute_ubls(d, sol0))
        d = {k: v.astype(np.complex64) for k, v in d.items()}
        sol0 = {k: v.astype(np.complex64) for k, v in sol0.items()}
        meta, sol = info.lincal(d, sol0, maxiter=12, conv_crit=1e-6)
        for bls in reds:
            ubl = sol[bls[0]]
            assert ubl.dtype == np.complex64
            for bl in bls:
                d_bl = d[bl]
                mdl = sol[(bl[0], 'Jxx')] * sol[(bl[1], 'Jxx')].conj() * ubl
                np.testing.assert_almost_equal(np.abs(d_bl), np.abs(mdl), decimal=6)
                np.testing.assert_almost_equal(np.angle(d_bl * mdl.conj()), 0, decimal=6)

    def test_lincal128(self):
        NANTS = 18
        antpos = linear_array(NANTS)
        reds = om.get_reds(antpos, pols=['xx'], pol_mode='1pol')
        info = om.RedundantCalibrator(reds)
        gains, true_vis, d = sim_red_data(reds, shape=(2, 1), gain_scatter=.0099999)
        w = dict([(k, 1.) for k in d.keys()])
        sol0 = dict([(k, np.ones_like(v)) for k, v in gains.items()])
        sol0.update(info.compute_ubls(d, sol0))
        d = {k: v.astype(np.complex128) for k, v in d.items()}
        sol0 = {k: v.astype(np.complex128) for k, v in sol0.items()}
        meta, sol = info.lincal(d, sol0, maxiter=12)
        for bls in reds:
            ubl = sol[bls[0]]
            assert ubl.dtype == np.complex128
            for bl in bls:
                d_bl = d[bl]
                mdl = sol[(bl[0], 'Jxx')] * sol[(bl[1], 'Jxx')].conj() * ubl
                np.testing.assert_almost_equal(np.abs(d_bl), np.abs(mdl), decimal=10)
                np.testing.assert_almost_equal(np.angle(d_bl * mdl.conj()), 0, decimal=10)

    def test_svd_convergence(self):
        for hexnum in (2, 3, 4):
            for dtype in (np.complex64, np.complex128):
                antpos = hex_array(hexnum, split_core=False, outriggers=0)
                reds = om.get_reds(antpos, pols=['xx'], pol_mode='1pol')
                rc = om.RedundantCalibrator(reds)
                gains, _, d = sim_red_data(reds, shape=(2, 1), gain_scatter=.01)
                d = {k: dk.astype(dtype) for k, dk in d.items()}
                w = {k: 1. for k in d.keys()}
                gains = {k: gk.astype(dtype) for k, gk in gains.items()}
                sol0 = {k: np.ones_like(gk) for k, gk in gains.items()}
                sol0.update(rc.compute_ubls(d, sol0))
                meta, sol = rc.lincal(d, sol0)  # should not raise 'np.linalg.linalg.LinAlgError: SVD did not converge'

    def test_remove_degen_firstcal_1D(self):
        pol = 'xx'
        xhat = np.array([1., 0, 0])
        dtau_dx = 10.
        antpos = linear_array(10)
        reds = om.get_reds(antpos, pols=[pol], pol_mode='1pol')
        rc = om.RedundantCalibrator(reds)
        # put in a linear slope in delays, see that it is taken out
        true_dlys = {(i, split_pol(pol)[0]): np.array([[np.dot(xhat, antpos[i]) * dtau_dx]]) for i in range(len(antpos))}
        dlys = rc.remove_degen_gains(true_dlys, mode='phase')
        for k in dlys:
            np.testing.assert_almost_equal(dlys[k], 0, decimal=10)
        dlys = rc.remove_degen_gains(true_dlys, degen_gains=true_dlys, mode='phase')
        for k in dlys:
            np.testing.assert_almost_equal(dlys[k], true_dlys[k], decimal=10)

    def test_remove_degen_firstcal_2D(self):
        pol = 'xx'
        xhat = np.array([1., 0, 0])
        yhat = np.array([0., 1, 0])
        dtau_dx = 10.
        dtau_dy = -5.
        antpos = hex_array(5, split_core=False, outriggers=0)
        reds = om.get_reds(antpos, pols=[pol], pol_mode='1pol')
        rc = om.RedundantCalibrator(reds)
        # put in a linear slope in delays, see that it is taken out
        true_dlys = {(i, split_pol(pol)[0]):
                     np.array([[np.dot(xhat, antpos[i]) * dtau_dx + np.dot(yhat, antpos[i]) * dtau_dy]])
                     for i in range(len(antpos))}
        dlys = rc.remove_degen_gains(true_dlys, mode='phase')
        for k in dlys:
            np.testing.assert_almost_equal(dlys[k], 0, decimal=10)

    def test_lincal_hex_end_to_end_1pol_with_remove_degen_and_firstcal(self):
        antpos = hex_array(3, split_core=False, outriggers=0)
        reds = om.get_reds(antpos, pols=['xx'], pol_mode='1pol')
        rc = om.RedundantCalibrator(reds)
        freqs = np.linspace(.1, .2, 10)
        gains, true_vis, d = sim_red_data(reds, gain_scatter=.1, shape=(1, len(freqs)))
        fc_delays = {ant: 100 * np.random.randn() for ant in gains.keys()}  # in ns
        fc_gains = {ant: np.reshape(np.exp(-2.0j * np.pi * freqs * delay), (1, len(freqs))) for ant, delay in fc_delays.items()}
        for ant1, ant2, pol in d.keys():
            d[(ant1, ant2, pol)] *= fc_gains[(ant1, split_pol(pol)[0])] * np.conj(fc_gains[(ant2, split_pol(pol)[1])])
        for ant in gains.keys():
            gains[ant] *= fc_gains[ant]

        w = dict([(k, 1.) for k in d.keys()])
        meta, sol0 = rc.logcal(d, sol0=fc_gains, wgts=w)
        meta, sol = rc.lincal(d, sol0, wgts=w)

        np.testing.assert_array_less(meta['iter'], 50 * np.ones_like(meta['iter']))
        np.testing.assert_almost_equal(meta['chisq'], np.zeros_like(meta['chisq']), decimal=10)

        np.testing.assert_almost_equal(meta['chisq'], 0, decimal=10)
        for i in range(len(antpos)):
            assert sol[(i, 'Jxx')].shape == (1, len(freqs))
        for bls in reds:
            ubl = sol[bls[0]]
            assert ubl.shape == (1, len(freqs))
            for bl in bls:
                d_bl = d[bl]
                mdl = sol[(bl[0], 'Jxx')] * sol[(bl[1], 'Jxx')].conj() * ubl
                np.testing.assert_almost_equal(np.abs(d_bl), np.abs(mdl), decimal=10)
                np.testing.assert_almost_equal(np.angle(d_bl * mdl.conj()), 0, decimal=10)

        sol_rd = rc.remove_degen(sol)
        g, v = om.get_gains_and_vis_from_sol(sol_rd)
        ants = [key for key in sol_rd.keys() if len(key) == 2]
        gainSols = np.array([sol_rd[ant] for ant in ants])
        meanSqAmplitude = np.mean([np.abs(g[key1] * g[key2]) for key1 in g.keys()
                                   for key2 in g.keys() if key1[1] == 'Jxx' and key2[1] == 'Jxx' and key1[0] != key2[0]], axis=0)
        np.testing.assert_almost_equal(meanSqAmplitude, 1, decimal=10)

        for bls in reds:
            ubl = sol_rd[bls[0]]
            assert ubl.shape == (1, len(freqs))
            for bl in bls:
                d_bl = d[bl]
                mdl = sol_rd[(bl[0], 'Jxx')] * sol_rd[(bl[1], 'Jxx')].conj() * ubl
                np.testing.assert_almost_equal(np.abs(d_bl), np.abs(mdl), decimal=10)
                np.testing.assert_almost_equal(np.angle(d_bl * mdl.conj()), 0, decimal=10)

        sol_rd = rc.remove_degen(sol, degen_sol=gains)
        g, v = om.get_gains_and_vis_from_sol(sol_rd)
        meanSqAmplitude = np.mean([np.abs(g[key1] * g[key2]) for key1 in g.keys()
                                   for key2 in g.keys() if key1[1] == 'Jxx' and key2[1] == 'Jxx' and key1[0] != key2[0]], axis=0)
        degenMeanSqAmplitude = np.mean([np.abs(gains[key1] * gains[key2]) for key1 in g.keys()
                                        for key2 in g.keys() if key1[1] == 'Jxx' and key2[1] == 'Jxx' and key1[0] != key2[0]], axis=0)
        np.testing.assert_almost_equal(meanSqAmplitude, degenMeanSqAmplitude, decimal=10)

        for key, val in sol_rd.items():
            if len(key) == 2:
                np.testing.assert_almost_equal(val, gains[key], decimal=10)
            if len(key) == 3:
                np.testing.assert_almost_equal(val, true_vis[key], decimal=10)

        rc.pol_mode = 'unrecognized_pol_mode'
        with pytest.raises(AssertionError):
            sol_rd = rc.remove_degen(sol)

    def test_lincal_hex_end_to_end_4pol_with_remove_degen_and_firstcal(self):
        antpos = hex_array(3, split_core=False, outriggers=0)
        reds = om.get_reds(antpos, pols=['xx', 'xy', 'yx', 'yy'], pol_mode='4pol')
        rc = om.RedundantCalibrator(reds)
        freqs = np.linspace(.1, .2, 10)
        gains, true_vis, d = sim_red_data(reds, gain_scatter=.09, shape=(1, len(freqs)))
        fc_delays = {ant: 100 * np.random.randn() for ant in gains.keys()}  # in ns
        fc_gains = {ant: np.reshape(np.exp(-2.0j * np.pi * freqs * delay), (1, len(freqs))) for ant, delay in fc_delays.items()}
        for ant1, ant2, pol in d.keys():
            d[(ant1, ant2, pol)] *= fc_gains[(ant1, split_pol(pol)[0])] * np.conj(fc_gains[(ant2, split_pol(pol)[1])])
        for ant in gains.keys():
            gains[ant] *= fc_gains[ant]

        w = dict([(k, 1.) for k in d.keys()])
        meta, sol0 = rc.logcal(d, sol0=fc_gains, wgts=w)
        meta, sol = rc.lincal(d, sol0, wgts=w)

        np.testing.assert_array_less(meta['iter'], 50 * np.ones_like(meta['iter']))
        np.testing.assert_almost_equal(meta['chisq'], np.zeros_like(meta['chisq']), decimal=10)

        np.testing.assert_almost_equal(meta['chisq'], 0, decimal=10)
        for i in range(len(antpos)):
            assert sol[(i, 'Jxx')].shape == (1, len(freqs))
            assert sol[(i, 'Jyy')].shape == (1, len(freqs))
        for bls in reds:
            for bl in bls:
                ubl = sol[bls[0]]
                assert ubl.shape == (1, len(freqs))
                d_bl = d[bl]
                mdl = sol[(bl[0], split_pol(bl[2])[0])] * sol[(bl[1], split_pol(bl[2])[1])].conj() * ubl
                np.testing.assert_almost_equal(np.abs(d_bl), np.abs(mdl), decimal=10)
                np.testing.assert_almost_equal(np.angle(d_bl * mdl.conj()), 0, decimal=10)

        sol_rd = rc.remove_degen(sol)

        ants = [key for key in sol_rd.keys() if len(key) == 2]
        gainPols = np.array([ant[1] for ant in ants])
        bl_pairs = [key for key in sol.keys() if len(key) == 3]
        visPols = np.array([[bl[2][0], bl[2][1]] for bl in bl_pairs])
        bl_vecs = np.array([antpos[bl_pair[0]] - antpos[bl_pair[1]] for bl_pair in bl_pairs])
        gainSols = np.array([sol_rd[ant] for ant in ants])
        g, v = om.get_gains_and_vis_from_sol(sol_rd)
        meanSqAmplitude = np.mean([np.abs(g[key1] * g[key2]) for key1 in g.keys()
                                   for key2 in g.keys() if key1[1] == 'Jxx' and key2[1] == 'Jxx' and key1[0] != key2[0]], axis=0)
        np.testing.assert_almost_equal(meanSqAmplitude, 1, decimal=10)
        meanSqAmplitude = np.mean([np.abs(g[key1] * g[key2]) for key1 in g.keys()
                                   for key2 in g.keys() if key1[1] == 'Jyy' and key2[1] == 'Jyy' and key1[0] != key2[0]], axis=0)
        np.testing.assert_almost_equal(meanSqAmplitude, 1, decimal=10)

        for bls in reds:
            for bl in bls:
                ubl = sol_rd[bls[0]]
                assert ubl.shape == (1, len(freqs))
                d_bl = d[bl]
                mdl = sol_rd[(bl[0], split_pol(bl[2])[0])] * sol_rd[(bl[1], split_pol(bl[2])[1])].conj() * ubl
                np.testing.assert_almost_equal(np.abs(d_bl), np.abs(mdl), decimal=10)
                np.testing.assert_almost_equal(np.angle(d_bl * mdl.conj()), 0, decimal=10)

        sol_rd = rc.remove_degen(sol, degen_sol=gains)
        g, v = om.get_gains_and_vis_from_sol(sol_rd)
        meanSqAmplitude = np.mean([np.abs(g[key1] * g[key2]) for key1 in g.keys()
                                   for key2 in g.keys() if key1[1] == 'Jxx' and key2[1] == 'Jxx' and key1[0] != key2[0]], axis=0)
        degenMeanSqAmplitude = np.mean([np.abs(gains[key1] * gains[key2]) for key1 in g.keys()
                                        for key2 in g.keys() if key1[1] == 'Jxx' and key2[1] == 'Jxx' and key1[0] != key2[0]], axis=0)
        np.testing.assert_almost_equal(meanSqAmplitude, degenMeanSqAmplitude, decimal=10)
        meanSqAmplitude = np.mean([np.abs(g[key1] * g[key2]) for key1 in g.keys()
                                   for key2 in g.keys() if key1[1] == 'Jyy' and key2[1] == 'Jyy' and key1[0] != key2[0]], axis=0)
        degenMeanSqAmplitude = np.mean([np.abs(gains[key1] * gains[key2]) for key1 in g.keys()
                                        for key2 in g.keys() if key1[1] == 'Jyy' and key2[1] == 'Jyy' and key1[0] != key2[0]], axis=0)
        np.testing.assert_almost_equal(meanSqAmplitude, degenMeanSqAmplitude, decimal=10)

        gainSols = np.array([sol_rd[ant] for ant in ants])
        degenGains = np.array([gains[ant] for ant in ants])
        np.testing.assert_almost_equal(np.mean(np.angle(gainSols[gainPols == 'Jxx']), axis=0),
                                       np.mean(np.angle(degenGains[gainPols == 'Jxx']), axis=0), decimal=10)
        np.testing.assert_almost_equal(np.mean(np.angle(gainSols[gainPols == 'Jyy']), axis=0),
                                       np.mean(np.angle(degenGains[gainPols == 'Jyy']), axis=0), decimal=10)

        for key, val in sol_rd.items():
            if len(key) == 2:
                np.testing.assert_almost_equal(val, gains[key], decimal=10)
            if len(key) == 3:
                np.testing.assert_almost_equal(val, true_vis[key], decimal=10)

    def test_lincal_hex_end_to_end_4pol_minV_with_remove_degen_and_firstcal(self):

        antpos = hex_array(3, split_core=False, outriggers=0)
        reds = om.get_reds(antpos, pols=['xx', 'xy', 'yx', 'yy'], pol_mode='4pol_minV')
        rc = om.RedundantCalibrator(reds)
        freqs = np.linspace(.1, .2, 10)
        gains, true_vis, d = sim_red_data(reds, gain_scatter=.1, shape=(1, len(freqs)))
        fc_delays = {ant: 100 * np.random.randn() for ant in gains.keys()}  # in ns
        fc_gains = {ant: np.reshape(np.exp(-2.0j * np.pi * freqs * delay), (1, len(freqs))) for ant, delay in fc_delays.items()}
        for ant1, ant2, pol in d.keys():
            d[(ant1, ant2, pol)] *= fc_gains[(ant1, split_pol(pol)[0])] * np.conj(fc_gains[(ant2, split_pol(pol)[1])])
        for ant in gains.keys():
            gains[ant] *= fc_gains[ant]

        w = dict([(k, 1.) for k in d.keys()])
        meta, sol0 = rc.logcal(d, sol0=fc_gains, wgts=w)
        meta, sol = rc.lincal(d, sol0, wgts=w)

        assert np.all(meta['iter'] < 50 * np.ones_like(meta['iter']))
        np.testing.assert_almost_equal(meta['chisq'], np.zeros_like(meta['chisq']), decimal=10)

        np.testing.assert_almost_equal(meta['chisq'], 0, decimal=10)
        for i in range(len(antpos)):
            assert sol[(i, 'Jxx')].shape == (1, len(freqs))
            assert sol[(i, 'Jyy')].shape == (1, len(freqs))
        for bls in reds:
            ubl = sol[bls[0]]
            for bl in bls:
                assert ubl.shape == (1, len(freqs))
                d_bl = d[bl]
                mdl = sol[(bl[0], split_pol(bl[2])[0])] * sol[(bl[1], split_pol(bl[2])[1])].conj() * ubl
                np.testing.assert_almost_equal(np.abs(d_bl), np.abs(mdl), decimal=10)
                np.testing.assert_almost_equal(np.angle(d_bl * mdl.conj()), 0, decimal=10)

        sol_rd = rc.remove_degen(sol)
        g, v = om.get_gains_and_vis_from_sol(sol_rd)
        ants = [key for key in sol_rd.keys() if len(key) == 2]
        gainPols = np.array([ant[1] for ant in ants])
        bl_pairs = [key for key in sol.keys() if len(key) == 3]
        visPols = np.array([[bl[2][0], bl[2][1]] for bl in bl_pairs])
        visPolsStr = np.array([bl[2] for bl in bl_pairs])
        bl_vecs = np.array([antpos[bl_pair[0]] - antpos[bl_pair[1]] for bl_pair in bl_pairs])
        gainSols = np.array([sol_rd[ant] for ant in ants])
        meanSqAmplitude = np.mean([np.abs(g[key1] * g[key2]) for key1 in g.keys()
                                   for key2 in g.keys() if key1[1] == 'Jxx' and key2[1] == 'Jxx' and key1[0] != key2[0]], axis=0)
        np.testing.assert_almost_equal(meanSqAmplitude, 1, decimal=10)
        meanSqAmplitude = np.mean([np.abs(g[key1] * g[key2]) for key1 in g.keys()
                                   for key2 in g.keys() if key1[1] == 'Jyy' and key2[1] == 'Jyy' and key1[0] != key2[0]], axis=0)
        np.testing.assert_almost_equal(meanSqAmplitude, 1, decimal=10)

        for bls in reds:
            ubl = sol_rd[bls[0]]
            for bl in bls:
                assert ubl.shape == (1, len(freqs))
                d_bl = d[bl]
                mdl = sol_rd[(bl[0], split_pol(bl[2])[0])] * sol_rd[(bl[1], split_pol(bl[2])[1])].conj() * ubl
                np.testing.assert_almost_equal(np.abs(d_bl), np.abs(mdl), decimal=10)
                np.testing.assert_almost_equal(np.angle(d_bl * mdl.conj()), 0, decimal=10)

        sol_rd = rc.remove_degen(sol, degen_sol=gains)
        g, v = om.get_gains_and_vis_from_sol(sol_rd)

        for bls in reds:
            ubl = sol_rd[bls[0]]
            for bl in bls:
                assert ubl.shape == (1, len(freqs))
                d_bl = d[bl]
                mdl = sol_rd[(bl[0], split_pol(bl[2])[0])] * sol_rd[(bl[1], split_pol(bl[2])[1])].conj() * ubl
                np.testing.assert_almost_equal(np.abs(d_bl), np.abs(mdl), decimal=10)
                np.testing.assert_almost_equal(np.angle(d_bl * mdl.conj()), 0, decimal=10)

        gainSols = np.array([sol_rd[ant] for ant in ants])
        degenGains = np.array([gains[ant] for ant in ants])
        np.testing.assert_almost_equal(np.mean(np.angle(gainSols), axis=0),
                                       np.mean(np.angle(degenGains), axis=0), decimal=10)

        meanSqAmplitude = np.mean([np.abs(g[key1] * g[key2]) for key1 in g.keys()
                                   for key2 in g.keys() if key1[1] == 'Jxx' and key2[1] == 'Jxx' and key1[0] != key2[0]], axis=0)
        degenMeanSqAmplitude = np.mean([np.abs(gains[key1] * gains[key2]) for key1 in g.keys()
                                        for key2 in g.keys() if key1[1] == 'Jxx' and key2[1] == 'Jxx' and key1[0] != key2[0]], axis=0)
        np.testing.assert_almost_equal(meanSqAmplitude, degenMeanSqAmplitude, decimal=10)
        meanSqAmplitude = np.mean([np.abs(g[key1] * g[key2]) for key1 in g.keys()
                                   for key2 in g.keys() if key1[1] == 'Jyy' and key2[1] == 'Jyy' and key1[0] != key2[0]], axis=0)
        degenMeanSqAmplitude = np.mean([np.abs(gains[key1] * gains[key2]) for key1 in g.keys()
                                        for key2 in g.keys() if key1[1] == 'Jyy' and key2[1] == 'Jyy' and key1[0] != key2[0]], axis=0)
        np.testing.assert_almost_equal(meanSqAmplitude, degenMeanSqAmplitude, decimal=10)

        visSols = np.array([sol_rd[bl] for bl in bl_pairs])
        degenVis = np.array([true_vis[bl] for bl in bl_pairs])
        np.testing.assert_almost_equal(np.mean(np.angle(visSols), axis=0),
                                       np.mean(np.angle(degenVis), axis=0), decimal=10)

        for key, val in sol_rd.items():
            if len(key) == 2:
                np.testing.assert_almost_equal(val, gains[key], decimal=10)
            if len(key) == 3:
                np.testing.assert_almost_equal(val, true_vis[key], decimal=10)

    def test_lincal_hex_end_to_end_2pol_with_remove_degen_and_firstcal(self):

        antpos = hex_array(3, split_core=False, outriggers=0)
        reds = om.get_reds(antpos, pols=['xx', 'yy'], pol_mode='2pol')
        rc = om.RedundantCalibrator(reds)
        freqs = np.linspace(.1, .2, 10)
        gains, true_vis, d = sim_red_data(reds, gain_scatter=.1, shape=(1, len(freqs)))
        fc_delays = {ant: 100 * np.random.randn() for ant in gains.keys()}  # in ns
        fc_gains = {ant: np.reshape(np.exp(-2.0j * np.pi * freqs * delay), (1, len(freqs))) for ant, delay in fc_delays.items()}
        for ant1, ant2, pol in d.keys():
            d[(ant1, ant2, pol)] *= fc_gains[(ant1, split_pol(pol)[0])] * np.conj(fc_gains[(ant2, split_pol(pol)[1])])
        for ant in gains.keys():
            gains[ant] *= fc_gains[ant]

        w = dict([(k, 1.) for k in d.keys()])
        meta, sol0 = rc.logcal(d, sol0=fc_gains, wgts=w)
        meta, sol = rc.lincal(d, sol0, wgts=w)

        assert np.all(meta['iter'] < 50 * np.ones_like(meta['iter']))
        np.testing.assert_almost_equal(meta['chisq'], np.zeros_like(meta['chisq']), decimal=10)

        np.testing.assert_almost_equal(meta['chisq'], 0, decimal=10)
        for i in range(len(antpos)):
            assert sol[(i, 'Jxx')].shape == (1, len(freqs))
            assert sol[(i, 'Jyy')].shape == (1, len(freqs))
        for bls in reds:
            for bl in bls:
                ubl = sol[bls[0]]
                assert ubl.shape == (1, len(freqs))
                d_bl = d[bl]
                mdl = sol[(bl[0], split_pol(bl[2])[0])] * sol[(bl[1], split_pol(bl[2])[1])].conj() * ubl
                np.testing.assert_almost_equal(np.abs(d_bl), np.abs(mdl), decimal=10)
                np.testing.assert_almost_equal(np.angle(d_bl * mdl.conj()), 0, decimal=10)

        sol_rd = rc.remove_degen(sol)

        ants = [key for key in sol_rd.keys() if len(key) == 2]
        gainPols = np.array([ant[1] for ant in ants])
        bl_pairs = [key for key in sol.keys() if len(key) == 3]
        visPols = np.array([[bl[2][0], bl[2][1]] for bl in bl_pairs])
        bl_vecs = np.array([antpos[bl_pair[0]] - antpos[bl_pair[1]] for bl_pair in bl_pairs])
        gainSols = np.array([sol_rd[ant] for ant in ants])
        g, v = om.get_gains_and_vis_from_sol(sol_rd)

        meanSqAmplitude = np.mean([np.abs(g[key1] * g[key2]) for key1 in g.keys()
                                   for key2 in g.keys() if key1[1] == 'Jxx' and key2[1] == 'Jxx' and key1[0] != key2[0]], axis=0)
        np.testing.assert_almost_equal(meanSqAmplitude, 1, decimal=10)
        meanSqAmplitude = np.mean([np.abs(g[key1] * g[key2]) for key1 in g.keys()
                                   for key2 in g.keys() if key1[1] == 'Jyy' and key2[1] == 'Jyy' and key1[0] != key2[0]], axis=0)
        np.testing.assert_almost_equal(meanSqAmplitude, 1, decimal=10)

        for bls in reds:
            for bl in bls:
                ubl = sol_rd[bls[0]]
                assert ubl.shape == (1, len(freqs))
                d_bl = d[bl]
                mdl = sol_rd[(bl[0], split_pol(bl[2])[0])] * sol_rd[(bl[1], split_pol(bl[2])[1])].conj() * ubl
                np.testing.assert_almost_equal(np.abs(d_bl), np.abs(mdl), decimal=10)
                np.testing.assert_almost_equal(np.angle(d_bl * mdl.conj()), 0, decimal=10)

        sol_rd = rc.remove_degen(sol, degen_sol=gains)
        g, v = om.get_gains_and_vis_from_sol(sol_rd)
        gainSols = np.array([sol_rd[ant] for ant in ants])
        degenGains = np.array([gains[ant] for ant in ants])

        meanSqAmplitude = np.mean([np.abs(g[key1] * g[key2]) for key1 in g.keys()
                                   for key2 in g.keys() if key1[1] == 'Jxx' and key2[1] == 'Jxx' and key1[0] != key2[0]], axis=0)
        degenMeanSqAmplitude = np.mean([np.abs(gains[key1] * gains[key2]) for key1 in g.keys()
                                        for key2 in g.keys() if key1[1] == 'Jxx' and key2[1] == 'Jxx' and key1[0] != key2[0]], axis=0)
        np.testing.assert_almost_equal(meanSqAmplitude, degenMeanSqAmplitude, decimal=10)
        meanSqAmplitude = np.mean([np.abs(g[key1] * g[key2]) for key1 in g.keys()
                                   for key2 in g.keys() if key1[1] == 'Jyy' and key2[1] == 'Jyy' and key1[0] != key2[0]], axis=0)
        degenMeanSqAmplitude = np.mean([np.abs(gains[key1] * gains[key2]) for key1 in g.keys()
                                        for key2 in g.keys() if key1[1] == 'Jyy' and key2[1] == 'Jyy' and key1[0] != key2[0]], axis=0)
        np.testing.assert_almost_equal(meanSqAmplitude, degenMeanSqAmplitude, decimal=10)

        np.testing.assert_almost_equal(np.mean(np.angle(gainSols[gainPols == 'Jxx']), axis=0),
                                       np.mean(np.angle(degenGains[gainPols == 'Jxx']), axis=0), decimal=10)
        np.testing.assert_almost_equal(np.mean(np.angle(gainSols[gainPols == 'Jyy']), axis=0),
                                       np.mean(np.angle(degenGains[gainPols == 'Jyy']), axis=0), decimal=10)

        for key, val in sol_rd.items():
            if len(key) == 2:
                np.testing.assert_almost_equal(val, gains[key], decimal=10)
            if len(key) == 3:
                np.testing.assert_almost_equal(val, true_vis[key], decimal=10)

    def test_count_degen(self):
        # 1 phase slope
        antpos = linear_array(10)
        reds = om.get_reds(antpos, pols=['xx'], pol_mode='1pol')
        assert om.RedundantCalibrator(reds).count_degens() == 3
        reds = om.get_reds(antpos, pols=['xx', 'yy'], pol_mode='2pol')
        assert om.RedundantCalibrator(reds).count_degens() == 6
        reds = om.get_reds(antpos, pols=['xx', 'yy', 'xy', 'yx'], pol_mode='4pol')
        assert om.RedundantCalibrator(reds).count_degens() == 5
        reds = om.get_reds(antpos, pols=['xx', 'yy', 'xy', 'yx'], pol_mode='4pol_minV')
        assert om.RedundantCalibrator(reds).count_degens() == 4

        # 2 phase slopes (fiducial case)
        antpos = hex_array(3, split_core=False, outriggers=0)
        reds = om.get_reds(antpos, pols=['xx'], pol_mode='1pol')
        assert om.RedundantCalibrator(reds).count_degens() == 4
        reds = om.get_reds(antpos, pols=['xx', 'yy'], pol_mode='2pol')
        assert om.RedundantCalibrator(reds).count_degens() == 8
        reds = om.get_reds(antpos, pols=['xx', 'yy', 'xy', 'yx'], pol_mode='4pol')
        assert om.RedundantCalibrator(reds).count_degens() == 6
        reds = om.get_reds(antpos, pols=['xx', 'yy', 'xy', 'yx'], pol_mode='4pol_minV')
        assert om.RedundantCalibrator(reds).count_degens() == 5

        # 4 phase slopes (not traditionally redundantly calibratable)
        antpos[0] += [5., 0, 0]
        antpos[7] += [0, 5., 0]
        reds = om.get_reds(antpos, pols=['xx'], pol_mode='1pol')
        assert om.RedundantCalibrator(reds).count_degens() == 6
        reds = om.get_reds(antpos, pols=['xx', 'yy'], pol_mode='2pol')
        assert om.RedundantCalibrator(reds).count_degens() == 12
        reds = om.get_reds(antpos, pols=['xx', 'yy', 'xy', 'yx'], pol_mode='4pol')
        assert om.RedundantCalibrator(reds).count_degens() == 8
        reds = om.get_reds(antpos, pols=['xx', 'yy', 'xy', 'yx'], pol_mode='4pol_minV')
        assert om.RedundantCalibrator(reds).count_degens() == 7

    def test_is_redundantly_calibratable(self):
        pos = hex_array(3, split_core=False, outriggers=0)
        assert om.is_redundantly_calibratable(pos, bl_error_tol=1)
        pos[0] += [.5, 0, 0]
        assert not om.is_redundantly_calibratable(pos, bl_error_tol=.1)
        assert om.is_redundantly_calibratable(pos, bl_error_tol=1)

        # test a very small array
        pos = hex_array(3, split_core=False, outriggers=0)
        pos = {ant: pos[ant] for ant in range(4)}
        assert not om.is_redundantly_calibratable(pos)

        # test disconnected redundant array
        pos = hex_array(5, split_core=False, outriggers=0)
        pos = {ant: pos[ant] for ant in pos if ant in [0, 1, 5, 6, 54, 55, 59, 60]}
        assert not om.is_redundantly_calibratable(pos)
        assert om.is_redundantly_calibratable(pos, require_coplanarity=False)

    def test_predict_chisq_per_bl(self):
        # This test shows that predicted chisq_per_bl make sense, given the known constraints.
        # See test_predict_chisq_statistically to see that the answer actually works

        # Test linear array
        antpos = linear_array(7)
        reds = om.get_reds(antpos)
        chisq_per_bl = om.predict_chisq_per_bl(reds)
        rc = om.RedundantCalibrator(reds)
        # show that the total degrees of freedom adds up for total chi^2 (see HERA memo #61)
        dof = len(antpos) * (len(antpos) - 1) / 2 - len(reds) - len(antpos) + rc.count_degens() / 2.0
        np.testing.assert_approx_equal(np.sum(list(chisq_per_bl.values())), dof)
        # show that all baselines have DoF less than 1 (this is just a sense check)
        np.testing.assert_array_less(list(chisq_per_bl.values()), 1.0)
        for red in reds:
            if len(red) == 1:
                # show that length 1 redundancies (unique baselines) have expected chi^2 of 0
                np.testing.assert_almost_equal(chisq_per_bl[red[0]], 1e-10)

        # Test hex array (see comments on analogous test above)
        antpos = hex_array(3, split_core=False, outriggers=0)
        reds = om.get_reds(antpos)
        chisq_per_bl = om.predict_chisq_per_bl(reds)
        rc = om.RedundantCalibrator(reds)
        dof = len(antpos) * (len(antpos) - 1) / 2 - len(reds) - len(antpos) + rc.count_degens() / 2.0
        np.testing.assert_approx_equal(np.sum(list(chisq_per_bl.values())), dof)
        np.testing.assert_array_less(list(chisq_per_bl.values()), 1.0)
        for red in reds:
            if len(red) == 1:
                np.testing.assert_almost_equal(chisq_per_bl[red[0]], 1e-10)

        # Test 2 pol array (see comments on analogous test above)
        antpos = hex_array(3, split_core=False, outriggers=0)
        reds = om.get_reds(antpos, pols=['xx', 'yy'])
        chisq_per_bl = om.predict_chisq_per_bl(reds)
        rc = om.RedundantCalibrator(reds)
        dof = 2.0 * len(antpos) * (len(antpos) - 1) / 2 - len(reds) - 2 * len(antpos) + rc.count_degens() / 2.0
        np.testing.assert_approx_equal(np.sum(list(chisq_per_bl.values())), dof)
        np.testing.assert_array_less(list(chisq_per_bl.values()), 1.0)
        for red in reds:
            if len(red) == 1:
                np.testing.assert_almost_equal(chisq_per_bl[red[0]], 1e-10)

    def test_predict_chisq_per_red(self):
        # This test shows that predicted chisq_per_red make sense, given the known constraints.
        # See test_predict_chisq_statistically to see that the answer actually works

        # Test linear array
        antpos = linear_array(7)
        reds = om.get_reds(antpos)
        nubl = len(reds)
        nbl = np.sum([len(red) for red in reds])
        nant = len(antpos)
        chisq_per_red = om.predict_chisq_per_red(reds)
        rc = om.RedundantCalibrator(reds)
        # show that the total degrees of freedom adds up for total chi^2 (see HERA memo #61)
        dof = len(antpos) * (len(antpos) - 1) / 2 - len(reds) - len(antpos) + rc.count_degens() / 2.0
        np.testing.assert_approx_equal(np.sum(list(chisq_per_red.values())), dof)
        non_degen_dof_per_ubl = {red[0]: len(red) - 1 - nant * (len(red) - 1.) / (nbl - nubl) for red in reds}
        for red in reds:
            # show that, up to the degeneracies, the above formula is a correct apportionment of the degrees of freedom
            # that are taken up by antennas and unique baselines
            assert chisq_per_red[red[0]] - non_degen_dof_per_ubl[red[0]] < 1
            if len(red) == 1:
                # show that if the length of the redundancy is 1, then then there are no DoF
                assert chisq_per_red[red[0]] < 1e-10
            else:
                # show that if the length of the redundancy is greater than 1, some of the degenerate DoF are present
                assert chisq_per_red[red[0]] - non_degen_dof_per_ubl[red[0]] > 0

        # Test hex array (see comments on analogous test above)
        antpos = hex_array(3, split_core=False, outriggers=0)
        reds = om.get_reds(antpos)
        nubl = len(reds)
        nbl = np.sum([len(red) for red in reds])
        nant = len(antpos)
        chisq_per_red = om.predict_chisq_per_red(reds)
        rc = om.RedundantCalibrator(reds)
        dof = len(antpos) * (len(antpos) - 1) / 2 - len(reds) - len(antpos) + rc.count_degens() / 2.0
        np.testing.assert_approx_equal(np.sum(list(chisq_per_red.values())), dof)
        non_degen_dof_per_ubl = {red[0]: len(red) - 1 - nant * (len(red) - 1.) / (nbl - nubl) for red in reds}
        for red in reds:
            assert chisq_per_red[red[0]] - non_degen_dof_per_ubl[red[0]] < 1
            if len(red) == 1:
                assert chisq_per_red[red[0]] < 1e-10
            else:
                assert chisq_per_red[red[0]] - non_degen_dof_per_ubl[red[0]] > 0

        # Test 2 pol array (see comments on analogous test above)
        antpos = hex_array(3, split_core=False, outriggers=0)
        reds = om.get_reds(antpos, pols=['xx', 'yy'])
        nubl = len(reds)
        nbl = np.sum([len(red) for red in reds])
        nant = len(antpos)
        chisq_per_red = om.predict_chisq_per_red(reds)
        rc = om.RedundantCalibrator(reds)
        dof = 2.0 * len(antpos) * (len(antpos) - 1) / 2 - len(reds) - 2 * len(antpos) + rc.count_degens() / 2.0
        np.testing.assert_approx_equal(np.sum(list(chisq_per_red.values())), dof)
        non_degen_dof_per_ubl = {red[0]: len(red) - 1 - nant * (len(red) - 1.) / (nbl / 2 - nubl / 2) for red in reds}
        for red in reds:
            assert chisq_per_red[red[0]] - non_degen_dof_per_ubl[red[0]] < 1
            if len(red) == 1:
                assert chisq_per_red[red[0]] < 1e-10
            else:
                assert chisq_per_red[red[0]] - non_degen_dof_per_ubl[red[0]] > 0

    def test_predict_chisq_per_ant(self):
        # This test shows that predicted chisq_per_ant make sense, given the known constraints.
        # See test_predict_chisq_statistically to see that the answer actually works

        # Test linear array
        antpos = linear_array(7)
        ants = [(ant, 'Jxx') for ant in antpos]
        reds = om.get_reds(antpos, pols=['xx'])
        nubl = len(reds)
        nbl = np.sum([len(red) for red in reds])
        nant = len(antpos)
        chisq_per_ant = om.predict_chisq_per_ant(reds)
        rc = om.RedundantCalibrator(reds)
        # show that the total degrees of freedom adds up for total chi^2 (see HERA memo #61)
        dof = len(antpos) * (len(antpos) - 1) / 2 - len(reds) - len(antpos) + rc.count_degens() / 2.0
        np.testing.assert_approx_equal(np.sum(list(chisq_per_ant.values())), 2 * dof)
        # factor of 2 comes from the fact that all baselines go into two antennas' chisq_per_ant
        non_degen_dof_per_ant = {ant: -2 for ant in ants}
        for red in reds:
            for bl in red:
                for ant in split_bl(bl):
                    non_degen_dof_per_ant[ant] += 1.0 - 1.0 / (len(red))
        for ant in ants:
            # show that the number of degrees of freedom (i.e. expected chi^2) per antenna is always a bit larger
            # than the number expected from just antenna and ubl DoF, but that each antenna gets some of the
            # degenerate DoF
            assert chisq_per_ant[ant] - non_degen_dof_per_ant[ant] < 2
            assert chisq_per_ant[ant] - non_degen_dof_per_ant[ant] > 0

        # Test hex array (see comments on analogous test above)
        antpos = hex_array(3, split_core=False, outriggers=0)
        ants = [(ant, 'Jxx') for ant in antpos]
        reds = om.get_reds(antpos, pols=['xx'])
        nubl = len(reds)
        nbl = np.sum([len(red) for red in reds])
        nant = len(antpos)
        chisq_per_ant = om.predict_chisq_per_ant(reds)
        rc = om.RedundantCalibrator(reds)
        dof = len(antpos) * (len(antpos) - 1) / 2 - len(reds) - len(antpos) + rc.count_degens() / 2.0
        np.testing.assert_approx_equal(np.sum(list(chisq_per_ant.values())), 2 * dof)
        non_degen_dof_per_ant = {ant: -2 for ant in ants}
        for red in reds:
            for bl in red:
                for ant in split_bl(bl):
                    non_degen_dof_per_ant[ant] += 1.0 - 1.0 / (len(red))
        for ant in ants:
            assert chisq_per_ant[ant] - non_degen_dof_per_ant[ant] < 2
            assert chisq_per_ant[ant] - non_degen_dof_per_ant[ant] > 0

        # Test 2 pol array (see comments on analogous test above)
        antpos = hex_array(3, split_core=False, outriggers=0)
        ants = [(ant, pol) for ant in antpos for pol in ['Jxx', 'Jyy']]
        reds = om.get_reds(antpos, pols=['xx', 'yy'])
        nubl = len(reds)
        nbl = np.sum([len(red) for red in reds])
        nant = len(antpos)
        chisq_per_ant = om.predict_chisq_per_ant(reds)
        rc = om.RedundantCalibrator(reds)
        dof = 2.0 * len(antpos) * (len(antpos) - 1) / 2 - len(reds) - 2 * len(antpos) + rc.count_degens() / 2.0
        np.testing.assert_approx_equal(np.sum(list(chisq_per_ant.values())), 2 * dof)
        non_degen_dof_per_ant = {ant: -2 for ant in ants}
        for red in reds:
            for bl in red:
                for ant in split_bl(bl):
                    non_degen_dof_per_ant[ant] += 1.0 - 1.0 / (len(red))
        for ant in ants:
            assert chisq_per_ant[ant] - non_degen_dof_per_ant[ant] < 2
            assert chisq_per_ant[ant] - non_degen_dof_per_ant[ant] > 0

    def test_predict_chisq_statistically(self):
        # Show that chisq prediction works pretty well for small arrays
        np.random.seed(21)
        antpos = hex_array(2, split_core=False, outriggers=0)
        reds = om.get_reds(antpos, pols=['xx'])
        freqs = np.linspace(100e6, 200e6, 64, endpoint=False)
        times = np.linspace(0, 600. / 60 / 60 / 24, 60, endpoint=False)
        df = np.median(np.diff(freqs))
        dt = np.median(np.diff(times)) * 3600. * 24

        # Simulate redundant data with noise
        noise_var = .001
        g, tv, d = sim_red_data(reds, shape=(len(times), len(freqs)), gain_scatter=.00)
        ants = g.keys()
        n = DataContainer({bl: np.sqrt(noise_var / 2) * (np.random.randn(*vis.shape) + 1j * np.random.randn(*vis.shape)) for bl, vis in d.items()})
        noisy_data = n + DataContainer(d)

        # Set up autocorrelations so that the predicted noise variance is the actual simulated noise variance
        for antnum in antpos.keys():
            noisy_data[(antnum, antnum, 'xx')] = np.ones((len(times), len(freqs))) * np.sqrt(noise_var * dt * df)
        noisy_data.freqs = deepcopy(freqs)
        noisy_data.times_by_bl = {bl[0:2]: deepcopy(times) for bl in noisy_data.keys()}
        cal = om.redundantly_calibrate(noisy_data, reds)

        # Compute various chi^2s
        chisq_per_bl = {}
        chisq_per_red = {red[0]: 0.0 for red in reds}
        chisq_per_ant = {ant: 0.0 for ant in ants}
        for red in reds:
            for bl in red:
                d_here = noisy_data[bl]
                ant0, ant1 = split_bl(bl)
                g1, g2 = cal['g_omnical'][ant0], cal['g_omnical'][ant1]
                v_here = cal['v_omnical'][red[0]]
                chisq_per_bl[bl] = np.abs(d_here - g1 * np.conj(g2) * v_here)**2 / noise_var
                chisq_per_red[red[0]] += chisq_per_bl[bl]
                chisq_per_ant[ant0] += chisq_per_bl[bl]
                chisq_per_ant[ant1] += chisq_per_bl[bl]

        # compare predictions at the 2% level
        predicted_chisq_per_bl = om.predict_chisq_per_bl(reds)
        for bl in chisq_per_bl:
            np.testing.assert_almost_equal(np.mean(chisq_per_bl[bl]), predicted_chisq_per_bl[bl], -np.log10(.02))

        predicted_chisq_per_red = om.predict_chisq_per_red(reds)
        for red in chisq_per_red:
            np.testing.assert_almost_equal(np.mean(chisq_per_red[red]), predicted_chisq_per_red[red], -np.log10(.02))

        predicted_chisq_per_ant = om.predict_chisq_per_ant(reds)
        for ant in chisq_per_ant:
            np.testing.assert_almost_equal(np.mean(chisq_per_ant[ant]), predicted_chisq_per_ant[ant], -np.log10(.02))

    def test_predict_chisq_statistically_with_excluded_antenna(self):
        np.random.seed(21)
        antpos = hex_array(2, split_core=False, outriggers=0)
        reds = om.get_reds(antpos, pols=['xx'])
        freqs = np.linspace(100e6, 200e6, 64, endpoint=False)
        times = np.linspace(0, 600. / 60 / 60 / 24, 60, endpoint=False)
        df = np.median(np.diff(freqs))
        dt = np.median(np.diff(times)) * 3600. * 24

        # Simulate redundant data with noise
        noise_var = .001
        g, tv, d = sim_red_data(reds, shape=(len(times), len(freqs)), gain_scatter=.00)
        ants = g.keys()
        n = DataContainer({bl: np.sqrt(noise_var / 2) * (np.random.randn(*vis.shape) + 1j * np.random.randn(*vis.shape)) for bl, vis in d.items()})
        noisy_data = n + DataContainer(d)
        nsamples = DataContainer({bl: np.ones_like(d[bl], dtype=float) for bl in d})

        # Set up autocorrelations so that the predicted noise variance is the actual simulated noise variance
        for antnum in antpos.keys():
            noisy_data[(antnum, antnum, 'xx')] = np.ones((len(times), len(freqs))) * np.sqrt(noise_var * dt * df)
        noisy_data.freqs = deepcopy(freqs)
        noisy_data.times_by_bl = {bl[0:2]: deepcopy(times) for bl in noisy_data.keys()}
        filtered_reds = om.filter_reds(reds, ex_ants=[6])
        cal = om.redundantly_calibrate(noisy_data, filtered_reds)
        cal_copy = deepcopy(cal)
        om.expand_omni_sol(cal, reds, noisy_data, nsamples)
        # expand_omni_sol(cal_copy, reds, noisy_data, nsamples)

        # Compute various chi^2s
        chisq_per_bl = {}
        chisq_per_red = {red[0]: 0.0 for red in filtered_reds}
        chisq_per_ant = {ant: 0.0 for ant in ants}
        for red in filtered_reds:
            for bl in red:
                d_here = noisy_data[bl]
                ant0, ant1 = split_bl(bl)
                g1, g2 = cal['g_omnical'][ant0], cal['g_omnical'][ant1]
                v_here = cal['v_omnical'][red[0]]
                chisq_per_bl[bl] = np.abs(d_here - g1 * np.conj(g2) * v_here)**2 / noise_var
                chisq_per_red[red[0]] += chisq_per_bl[bl]
                chisq_per_ant[ant0] += chisq_per_bl[bl]
                chisq_per_ant[ant1] += chisq_per_bl[bl]

        # compare predictions at the 3% level for non-excluded antennas
        np.testing.assert_almost_equal(np.mean(cal['chisq']['Jxx']), 1.0, -np.log10(.03))

        predicted_chisq_per_bl = om.predict_chisq_per_bl(filtered_reds)
        for bl in predicted_chisq_per_bl:
            np.testing.assert_almost_equal(np.mean(chisq_per_bl[bl]), predicted_chisq_per_bl[bl], -np.log10(.03))

        predicted_chisq_per_red = om.predict_chisq_per_red(filtered_reds)
        for red in predicted_chisq_per_red:
            np.testing.assert_almost_equal(np.mean(chisq_per_red[red]), predicted_chisq_per_red[red], -np.log10(.03))

        predicted_chisq_per_ant = om.predict_chisq_per_ant(filtered_reds)
        for ant in predicted_chisq_per_ant:
            np.testing.assert_almost_equal(np.mean(chisq_per_ant[ant]), predicted_chisq_per_ant[ant], -np.log10(.03))

        # make sure excluded antenna has the highest chi^2, but not inexplicably large
        assert np.mean(cal['chisq_per_ant'][6, 'Jxx']) <= len(antpos)
        for ant in cal['chisq_per_ant']:
            assert np.mean(cal['chisq_per_ant'][ant]) <= np.mean(cal['chisq_per_ant'][6, 'Jxx'])


class TestRedcalAndAbscal(object):

    @pytest.mark.parametrize("fc_min_vis_per_ant", [16, None])
    def test_post_redcal_abscal(self, fc_min_vis_per_ant):
        '''This test shows that performing a combination of redcal and abscal recovers the exact input gains
        up to an overall phase (which is handled by using a reference antenna).'''
        # Simulate Redundant Data
        np.random.seed(21)
        antpos = hex_array(3, split_core=False, outriggers=0)
        reds = om.get_reds(antpos, pols=['xx'], pol_mode='1pol')
        rc = om.RedundantCalibrator(reds)
        freqs = np.linspace(1e8, 2e8, 256)
        gains, true_vis, d = sim_red_data(reds, gain_scatter=.1, shape=(2, len(freqs)))
        d = DataContainer(d)
        fc_delays = {ant: 100e-9 * np.random.randn() for ant in gains.keys()}  # in s
        fc_offsets = {ant: 2 * np.pi * np.random.rand() for ant in gains.keys()}  # random phase offsets
        fc_gains = {ant: np.reshape(np.exp(2.0j * np.pi * freqs * delay + 1.0j * fc_offsets[ant]),
                                    (1, len(freqs))) for ant, delay in fc_delays.items()}
        for ant1, ant2, pol in d.keys():
            d[(ant1, ant2, pol)] *= fc_gains[(ant1, split_pol(pol)[0])] * np.conj(fc_gains[(ant2, split_pol(pol)[1])])
        for ant in gains.keys():
            gains[ant] *= fc_gains[ant]
        true_gains = deepcopy(gains)
        for ant in antpos.keys():
            for pol in ['xx']:
                d[ant, ant, pol] = np.ones_like(d[0, 1, 'xx'])  # these are used only for calculating chi^2 and not relevant
        d.freqs = freqs
        d.times_by_bl = {bl[0:2]: np.array([2458110.18523274, 2458110.18535701]) for bl in d.keys()}
        d.antpos = antpos

        # run redcal
        cal = om.redundantly_calibrate(d, reds, fc_min_vis_per_ant=fc_min_vis_per_ant, oc_conv_crit=1e-13, oc_maxiter=5000)

        # set up abscal
        d_omnicaled = deepcopy(d)
        f_omnicaled = DataContainer({bl: np.zeros_like(d[bl], dtype=bool) for bl in d.keys()})
        calibrate_in_place(d_omnicaled, cal['g_omnical'], data_flags=f_omnicaled, cal_flags=cal['gf_omnical'])
        wgts = DataContainer({k: (~f_omnicaled[k]).astype(float) for k in f_omnicaled.keys()})
        model = DataContainer({bl: true_vis[red[0]] for red in reds for bl in red})
        model.freqs = freqs
        model.times_by_bl = {bl[0:2]: np.array([2458110.18523274, 2458110.18535701]) for bl in model.keys()}
        model.antpos = antpos

        # run abscal
        abscal_delta_gains = abscal.post_redcal_abscal(model, d_omnicaled, wgts, cal['gf_omnical'], verbose=True, phs_max_iter=200, phs_conv_crit=1e-10)

        # evaluate solutions, rephasing to antenna 0 as a reference
        abscal_gains = {ant: cal['g_omnical'][ant] * abscal_delta_gains[ant] for ant in cal['g_omnical']}
        refant = {'Jxx': (0, 'Jxx'), 'Jyy': (0, 'Jyy')}
        agr = {ant: abscal_gains[ant] * np.abs(abscal_gains[refant[ant[1]]]) / abscal_gains[refant[ant[1]]]
               for ant in abscal_gains.keys()}
        tgr = {ant: true_gains[ant] * np.abs(true_gains[refant[ant[1]]]) / true_gains[refant[ant[1]]]
               for ant in true_gains.keys()}
        gain_errors = [agr[ant] - tgr[ant] for ant in tgr if ant[1] == 'Jxx']
        np.testing.assert_almost_equal(np.abs(gain_errors), 0, decimal=10)


@pytest.mark.filterwarnings("ignore:It seems that the latitude and longitude are in radians")
@pytest.mark.filterwarnings("ignore:The default for the `center` keyword has changed")
class TestRunMethods(object):

    def test_get_pol_load_list(self):
        assert om._get_pol_load_list(['xx'], pol_mode='1pol') == [['xx']]
        assert om._get_pol_load_list(['xx', 'yy'], pol_mode='2pol') == [['xx'], ['yy']]
        assert om._get_pol_load_list(['xx', 'yy', 'xy', 'yx'], pol_mode='4pol') == [['xx', 'yy', 'xy', 'yx']]
        assert om._get_pol_load_list(['xx', 'yy', 'xy', 'yx'], pol_mode='4pol_minV') == [['xx', 'yy', 'xy', 'yx']]
        with pytest.raises(AssertionError):
            om._get_pol_load_list(['xx'], pol_mode='4pol')
        with pytest.raises(ValueError):
            om._get_pol_load_list(['xx'], pol_mode='10pol')

    def test_redundantly_calibrate(self):
        hd = io.HERAData(os.path.join(DATA_PATH, 'zen.2458098.43124.downsample.uvh5'))
        data, flags, nsamples = hd.read()
        nTimes, nFreqs = len(hd.times), len(hd.freqs)

        for pol_mode in ['2pol', '4pol']:
            pol_load_list = om._get_pol_load_list(hd.pols, pol_mode=pol_mode)
            ant_nums = np.unique(np.append(hd.ant_1_array, hd.ant_2_array))
            all_reds = om.get_reds({ant: hd.antpos[ant] for ant in ant_nums}, pol_mode=pol_mode,
                                   pols=set([pol for pols in pol_load_list for pol in pols]))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rv = om.redundantly_calibrate(data, all_reds)
                for r in ['v_omnical', 'chisq_per_ant', 'omni_meta', 'g_firstcal', 'chisq', 'gf_firstcal', 'g_omnical', 'gf_omnical', 'vf_omnical']:
                    assert r in rv
                for r in ['v_omnical', 'g_firstcal', 'gf_firstcal', 'g_omnical', 'gf_omnical', 'vf_omnical', 'chisq_per_ant']:
                    for val in rv[r].values():
                        assert val.shape == (nTimes, nFreqs)
                        if r in ['v_omnical', 'g_firstcal', 'g_omnical']:
                            assert val.dtype == np.complex64
                        elif r in ['vf_omnical', 'gf_firstcal', 'gf_omnical']:
                            assert val.dtype == bool

                for flag in rv['gf_firstcal'].values():
                    np.testing.assert_array_equal(flag, 0)
                for k, flag in rv['gf_omnical'].items():
                    np.testing.assert_array_equal(rv['g_omnical'][k][flag], 1.)
                for k, flag in rv['vf_omnical'].items():
                    np.testing.assert_array_equal(rv['v_omnical'][k][flag], 0)

        if pol_mode == '4pol':
            assert rv['chisq'].shape == (nTimes, nFreqs)
        else:
            assert len(rv['chisq']) == 2
            for val in rv['chisq'].values():
                assert val.shape == (nTimes, nFreqs)

    def test_expand_omni_sol(self):
        # noise free test of dead antenna resurrection
        ex_ants = [0, 13, 2, 18]
        antpos = hex_array(3, split_core=False, outriggers=0)
        pols = ['xx', 'yy']
        reds = om.get_reds(antpos, pols=pols)
        np.random.seed(21)
        freqs = np.linspace(100e6, 200e6, 64, endpoint=False)
        times = np.linspace(0, 600. / 60 / 60 / 24, 3, endpoint=False)
        df = np.median(np.diff(freqs))
        dt = np.median(np.diff(times)) * 3600. * 24

        g, tv, d = sim_red_data(reds, shape=(len(times), len(freqs)), gain_scatter=.01)
        tv, d = DataContainer(tv), DataContainer(d)
        nsamples = DataContainer({bl: np.ones_like(d[bl], dtype=float) for bl in d})

        for antnum in antpos.keys():
            for pol in pols:
                d[(antnum, antnum, pol)] = np.ones((len(times), len(freqs)), dtype=complex)
        d.freqs = deepcopy(freqs)
        d.times_by_bl = {bl[0:2]: deepcopy(times) for bl in d.keys()}

        filtered_reds = om.filter_reds(reds, ex_ants=ex_ants, antpos=antpos, max_bl_cut=30)
        cal = om.redundantly_calibrate(d, filtered_reds)
        om.expand_omni_sol(cal, reds, d, nsamples)

        # test that all chisqs are 0
        for red in reds:
            for bl in red:
                ant0, ant1 = split_bl(bl)
                np.testing.assert_array_almost_equal(d[bl], cal['g_omnical'][ant0] * np.conj(cal['g_omnical'][ant1]) * cal['v_omnical'][red[0]])
        assert len(pols) * len(antpos) == len(cal['g_omnical'])
        for ant in cal['chisq_per_ant']:
            np.testing.assert_array_less(cal['chisq_per_ant'][ant], 1e-10)

    def test_redcal_iteration(self):
        hd = io.HERAData(os.path.join(DATA_PATH, 'zen.2458098.43124.downsample.uvh5'))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rv = om.redcal_iteration(hd, nInt_to_load=1, flag_nchan_high=40, flag_nchan_low=30)
        for t in range(len(hd.times)):
            for flag in rv['gf_omnical'].values():
                assert not np.all(flag[t, :])
                assert np.all(flag[t, 0:30])
                assert np.all(flag[t, -40:])

        hd = io.HERAData(os.path.join(DATA_PATH, 'zen.2458098.43124.downsample.uvh5'))  # test w/o partial loading
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rv = om.redcal_iteration(hd, flag_nchan_high=40, flag_nchan_low=30)
        for t in range(len(hd.times)):
            for flag in rv['gf_omnical'].values():
                assert not np.all(flag[t, :])
                assert np.all(flag[t, 0:30])
                assert np.all(flag[t, -40:])

        hd = io.HERAData(os.path.join(DATA_PATH, 'zen.2458098.43124.downsample.uvh5'))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rv = om.redcal_iteration(hd, pol_mode='4pol')
        np.testing.assert_array_equal(rv['chisq']['Jee'], rv['chisq']['Jnn'])

        hd.telescope_location_lat_lon_alt_degrees = (-30.7, 121.4, 1051.7)  # move array longitude
        rv = om.redcal_iteration(hd, solar_horizon=0.0)
        for flag in rv['gf_firstcal'].values():
            np.testing.assert_array_equal(flag, True)
        for flag in rv['gf_omnical'].values():
            np.testing.assert_array_equal(flag, True)
        for flag in rv['vf_omnical'].values():
            np.testing.assert_array_equal(flag, True)
        for nsamples in rv['vns_omnical'].values():
            np.testing.assert_array_equal(nsamples, 0)

        # this tests redcal.expand_omni_sol
        hd = io.HERAData(os.path.join(DATA_PATH, 'zen.2458098.43124.downsample.uvh5'))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rv = om.redcal_iteration(hd, pol_mode='2pol', ex_ants=[1, 27], min_bl_cut=15)
        for pol in ['ee', 'nn']:
            for key in ['v_omnical', 'vf_omnical', 'vns_omnical']:
                # test that the unique baseline is keyed by the first entry in all_reds, not filtered_reds
                assert (1, 12, pol) in rv[key].keys()
                # test that completely excluded baselines from redcal are still represented
                assert (23, 27, pol) in rv[key].keys()
            # test redundant baseline counting
            np.testing.assert_array_equal(rv['vns_omnical'][(1, 12, pol)][~rv['vf_omnical'][(1, 12, pol)]], 4.0)
            np.testing.assert_array_equal(rv['vns_omnical'][(23, 27, pol)], 0.0)
            np.testing.assert_array_equal(rv['vns_omnical'][(1, 27, pol)], 0.0)
        for ant in [(1, 'Jee'), (1, 'Jnn'), (27, 'Jee'), (27, 'Jnn')]:
            assert not np.all(rv['g_omnical'][ant] == 1.0)
            assert not np.all(rv['chisq_per_ant'][ant] == 0.0)
            np.testing.assert_array_equal(rv['gf_omnical'][ant], True)

    def test_redcal_run(self):
        input_data = os.path.join(DATA_PATH, 'zen.2458098.43124.downsample.uvh5')
        ant_metrics_file = os.path.join(DATA_PATH, 'test_input/zen.2458098.43124.HH.uv.ant_metrics.json')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sys.stdout = open(os.devnull, 'w')
            cal = om.redcal_run(input_data, verbose=True, ant_z_thresh=1.8, add_to_history='testing',
                                a_priori_ex_ants_yaml=os.path.join(DATA_PATH, 'test_input', 'a_priori_flags_sample.yaml'),
                                iter0_prefix='.iter0', metrics_files=ant_metrics_file, clobber=True)

            hd = io.HERAData(input_data)
            cal0 = om.redcal_iteration(hd, ex_ants=[11, 50])

            sys.stdout = sys.__stdout__

        for prefix, cal_here, bad_ants in [('', cal, [11, 50, 12]), ('.iter0', cal0, [11, 50])]:
            # bad_ants is based on experiments with this particular file
            hc = io.HERACal(os.path.splitext(input_data)[0] + prefix + '.first.calfits')
            gains, flags, quals, total_qual = hc.read()
            np.testing.assert_almost_equal(np.unique(hc.lst_array), np.unique(hd.lst_array))
            np.testing.assert_almost_equal(hc.telescope_location, hd.telescope_location)
            for antnum, antpos in zip(hc.antenna_numbers, hc.antenna_positions):
                np.testing.assert_almost_equal(antpos, hd.antenna_positions[hd.antenna_numbers == antnum].flatten())
            for ant in gains.keys():
                np.testing.assert_almost_equal(gains[ant], cal_here['g_firstcal'][ant])
                np.testing.assert_almost_equal(flags[ant], cal_here['gf_firstcal'][ant])
                if ant[0] in bad_ants:
                    np.testing.assert_array_equal(gains[ant], 1.0)
                    np.testing.assert_array_equal(flags[ant], True)
            assert 'testing' in hc.history.replace('\n', '').replace(' ', '')
            if prefix == '':
                assert 'Throwingoutantenna12' in hc.history.replace('\n', '').replace(' ', '')
            else:
                assert 'Iteration0Results.' in hc.history.replace('\n', '').replace(' ', '')
            assert 'Thisfilewasproducedbythefunction' in hc.history.replace('\n', '').replace(' ', '')

            hc = io.HERACal(os.path.splitext(input_data)[0] + prefix + '.omni.calfits')
            gains, flags, quals, total_qual = hc.read()
            np.testing.assert_almost_equal(np.unique(hc.lst_array), np.unique(hd.lst_array))
            np.testing.assert_almost_equal(hc.telescope_location, hd.telescope_location)
            for antnum, antpos in zip(hc.antenna_numbers, hc.antenna_positions):
                np.testing.assert_almost_equal(antpos, hd.antenna_positions[hd.antenna_numbers == antnum].flatten())
            for ant in gains.keys():
                zero_check = np.isclose(cal_here['g_omnical'][ant], 0, rtol=1e-10, atol=1e-10)
                np.testing.assert_array_almost_equal(gains[ant][~zero_check], cal_here['g_omnical'][ant][~zero_check])
                np.testing.assert_array_almost_equal(flags[ant][~zero_check], cal_here['gf_omnical'][ant][~zero_check])
                if np.sum(zero_check) > 0:
                    np.testing.assert_array_equal(flags[ant][zero_check], True)
                np.testing.assert_array_almost_equal(quals[ant][~zero_check], cal_here['chisq_per_ant'][ant][~zero_check])
                if ant[0] in bad_ants:
                    np.testing.assert_array_equal(flags[ant], True)
            for antpol in total_qual.keys():
                np.testing.assert_array_almost_equal(total_qual[antpol], cal_here['chisq'][antpol])
            assert 'testing' in hc.history.replace('\n', '').replace(' ', '')
            if prefix == '':
                assert 'Throwingoutantenna12' in hc.history.replace('\n', '').replace(' ', '')
            else:
                assert 'Iteration0Results.' in hc.history.replace('\n', '').replace(' ', '')
            assert 'Thisfilewasproducedbythefunction' in hc.history.replace('\n', '').replace(' ', '')

            hd = io.HERAData(os.path.splitext(input_data)[0] + prefix + '.omni_vis.uvh5')
            data, flags, nsamples = hd.read()
            for bl in data.keys():
                np.testing.assert_array_almost_equal(data[bl], cal_here['v_omnical'][bl])
                np.testing.assert_array_almost_equal(flags[bl], cal_here['vf_omnical'][bl])
                np.testing.assert_array_almost_equal(nsamples[bl], cal_here['vns_omnical'][bl])
            assert 'testing' in hd.history.replace('\n', '').replace(' ', '')
            if prefix == '':
                assert 'Throwingoutantenna12' in hc.history.replace('\n', '').replace(' ', '')
            else:
                assert 'Iteration0Results.' in hc.history.replace('\n', '').replace(' ', '')
            assert 'Thisfilewasproducedbythefunction' in hd.history.replace('\n', '').replace(' ', '')

            meta_file = os.path.splitext(input_data)[0] + prefix + '.redcal_meta.hdf5'
            fc_meta, omni_meta, freqs, times, lsts, antpos, history = io.read_redcal_meta(meta_file)
            for key1 in fc_meta:
                for key2 in fc_meta[key1]:
                    np.testing.assert_array_almost_equal(fc_meta[key1][key2], cal_here['fc_meta'][key1][key2])
            for key1 in omni_meta:
                for key2 in omni_meta[key1]:
                    np.testing.assert_array_almost_equal(omni_meta[key1][key2], cal_here['omni_meta'][key1][key2])
            np.testing.assert_array_almost_equal(freqs, hd.freqs)
            np.testing.assert_array_almost_equal(times, hd.times)
            np.testing.assert_array_almost_equal(lsts, hd.lsts)
            for ant in antpos:
                np.testing.assert_array_almost_equal(antpos[ant], hd.antpos[ant])
            if prefix == '':
                assert 'Throwingoutantenna12' in history.replace('\n', '').replace(' ', '')
            else:
                assert 'Iteration0Results.' in history.replace('\n', '').replace(' ', '')
            assert 'Thisfilewasproducedbythefunction' in history.replace('\n', '').replace(' ', '')

        os.remove(os.path.splitext(input_data)[0] + '.first.calfits')
        os.remove(os.path.splitext(input_data)[0] + '.omni.calfits')
        os.remove(os.path.splitext(input_data)[0] + '.omni_vis.uvh5')
        os.remove(os.path.splitext(input_data)[0] + '.redcal_meta.hdf5')
        os.remove(os.path.splitext(input_data)[0] + '.iter0.first.calfits')
        os.remove(os.path.splitext(input_data)[0] + '.iter0.omni.calfits')
        os.remove(os.path.splitext(input_data)[0] + '.iter0.omni_vis.uvh5')
        os.remove(os.path.splitext(input_data)[0] + '.iter0.redcal_meta.hdf5')

        hd = io.HERAData(input_data)
        hd.read()
        hd.channel_width = np.median(np.diff(hd.freqs))
        hd.write_miriad(os.path.join(DATA_PATH, 'test_output/temp.uv'))
        hd = io.HERAData(os.path.join(DATA_PATH, 'test_output/temp.uv'), filetype='miriad')
        hd.read()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sys.stdout = open(os.devnull, 'w')
            cal = om.redcal_run(hd, metrics_files=ant_metrics_file, clobber=True)
            sys.stdout = sys.__stdout__
        assert len(cal) != 0
        shutil.rmtree(os.path.join(DATA_PATH, 'test_output/temp.uv'))
        os.remove(os.path.join(DATA_PATH, 'test_output/temp.first.calfits'))
        os.remove(os.path.join(DATA_PATH, 'test_output/temp.omni.calfits'))
        os.remove(os.path.join(DATA_PATH, 'test_output/temp.omni_vis.uvh5'))

        with pytest.raises(TypeError):
            cal = om.redcal_run({})

    def test_redcal_run_bda(self):
        uvh5_bda = os.path.join(DATA_PATH, "zen.2459122.30030.sum.bda.downsampled.uvh5")
        # test that gains have 8 times when we're upsampling
        cal = om.redcal_run(uvh5_bda, upsample=True, clobber=True)
        for gain in cal['g_omnical'].values():
            assert gain.shape[0] == 8
        # test that gains have 1 time when we're downsampling
        cal = om.redcal_run(uvh5_bda, downsample=True, clobber=True)
        for gain in cal['g_omnical'].values():
            assert gain.shape[0] == 1
        os.remove(os.path.join(DATA_PATH, 'zen.2459122.30030.sum.bda.downsampled.first.calfits'))
        os.remove(os.path.join(DATA_PATH, 'zen.2459122.30030.sum.bda.downsampled.omni.calfits'))
        os.remove(os.path.join(DATA_PATH, 'zen.2459122.30030.sum.bda.downsampled.omni_vis.uvh5'))
        os.remove(os.path.join(DATA_PATH, 'zen.2459122.30030.sum.bda.downsampled.redcal_meta.hdf5'))

    def test_redcal_argparser(self):
        sys.argv = [sys.argv[0], 'a', '--metrics_files', 'b', '--ex_ants', '5', '6', '--verbose']
        a = om.redcal_argparser()
        assert a.input_data == 'a'
        assert a.metrics_files == ['b']
        assert a.ex_ants == [5, 6]
        assert a.gain == 0.4
        assert a.verbose is True
