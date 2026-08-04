# -*- coding: utf-8 -*-
# Copyright 2026 the HERA Project
# Licensed under the MIT License

import pytest
import numpy as np
from scipy.optimize import least_squares
import linsolve
from hera_filters import dspec

from .. import skycal
from ..datacontainer import DataContainer
from ..apply_cal import calibrate_in_place
from .. import utils


def build_sim(nants=7, nfreqs=128, ntimes=2, pol='ee', seed=21, dlys=None,
              offsets=None, amp_ripple=0.1, phs_ripple=0.05, noise=0.0,
              snap_suppression=None, ant_to_SNAP_dict=None, dt=10.0, df=122e3):
    '''Build a synthetic dataset with known gains: V_ij = g_i g_j^* M_ij and
    autos_i = |g_i|^2 * A_sky. Returns a dict of everything needed for tests.
    If snap_suppression is given (dict mapping SNAP ID to loss fraction p),
    inter-SNAP cross visibilities are multiplied by (1 - p_i)(1 - p_j) while
    autos and intra-SNAP baselines are left untouched (mimicking correlator
    decoherence).'''
    rng = np.random.default_rng(seed)
    freqs = 100e6 + np.arange(nfreqs) * df
    ants = list(range(nants))
    if dlys is None:
        dlys = {ant: 0.0 for ant in ants}
    if offsets is None:
        offsets = {ant: 0.0 for ant in ants}

    # smooth true gains: few-mode ripples on top of delay/offset phases
    def smooth_ripple(scale):
        modes = rng.normal(size=3) * scale
        x = np.linspace(0, 1, nfreqs)
        return sum(m * np.cos((k + 1) * np.pi * x + rng.uniform(0, np.pi))
                   for k, m in enumerate(modes))

    true_gains = {}
    for ant in ants:
        amp = 1.0 + smooth_ripple(amp_ripple)
        phs = (2 * np.pi * freqs * dlys[ant] + offsets[ant]
               + smooth_ripple(phs_ripple))
        g = (amp * np.exp(1j * phs))[None, :] * np.ones((ntimes, 1))
        true_gains[(ant, utils.split_pol(pol)[0])] = g

    # random smooth-ish model visibilities and a common sky auto spectrum
    sky_auto = 200.0 * (1.0 + 0.3 * np.cos(np.linspace(0, 3, nfreqs)))
    data, model = {}, {}
    for i in ants:
        gi = true_gains[(i, utils.split_pol(pol)[0])]
        data[(i, i, pol)] = (np.abs(gi)**2 * sky_auto[None, :]
                             * np.ones((ntimes, 1))).astype(complex)
    for i in ants:
        gi = true_gains[(i, utils.split_pol(pol)[0])]
        for j in ants:
            if j <= i:
                continue
            gj = true_gains[(j, utils.split_pol(pol)[0])]
            amp = 10.0 * (0.5 + rng.uniform(size=nfreqs))
            phs = rng.uniform(0, 2 * np.pi) + np.linspace(
                0, rng.uniform(-3, 3), nfreqs)
            mvis = (amp * np.exp(1j * phs))[None, :] * np.ones((ntimes, 1))
            model[(i, j, pol)] = mvis
            vis = gi * np.conj(gj) * mvis
            if snap_suppression is not None:
                si, sj = ant_to_SNAP_dict[i], ant_to_SNAP_dict[j]
                if si != sj:
                    pi_, pj_ = snap_suppression.get(si, 0), \
                        snap_suppression.get(sj, 0)
                    vis = vis * (1 - pi_) * (1 - pj_)
            if noise > 0:
                sigma = np.sqrt(np.abs(data[(i, i, pol)] * data[(j, j, pol)])
                                / dt / df)
                vis = vis + noise * sigma * (rng.normal(size=vis.shape)
                                             + 1j * rng.normal(size=vis.shape)
                                             ) / np.sqrt(2)
            data[(i, j, pol)] = vis
    return {'data': DataContainer(data), 'model': DataContainer(model),
            'true_gains': true_gains, 'freqs': freqs, 'ants': ants,
            'pol': pol, 'dt': dt, 'df': df, 'ntimes': ntimes}


class TestSolvePerAntennaWeightedLeastSquares:
    def setup_method(self):
        self.rng = np.random.default_rng(21)

    def test_recovery_and_degeneracy_fixing(self):
        nants = 6
        x = self.rng.normal(size=nants)
        x -= x.mean()
        bls = [(i, j) for i in range(nants) for j in range(i + 1, nants)]
        ant_i = np.array([bl[0] for bl in bls])
        ant_j = np.array([bl[1] for bl in bls])
        vals = x[ant_i] - x[ant_j]
        wgts = self.rng.uniform(0.5, 2.0, size=len(bls))
        sol = skycal._solve_per_antenna_weighted_least_squares(vals, wgts, ant_i, ant_j, nants)
        np.testing.assert_allclose(sol, x, atol=1e-8)
        assert np.abs(np.mean(sol)) < 1e-10

    def test_unsolvable_antenna_is_nan(self):
        # antenna 3 appears in no baselines
        bls = [(0, 1), (0, 2), (1, 2)]
        ant_i = np.array([bl[0] for bl in bls])
        ant_j = np.array([bl[1] for bl in bls])
        vals = np.array([1.0, 2.0, 1.0])
        wgts = np.ones(3)
        sol = skycal._solve_per_antenna_weighted_least_squares(vals, wgts, ant_i, ant_j, 4)
        assert np.isnan(sol[3])
        assert np.all(np.isfinite(sol[:3]))


class TestBuildDataModelRatio:
    def setup_method(self):
        self.sim = build_sim(nants=5, nfreqs=32, ntimes=2, seed=1)

    def test_missing_autos_raise(self):
        # a clear error, rather than a KeyError deep inside noise.py
        sim = self.sim
        autos = DataContainer({bl: sim['data'][bl] for bl in sim['data']
                               if bl[0] == bl[1] and bl[0] != 0})
        with pytest.raises(ValueError, match='Autocorrelations'):
            skycal.build_data_model_ratio(sim['data'], sim['model'],
                                          autos=autos, dt=sim['dt'],
                                          df=sim['df'])

    def test_perfect_data_gives_unity_ratio(self):
        # unity gains: data equals model on crosses
        sim = build_sim(nants=5, nfreqs=32, ntimes=2, seed=2, amp_ripple=0,
                        phs_ripple=0)
        ratio, wgts = skycal.build_data_model_ratio(
            sim['data'], sim['model'], dt=sim['dt'], df=sim['df'])
        for bl in ratio:
            np.testing.assert_allclose(ratio[bl], 1.0, atol=1e-10)
            sigma2 = np.abs(sim['data'][(bl[0], bl[0], bl[2])]
                            * sim['data'][(bl[1], bl[1], bl[2])]
                            ) / sim['dt'] / sim['df']
            np.testing.assert_allclose(
                wgts[bl], np.abs(sim['model'][bl])**2 / sigma2, rtol=1e-10)

    def test_flag_propagation(self):
        sim = self.sim
        pol = sim['pol']
        bl = (0, 1, pol)
        data_flags = DataContainer({k: np.zeros_like(sim['data'][k], dtype=bool)
                                    for k in sim['data'] if k[0] != k[1]})
        data_flags[bl][0, 5] = True
        antpol = utils.split_pol(pol)[0]
        ant_flags = {(2, antpol): np.zeros((sim['ntimes'], 32), dtype=bool)}
        ant_flags[(2, antpol)][1, 7] = True
        ratio, wgts = skycal.build_data_model_ratio(
            sim['data'], sim['model'], data_flags=data_flags,
            ant_flags=ant_flags, dt=sim['dt'], df=sim['df'])
        assert np.isnan(ratio[bl][0, 5])
        assert wgts[bl][0, 5] == 0
        for other in ratio:
            if 2 in other[:2]:
                assert np.isnan(ratio[other][1, 7])
                assert wgts[other][1, 7] == 0


class TestModelBasedFirstcal:
    def test_signed_delay_and_offset_recovery(self):
        # includes a delay that wraps many times across the band
        nants = 6
        dlys_true = {0: 0.0, 1: 30e-9, 2: -55e-9, 3: 500e-9, 4: -100e-9,
                     5: 10e-9}
        mean_dly = np.mean(list(dlys_true.values()))
        dlys_true = {ant: d - mean_dly for ant, d in dlys_true.items()}
        offsets_true = {0: 0.1, 1: -0.4, 2: 0.3, 3: 0.0, 4: -0.2, 5: 0.2}
        mean_off = np.mean(list(offsets_true.values()))
        offsets_true = {ant: o - mean_off for ant, o in offsets_true.items()}
        sim = build_sim(nants=nants, nfreqs=256, ntimes=2, seed=3,
                        dlys=dlys_true, offsets=offsets_true,
                        amp_ripple=0, phs_ripple=0)
        ratio, wgts = skycal.build_data_model_ratio(
            sim['data'], sim['model'], dt=sim['dt'], df=sim['df'])
        dlys, offsets = skycal.model_based_firstcal(ratio, wgts, sim['freqs'])
        antpol = utils.split_pol(sim['pol'])[0]
        for ant in sim['ants']:
            # per-integration solves: (Ntimes, 1) arrays
            assert dlys[(ant, antpol)].shape == (sim['ntimes'], 1)
            # ~ns-level accuracy: Quinn interpolation is slightly biased by
            # the frequency structure of the weights, but this is already
            # better than a pad-8 FFT grid and far tighter than needed (the
            # per-channel refinement solves phases exactly and its phase-sync
            # initialization is wrap-immune). The signed comparison locks the
            # delay sign convention.
            assert np.max(np.abs(dlys[(ant, antpol)]
                                 - dlys_true[ant])) < 2e-9
            # delays and offsets covary through the absolute-frequency lever
            # arm (a delay error dtau shifts the fitted offset by ~2 pi f0
            # dtau), so only the total phase model over the band is
            # meaningful — and it's what firstcal_gains actually uses
            phase_err = (2 * np.pi * sim['freqs']
                         * (dlys[(ant, antpol)] - dlys_true[ant])
                         + offsets[(ant, antpol)] - offsets_true[ant])
            # bound: far inside wrap-safety (pi), small enough that the
            # per-channel refinement takes over from an excellent start
            assert np.max(np.abs(np.angle(np.exp(1j * phase_err)))) < 0.3

    def test_firstcal_gains_expression(self):
        freqs = 100e6 + np.arange(16) * 1e5
        dlys = {(0, 'Jee'): np.full((3, 1), 25e-9)}
        offsets = {(0, 'Jee'): np.full((3, 1), 0.3)}
        gains = skycal.firstcal_gains(dlys, offsets, freqs)
        expected = np.exp(2j * np.pi * 25e-9 * freqs + 1j * 0.3)
        assert gains[(0, 'Jee')].shape == (3, 16)
        np.testing.assert_allclose(gains[(0, 'Jee')][2], expected, rtol=1e-12)


class TestCalibrateAbsAmpFromAutos:
    def test_amplitude_recovery(self):
        sim = build_sim(nants=6, nfreqs=32, ntimes=2, seed=4, amp_ripple=0.2,
                        phs_ripple=0)
        gains = skycal.calibrate_abs_amp_from_autos(sim['data'])
        antpol = utils.split_pol(sim['pol'])[0]
        # recovered amps should equal true amps up to a per-channel rescaling
        # (the median reference) that is COMMON to all antennas
        ratios = np.asarray([np.abs(gains[(ant, antpol)])
                             / np.abs(sim['true_gains'][(ant, antpol)])
                             for ant in sim['ants']])
        np.testing.assert_allclose(
            ratios, np.broadcast_to(ratios[0:1], ratios.shape), rtol=1e-10)

    def test_flagged_cells_excluded_from_reference(self):
        sim = build_sim(nants=6, nfreqs=32, ntimes=2, seed=5)
        pol = sim['pol']
        auto_flags = {(0, 0, pol): np.zeros((sim['ntimes'], 32), dtype=bool)}
        auto_flags[(0, 0, pol)][0, 3] = True
        # with the cell flagged, corrupting it must not change anyone's gains
        gains_before = skycal.calibrate_abs_amp_from_autos(
            sim['data'], auto_flags=auto_flags)
        sim['data'][(0, 0, pol)][0, 3] *= 100
        gains_after = skycal.calibrate_abs_amp_from_autos(
            sim['data'], auto_flags=auto_flags)
        antpol = utils.split_pol(pol)[0]
        for ant in sim['ants'][1:]:
            np.testing.assert_allclose(gains_after[(ant, antpol)],
                                       gains_before[(ant, antpol)], rtol=1e-8)


class TestRefineGainsCore:
    def setup_method(self):
        self.rng = np.random.default_rng(21)

    def _setup_arrays(self, nants=5, nfreqs=8, phase_scale=0.3, amp_scale=0.2,
                      noise=0.0):
        true_h = ((1.0 + amp_scale * self.rng.uniform(-1, 1, (nants, nfreqs)))
                  * np.exp(1j * phase_scale
                           * self.rng.uniform(-1, 1, (nants, nfreqs))))
        # remove the truth's degeneracy: mean phase 0 per channel
        true_h *= np.exp(-1j * np.angle(true_h /
                                        np.abs(true_h)).mean(axis=0))[None, :]
        bls = [(i, j) for i in range(nants) for j in range(i + 1, nants)]
        ant_i = np.array([bl[0] for bl in bls])
        ant_j = np.array([bl[1] for bl in bls])
        vis_ratio = true_h[ant_i] * np.conj(true_h[ant_j])
        if noise > 0:
            vis_ratio = vis_ratio + noise * (
                self.rng.normal(size=vis_ratio.shape)
                + 1j * self.rng.normal(size=vis_ratio.shape)) / np.sqrt(2)
        wgts = self.rng.uniform(0.5, 2.0, size=vis_ratio.shape)
        return true_h, vis_ratio, wgts, ant_i, ant_j, nants

    def test_zero_on_unflagged_cell_raises(self):
        # bad data stored as 0 without flags must fail loudly: log-amplitude
        # and unit-phasor initialization are both undefined at zero
        true_h, vis_ratio, wgts, ant_i, ant_j, nants = self._setup_arrays()
        vis_ratio[3, 2] = 0.0
        with pytest.raises(ValueError, match='zero or non-finite'):
            skycal._refine_gains_single_pol_time(vis_ratio, wgts, ant_i,
                                                 ant_j, nants)

    def test_known_gain_recovery(self):
        true_h, vis_ratio, wgts, ant_i, ant_j, nants = self._setup_arrays()
        gains, meta = skycal._refine_gains_single_pol_time(
            vis_ratio, wgts, ant_i, ant_j, nants)
        np.testing.assert_allclose(gains, true_h, atol=1e-7)
        assert np.nanmax(meta['conv_crit']) < 1e-7

    def test_wrap_immunity(self):
        # phases uniform on (-pi, pi]: a linearized phase solve would fail
        true_h, vis_ratio, wgts, ant_i, ant_j, nants = self._setup_arrays(
            phase_scale=np.pi, amp_scale=0.1)
        gains, meta = skycal._refine_gains_single_pol_time(
            vis_ratio, wgts, ant_i, ant_j, nants)
        # compare baseline-level products (immune to the phase degeneracy)
        np.testing.assert_allclose(gains[ant_i] * np.conj(gains[ant_j]),
                                   vis_ratio, atol=1e-7)

    def test_brute_force_cross_check(self):
        true_h, vis_ratio, wgts, ant_i, ant_j, nants = self._setup_arrays(
            nants=5, nfreqs=3, noise=0.05)
        gains, meta = skycal._refine_gains_single_pol_time(
            vis_ratio, wgts, ant_i, ant_j, nants)

        # independent solve: scipy least_squares per channel with ant 0's
        # phase pinned, then degeneracy re-fixed to mean phase = 0
        for chan in range(3):
            z, w = vis_ratio[:, chan], wgts[:, chan]

            def resid(x):
                amps = x[:nants]
                phases = np.concatenate([[0], x[nants:]])
                h = amps * np.exp(1j * phases)
                r = np.sqrt(w) * (z - h[ant_i] * np.conj(h[ant_j]))
                return np.concatenate([r.real, r.imag])

            x0 = np.concatenate([np.ones(nants), np.zeros(nants - 1)])
            sol = least_squares(resid, x0, xtol=1e-15, ftol=1e-15, gtol=1e-15)
            h_bf = sol.x[:nants] * np.exp(
                1j * np.concatenate([[0], sol.x[nants:]]))
            h_bf = h_bf * np.exp(-1j * np.angle(h_bf / np.abs(h_bf)).mean())
            np.testing.assert_allclose(gains[:, chan], h_bf, atol=1e-6)

    def test_uniform_flag_structure_accepted(self):
        true_h, vis_ratio, wgts, ant_i, ant_j, nants = self._setup_arrays()
        # the two supported flag types: a whole antenna out (all its
        # baselines) and a channel flagged for ALL baselines
        wgts[(ant_i == 2) | (ant_j == 2), :] = 0
        wgts[:, 4] = 0
        gains, meta = skycal._refine_gains_single_pol_time(
            vis_ratio, wgts, ant_i, ant_j, nants)
        assert np.all(np.isnan(gains[2]))
        assert np.all(np.isnan(gains[:, 4]))
        solved_ants = np.ones(nants, dtype=bool)
        solved_ants[2] = False
        good = np.ones(gains.shape[1], dtype=bool)
        good[4] = False
        # remaining antennas/channels still recovered (the degeneracy is
        # fixed over the surviving antennas, so it differs from the truth's;
        # compare baseline-level products, which are degeneracy-invariant)
        for i in np.where(solved_ants)[0]:
            for j in np.where(solved_ants)[0]:
                np.testing.assert_allclose(
                    (gains[i] * np.conj(gains[j]))[good],
                    (true_h[i] * np.conj(true_h[j]))[good], atol=1e-6)

    def test_nonuniform_flags_raise(self):
        true_h, vis_ratio, wgts, ant_i, ant_j, nants = self._setup_arrays()
        # a channel flagged for only SOME baselines is not a supported
        # pattern: it must be fixed upstream, not silently repaired
        wgts[(ant_i == 2) | (ant_j == 2), 4] = 0
        with pytest.raises(ValueError):
            skycal._refine_gains_single_pol_time(
                vis_ratio, wgts, ant_i, ant_j, nants)

    def test_whole_baseline_flag_raises(self):
        true_h, vis_ratio, wgts, ant_i, ant_j, nants = self._setup_arrays()
        # a fully-flagged baseline between two otherwise-unflagged antennas
        # is also unsupported: baselines are excluded by omission, and only
        # whole ANTENNAS may be flagged
        wgts[0, :] = 0
        with pytest.raises(ValueError):
            skycal._refine_gains_single_pol_time(
                vis_ratio, wgts, ant_i, ant_j, nants)

    def test_convergence_failure_raises(self):
        true_h, vis_ratio, wgts, ant_i, ant_j, nants = self._setup_arrays(
            noise=0.3)
        with pytest.raises(RuntimeError):
            skycal._refine_gains_single_pol_time(
                vis_ratio, wgts, ant_i, ant_j, nants, refine_maxiter=0)

    def test_mean_phase_degeneracy_fixing(self):
        true_h, vis_ratio, wgts, ant_i, ant_j, nants = self._setup_arrays(
            noise=0.02)
        gains, meta = skycal._refine_gains_single_pol_time(
            vis_ratio, wgts, ant_i, ant_j, nants)
        # the solver zeroes the arithmetic mean of the accumulated phase
        # updates (same convention as redcal.remove_degen_gains)
        mean_phase = np.angle(gains).mean(axis=0)
        np.testing.assert_allclose(mean_phase, 0, atol=1e-7)

    def test_amp_updates_match_linsolve(self):
        # the scatter-built batched normal matrices are a pure speed
        # optimization (linsolve on the same equations is ~20x slower at
        # production scale); this cross-check keeps linsolve as the
        # independent referee for the custom machinery
        nsel, nchan = 8, 4
        bls = [(i, j) for i in range(nsel) for j in range(i + 1, nsel)]
        nbl = len(bls)
        wgts2d = self.rng.uniform(0.5, 2.0, (nbl, nchan))
        resids2d = self.rng.normal(size=(nbl, nchan))
        bl_inds, chan_inds = np.divmod(np.arange(nbl * nchan), nchan)
        ei = np.array([bls[k][0] for k in bl_inds])
        ej = np.array([bls[k][1] for k in bl_inds])
        w = wgts2d[bl_inds, chan_inds]
        r = resids2d[bl_inds, chan_inds]
        amp_mats, _ = skycal._build_normal_matrices(
            nchan, nsel, chan_inds, ei, ej, w)
        ours = skycal._solve_logamp_updates(
            amp_mats, chan_inds, ei, ej, w, r, nchan, nsel)
        ls_data = {f'e_{i} + e_{j}': resids2d[k]
                   for k, (i, j) in enumerate(bls)}
        ls_wgts = {f'e_{i} + e_{j}': wgts2d[k]
                   for k, (i, j) in enumerate(bls)}
        sol = linsolve.LinearSolver(ls_data, wgts=ls_wgts).solve()
        theirs = np.array([sol[f'e_{i}'] for i in range(nsel)]).T
        np.testing.assert_allclose(ours, theirs, atol=1e-8)

    def test_low_snr_amplitude_unbiased(self):
        # the converged solution is the stationary point of the complex
        # least-squares objective, NOT the log-space optimum, so it should
        # not show the low-SNR amplitude bias of logcal (Liu et al. 2010);
        # at this noise level a log-space estimator's bias benchmark is
        # ~sigma^2/2 = 0.045, which the bound below excludes
        nants, nfreqs, sigma = 6, 2, 0.3
        bls = [(i, j) for i in range(nants) for j in range(i + 1, nants)]
        ant_i = np.array([bl[0] for bl in bls])
        ant_j = np.array([bl[1] for bl in bls])
        wgts = np.full((len(bls), nfreqs), 1 / sigma**2)
        biases = []
        for trial in range(200):
            true_g = ((1.0 + 0.1 * self.rng.uniform(-1, 1, (nants, nfreqs)))
                      * np.exp(0.3j * self.rng.uniform(-1, 1,
                                                       (nants, nfreqs))))
            vis_ratio = (true_g[ant_i] * np.conj(true_g[ant_j])
                         + sigma * (self.rng.normal(size=wgts.shape)
                                    + 1j * self.rng.normal(size=wgts.shape))
                         / np.sqrt(2))
            gains, _ = skycal._refine_gains_single_pol_time(
                vis_ratio, wgts, ant_i, ant_j, nants, refine_maxiter=500)
            biases.append(np.mean(np.abs(gains) - np.abs(true_g)))
        assert np.abs(np.mean(biases)) < 0.015

    def test_wide_weight_dynamic_range(self):
        # stress test: weights spanning 3 decades, amplitudes 0.1-1.9x, and
        # fully wrapped phases (real inverse-variance weights span decades
        # across the band, and this regime once exposed a solver divergence)
        rng = np.random.default_rng(5)
        nants, nfreqs = 20, 16
        true_g = ((1.0 + 0.9 * rng.uniform(-1, 1, (nants, nfreqs)))
                  * np.exp(1j * np.pi
                           * rng.uniform(-1, 1, (nants, nfreqs))))
        bls = [(i, j) for i in range(nants) for j in range(i + 1, nants)]
        ant_i = np.array([bl[0] for bl in bls])
        ant_j = np.array([bl[1] for bl in bls])
        vis_ratio = true_g[ant_i] * np.conj(true_g[ant_j])
        vis_ratio = vis_ratio + 0.3 * (
            rng.normal(size=vis_ratio.shape)
            + 1j * rng.normal(size=vis_ratio.shape)) / np.sqrt(2)
        wgts = 10.0 ** rng.uniform(0, 3, size=vis_ratio.shape)
        gains, meta = skycal._refine_gains_single_pol_time(
            vis_ratio, wgts, ant_i, ant_j, nants)
        assert np.isfinite(gains).all()
        assert np.nanmax(meta['conv_crit']) < 1e-6


class TestRefineGains:
    def setup_method(self):
        self.ant_to_SNAP_dict = {0: 'A', 1: 'A', 2: 'A', 3: 'B', 4: 'B',
                                 5: 'B', 6: 'C'}

    def test_missing_snap_raises(self):
        sim = build_sim(nants=7, nfreqs=16, ntimes=1, seed=6)
        ratio, wgts = skycal.build_data_model_ratio(
            sim['data'], sim['model'], dt=sim['dt'], df=sim['df'])
        antpol = utils.split_pol(sim['pol'])[0]
        g0 = {(ant, antpol): np.ones((1, 16), dtype=complex)
              for ant in sim['ants']}
        incomplete = {ant: 'A' for ant in sim['ants'][:-1]}
        with pytest.raises(ValueError):
            skycal.refine_gains(ratio, wgts, g0,
                                ant_to_SNAP_dict=incomplete)

    def test_intra_snap_corruption_invariance(self):
        sim = build_sim(nants=7, nfreqs=16, ntimes=1, seed=7)
        antpol = utils.split_pol(sim['pol'])[0]
        g0 = {(ant, antpol): np.ones((1, 16), dtype=complex)
              for ant in sim['ants']}
        ratio, wgts = skycal.build_data_model_ratio(
            sim['data'], sim['model'], dt=sim['dt'], df=sim['df'])
        clean, _ = skycal.refine_gains(
            ratio, wgts, g0, ant_to_SNAP_dict=self.ant_to_SNAP_dict)
        # corrupt every intra-SNAP visibility: solution must not change
        for bl in sim['data']:
            if (bl[0] != bl[1] and self.ant_to_SNAP_dict[bl[0]]
                    == self.ant_to_SNAP_dict[bl[1]]):
                sim['data'][bl] *= 17.0
        ratio2, wgts2 = skycal.build_data_model_ratio(
            sim['data'], sim['model'], dt=sim['dt'], df=sim['df'])
        corrupted, _ = skycal.refine_gains(
            ratio2, wgts2, g0, ant_to_SNAP_dict=self.ant_to_SNAP_dict)
        for key in clean:
            np.testing.assert_allclose(corrupted[key], clean[key], atol=1e-9)

    def test_amplitude_exemption_property(self):
        # the physics headline: per-SNAP suppression applied to inter-SNAP
        # crosses only (autos untouched) must land entirely in the refined
        # gains, not in the auto-derived amplitudes
        suppression = {'A': 0.05, 'B': 0.0, 'C': 0.02}
        sim = build_sim(nants=7, nfreqs=16, ntimes=1, seed=8, amp_ripple=0,
                        phs_ripple=0, snap_suppression=suppression,
                        ant_to_SNAP_dict=self.ant_to_SNAP_dict)
        antpol = utils.split_pol(sim['pol'])[0]
        amp_gains = skycal.calibrate_abs_amp_from_autos(sim['data'])
        for ant in sim['ants']:
            np.testing.assert_allclose(np.abs(amp_gains[(ant, antpol)]), 1.0,
                                       atol=1e-10)
        ratio, wgts = skycal.build_data_model_ratio(
            sim['data'], sim['model'], dt=sim['dt'], df=sim['df'])
        refined, _ = skycal.refine_gains(
            ratio, wgts, amp_gains, ant_to_SNAP_dict=self.ant_to_SNAP_dict)
        for ant in sim['ants']:
            expected = 1 - suppression[self.ant_to_SNAP_dict[ant]]
            np.testing.assert_allclose(np.abs(refined[(ant, antpol)]),
                                       expected, atol=1e-8)


class TestSkyCalibrate:
    def test_unrefinable_antennas_raise(self):
        # all antennas on one SNAP -> every baseline is intra-SNAP -> nothing
        # to refine; must error rather than silently drop antennas at the
        # final merge (whole antennas are only ever excluded by omission)
        sim = build_sim(nants=5, nfreqs=32, ntimes=1, seed=11)
        with pytest.raises(ValueError, match='refinement solve'):
            skycal.sky_calibrate(
                sim['data'], sim['model'], freqs=sim['freqs'],
                dt=sim['dt'], df=sim['df'],
                ant_to_SNAP_dict={ant: 'A' for ant in sim['ants']})

    def test_end_to_end_recovery(self):
        dlys_true = {ant: d for ant, d in enumerate(
            [0.0, 20e-9, -30e-9, 45e-9, -35e-9, 10e-9, -10e-9])}
        sim = build_sim(nants=7, nfreqs=128, ntimes=2, seed=9, dlys=dlys_true,
                        amp_ripple=0.1, phs_ripple=0.05, noise=0.1)
        gains, meta = skycal.sky_calibrate(
            sim['data'], sim['model'], freqs=sim['freqs'], dt=sim['dt'],
            df=sim['df'])
        # calibrating the data with the returned gains should recover the
        # model on all cross baselines (baseline products are invariant to
        # how the per-channel mean-phase degeneracy is fixed)
        data = DataContainer({bl: np.array(sim['data'][bl])
                              for bl in sim['data'] if bl[0] != bl[1]})
        calibrate_in_place(data, gains)
        for bl in data:
            np.testing.assert_allclose(data[bl], sim['model'][bl],
                                       rtol=2e-2, atol=0)
        # amplitudes are recovered absolutely; phases up to the degeneracy
        antpol = utils.split_pol(sim['pol'])[0]
        for ant in sim['ants']:
            np.testing.assert_allclose(
                np.abs(gains[(ant, antpol)]),
                np.abs(sim['true_gains'][(ant, antpol)]), rtol=2e-2)

    def test_meta_completeness(self):
        sim = build_sim(nants=5, nfreqs=32, ntimes=1, seed=10)
        gains, meta = skycal.sky_calibrate(
            sim['data'], sim['model'], freqs=sim['freqs'], dt=sim['dt'],
            df=sim['df'])
        for key in ['data_model_ratio', 'wgts', 'dlys', 'offsets',
                    'abs_amp_gains', 'g0', 'refined_gains', 'iter',
                    'conv_crit']:
            assert key in meta
        assert (0, sim['pol']) in meta['iter']
        assert np.nanmax(meta['conv_crit'][(0, sim['pol'])]) < 1e-6


def build_decoherence_sim(ptil_by_snap, nants_per_snap=3, nfreqs=256,
                          nchans_per_block=32, ntimes=1, noise_ln=0.005,
                          seed=11):
    '''Build synthetic (gains, logamp_wgts, ant_to_SNAP_dict, freqs) for
    decoherence-estimator tests. ptil_by_snap maps SNAP ID to an (Nblocks,)
    array of true log-suppressions -ln(1-p); the corresponding staircase is
    imprinted on |gains| of every antenna on that SNAP, on top of a smooth
    per-antenna bandpass, with per-channel ln|g| scatter noise_ln matching
    the (flat) reported inverse variances. Frequencies are chosen so the
    default-test band split (107.9 MHz, between chans 95 and 96) falls on a
    block boundary.'''
    rng = np.random.default_rng(seed)
    freqs = 60e6 + np.arange(nfreqs) * 0.5e6
    snaps = sorted(ptil_by_snap)
    ants, ant_to_SNAP_dict = [], {}
    for si, snap in enumerate(snaps):
        for k in range(nants_per_snap):
            ant = 10 * si + k
            ants.append(ant)
            ant_to_SNAP_dict[ant] = snap
    chan_to_block = np.arange(nfreqs) // nchans_per_block
    x = np.linspace(0, 1, nfreqs)

    gains, logamp_wgts = {}, {}
    for ant in ants:
        smooth_amp = 1.0 + 0.05 * np.cos(2 * np.pi * x + rng.uniform(0, 6))
        stair = ptil_by_snap[ant_to_SNAP_dict[ant]][chan_to_block]
        log_amp = (np.log(smooth_amp)[None, :] - stair
                   + noise_ln * rng.normal(size=(ntimes, nfreqs)))
        phase = 2 * np.pi * rng.uniform(size=1)
        gains[(ant, 'Jee')] = np.exp(log_amp + 1j * phase)
        logamp_wgts[(ant, 'Jee')] = np.full((ntimes, nfreqs),
                                            1 / noise_ln**2)
    return dict(gains=gains, logamp_wgts=logamp_wgts,
                ant_to_SNAP_dict=ant_to_SNAP_dict, freqs=freqs,
                nchans_per_block=nchans_per_block,
                nblocks=nfreqs // nchans_per_block)


# split between channels 95 (107.5 MHz) and 96 (108 MHz), on the 32-channel
# test block boundary (get_minimal_slices assigns a channel exactly at the
# cut to neither band, so the split is placed between channels)
DECO_KWARGS = dict(band_split_freq=107.9e6)


class TestDecoherenceHelpers:
    def test_block_design_matrix(self):
        chan_to_block, design = skycal._block_design_matrix(96, 32)
        assert design.shape == (96, 3)
        np.testing.assert_array_equal(chan_to_block, np.arange(96) // 32)
        np.testing.assert_allclose(design.sum(axis=1), 1.0)
        assert design[0, 0] == 1 and design[95, 2] == 1

    def test_mcp_firm_threshold_zones(self):
        # diagonal system: the unconstrained update equals -rhs, so targets
        # map straight through the firm threshold
        targets = np.array([0.5, 1.5, 3.0, -1.0])
        normal_mat = np.eye(4)
        rhs = -targets
        zero_below = np.ones(4)
        unbiased_above = np.full(4, 2.0)
        fit = skycal._mcp_penalized_nnls(normal_mat, rhs, zero_below,
                                         unbiased_above)
        # below the corner -> exactly 0; in the shrinkage zone ->
        # (pu - T1) * T2 / (T2 - T1); beyond -> unbiased; negative -> 0
        np.testing.assert_allclose(fit, [0.0, 1.0, 3.0, 0.0], atol=1e-9)
        # with no thresholds this is plain NNLS
        fit = skycal._mcp_penalized_nnls(normal_mat, rhs, np.zeros(4),
                                         np.full(4, np.inf))
        np.testing.assert_allclose(fit, np.maximum(targets, 0), atol=1e-9)

    def test_project_out_smooth_removes_smooth_keeps_steps(self):
        freqs = 60e6 + np.arange(256) * 0.5e6
        band_slices = [slice(0, 96), slice(96, 256)]   # split at 108 MHz
        bases = skycal._dpss_bases(freqs, band_slices, 100e-9, 1e-12)
        wgts = np.ones(256)
        x = np.linspace(0, 1, 256)
        smooth = 0.3 * np.cos(2 * np.pi * x) + 0.1 * x
        resid = skycal._project_out_smooth(smooth, wgts, band_slices, bases)
        assert np.abs(resid).max() < 1e-6
        step = np.where(np.arange(256) // 32 == 5, 0.1, 0.0)
        resid = skycal._project_out_smooth(step, wgts, band_slices, bases)
        assert np.abs(resid).max() > 0.01

    def test_project_out_smooth_matches_fourier_filter(self):
        # referee: on unflagged channels the bespoke projection must match
        # hera_filters' canonical DPSS least-squares filter. Flagged
        # channels may differ (both extrapolate the smooth model into gaps
        # with different conditioning choices), but the estimator multiplies
        # those channels by weight zero everywhere, so they never matter.
        rng = np.random.default_rng(3)
        freqs = 60e6 + np.arange(256) * 0.5e6
        band_slices = [slice(0, 96), slice(96, 256)]
        bases = skycal._dpss_bases(freqs, band_slices, 100e-9, 1e-12)
        vals = (0.2 * np.cos(2 * np.pi * np.linspace(0, 3, 256))
                + rng.normal(size=256) * 0.05)
        wgts = rng.uniform(0.5, 2.0, size=256)
        wgts[40:60] = 0   # a flag gap inside the low band
        mine = skycal._project_out_smooth(vals, wgts, band_slices, bases)
        for band in band_slices:
            _, resid, _ = dspec.fourier_filter(
                freqs[band], vals[band].astype(complex), wgts[band],
                filter_centers=[0.0], filter_half_widths=[100e-9],
                mode='dpss_leastsq', eigenval_cutoff=[1e-12])
            unflagged = wgts[band] > 0
            np.testing.assert_allclose(mine[band][unflagged],
                                       resid.real[unflagged], atol=1e-8)

    def test_log_gain_inverse_variance_ensemble_calibration(self):
        # statistical referee for the formula itself: the scatter of
        # ln|refined| across noise realizations of the ACTUAL solver must
        # match 1/sqrt(inv_var). The diagonal-Fisher approximation biases
        # the prediction slightly low (measured ~13% in sigma at 10
        # antennas, worse where few partners dominate; the HAC noise model
        # downstream absorbs that deficit as measured excess), so the
        # bounds allow it — but a missing factor of 2 (mean ratio ~1.6) or
        # a missing |g0|^2 propagation (per-cell spread ~16x) would fail.
        rng = np.random.default_rng(42)
        nants, nfreqs, nreal = 10, 4, 150
        ants = list(range(nants))
        bls = [(i, j, 'ee') for i in ants for j in ants if j > i]
        ant_i = np.array([bl[0] for bl in bls])
        ant_j = np.array([bl[1] for bl in bls])
        g0_arr = (rng.uniform(0.5, 2.0, (nants, nfreqs))
                  * np.exp(2j * np.pi * rng.uniform(size=(nants, nfreqs))))
        wgts_arr = 10.0 ** rng.uniform(2, 4, size=(len(bls), nfreqs))
        g0_ij = g0_arr[ant_i] * np.conj(g0_arr[ant_j])
        ratio_wgts = wgts_arr * np.abs(g0_ij)**2
        log_refined = np.zeros((nreal, nants, nfreqs))
        for r in range(nreal):
            noise = (rng.normal(size=(len(bls), nfreqs))
                     + 1j * rng.normal(size=(len(bls), nfreqs))
                     ) / np.sqrt(2)
            vis_ratio = 1.0 + noise / np.sqrt(wgts_arr) / g0_ij
            gains, _ = skycal._refine_gains_single_pol_time(
                vis_ratio, ratio_wgts, ant_i, ant_j, nants)
            log_refined[r] = np.log(np.abs(gains))
        measured_sigma = log_refined.std(axis=0)
        wgts_dc = DataContainer({bl: wgts_arr[k][None, :]
                                 for k, bl in enumerate(bls)})
        g0_dict = {(a, 'Jee'): g0_arr[a][None, :] for a in ants}
        ones = {(a, 'Jee'): np.ones((1, nfreqs), dtype=complex)
                for a in ants}
        inv_var = skycal.log_gain_inverse_variance(
            wgts_dc, g0_dict, ones, {a: f'S{a}' for a in ants})
        predicted_sigma = np.array([1 / np.sqrt(inv_var[(a, 'Jee')][0])
                                    for a in ants])
        ratio = measured_sigma / predicted_sigma
        assert 1.0 < ratio.mean() < 1.35
        assert ratio.min() > 0.8 and ratio.max() < 1.7

    def test_log_gain_inverse_variance(self):
        # 3 antennas, ants 0 and 2 share SNAP A: the intra-SNAP baseline
        # (0, 2) must NOT contribute; each inter-SNAP entry contributes
        # 2 |H_a|^2 * w * |g0_a g0_b|^2 * |H_b|^2
        shape = (1, 4)
        g0 = {(0, 'Jee'): np.full(shape, 2.0 * np.exp(0.3j)),
              (1, 'Jee'): np.full(shape, 1.0 + 0j),
              (2, 'Jee'): np.full(shape, 0.5 * np.exp(-1.1j))}
        refined = {(0, 'Jee'): np.full(shape, 0.9 + 0j),
                   (1, 'Jee'): np.full(shape, 1.1 + 0j),
                   (2, 'Jee'): np.full(shape, 1.0 + 0j)}
        wgts = DataContainer({(0, 1, 'ee'): np.full(shape, 3.0),
                              (0, 2, 'ee'): np.full(shape, 7.0),
                              (1, 2, 'ee'): np.full(shape, 5.0)})
        ant_to_SNAP_dict = {0: 'A', 1: 'B', 2: 'A'}
        inv_var = skycal.log_gain_inverse_variance(wgts, g0, refined,
                                                   ant_to_SNAP_dict)
        w01 = 3.0 * np.abs(2.0 * 1.0)**2
        w12 = 5.0 * np.abs(1.0 * 0.5)**2
        np.testing.assert_allclose(inv_var[(0, 'Jee')],
                                   2 * 0.9**2 * w01 * 1.1**2)
        np.testing.assert_allclose(inv_var[(1, 'Jee')],
                                   2 * 1.1**2 * (w01 * 0.9**2 + w12 * 1.0))
        np.testing.assert_allclose(inv_var[(2, 'Jee')],
                                   2 * 1.0 * w12 * 1.1**2)


class TestEstimateSNAPDecoherence:
    def test_staircase_recovery(self):
        nblocks = 8
        ptil_a = np.zeros(nblocks)
        ptil_a[1], ptil_a[5] = 0.06, 0.12
        sim = build_decoherence_sim({'A': ptil_a, 'B': np.zeros(nblocks)})
        deco, meta = skycal.estimate_SNAP_decoherence(
            sim['gains'], sim['logamp_wgts'],
            sim['ant_to_SNAP_dict'], sim['freqs'],
            nchans_per_block=sim['nchans_per_block'], **DECO_KWARGS)
        log_supp = meta['log_suppression']
        # suppressed blocks recovered absolutely (each band retains an
        # unsuppressed floor block), clean blocks exactly zero
        assert abs(log_supp['A'][0, 1] - 0.06) < 0.01
        assert abs(log_supp['A'][0, 5] - 0.12) < 0.01
        clean = [b for b in range(nblocks) if b not in (1, 5)]
        np.testing.assert_allclose(log_supp['A'][0, clean], 0, atol=1e-10)
        np.testing.assert_allclose(log_supp['B'][0], 0, atol=1e-10)
        # decoherence is 1 - exp(-log_suppression)
        np.testing.assert_allclose(deco['A'][0],
                                   1 - np.exp(-log_supp['A'][0]), atol=1e-12)
        # sigma is reported on active blocks and plausibly sized
        assert np.isfinite(meta['log_suppression_sigma']['A'][0, 1])
        assert 0 < meta['log_suppression_sigma']['A'][0, 1] < 0.05
        # refit agrees with the fit well above threshold
        assert abs(meta['log_suppression_refit']['A'][0, 5]
                   - log_supp['A'][0, 5]) < 0.01

    def test_floor_degeneracy(self):
        # a suppression common to ALL covered blocks of a band is
        # degenerate with smooth structure: the estimator must report the
        # floor-relative value, i.e. ~0
        nblocks = 8
        ptil = np.zeros(nblocks)
        ptil[3:] = 0.05   # every block of the high band (blocks 3-7)
        sim = build_decoherence_sim({'A': ptil, 'B': np.zeros(nblocks)},
                                    seed=12)
        _, meta = skycal.estimate_SNAP_decoherence(
            sim['gains'], sim['logamp_wgts'],
            sim['ant_to_SNAP_dict'], sim['freqs'],
            nchans_per_block=sim['nchans_per_block'], **DECO_KWARGS)
        assert np.nanmax(meta['log_suppression']['A'][0, 3:]) < 0.01

    def test_flagged_band_edges_and_fully_flagged_band(self):
        # exterior flagged channels must be trimmed from the band slices
        # (get_minimal_slices) and a fully-flagged band skipped entirely
        nblocks = 8
        ptil = np.zeros(nblocks)
        ptil[5] = 0.1
        sim = build_decoherence_sim({'A': ptil, 'B': np.zeros(nblocks)},
                                    seed=13)
        for key in sim['logamp_wgts']:
            sim['logamp_wgts'][key][:, :96] = 0     # whole low band
            sim['logamp_wgts'][key][:, 96:100] = 0  # high-band leading edge
            sim['logamp_wgts'][key][:, -5:] = 0     # high-band trailing edge
        _, meta = skycal.estimate_SNAP_decoherence(
            sim['gains'], sim['logamp_wgts'],
            sim['ant_to_SNAP_dict'], sim['freqs'],
            nchans_per_block=sim['nchans_per_block'], **DECO_KWARGS)
        # low-band blocks have no data at all -> NaN, not zero
        assert np.all(np.isnan(meta['log_suppression']['A'][0, :3]))
        # the suppressed high-band block is still recovered
        assert abs(meta['log_suppression']['A'][0, 5] - 0.1) < 0.01

    def test_missing_snap_or_wgts_raises(self):
        sim = build_decoherence_sim({'A': np.zeros(8), 'B': np.zeros(8)})
        incomplete = dict(sim['ant_to_SNAP_dict'])
        del incomplete[0]
        with pytest.raises(ValueError, match='missing antennas'):
            skycal.estimate_SNAP_decoherence(
                sim['gains'], sim['logamp_wgts'], incomplete,
                sim['freqs'], nchans_per_block=sim['nchans_per_block'],
                **DECO_KWARGS)
        partial_wgts = dict(sim['logamp_wgts'])
        del partial_wgts[(0, 'Jee')]
        with pytest.raises(ValueError, match='missing keys'):
            skycal.estimate_SNAP_decoherence(
                sim['gains'], partial_wgts, sim['ant_to_SNAP_dict'],
                sim['freqs'], nchans_per_block=sim['nchans_per_block'],
                **DECO_KWARGS)

    def test_meta_completeness(self):
        sim = build_decoherence_sim({'A': np.zeros(8), 'B': np.zeros(8)})
        deco, meta = skycal.estimate_SNAP_decoherence(
            sim['gains'], sim['logamp_wgts'],
            sim['ant_to_SNAP_dict'], sim['freqs'],
            nchans_per_block=sim['nchans_per_block'], **DECO_KWARGS)
        for key in ['log_suppression', 'log_suppression_refit',
                    'log_suppression_sigma',
                    'fgls_iterations', 'n_spectra_per_snap',
                    'sigma_over_thermal', 'covered_blocks', 'edge_blocks',
                    'chan_to_block']:
            assert key in meta
        for snap in ['A', 'B']:
            assert meta['log_suppression'][snap].shape == (1, sim['nblocks'])
            assert deco[snap].shape == (1, sim['nblocks'])
            assert meta['fgls_iterations'][snap].shape == (1,)
            assert meta['n_spectra_per_snap'][snap][0] == 3
            assert meta['sigma_over_thermal'][snap].shape == (1, 2)
        assert meta['covered_blocks'].all()
        np.testing.assert_array_equal(meta['chan_to_block'],
                                      np.arange(256) // 32)
