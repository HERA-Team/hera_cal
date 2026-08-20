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
              SNAP_suppression=None, ant_to_SNAP_dict=None, dt=10.0, df=122e3):
    '''Build a synthetic dataset with known gains: V_ij = g_i g_j^* M_ij and
    autos_i = |g_i|^2 * A_sky. Returns a dict of everything needed for tests.
    If SNAP_suppression is given (dict mapping SNAP ID to loss fraction p),
    inter-SNAP cross visibilities are multiplied by (1 - p_i)(1 - p_j) while
    autos and intra-SNAP baselines are left untouched (mimicking correlator
    decoherence).'''
    rng = np.random.default_rng(seed)
    freqs = 100e6 + np.arange(nfreqs) * df
    antnums = list(range(nants))
    if dlys is None:
        dlys = {antnum: 0.0 for antnum in antnums}
    if offsets is None:
        offsets = {antnum: 0.0 for antnum in antnums}

    # smooth true gains: few-mode ripples on top of delay/offset phases
    def smooth_ripple(scale):
        modes = rng.normal(size=3) * scale
        x = np.linspace(0, 1, nfreqs)
        return sum(m * np.cos((k + 1) * np.pi * x + rng.uniform(0, np.pi))
                   for k, m in enumerate(modes))

    true_gains = {}
    for antnum in antnums:
        amp = 1.0 + smooth_ripple(amp_ripple)
        phs = (2 * np.pi * freqs * dlys[antnum] + offsets[antnum]
               + smooth_ripple(phs_ripple))
        g = (amp * np.exp(1j * phs))[None, :] * np.ones((ntimes, 1))
        true_gains[(antnum, utils.split_pol(pol)[0])] = g

    # random smooth-ish model visibilities and a common sky auto spectrum
    sky_auto = 200.0 * (1.0 + 0.3 * np.cos(np.linspace(0, 3, nfreqs)))
    data, model = {}, {}
    for i in antnums:
        gi = true_gains[(i, utils.split_pol(pol)[0])]
        data[(i, i, pol)] = (np.abs(gi)**2 * sky_auto[None, :]
                             * np.ones((ntimes, 1))).astype(complex)
    for i in antnums:
        gi = true_gains[(i, utils.split_pol(pol)[0])]
        for j in antnums:
            if j <= i:
                continue
            gj = true_gains[(j, utils.split_pol(pol)[0])]
            amp = 10.0 * (0.5 + rng.uniform(size=nfreqs))
            phs = rng.uniform(0, 2 * np.pi) + np.linspace(
                0, rng.uniform(-3, 3), nfreqs)
            mvis = (amp * np.exp(1j * phs))[None, :] * np.ones((ntimes, 1))
            model[(i, j, pol)] = mvis
            vis = gi * np.conj(gj) * mvis
            if SNAP_suppression is not None:
                si, sj = ant_to_SNAP_dict[i], ant_to_SNAP_dict[j]
                if si != sj:
                    pi_, pj_ = SNAP_suppression.get(si, 0), \
                        SNAP_suppression.get(sj, 0)
                    vis = vis * (1 - pi_) * (1 - pj_)
            if noise > 0:
                sigma = np.sqrt(np.abs(data[(i, i, pol)] * data[(j, j, pol)])
                                / dt / df)
                vis = vis + noise * sigma * (rng.normal(size=vis.shape)
                                             + 1j * rng.normal(size=vis.shape)
                                             ) / np.sqrt(2)
            data[(i, j, pol)] = vis
    return {'data': DataContainer(data), 'model': DataContainer(model),
            'true_gains': true_gains, 'freqs': freqs, 'antnums': antnums,
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

    def test_solve_mode_kwarg(self):
        # the linsolve mode is plumbed through, and alternate modes solve
        # the same (singular) difference system: 'lsqr' and 'pinv' agree
        # once the mean-zero degeneracy fixing is applied
        nants = 6
        x = self.rng.normal(size=nants)
        x -= x.mean()
        bls = [(i, j) for i in range(nants) for j in range(i + 1, nants)]
        ant_i = np.array([bl[0] for bl in bls])
        ant_j = np.array([bl[1] for bl in bls])
        vals = x[ant_i] - x[ant_j]
        wgts = self.rng.uniform(0.5, 2.0, size=len(bls))
        sol_pinv = skycal._solve_per_antenna_weighted_least_squares(
            vals, wgts, ant_i, ant_j, nants, mode='pinv')
        sol_lsqr = skycal._solve_per_antenna_weighted_least_squares(
            vals, wgts, ant_i, ant_j, nants, mode='lsqr')
        np.testing.assert_allclose(sol_pinv, sol_lsqr, atol=1e-6)
        np.testing.assert_allclose(sol_lsqr, x, atol=1e-6)

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
        dlys_true = {antnum: d - mean_dly for antnum, d in dlys_true.items()}
        offsets_true = {0: 0.1, 1: -0.4, 2: 0.3, 3: 0.0, 4: -0.2, 5: 0.2}
        mean_off = np.mean(list(offsets_true.values()))
        offsets_true = {antnum: o - mean_off for antnum, o in offsets_true.items()}
        sim = build_sim(nants=nants, nfreqs=256, ntimes=2, seed=3,
                        dlys=dlys_true, offsets=offsets_true,
                        amp_ripple=0, phs_ripple=0)
        ratio, wgts = skycal.build_data_model_ratio(
            sim['data'], sim['model'], dt=sim['dt'], df=sim['df'])
        dlys, offsets = skycal.model_based_firstcal(ratio, wgts, sim['freqs'])
        antpol = utils.split_pol(sim['pol'])[0]
        for antnum in sim['antnums']:
            # per-integration solves: (Ntimes, 1) arrays
            assert dlys[(antnum, antpol)].shape == (sim['ntimes'], 1)
            # ~ns-level accuracy: Quinn interpolation is slightly biased by
            # the frequency structure of the weights, but this is already
            # better than a pad-8 FFT grid and far tighter than needed (the
            # per-channel refinement solves phases exactly and its phase-sync
            # initialization is wrap-immune). The signed comparison locks the
            # delay sign convention.
            assert np.max(np.abs(dlys[(antnum, antpol)]
                                 - dlys_true[antnum])) < 2e-9
            # delays and offsets covary through the absolute-frequency lever
            # arm (a delay error dtau shifts the fitted offset by ~2 pi f0
            # dtau), so only the total phase model over the band is
            # meaningful — and it's what firstcal_gains actually uses
            phase_err = (2 * np.pi * sim['freqs']
                         * (dlys[(antnum, antpol)] - dlys_true[antnum])
                         + offsets[(antnum, antpol)] - offsets_true[antnum])
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
        ratios = np.asarray([np.abs(gains[(antnum, antpol)])
                             / np.abs(sim['true_gains'][(antnum, antpol)])
                             for antnum in sim['antnums']])
        np.testing.assert_allclose(
            ratios, np.broadcast_to(ratios[0:1], ratios.shape), rtol=1e-10)

    def test_cross_pol_autos_ignored(self):
        # a full-Stokes DataContainer has (ant, ant, 'en') keys whose antpol
        # is also 'Jee': they must not contribute to the co-pol amplitudes
        sim = build_sim(nants=6, nfreqs=32, ntimes=2, seed=4, amp_ripple=0.2)
        gains = skycal.calibrate_abs_amp_from_autos(sim['data'])
        rng = np.random.default_rng(0)
        for antnum in sim['antnums']:
            for xpol in ['en', 'ne']:
                sim['data'][(antnum, antnum, xpol)] = (
                    1e-3 * rng.normal(size=(sim['ntimes'], 32)).astype(complex))
        gains_with_xpol = skycal.calibrate_abs_amp_from_autos(sim['data'])
        assert set(gains_with_xpol) == set(gains)
        for ant in gains:
            np.testing.assert_array_equal(gains_with_xpol[ant], gains[ant])

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
        for antnum in sim['antnums'][1:]:
            np.testing.assert_allclose(gains_after[(antnum, antpol)],
                                       gains_before[(antnum, antpol)], rtol=1e-8)


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

        # independent solve: scipy least_squares per channel with antnum 0's
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

    def test_missing_SNAP_raises(self):
        sim = build_sim(nants=7, nfreqs=16, ntimes=1, seed=6)
        ratio, wgts = skycal.build_data_model_ratio(
            sim['data'], sim['model'], dt=sim['dt'], df=sim['df'])
        antpol = utils.split_pol(sim['pol'])[0]
        g0 = {(antnum, antpol): np.ones((1, 16), dtype=complex)
              for antnum in sim['antnums']}
        incomplete = {antnum: 'A' for antnum in sim['antnums'][:-1]}
        with pytest.raises(ValueError):
            skycal.refine_gains(ratio, wgts, g0,
                                ant_to_SNAP_dict=incomplete)

    def test_intra_SNAP_corruption_invariance(self):
        sim = build_sim(nants=7, nfreqs=16, ntimes=1, seed=7)
        antpol = utils.split_pol(sim['pol'])[0]
        g0 = {(antnum, antpol): np.ones((1, 16), dtype=complex)
              for antnum in sim['antnums']}
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
                        phs_ripple=0, SNAP_suppression=suppression,
                        ant_to_SNAP_dict=self.ant_to_SNAP_dict)
        antpol = utils.split_pol(sim['pol'])[0]
        amp_gains = skycal.calibrate_abs_amp_from_autos(sim['data'])
        for antnum in sim['antnums']:
            np.testing.assert_allclose(np.abs(amp_gains[(antnum, antpol)]), 1.0,
                                       atol=1e-10)
        ratio, wgts = skycal.build_data_model_ratio(
            sim['data'], sim['model'], dt=sim['dt'], df=sim['df'])
        refined, _ = skycal.refine_gains(
            ratio, wgts, amp_gains, ant_to_SNAP_dict=self.ant_to_SNAP_dict)
        for antnum in sim['antnums']:
            expected = 1 - suppression[self.ant_to_SNAP_dict[antnum]]
            np.testing.assert_allclose(np.abs(refined[(antnum, antpol)]),
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
                ant_to_SNAP_dict={antnum: 'A' for antnum in sim['antnums']})

    def test_end_to_end_recovery(self):
        dlys_true = {antnum: d for antnum, d in enumerate(
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
        for antnum in sim['antnums']:
            np.testing.assert_allclose(
                np.abs(gains[(antnum, antpol)]),
                np.abs(sim['true_gains'][(antnum, antpol)]), rtol=2e-2)

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


def build_decoherence_sim(ptil_by_SNAP, nants_per_SNAP=3, nfreqs=256,
                          nchans_per_block=32, ntimes=1, noise_ln=0.005,
                          seed=11):
    '''Build synthetic (gains, logamp_wgts, ant_to_SNAP_dict, freqs) for
    decoherence-estimator tests. ptil_by_SNAP maps SNAP ID to an (Nblocks,)
    array of true log-suppressions -ln(1-p); the corresponding staircase is
    imprinted on |gains| of every antenna on that SNAP, on top of a smooth
    per-antenna bandpass, with per-channel ln|g| scatter noise_ln matching
    the (flat) reported inverse variances. Frequencies are chosen so the
    default-test band split (107.9 MHz, between chans 95 and 96) falls on a
    block boundary.'''
    rng = np.random.default_rng(seed)
    freqs = 60e6 + np.arange(nfreqs) * 0.5e6
    SNAPs = sorted(ptil_by_SNAP)
    antnums, ant_to_SNAP_dict = [], {}
    for si, SNAP in enumerate(SNAPs):
        for k in range(nants_per_SNAP):
            antnum = 10 * si + k
            antnums.append(antnum)
            ant_to_SNAP_dict[antnum] = SNAP
    chan_to_block = np.arange(nfreqs) // nchans_per_block
    x = np.linspace(0, 1, nfreqs)

    gains, logamp_wgts = {}, {}
    for antnum in antnums:
        smooth_amp = 1.0 + 0.05 * np.cos(2 * np.pi * x + rng.uniform(0, 6))
        stair = ptil_by_SNAP[ant_to_SNAP_dict[antnum]][chan_to_block]
        log_amp = (np.log(smooth_amp)[None, :] - stair
                   + noise_ln * rng.normal(size=(ntimes, nfreqs)))
        phase = 2 * np.pi * rng.uniform(size=1)
        gains[(antnum, 'Jee')] = np.exp(log_amp + 1j * phase)
        logamp_wgts[(antnum, 'Jee')] = np.full((ntimes, nfreqs),
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
        antnums = list(range(nants))
        bls = [(i, j, 'ee') for i in antnums for j in antnums if j > i]
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
        g0_dict = {(a, 'Jee'): g0_arr[a][None, :] for a in antnums}
        ones = {(a, 'Jee'): np.ones((1, nfreqs), dtype=complex)
                for a in antnums}
        inv_var = skycal.log_gain_inverse_variance(
            wgts_dc, g0_dict, ones, {a: f'S{a}' for a in antnums})
        predicted_sigma = np.array([1 / np.sqrt(inv_var[(a, 'Jee')][0])
                                    for a in antnums])
        ratio = measured_sigma / predicted_sigma
        assert 1.0 < ratio.mean() < 1.35
        assert ratio.min() > 0.8 and ratio.max() < 1.7

    def test_log_gain_inverse_variance(self):
        # 3 antennas, antnums 0 and 2 share SNAP A: the intra-SNAP baseline
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

    def test_missing_SNAP_or_wgts_raises(self):
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
                    'fgls_iterations', 'n_spectra_per_SNAP',
                    'sigma_over_thermal', 'covered_blocks', 'edge_blocks',
                    'chan_to_block', 'band_slices']:
            assert key in meta
        for SNAP in ['A', 'B']:
            assert meta['log_suppression'][SNAP].shape == (1, sim['nblocks'])
            assert deco[SNAP].shape == (1, sim['nblocks'])
            assert meta['fgls_iterations'][SNAP].shape == (1,)
            assert meta['n_spectra_per_SNAP'][SNAP][0] == 3
            assert meta['sigma_over_thermal'][SNAP].shape == (1, 2)
        assert meta['covered_blocks'].all()
        np.testing.assert_array_equal(meta['chan_to_block'],
                                      np.arange(256) // 32)
        # no flagging in this sim, so the two bands tile the full axis,
        # splitting at band_split_freq
        assert meta['band_slices'] == [slice(0, 96), slice(96, 256)]


class TestCoverageGaps:
    '''Targeted tests for previously-untested branches.'''

    def test_model_flags_propagation(self):
        sim = build_sim(nants=4, nfreqs=32, ntimes=2, seed=3)
        bl = (0, 1, sim['pol'])
        model_flags = DataContainer(
            {k: np.zeros((2, 32), dtype=bool) for k in sim['model']})
        model_flags[bl][1, 3] = True
        ratio, wgts = skycal.build_data_model_ratio(
            sim['data'], sim['model'], model_flags=model_flags,
            dt=sim['dt'], df=sim['df'])
        assert np.isnan(ratio[bl][1, 3])
        assert wgts[bl][1, 3] == 0

    def test_duplicate_baseline_equations_weighted_average(self):
        # repeated (i, j) measurements must be merged by weighted average
        ant_i = np.array([0, 0, 1])
        ant_j = np.array([1, 1, 2])
        vals = np.array([1.0, 1.2, 2.0])
        wgts = np.array([1.0, 3.0, 1.0])
        sol = skycal._solve_per_antenna_weighted_least_squares(
            vals, wgts, ant_i, ant_j, 3)
        merged = skycal._solve_per_antenna_weighted_least_squares(
            np.array([1.15, 2.0]), np.array([4.0, 1.0]),
            np.array([0, 1]), np.array([1, 2]), 3)
        np.testing.assert_allclose(sol, merged, atol=1e-10)

    def test_firstcal_unsolvable_antenna_gets_zero(self):
        sim = build_sim(nants=4, nfreqs=32, ntimes=2, seed=4,
                        dlys={0: 0.0, 1: 20e-9, 2: -20e-9, 3: 5e-9})
        ratio, wgts = skycal.build_data_model_ratio(
            sim['data'], sim['model'], dt=sim['dt'], df=sim['df'])
        for bl in wgts:
            if 0 in bl[:2]:
                wgts[bl] *= 0
        dlys, offsets = skycal.model_based_firstcal(ratio, wgts,
                                                    sim['freqs'])
        antpol = utils.split_pol(sim['pol'])[0]
        np.testing.assert_array_equal(dlys[(0, antpol)], 0.0)
        np.testing.assert_array_equal(offsets[(0, antpol)], 0.0)
        assert np.all(np.isfinite(dlys[(1, antpol)]))

    def test_all_baselines_flagged_shared_channel_flags(self):
        flagged_ants, chan_flags = skycal._shared_channel_flags(
            np.zeros((3, 4)), np.array([0, 0, 1]), np.array([1, 2, 2]), 3)
        assert flagged_ants.all()
        assert chan_flags.all()

    def test_refinement_divergence_raises(self):
        # pathological inputs (weights spanning 4 decades, gains 0.1-1.9x,
        # heavy noise) drive undamped Gauss-Newton to non-finite gains,
        # which must raise rather than return garbage
        rng = np.random.default_rng(0)
        nants, nfreqs = 20, 8
        tg = ((1.0 + 0.9 * rng.uniform(-1, 1, (nants, nfreqs)))
              * np.exp(1j * np.pi * rng.uniform(-1, 1, (nants, nfreqs))))
        bls = [(i, j) for i in range(nants) for j in range(i + 1, nants)]
        ant_i = np.array([b[0] for b in bls])
        ant_j = np.array([b[1] for b in bls])
        vis = tg[ant_i] * np.conj(tg[ant_j])
        vis = vis + 0.5 * (rng.normal(size=vis.shape)
                           + 1j * rng.normal(size=vis.shape)) / np.sqrt(2)
        wgts = 10.0 ** rng.uniform(0, 4, size=vis.shape)
        with pytest.raises(RuntimeError, match='diverged'):
            skycal._refine_gains_single_pol_time(vis, wgts, ant_i, ant_j,
                                                 nants)

    def test_sky_calibrate_infers_freqs(self):
        sim = build_sim(nants=5, nfreqs=32, ntimes=1, seed=5)
        sim['data'].freqs = sim['freqs']
        gains, meta = skycal.sky_calibrate(sim['data'], sim['model'],
                                           dt=sim['dt'], df=sim['df'])
        antpol = utils.split_pol(sim['pol'])[0]
        assert (0, antpol) in gains

    def test_mcp_nonconvergence_raises(self):
        normal_mat = np.array([[2.0, 1.0], [1.0, 2.0]])
        rhs = -np.array([1.0, 1.0])
        with pytest.raises(RuntimeError, match='did not converge'):
            skycal._mcp_penalized_nnls(normal_mat, rhs, np.zeros(2),
                                       np.full(2, np.inf), maxiter=1)

    def test_relative_staircase_and_multiblock_sigma(self):
        # an all-suppressed band: the fitter reports the pattern RELATIVE
        # to the least-suppressed block (which lands at exactly 0), and
        # the HAC covariance covers multiple active blocks in one band
        nblocks = 8
        ptil = np.zeros(nblocks)
        ptil[3:] = [0.30, 0.35, 0.40, 0.45, 0.50]
        sim = build_decoherence_sim({'A': ptil, 'B': np.zeros(nblocks)},
                                    seed=21, noise_ln=0.003)
        deco, meta = skycal.estimate_SNAP_decoherence(
            sim['gains'], sim['logamp_wgts'], sim['ant_to_SNAP_dict'],
            sim['freqs'], nchans_per_block=sim['nchans_per_block'],
            **DECO_KWARGS)
        log_supp = meta['log_suppression']['A'][0]
        np.testing.assert_allclose(log_supp[3:], ptil[3:] - 0.30, atol=0.02)
        assert np.isclose(log_supp[3:].min(), 0)
        # >= 2 active blocks in one band: covariance cross-terms exercised
        assert np.isfinite(meta['log_suppression_sigma']['A'][0, 4:]).all()

    def test_heterogeneous_SNAP_flags(self):
        # SNAP B: dead low band and one dead block; SNAP C: fully dead
        nblocks = 8
        ptil = np.zeros(nblocks)
        ptil[5] = 0.1
        sim = build_decoherence_sim(
            {'A': ptil, 'B': np.zeros(nblocks), 'C': np.zeros(nblocks)},
            seed=6)
        for key in sim['logamp_wgts']:
            antnum = key[0]
            SNAP = sim['ant_to_SNAP_dict'][antnum]
            if SNAP == 'B':
                sim['logamp_wgts'][key][:, :96] = 0     # low band dead
                sim['logamp_wgts'][key][:, 192:224] = 0  # block 6 dead
            elif SNAP == 'C':
                sim['logamp_wgts'][key][:] = 0
        deco, meta = skycal.estimate_SNAP_decoherence(
            sim['gains'], sim['logamp_wgts'], sim['ant_to_SNAP_dict'],
            sim['freqs'], nchans_per_block=sim['nchans_per_block'],
            **DECO_KWARGS)
        log_supp = meta['log_suppression']
        assert abs(log_supp['A'][0, 5] - 0.1) < 0.01
        assert np.all(np.isnan(log_supp['B'][0, :3]))   # dead band
        assert np.isnan(log_supp['B'][0, 6])            # dead block
        assert np.all(np.isnan(log_supp['C'][0]))       # no spectra
        assert meta['n_spectra_per_SNAP']['C'][0] == 0

    def test_SNAP_with_no_fittable_blocks(self):
        # B alive only in 20 chans of block 2, everyone else dead there:
        # block 2 falls below min_block_coverage -> B has spectra but no
        # fittable blocks (all NaN); A just loses block 2
        nblocks = 8
        sim = build_decoherence_sim({'A': np.zeros(nblocks),
                                     'B': np.zeros(nblocks)}, seed=8)
        for key in sorted(sim['logamp_wgts']):
            if sim['ant_to_SNAP_dict'][key[0]] == 'A':
                sim['logamp_wgts'][key][:, 64:96] = 0    # block 2 dead
            else:
                wgt_val = sim['logamp_wgts'][key][0, 0]
                sim['logamp_wgts'][key][:] = 0
                sim['logamp_wgts'][key][:, 64:84] = wgt_val
        deco, meta = skycal.estimate_SNAP_decoherence(
            sim['gains'], sim['logamp_wgts'], sim['ant_to_SNAP_dict'],
            sim['freqs'], nchans_per_block=sim['nchans_per_block'],
            min_block_coverage=0.7, **DECO_KWARGS)
        log_supp = meta['log_suppression']
        assert np.all(np.isnan(log_supp['B'][0]))
        assert meta['n_spectra_per_SNAP']['B'][0] == 3
        assert np.isnan(log_supp['A'][0, 2])
        assert np.isfinite(log_supp['A'][0, [0, 1, 3, 4, 5, 6, 7]]).all()

    def test_underdetermined_band_raises(self):
        # fewer unflagged channels in a band than DPSS modes ->
        # interpolatory fit, no information: must raise
        nblocks = 8
        sim = build_decoherence_sim({'A': np.zeros(nblocks),
                                     'B': np.zeros(nblocks)}, seed=6)
        key = sorted(sim['logamp_wgts'])[0]
        wgt_val = sim['logamp_wgts'][key][0, 0]
        sim['logamp_wgts'][key][:] = 0
        sim['logamp_wgts'][key][:, 130] = wgt_val   # a single live channel
        with pytest.raises(ValueError, match='interpolatory'):
            skycal.estimate_SNAP_decoherence(
                sim['gains'], sim['logamp_wgts'], sim['ant_to_SNAP_dict'],
                sim['freqs'], nchans_per_block=sim['nchans_per_block'],
                **DECO_KWARGS)

    def test_sparse_block_not_fit(self):
        # a block below min_block_coverage must come back NaN, not fit
        # with under-inflated errors
        nblocks = 8
        sim = build_decoherence_sim({'A': np.zeros(nblocks),
                                     'B': np.zeros(nblocks)}, seed=7)
        keys = sorted(sim['logamp_wgts'])
        for key in keys:
            sim['logamp_wgts'][key][:, 64:96] = 0   # block 2 dead...
        # ...except one channel on one antenna: coverage 1/32 < 0.05
        sim['logamp_wgts'][keys[0]][:, 64] = 1.0
        deco, meta = skycal.estimate_SNAP_decoherence(
            sim['gains'], sim['logamp_wgts'], sim['ant_to_SNAP_dict'],
            sim['freqs'], nchans_per_block=sim['nchans_per_block'],
            **DECO_KWARGS)
        for SNAP in ['A', 'B']:
            assert np.isnan(meta['log_suppression'][SNAP][0, 2])

    def test_log_gain_inverse_variance_skips(self):
        # autos in wgts and baselines with antennas missing from
        # refined_gains must contribute nothing
        shape = (1, 4)
        g0 = {(a, 'Jee'): np.ones(shape, dtype=complex) for a in range(4)}
        refined = {(a, 'Jee'): np.ones(shape, dtype=complex)
                   for a in range(3)}
        base = DataContainer({(0, 1, 'ee'): np.full(shape, 3.0)})
        extra = DataContainer({(0, 1, 'ee'): np.full(shape, 3.0),
                               (1, 1, 'ee'): np.full(shape, 9.0),
                               (0, 3, 'ee'): np.full(shape, 9.0)})
        SNAP_map = {0: 'A', 1: 'B', 2: 'A', 3: 'B'}
        out_base = skycal.log_gain_inverse_variance(base, g0, refined,
                                                    SNAP_map)
        out_extra = skycal.log_gain_inverse_variance(extra, g0, refined,
                                                     SNAP_map)
        for key in out_base:
            np.testing.assert_allclose(out_base[key], out_extra[key])

    def test_fix_band_floors(self):
        # the floor safety net, tested directly since the MCP solve's
        # nonnegativity boundary makes it unreachable through the driver:
        # an all-positive band is shifted so its minimum is exactly 0; a
        # band already containing a zero is untouched
        vals = np.array([0.0, 0.2, 0.1, 0.3, 0.45, 0.35, 0.0, 0.0])
        band_blocks = [[0, 1, 2], [3, 4, 5]]
        skycal._fix_band_floors(vals, band_blocks)
        np.testing.assert_allclose(vals[:3], [0.0, 0.2, 0.1])   # untouched
        np.testing.assert_allclose(vals[3:6], [0.0, 0.15, 0.05])  # pinned
        # empty bands are a no-op
        skycal._fix_band_floors(vals, [[]])
        np.testing.assert_allclose(vals[3:6], [0.0, 0.15, 0.05])


class TestDivergentChannelTolerance:
    def test_solve_or_nan(self):
        '''A singular member of a batched solve yields np.nan for that system only,
        deterministically on every platform, rather than raising for the whole batch.'''
        mats = np.array([np.eye(3), np.zeros((3, 3))])
        rhs = np.ones((2, 3, 1))
        solutions = skycal._solve_or_nan(mats, rhs)
        np.testing.assert_array_equal(solutions[0], rhs[0])
        assert np.all(np.isnan(solutions[1]))

    '''Tests for max_divergent_chan_frac: failed channels may be dropped and
    returned as nan instead of failing the whole solve, which lets coarse
    antenna-exclusion rounds (whose per-antenna chi^2 is a median over
    channels) proceed and remove whatever made those channels fail.'''

    def setup_method(self):
        self.rng = np.random.default_rng(5)

    def _arrays(self, nants=6, nfreqs=16, noise=0.0):
        true_h = ((1.0 + 0.2 * self.rng.uniform(-1, 1, (nants, nfreqs)))
                  * np.exp(1j * 0.3 * self.rng.uniform(-1, 1, (nants, nfreqs))))
        true_h *= np.exp(-1j * np.angle(true_h / np.abs(true_h)).mean(axis=0))[None, :]
        bls = [(i, j) for i in range(nants) for j in range(i + 1, nants)]
        ant_i = np.array([bl[0] for bl in bls])
        ant_j = np.array([bl[1] for bl in bls])
        vis_ratio = true_h[ant_i] * np.conj(true_h[ant_j])
        if noise > 0:
            vis_ratio = vis_ratio + noise * (self.rng.normal(size=vis_ratio.shape)
                                             + 1j * self.rng.normal(size=vis_ratio.shape)) / np.sqrt(2)
        wgts = np.ones_like(vis_ratio, dtype=float)
        return true_h, vis_ratio, wgts, ant_i, ant_j, nants

    def test_maxiter_failures_tolerated_and_reported(self):
        true_h, vis_ratio, wgts, ant_i, ant_j, nants = self._arrays()
        # refine_maxiter=0 fails every channel: intolerable by default...
        with pytest.raises(RuntimeError, match='did not converge'):
            skycal._refine_gains_single_pol_time(vis_ratio, wgts, ant_i, ant_j, nants,
                                                 refine_maxiter=0)
        # ...but tolerated in full, and every failed channel comes back nan
        gains, meta = skycal._refine_gains_single_pol_time(
            vis_ratio, wgts, ant_i, ant_j, nants, refine_maxiter=0,
            max_divergent_chan_frac=1.0)
        assert np.all(np.isnan(gains))
        np.testing.assert_array_equal(meta['divergent_chans'], np.arange(vis_ratio.shape[1]))

    def test_surviving_channels_are_unaffected(self):
        '''A few bad channels neither corrupt nor perturb the good ones.'''
        true_h, vis_ratio, wgts, ant_i, ant_j, nants = self._arrays()
        clean, _ = skycal._refine_gains_single_pol_time(vis_ratio, wgts, ant_i, ant_j, nants)

        # wreck two channels by making one antenna's visibilities inconsistent
        # with any per-antenna gain model (sign flips on a subset of its bls)
        bad_chans = [3, 11]
        wrecked = vis_ratio.copy()
        touches_0 = np.where(ant_i == 0)[0]
        for c in bad_chans:
            wrecked[touches_0[::2], c] *= -1e6
        with pytest.raises(RuntimeError):
            skycal._refine_gains_single_pol_time(wrecked, wgts, ant_i, ant_j, nants)
        gains, meta = skycal._refine_gains_single_pol_time(
            wrecked, wgts, ant_i, ant_j, nants, max_divergent_chan_frac=0.25)

        failed = set(meta['divergent_chans'].tolist())
        assert failed and failed.issubset(set(bad_chans))
        good = [c for c in range(vis_ratio.shape[1]) if c not in failed]
        assert np.all(np.isnan(gains[:, sorted(failed)]))
        assert np.all(np.isfinite(gains[:, good]))
        # the surviving channels are bit-identical to the all-clean solve
        np.testing.assert_allclose(gains[:, good], clean[:, good], atol=1e-10)

    def test_frac_bounds_the_damage(self):
        '''Tolerance is a budget: exceeding it still raises, naming channels.'''
        true_h, vis_ratio, wgts, ant_i, ant_j, nants = self._arrays()
        wrecked = vis_ratio.copy()
        touches_0 = np.where(ant_i == 0)[0]
        for c in [2, 5, 9, 13]:
            wrecked[touches_0[::2], c] *= -1e6
        # 1/16 of channels allowed, 4 fail
        with pytest.raises(RuntimeError, match='diverged'):
            skycal._refine_gains_single_pol_time(wrecked, wgts, ant_i, ant_j, nants,
                                                 max_divergent_chan_frac=1 / 16)
        gains, meta = skycal._refine_gains_single_pol_time(
            wrecked, wgts, ant_i, ant_j, nants, max_divergent_chan_frac=0.5)
        assert len(meta['divergent_chans']) == 4

    @pytest.mark.parametrize('frac', [-0.1, 1.5, np.nan, np.inf])
    def test_invalid_frac_raises(self, frac):
        '''Out-of-range values -- including a channel count mistaken for a
        fraction -- fail legibly instead of deep inside the solver.'''
        true_h, vis_ratio, wgts, ant_i, ant_j, nants = self._arrays()
        with pytest.raises(ValueError, match='fraction between'):
            skycal._refine_gains_single_pol_time(vis_ratio, wgts, ant_i, ant_j, nants,
                                                 max_divergent_chan_frac=frac)

    def test_failed_channels_marked_in_iter(self):
        '''meta['iter'] distinguishes failed channels (-1) from flagged (0).'''
        true_h, vis_ratio, wgts, ant_i, ant_j, nants = self._arrays()
        wgts[:, 7] = 0   # channel 7 flagged for every baseline
        wrecked = vis_ratio.copy()
        touches_0 = np.where(ant_i == 0)[0]
        wrecked[touches_0[::2], 3] *= -1e6
        gains, meta = skycal._refine_gains_single_pol_time(
            wrecked, wgts, ant_i, ant_j, nants, max_divergent_chan_frac=0.25)
        assert meta['iter'][7] == 0            # flagged
        assert meta['iter'][3] == -1           # failed
        assert np.all(meta['iter'][[0, 1, 2]] > 0)   # solved
        np.testing.assert_array_equal(meta['divergent_chans'], [3])

    def test_normal_matrices_stay_finite(self, monkeypatch):
        '''However badly a channel diverges, the linear solver is never handed a
        non-finite normal matrix. LAPACK reports those platform-dependently (as a
        LinAlgError on Linux but not macOS), so the solver detects the collapse
        itself rather than letting that escape.'''
        true_h, vis_ratio, wgts, ant_i, ant_j, nants = self._arrays()
        wrecked = vis_ratio.copy()
        touches_0 = np.where(ant_i == 0)[0]
        for c in [3, 11]:
            wrecked[touches_0[::2], c] *= -1e6

        original = skycal._build_normal_matrices
        all_finite = []

        def checked(*args, **kwargs):
            amp_mats, phase_mats = original(*args, **kwargs)
            all_finite.append(bool(np.isfinite(amp_mats).all() and np.isfinite(phase_mats).all()))
            return amp_mats, phase_mats

        monkeypatch.setattr(skycal, '_build_normal_matrices', checked)
        skycal._refine_gains_single_pol_time(wrecked, wgts, ant_i, ant_j, nants,
                                             max_divergent_chan_frac=0.25)
        assert len(all_finite) > 1 and all(all_finite)

    def test_default_is_unchanged_behavior(self):
        '''With the default, a clean solve is bit-identical and reports nothing.'''
        true_h, vis_ratio, wgts, ant_i, ant_j, nants = self._arrays(noise=0.01)
        g1, m1 = skycal._refine_gains_single_pol_time(vis_ratio, wgts, ant_i, ant_j, nants)
        g2, m2 = skycal._refine_gains_single_pol_time(vis_ratio, wgts, ant_i, ant_j, nants,
                                                      max_divergent_chan_frac=0.5)
        np.testing.assert_array_equal(g1, g2)
        assert len(m1['divergent_chans']) == 0 and len(m2['divergent_chans']) == 0

    def test_passes_through_sky_calibrate(self):
        '''The kwarg reaches the solver from the top-level driver.'''
        sim = build_sim(nants=7, nfreqs=32, seed=4)
        gains, meta = skycal.sky_calibrate(
            sim['data'], sim['model'], freqs=sim['freqs'], dt=sim['dt'], df=sim['df'],
            max_divergent_chan_frac=0.1)
        assert 'divergent_chans' in meta
        for key, chans in meta['divergent_chans'].items():
            assert len(chans) == 0  # clean sim: nothing should fail


class TestExpandSkyGains:
    def test_recovery_and_containment(self):
        sim = build_sim()
        gains = dict(sim['true_gains'])
        spectator = (3, 'Jee')
        del gains[spectator]
        solved_before = {ant: g.copy() for ant, g in gains.items()}
        chisq_per_ant = {}
        skycal.expand_sky_gains(sim['data'], sim['model'], gains, dt=sim['dt'], df=sim['df'],
                                chisq_per_ant=chisq_per_ant)
        # the spectator's true gain is recovered exactly and its chi^2 is ~0 (noiseless),
        # while every solved antenna's gain is bit-identical to before
        np.testing.assert_allclose(gains[spectator], sim['true_gains'][spectator], rtol=1e-8)
        assert np.all(chisq_per_ant[spectator] < 1e-16)
        assert set(chisq_per_ant) == {spectator}
        for ant, g in solved_before.items():
            np.testing.assert_array_equal(gains[ant], g)

    def test_two_spectators_never_chain(self):
        sim = build_sim()
        gains = dict(sim['true_gains'])
        # corrupt the baseline between the two spectators: it must never be used
        sim['data'][(2, 5, sim['pol'])] *= 100
        for spectator in [(2, 'Jee'), (5, 'Jee')]:
            del gains[spectator]
        skycal.expand_sky_gains(sim['data'], sim['model'], gains, dt=sim['dt'], df=sim['df'])
        for spectator in [(2, 'Jee'), (5, 'Jee')]:
            np.testing.assert_allclose(gains[spectator], sim['true_gains'][spectator], rtol=1e-8)

    def test_inter_SNAP_restriction(self):
        sim = build_sim()
        ant_to_SNAP_dict = {antnum: ('S0' if antnum < 4 else 'S1') for antnum in sim['antnums']}
        gains = dict(sim['true_gains'])
        del gains[(2, 'Jee')]
        # corrupt the spectator's intra-SNAP baselines: with the restriction they are
        # never used, so recovery stays exact; without it, they bias the solution
        for j in [0, 1, 3]:
            sim['data'][(min(2, j), max(2, j), sim['pol'])] *= 3
        skycal.expand_sky_gains(sim['data'], sim['model'], gains, dt=sim['dt'], df=sim['df'],
                                ant_to_SNAP_dict=ant_to_SNAP_dict)
        np.testing.assert_allclose(gains[(2, 'Jee')], sim['true_gains'][(2, 'Jee')], rtol=1e-8)
        gains_unrestricted = {ant: g for ant, g in sim['true_gains'].items() if ant != (2, 'Jee')}
        skycal.expand_sky_gains(sim['data'], sim['model'], gains_unrestricted,
                                dt=sim['dt'], df=sim['df'])
        assert not np.allclose(gains_unrestricted[(2, 'Jee')], sim['true_gains'][(2, 'Jee')],
                               rtol=1e-3)

    def test_corrupted_spectator_gets_large_chisq(self):
        sim = build_sim(noise=1.0, seed=5)
        gains = dict(sim['true_gains'])
        del gains[(3, 'Jee')]
        # a common rescaling would be absorbed by g_a, so corrupt the spectator's
        # baselines with baseline-DEPENDENT factors no per-antenna gain can explain
        for k, bl in enumerate(bl for bl in list(sim['data']) if 3 in bl[:2] and bl[0] != bl[1]):
            sim['data'][bl] *= (1 + 0.5 * (-1)**k)
        chisq_per_ant = {}
        skycal.expand_sky_gains(sim['data'], sim['model'], gains, dt=sim['dt'], df=sim['df'],
                                chisq_per_ant=chisq_per_ant)
        assert np.mean(chisq_per_ant[(3, 'Jee')]) > 100

    def test_nan_where_unusable(self):
        sim = build_sim()
        gains = dict(sim['true_gains'])
        del gains[(3, 'Jee')]
        all_flagged = np.ones((sim['ntimes'], len(sim['freqs'])), dtype=bool)
        chisq_per_ant = {}
        skycal.expand_sky_gains(sim['data'], sim['model'], gains, dt=sim['dt'], df=sim['df'],
                                ant_flags={ant: all_flagged for ant in gains},
                                chisq_per_ant=chisq_per_ant)
        assert np.all(np.isnan(gains[(3, 'Jee')]))
        assert np.all(np.isnan(chisq_per_ant[(3, 'Jee')]))

    def test_no_spectators_is_a_no_op(self):
        sim = build_sim()
        gains = dict(sim['true_gains'])
        skycal.expand_sky_gains(sim['data'], sim['model'], gains, dt=sim['dt'], df=sim['df'])
        assert set(gains) == set(sim['true_gains'])
