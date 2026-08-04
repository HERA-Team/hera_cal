# -*- coding: utf-8 -*-
# Copyright 2026 the HERA Project
# Licensed under the MIT License

"""Staged sky-model-based calibration.

This module calibrates raw visibilities against a sky model in stages designed
so that any per-antenna, non-smooth amplitude effect that only appears on cross
correlations (e.g. per-SNAP signal loss in the correlator, a.k.a. "decoherence")
lands in exactly one place — the final per-channel refinement gains — where it
can be measured and corrected downstream:

    1. Firstcal-style delay calibration: per-antenna delays and phase offsets
       from the phases of the data/model ratio on cross baselines. Phases only,
       blind to amplitude suppression.
    2. Amplitude calibration from the autocorrelations: each antenna's |gain|
       from its own autocorrelation referenced to the median over antennas.
       Autocorrelations are exempt from cross-correlation-only signal loss, so
       this amplitude scale cannot absorb it.
    3. Per-channel complex gain refinement solved on cross baselines — by
       default restricted to baselines whose antennas are on different SNAPs,
       since intra-SNAP baselines are likewise exempt from decoherence.
       Whatever amplitude structure the crosses need beyond stages 1-2 lands
       in these refined gains.

The final gain is the product of the stage 1-2 gains (g0) and the stage 3
refined gains.

The primary use case for the "model" is LST-stacked, redundantly-averaged, and
delay/fringe-rate-filtered visibilities from prior nights (e.g. the corner-
turned single-baseline sky-model files built for abscal): smooth by
construction and keyed by redundant group, so a RedDataContainer can map every
physical baseline to its group's model. Any time-aligned model visibilities
work in principle, but the noise weighting and the smoothness assumptions
implicit in downstream analysis are designed around that product.

This module works on already-loaded data: file I/O, sky-model file
selection/LST-matching, flag sourcing, and the M&C antenna-to-SNAP lookup are
all the caller's responsibility (hera_mc is deliberately not imported).
"""
import numpy as np
import linsolve
from hera_filters import dspec

from . import utils
from .noise import predict_noise_variance_from_autos
from .datacontainer import DataContainer
from .abscal import merge_gains


########################################################################
# Shared utilities
########################################################################


def _pack_baseline_arrays(container, bls):
    '''Stack per-baseline waterfalls into a single array plus antenna indexing.

    Arguments:
        container: DataContainer (or dict) mapping baseline keys to
            (Ntimes, Nfreqs) waterfalls
        bls: list of baseline tuples like (0, 1, 'ee') to stack, in order

    Returns:
        stacked: (Nbls, Ntimes, Nfreqs) ndarray of the waterfalls, in the
            order given by bls
        ant_i_idx: int ndarray of first-antenna indices into ants, per baseline
        ant_j_idx: int ndarray of second-antenna indices into ants, per baseline
        ants: sorted list of antenna numbers appearing in bls
    '''
    ants = sorted({ant for bl in bls for ant in bl[:2]})
    idx = {ant: i for i, ant in enumerate(ants)}
    stacked = np.asarray([container[bl] for bl in bls])
    ant_i_idx = np.array([idx[bl[0]] for bl in bls], dtype=int)
    ant_j_idx = np.array([idx[bl[1]] for bl in bls], dtype=int)
    return stacked, ant_i_idx, ant_j_idx, ants


########################################################################
# Data / model ratio construction
########################################################################


def build_data_model_ratio(data, model, autos=None, data_flags=None,
                           model_flags=None, ant_flags=None, bls=None,
                           dt=None, df=None):
    '''Build the matched-filter data/model ratio and its inverse-variance
    weights: ratio = V * M^* / |M|^2 (equal to V / M in the low-noise limit)
    and wgts = |M|^2 / sigma^2 with sigma^2 = auto_i * auto_j / (dt * df) from
    noise.predict_noise_variance_from_autos. In these conventions the weighted
    average of ratio over any set of cells is the inverse-variance estimate of
    the (assumed common) gain factor relating data to model.

    Arguments:
        data: DataContainer of visibilities (cross-correlations; may also
            contain the autocorrelations used for noise weights)
        model: DataContainer or RedDataContainer of model visibilities,
            time-aligned to data. Primarily intended to be LST-stacked,
            redundantly-averaged, and filtered visibilities from prior nights
            (smooth by construction); a RedDataContainer resolves any baseline
            in a redundant group to its prototype, conjugating as needed.
        autos: DataContainer containing autocorrelations for noise weights.
            Default None uses data.
        data_flags: optional DataContainer of per-baseline flag waterfalls
        model_flags: optional DataContainer of model flag waterfalls
        ant_flags: optional dict mapping (ant, antpol) e.g. (0, 'Jee') to flag
            waterfalls, applied to both antennas of each baseline
        bls: optional list of baselines to include. Default None uses all
            cross baselines in data that have corresponding models.
        dt: integration time in seconds. Default None infers from autos.
        df: channel width in Hz. Default None infers from autos.

    Returns:
        data_model_ratio: DataContainer of V * M^* / |M|^2 waterfalls, np.nan
            where flagged or where the model is missing/zero
        wgts: DataContainer of |M|^2 / sigma^2 waterfalls, 0 where flagged
    '''
    if autos is None:
        autos = data
    if bls is None:
        bls = [bl for bl in data if bl[0] != bl[1] and bl in model]
    # fail early and legibly if any noise-weight autocorrelation is missing
    # (otherwise this surfaces as an opaque KeyError inside noise.py)
    missing_autos = sorted({utils.join_bl(ant, ant) for bl in bls
                            for ant in utils.split_bl(bl)
                            if utils.join_bl(ant, ant) not in autos})
    if len(missing_autos) > 0:
        raise ValueError('Autocorrelations needed for noise weights are '
                         f'missing from autos: {missing_autos}.')

    ratio_here, wgts_here = {}, {}
    for bl in bls:
        mvis = np.asarray(model[bl])
        flgs = ~np.isfinite(mvis)
        if data_flags is not None:
            flgs = flgs | data_flags[bl]
        if model_flags is not None:
            flgs = flgs | np.asarray(model_flags[bl])
        if ant_flags is not None:
            ant1, ant2 = utils.split_bl(bl)
            if ant1 in ant_flags:
                flgs = flgs | ant_flags[ant1]
            if ant2 in ant_flags:
                flgs = flgs | ant_flags[ant2]
        mvis = np.nan_to_num(mvis)
        m2 = np.abs(mvis)**2
        sigma2 = predict_noise_variance_from_autos(bl, autos, dt=dt, df=df)
        good = (~flgs) & (m2 > 0) & np.isfinite(sigma2) & (sigma2 > 0)
        with np.errstate(invalid='ignore', divide='ignore'):
            ratio_here[bl] = np.where(good, data[bl] * np.conj(mvis)
                                      / np.where(m2 > 0, m2, 1), np.nan)
            wgts_here[bl] = np.where(good, m2 / np.where(good, sigma2, 1), 0)
    return DataContainer(ratio_here), DataContainer(wgts_here)


########################################################################
# Model-based firstcal: per-antenna delays and phase offsets
########################################################################


def _solve_per_antenna_weighted_least_squares(vals, wgts, ant_i_idx,
                                              ant_j_idx, nants):
    '''Weighted least-squares solve of per-baseline differences for per-antenna
    values, i.e. vals[bl] ~ x[i] - x[j], with the mean over solved antennas
    fixed to 0 (an overall offset is unobservable in differences).

    Arguments:
        vals: ndarray of per-baseline measurements (e.g. delays tau_i - tau_j)
        wgts: ndarray of per-baseline weights (non-positive weights are ignored)
        ant_i_idx / ant_j_idx: int ndarrays indexing each baseline's antennas
        nants: total number of antennas indexed

    Returns:
        x: ndarray of length nants with per-antenna solutions; antennas with no
            usable baselines are np.nan
    '''
    ok = np.isfinite(vals) & (wgts > 0)
    ls_data, ls_wgts = {}, {}
    for bi in np.where(ok)[0]:
        eqn = f'x_{ant_i_idx[bi]} - x_{ant_j_idx[bi]}'
        if eqn in ls_data:
            # weighted-average repeated measurements of the same difference
            wtot = ls_wgts[eqn] + wgts[bi]
            ls_data[eqn] = (ls_data[eqn] * ls_wgts[eqn]
                            + vals[bi] * wgts[bi]) / wtot
            ls_wgts[eqn] = wtot
        else:
            ls_data[eqn] = vals[bi]
            ls_wgts[eqn] = wgts[bi]
    x = np.full(nants, np.nan)
    if len(ls_data) > 0:
        # pinv gives the minimum-norm solution of the (singular) difference
        # system; the explicit mean subtraction then removes the degeneracy
        # by fixing the mean to 0
        sol = linsolve.LinearSolver(ls_data, wgts=ls_wgts).solve(mode='pinv')
        for i in range(nants):
            if f'x_{i}' in sol:
                x[i] = float(sol[f'x_{i}'])
        solved = np.isfinite(x)
        x[solved] -= x[solved].mean()
    return x


def model_based_firstcal(data_model_ratio, wgts, freqs, verbose=False):
    '''Solve for per-antenna delays and phase offsets from the phases of the
    data/model ratio (phases only; amplitudes are untouched), independently
    for each integration. Per baseline and integration, the weighted FFT of
    the ratio across frequency peaks at tau_i - tau_j; per-antenna delays
    follow by weighted least squares with the mean delay fixed to 0 (an
    overall delay is unobservable). Per-antenna phase offsets are then solved
    the same way from the delay-corrected mean phases.

    Arguments:
        data_model_ratio: DataContainer from build_data_model_ratio
        wgts: DataContainer of weights from build_data_model_ratio
        freqs: ndarray of frequencies in Hz
        verbose: print statements if True

    Returns:
        dlys: dict mapping (ant, antpol) e.g. (0, 'Jee') to (Ntimes, 1)
            ndarrays of delays in seconds. Antennas with no usable
            cross-baseline data get 0.0.
        offsets: dict mapping (ant, antpol) to (Ntimes, 1) ndarrays of phase
            offsets in radians. Antennas with no usable cross-baseline data
            get 0.0.
    '''
    freqs = np.asarray(freqs)
    df = np.median(np.diff(freqs))
    dlys, offsets = {}, {}
    for pol in sorted({bl[2] for bl in data_model_ratio}):
        bls_here = [bl for bl in data_model_ratio
                    if bl[2] == pol and bl[0] != bl[1]]
        ratio, ant_i_idx, ant_j_idx, ants = _pack_baseline_arrays(
            data_model_ratio, bls_here)
        wgt_arr = np.asarray([wgts[bl] for bl in bls_here])
        nants = len(ants)
        nbls, ntimes = ratio.shape[0], ratio.shape[1]
        wgtd_ratio = np.nan_to_num(ratio) * wgt_arr
        solve_wgts = wgt_arr.sum(axis=2)

        # per-(baseline, integration) delays from the FFT peak
        flat_ratio = wgtd_ratio.reshape(nbls * ntimes, -1)
        flat_wgts = solve_wgts.reshape(nbls * ntimes)
        bl_dlys = np.full(nbls * ntimes, np.nan)
        has_wgt = flat_wgts > 0
        if has_wgt.any():
            bl_dlys[has_wgt] = utils.fft_dly(flat_ratio[has_wgt],
                                             df)[0].ravel()
        bl_dlys = bl_dlys.reshape(nbls, ntimes)

        # per-antenna delays (degeneracy fixed: mean delay = 0), then
        # per-antenna offsets from delay-corrected mean phases,
        # independently per integration
        ant_dlys = np.full((ntimes, nants), np.nan)
        ant_offsets = np.full((ntimes, nants), np.nan)
        for tind in range(ntimes):
            ant_dlys[tind] = _solve_per_antenna_weighted_least_squares(
                bl_dlys[:, tind], solve_wgts[:, tind], ant_i_idx, ant_j_idx,
                nants)
            with np.errstate(invalid='ignore'):
                dly_phasor = np.exp(
                    -2j * np.pi * freqs[None, :]
                    * np.nan_to_num(ant_dlys[tind, ant_i_idx]
                                    - ant_dlys[tind, ant_j_idx])[:, None])
            resid_phases = np.angle(
                (wgtd_ratio[:, tind] * dly_phasor).sum(axis=1))
            ant_offsets[tind] = _solve_per_antenna_weighted_least_squares(
                resid_phases, solve_wgts[:, tind], ant_i_idx, ant_j_idx,
                nants)

        n_unsolved = int(np.sum(~np.isfinite(ant_dlys).any(axis=0)))
        if n_unsolved > 0:
            utils.echo(f'{n_unsolved} antennas in {pol} have no usable '
                       'cross-baseline data; setting delays/offsets to 0.',
                       verbose=verbose)
        antpol = utils.split_pol(pol)[0]
        for i, ant in enumerate(ants):
            dlys[(ant, antpol)] = np.nan_to_num(ant_dlys[:, i])[:, None]
            offsets[(ant, antpol)] = np.nan_to_num(ant_offsets[:, i])[:, None]
    return dlys, offsets


def firstcal_gains(dlys, offsets, freqs):
    '''Expand per-antenna delays and offsets into unit-amplitude gain
    waterfalls.

    Arguments:
        dlys: dict mapping (ant, antpol) to (Ntimes, 1) ndarrays of delays in
            seconds, as returned by model_based_firstcal
        offsets: dict mapping (ant, antpol) to (Ntimes, 1) ndarrays of phase
            offsets in radians, as returned by model_based_firstcal
        freqs: ndarray of frequencies in Hz

    Returns:
        gains: dict mapping (ant, antpol) to complex (Ntimes, len(freqs))
            gain waterfalls, exp(2j pi freqs dly + 1j offset)
    '''
    # same convention as redcal.RedundantCalibrator.firstcal
    return {ant: np.exp(2j * np.pi * dly * np.asarray(freqs) + 1j * offsets[ant])
            for ant, dly in dlys.items()}


########################################################################
# Amplitude calibration from the autocorrelations
########################################################################


def calibrate_abs_amp_from_autos(autos, auto_flags=None):
    '''Derive real, positive per-antenna gain amplitudes from the
    autocorrelations: |g_i| = sqrt(auto_i / median-over-antennas auto). The
    median reference divides out the sky, so each antenna's amplitude carries
    its bandpass and receiver temperature relative to the array median. The
    job of this stage is to start the per-channel refinement from roughly the
    right amplitude scale; the refinement then determines the amplitudes
    absolutely from the cross-correlations. Note that these amplitudes are
    BIASED wherever receiver temperatures differ from antenna to antenna:
    each autocorrelation measures |g|^2 * (T_sky + T_rx) while the
    cross-correlations see only the sky signal, so referencing to the median
    auto effectively assumes one common T_rx. That per-antenna (and mostly
    spectrally smooth) error is subsequently absorbed by the refinement.
    (A useful side effect of using the autos: because autocorrelations are
    exempt from cross-correlation-only signal loss ("decoherence"), none of
    that loss is baked into these starting amplitudes, so it appears cleanly
    in the refined gains instead.)

    Arguments:
        autos: DataContainer containing autocorrelations, keys (ant, ant, pol)
        auto_flags: optional DataContainer/dict of flag waterfalls with the
            same keys; flagged cells are excluded from the median reference

    Returns:
        gains: dict mapping (ant, antpol) to real-positive (Ntimes, Nfreqs)
            gain amplitude waterfalls (complex dtype for downstream merging)
    '''
    gains = {}
    for pol in sorted({bl[2] for bl in autos if bl[0] == bl[1]}):
        auto_bls = sorted(bl for bl in autos if bl[0] == bl[1]
                          and bl[2] == pol)
        stack = np.asarray([np.abs(autos[bl]) for bl in auto_bls], dtype=float)
        if auto_flags is not None:
            for i, bl in enumerate(auto_bls):
                if bl in auto_flags:
                    stack[i][np.asarray(auto_flags[bl])] = np.nan
        ref_auto = np.nanmedian(stack, axis=0)
        antpol = utils.split_pol(pol)[0]
        for bl in auto_bls:
            with np.errstate(invalid='ignore', divide='ignore'):
                amp = np.sqrt(np.abs(autos[bl])
                              / np.where(ref_auto > 0, ref_auto, np.nan))
            gains[(bl[0], antpol)] = amp.astype(complex)
    return gains


########################################################################
# Per-channel complex gain refinement
########################################################################


def _shared_channel_flags(ratio_wgts, ant_i_idx, ant_j_idx, nants):
    '''Validate that the flagging pattern (encoded as zero weights) has the
    uniform structure this solver requires and reduce it to a single shared
    flag waterfall. Acceptable flags are combinations of exactly two kinds:
    whole ANTENNAS flagged (every channel of every baseline containing them
    at this time) and channels flagged for ALL baselines at once (e.g.
    array-wide RFI flags). Under that structure every surviving baseline
    sees the same channels, so every participating antenna appears in every
    kept channel and the per-channel systems are exactly nonsingular with no
    regularization. Anything else — a channel flagged for a strict subset of
    baselines, or a fully-flagged baseline between two otherwise-unflagged
    antennas — raises a ValueError rather than being silently repaired: fix
    the flags upstream (the intended file_calibration flagging produces
    exactly the required structure), and exclude unwanted baselines from the
    solve by omitting them, not by flagging them.

    Arguments:
        ratio_wgts: float ndarray (Nbls, Nfreqs) of weights; 0 = flagged
        ant_i_idx / ant_j_idx: int ndarrays of length Nbls indexing antennas
        nants: total number of antennas indexed

    Returns:
        flagged_ants: boolean ndarray over antennas that are entirely flagged
        chan_flags: boolean ndarray over channels; the OR of the surviving
            baselines' flags (identical for each of them, by validation)
    '''
    unflagged = ratio_wgts > 0
    flagged_bls = ~unflagged.any(axis=1)
    # an antenna is flagged if and only if all of its baselines are
    ant_live = np.zeros(nants, dtype=bool)
    np.logical_or.at(ant_live, ant_i_idx, ~flagged_bls)
    np.logical_or.at(ant_live, ant_j_idx, ~flagged_bls)
    flagged_ants = ~ant_live
    unexplained = flagged_bls & ~(flagged_ants[ant_i_idx]
                                  | flagged_ants[ant_j_idx])
    if unexplained.any():
        raise ValueError(
            'Flagging pattern is not uniform: some baselines are entirely '
            'flagged although neither of their antennas is entirely flagged. '
            'Only whole-antenna flags and channels flagged for ALL baselines '
            'are supported; exclude baselines from the solve by omitting '
            'them, not by flagging them.')
    if flagged_bls.all():
        return flagged_ants, np.ones(ratio_wgts.shape[1], dtype=bool)
    chan_flags = ~unflagged[~flagged_bls].all(axis=0)
    if not (unflagged[~flagged_bls] == ~chan_flags[None, :]).all():
        raise ValueError(
            'Flagging pattern is not uniform: only whole-antenna flags and '
            'channels flagged for ALL baselines are supported, but some '
            'channels are flagged for a strict subset of baselines. Fix the '
            'flags upstream (e.g. broadcast per-antenna channel flags to '
            'all antennas) before calling this solver.')
    return flagged_ants, chan_flags


def _build_normal_matrices(nchan_active, nsel, chan_pos, ei, ej, round_wgts):
    '''Build the batched per-channel normal matrices (the matrices of the two
    weighted least-squares systems solved each round). Amplitude equations
    have the form resid ~ eta_i + eta_j — log-amplitudes ADD, because
    |g_i * conj(g_j)| = |g_i| |g_j| — giving matrices with POSITIVE
    off-diagonal couplings that are positive definite under uniform flagging.
    Phase equations have the form resid ~ phi_i - phi_j — phases SUBTRACT,
    from the conjugation of antenna j's gain — giving graph-Laplacian
    matrices whose one exact null direction (a constant added to every
    phase) is removed by pinning the first participating antenna to 0; the
    caller then removes the degeneracy of the resulting update by setting
    its mean phase to 0.

    Arguments:
        nchan_active: number of channels in the batch
        nsel: number of participating antennas
        chan_pos / ei / ej: int ndarrays over unflagged (baseline, channel)
            entries giving each entry's position in the channel batch and
            its two antenna indices
        round_wgts: ndarray of per-entry weights for this round

    Returns:
        amp_mats: (nchan_active, nsel, nsel) log-amplitude system matrices
        phase_mats: (nchan_active, nsel - 1, nsel - 1) phase system matrices
            with the first participating antenna pinned
    '''
    # accumulate with np.bincount on flattened indices (several times faster
    # than np.add.at); the amplitude and phase matrices share their diagonal
    # blocks and differ only in the sign of the off-diagonal couplings
    n2 = nsel * nsel
    base = chan_pos.astype(np.int64) * n2
    minlength = nchan_active * n2
    diag = (np.bincount(base + ei * (nsel + 1), weights=round_wgts,
                        minlength=minlength)
            + np.bincount(base + ej * (nsel + 1), weights=round_wgts,
                          minlength=minlength))
    off = (np.bincount(base + ei * nsel + ej, weights=round_wgts,
                       minlength=minlength)
           + np.bincount(base + ej * nsel + ei, weights=round_wgts,
                         minlength=minlength))
    amp_mats = (diag + off).reshape(nchan_active, nsel, nsel)
    phase_mats = (diag - off).reshape(nchan_active, nsel, nsel)
    return amp_mats, phase_mats[:, 1:, 1:]


def _phase_sync_init(phasors, round_wgts, chan_pos, ei, ej, nchan_active,
                     nsel, sync_tol=1e-3, sync_maxiter=200):
    '''Wrap-immune initialization of per-antenna phases by eigenvector phase
    synchronization, batched over channels.

    The problem: given noisy unit-modulus baseline measurements
    u_ij ~ exp(1j * (phi_i - phi_j)) with weights w_ij, estimate per-antenna
    phases phi_i. The obvious approach — weighted least squares on the
    measured angles, angle(u_ij) ~ phi_i - phi_j — is WRAP-BLIND: each
    measured angle determines phi_i - phi_j only modulo 2 pi, so once the
    true phases span more than ~1 radian the linear solve lands on the wrong
    branch. This initialization instead works with the phasors themselves,
    so no angle is ever unwrapped: build the Hermitian matrix
    S[i, j] = w_ij * u_ij (with S[j, i] its conjugate). Up to noise,
    S[i, j] = w_ij * v_i * conj(v_j) for the true phasor vector
    v_i = exp(1j * phi_i), which makes v the leading eigenvector of S; power
    iteration converges to it, and the angles of its entries are the desired
    phases. (This is the standard eigenvector method for the angular
    synchronization problem — see e.g. Singer 2011, Applied and
    Computational Harmonic Analysis, 30, 20.) Iteration stops when the
    largest per-antenna angle change falls below sync_tol. The returned
    angles have their mean over antennas subtracted, since an overall phase
    is unobservable in baseline differences.

    Arguments:
        phasors: complex ndarray of unit-modulus per-entry measurements u_ij
        round_wgts: ndarray of per-entry weights
        chan_pos / ei / ej: int ndarrays giving each entry's position in the
            channel batch and its two antenna indices
        nchan_active: number of channels in the batch
        nsel: number of participating antennas
        sync_tol: convergence threshold in radians
        sync_maxiter: iteration cap

    Returns:
        phases: (nchan_active, nsel) ndarray of initial per-antenna phases,
            mean 0 over antennas in each channel
    '''
    sync_mats = np.zeros((nchan_active, nsel, nsel), dtype=np.complex64)
    np.add.at(sync_mats, (chan_pos, ei, ej),
              (round_wgts * phasors).astype(np.complex64))
    np.add.at(sync_mats, (chan_pos, ej, ei),
              (round_wgts * np.conj(phasors)).astype(np.complex64))
    eigvec = np.ones((nchan_active, nsel, 1), dtype=np.complex64)
    phases = np.zeros((nchan_active, nsel))
    for _ in range(sync_maxiter):
        # one power-iteration step toward the leading eigenvector
        eigvec = sync_mats @ eigvec
        eigvec /= np.linalg.norm(eigvec, axis=1, keepdims=True) + 1e-30
        new_phases = np.angle(eigvec[:, :, 0])
        new_phases -= new_phases.mean(axis=1, keepdims=True)
        max_change = np.max(np.abs(np.angle(np.exp(
            1j * (new_phases - phases)))))
        phases = new_phases
        if max_change < sync_tol:
            break
    return phases


def _solve_logamp_updates(amp_mats, chan_pos, ei, ej, round_wgts, resids,
                          nchan_active, nsel):
    '''Solve the batched linear weighted least-squares systems for the
    per-antenna log-amplitude updates. Each entry contributes the equation
    resid ~ eta_i + eta_j, so both antennas' right-hand sides accumulate the
    residual with POSITIVE sign (see _build_normal_matrices).

    Returns: (nchan_active, nsel) ndarray of log-amplitude updates eta.'''
    base = chan_pos.astype(np.int64) * nsel
    wgtd_resids = round_wgts * resids
    amp_rhs = (np.bincount(base + ei, weights=wgtd_resids,
                           minlength=nchan_active * nsel)
               + np.bincount(base + ej, weights=wgtd_resids,
                             minlength=nchan_active * nsel)
               ).reshape(nchan_active, nsel)
    return np.linalg.solve(amp_mats, amp_rhs[:, :, None])[:, :, 0]


def _solve_phase_updates(phase_mats, chan_pos, ei, ej, round_wgts, resids,
                         nchan_active, nsel):
    '''Solve the batched linear weighted least-squares systems for the
    per-antenna phase updates. Each entry contributes the equation
    resid ~ phi_i - phi_j (the sign flip comes from the conjugation of
    antenna j's gain). The systems are the reduced ones with the first
    participating antenna pinned to 0 (see _build_normal_matrices); the
    caller removes the degeneracy of the full update by setting its mean
    phase to 0.

    Returns: (nchan_active, nsel) ndarray of phase updates phi, with the
        pinned antenna's entry equal to 0.'''
    base = chan_pos.astype(np.int64) * nsel
    wgtd_resids = round_wgts * resids
    phase_rhs = (np.bincount(base + ei, weights=wgtd_resids,
                             minlength=nchan_active * nsel)
                 - np.bincount(base + ej, weights=wgtd_resids,
                               minlength=nchan_active * nsel)
                 ).reshape(nchan_active, nsel)
    delta_phase = np.zeros((nchan_active, nsel))
    delta_phase[:, 1:] = np.linalg.solve(phase_mats,
                                         phase_rhs[:, 1:, None])[:, :, 0]
    return delta_phase


def _stationarity_residual(vis_ratio, ratio_wgts, full_gains, ant_i_idx,
                           ant_j_idx):
    '''Compute the convergence certificate: the per-channel MAXIMUM
    fixed-point residual over all solved antennas. At the exact weighted
    least-squares optimum, every gain equals the weighted projection of the
    data onto the other antennas' gains,
    g_i = sum_j(w_ij * z_ij * g_j) / sum_j(w_ij * |g_j|^2) = U / D, so
    max |U/D - g| / |g| measures how far each cell is from stationarity,
    independently of the solver's own update sizes. Certifying on maxima
    (never a median or percentile) is deliberate: a median criterion can
    declare victory while an entire contiguous band remains unconverged.

    Returns: (Nfreqs,) ndarray of the max residual over solved antennas in
        each channel; np.nan for channels with no solved cells.'''
    wgtd_ratio = np.nan_to_num(vis_ratio) * ratio_wgts
    numerator = np.zeros(full_gains.shape, dtype=complex)
    denominator = np.zeros(full_gains.shape)
    gains_zeroed = np.nan_to_num(full_gains)
    np.add.at(numerator, ant_i_idx, wgtd_ratio * gains_zeroed[ant_j_idx])
    np.add.at(denominator, ant_i_idx,
              ratio_wgts * np.abs(gains_zeroed[ant_j_idx])**2)
    np.add.at(numerator, ant_j_idx,
              np.conj(wgtd_ratio) * gains_zeroed[ant_i_idx])
    np.add.at(denominator, ant_j_idx,
              ratio_wgts * np.abs(gains_zeroed[ant_i_idx])**2)
    with np.errstate(invalid='ignore', divide='ignore'):
        resid = np.abs(numerator / np.maximum(denominator, 1e-30)
                       - full_gains) / np.abs(full_gains)
    solved_cells = np.isfinite(full_gains) & (denominator > 0)
    per_chan = np.full(full_gains.shape[1], np.nan)
    solved_chans = solved_cells.any(axis=0)
    per_chan[solved_chans] = np.where(solved_cells, resid,
                                      -np.inf).max(axis=0)[solved_chans]
    return per_chan


def _refine_gains_single_pol_time(vis_ratio, ratio_wgts, ant_i_idx, ant_j_idx,
                                  nants, refine_tol=1e-8,
                                  refine_maxiter=100, sync_tol=1e-3,
                                  sync_maxiter=200, verbose=False):
    '''Solve for per-antenna, per-channel complex gains g such that
    vis_ratio ~ g_i * conj(g_j) on each baseline, for a single
    (time, polarization).

    The model is BILINEAR in the gains, so the problem itself is not linear.
    It is solved by iteratively linearized weighted least squares
    (Gauss-Newton) — the same strategy as redcal's lincal/omnical — with
    every channel independent and solved as one batch of small dense
    systems:

        1. Initialization: log-amplitudes come from the EXACTLY linear
           system log|vis_ratio_ij| = log|g_i| + log|g_j|; phases come from
           wrap-immune eigenvector phase synchronization (_phase_sync_init),
           because a linearized solve on measured angles cannot be used to
           start — angles are only known modulo 2 pi.
        2. Gauss-Newton rounds: writing the update as
           g <- g * exp(eta + 1j * phi), the residual ratio
           vis_ratio / (g_i * conj(g_j)) = exp(eta_i + eta_j
           + 1j * (phi_i - phi_j)), which is approximately
           1 + (eta_i + eta_j) + 1j * (phi_i - phi_j) for small updates. To
           first order the two kinds of unknowns separate: the REAL part
           minus 1 gives a linear system for the log-amplitude updates and
           the IMAGINARY part gives a linear system for the phase updates.
           Both are solved exactly each round; iterating drives the
           linearization error to zero. Channels whose largest update falls
           below refine_tol drop out of the batch early.

    Relation to redcal's estimators and the logcal bias: only the
    initialization is logcal-like. Linearizing g_i * conj(g_j) shows that
    each Gauss-Newton round solves the normal equations of the UNTRANSFORMED
    complex objective chi^2 = sum_ij w_ij |vis_ratio_ij - g_i * conj(g_j)|^2
    (the (|g_i| |g_j|)^2 weight factor is that objective's Jacobian), so the
    converged solution is the stationary point of the complex chi^2 —
    exactly what _stationarity_residual certifies — independent of the
    starting point. The low-signal-to-noise bias of logcal identified by
    Liu et al. (2010, MNRAS 408, 1029) afflicts estimators whose FINAL
    answer is the log-space optimum; here the log-space solve only starts
    the iteration. (Iterating in log space instead was observed, during this
    algorithm's development, to converge to a measurably offset optimum at
    low signal-to-noise — which is why the rounds target the complex
    objective.) The remaining bias is the generic second-order-in-noise bias
    of any nonlinear least-squares fit, shared with redcal's lincal/omnical.

    The flags (zero weights) are required to have a uniform structure —
    whole antennas flagged plus channels flagged for all baselines at once —
    which is validated and reduced to a single shared flag waterfall
    (_shared_channel_flags); this makes every system exactly nonsingular
    with no regularization. The overall phase degeneracy is fixed by pinning
    one antenna inside the solves and then removing the degeneracy from each
    update by setting its mean phase to 0 over participating antennas — the
    same convention as redcal.remove_degen_gains. Convergence is certified by the maximum
    fixed-point residual over ALL solved cells (_stationarity_residual) and
    enforced with a RuntimeError.

    Arguments:
        vis_ratio: complex ndarray (Nbls, Nfreqs) of data/model ratios
            divided by g0_i * conj(g0_j); np.nan excludes a cell. Baselines
            that should not participate (e.g. intra-SNAP) must be omitted
            from the input, not flagged.
        ratio_wgts: float ndarray (Nbls, Nfreqs) of inverse-variance weights
            for vis_ratio; 0 excludes a cell
        ant_i_idx / ant_j_idx: int ndarrays of length Nbls indexing antennas
        nants: total number of antennas indexed
        refine_tol: convergence threshold on the largest per-(antenna,
            channel) update
        refine_maxiter: maximum number of Gauss-Newton rounds
        sync_tol: convergence threshold (radians) for the phase
            synchronization initialization (see _phase_sync_init)
        sync_maxiter: iteration cap for the phase synchronization
        verbose: print statements if True

    Returns:
        gains: complex ndarray (nants, Nfreqs); np.nan where unsolved
        meta: dict of per-channel (Nfreqs,) arrays, following redcal's
            solve_iteratively convention: 'iter' (int rounds each channel
            used before its early exit; 0 where flagged) and 'conv_crit'
            (max fixed-point residual over antennas in each channel;
            np.nan where flagged)

    Raises:
        ValueError: if the flagging pattern is not uniform (see
            _shared_channel_flags), or if any unflagged cell has zero or
            non-finite data/model ratio (flags or zero weights must cover
            all bad data)
        RuntimeError: if any channel has not converged after refine_maxiter
            rounds. Do not proceed with unconverged gains: partially
            converged channels retain memory of the initialization and can
            imprint coherent per-antenna spectral structure.
    '''
    nfreqs = vis_ratio.shape[1]

    # validate the flags and reduce them to one shared waterfall: whole
    # antennas out, plus channels flagged for everyone; guarantees that the
    # linear systems below are exactly nonsingular (no regularization)
    flagged_ants, chan_flags = _shared_channel_flags(ratio_wgts, ant_i_idx,
                                                     ant_j_idx, nants)
    good_chans = ~chan_flags
    sel_ants = np.where(~flagged_ants)[0]
    nsel = len(sel_ants)
    unflagged = ratio_wgts > 0

    # flatten the unflagged (baseline, channel) cells into 1D entry arrays,
    # each tagged with its kept-channel index and its two antenna positions
    ant_map = np.full(nants, -1)
    ant_map[sel_ants] = np.arange(nsel)
    good_chan_idx = np.where(good_chans)[0]
    nchans = len(good_chan_idx)
    bl_inds, chan_inds = np.where(unflagged[:, good_chan_idx])
    sel_i = ant_map[ant_i_idx[bl_inds]]
    sel_j = ant_map[ant_j_idx[bl_inds]]
    entry_wgts = ratio_wgts[:, good_chan_idx][bl_inds, chan_inds]
    entry_ratio = vis_ratio[:, good_chan_idx][bl_inds, chan_inds]

    # exact zeros are as fatal as NaNs here (log-amplitudes and unit phasors
    # are both undefined) and typically mean bad data stored as 0 without
    # accompanying flags, so fail loudly rather than diverge confusingly
    bad_entries = ~np.isfinite(entry_ratio) | (entry_ratio == 0)
    if bad_entries.any():
        raise ValueError(
            f'{int(bad_entries.sum())} unflagged (baseline, channel) cells '
            'have zero or non-finite data/model ratio. Flags or zero '
            'weights must cover all bad data.')

    gains = np.ones((nsel, nchans), dtype=complex)
    active_chans = np.ones(nchans, dtype=bool)   # channels still iterating
    chan_iters = np.zeros(nfreqs, dtype=int)     # rounds each channel used
    niter = 0
    while niter <= refine_maxiter and active_chans.any():
        # restrict the flattened entries to channels still iterating,
        # renumbering channels to positions within this round's batch
        active_idx = np.where(active_chans)[0]
        remap = np.full(nchans, -1)
        remap[active_idx] = np.arange(len(active_idx))
        in_active = active_chans[chan_inds]
        chan_pos = remap[chan_inds[in_active]]
        ei, ej = sel_i[in_active], sel_j[in_active]
        echan = chan_inds[in_active]
        wgts_here = entry_wgts[in_active]
        nactive = len(active_idx)

        # residual ratio: the data with the current gain model divided out;
        # exactly exp(eta_i + eta_j + 1j * (phi_i - phi_j)) for the updates
        # (eta, phi) that remain to be solved
        gi, gj = gains[ei, echan], gains[ej, echan]
        with np.errstate(invalid='ignore', divide='ignore'):
            resid_ratio = entry_ratio[in_active] / (gi * np.conj(gj))
        if not np.isfinite(resid_ratio).all():
            # inputs were validated above, so this can only be divergence
            raise RuntimeError('Per-channel gain refinement diverged: '
                               f'non-finite gains in round {niter}.')

        if niter == 0:
            # ---- initialization round ----
            round_wgts = wgts_here
            amp_mats, phase_mats = _build_normal_matrices(
                nactive, nsel, chan_pos, ei, ej, round_wgts)
            # log-amplitudes: log|resid_ratio| = eta_i + eta_j is exactly
            # linear in the log-amplitudes — no approximation needed
            amp_resids = np.log(np.abs(resid_ratio))
            # phases: wrap-immune eigenvector synchronization (see helper)
            delta_phase = _phase_sync_init(
                resid_ratio / np.abs(resid_ratio), round_wgts, chan_pos,
                ei, ej, nactive, nsel, sync_tol=sync_tol,
                sync_maxiter=sync_maxiter)
        else:
            # ---- Gauss-Newton round ----
            # propagating the data weights through the division by the
            # current gains multiplies each weight by (|g_i| |g_j|)^2
            round_wgts = wgts_here * (np.abs(gi) * np.abs(gj))**2
            amp_mats, phase_mats = _build_normal_matrices(
                nactive, nsel, chan_pos, ei, ej, round_wgts)
            # linearize: Im(resid_ratio) ~ phi_i - phi_j and
            # Re(resid_ratio) - 1 ~ eta_i + eta_j
            delta_phase = _solve_phase_updates(
                phase_mats, chan_pos, ei, ej, round_wgts,
                np.imag(resid_ratio), nactive, nsel)
            amp_resids = np.real(resid_ratio) - 1.0
        delta_logamp = _solve_logamp_updates(
            amp_mats, chan_pos, ei, ej, round_wgts, amp_resids, nactive,
            nsel)

        # remove the phase degeneracy by setting the update's mean over
        # participating antennas to 0 (same convention as
        # redcal.remove_degen_gains), then apply both updates
        # multiplicatively
        delta_phase -= delta_phase.mean(axis=1, keepdims=True)
        gains[:, active_idx] = gains[:, active_idx] \
            * np.exp(delta_logamp.T + 1j * delta_phase.T)

        # channels whose largest update is below tolerance are converged
        # and drop out of subsequent rounds; record the rounds each used
        # (redcal's solve_iteratively likewise reports per-channel counts)
        if niter > 0:
            max_update = np.hypot(delta_logamp, delta_phase).max(axis=1)
            dropped = active_idx[max_update < refine_tol]
            active_chans[dropped] = False
            chan_iters[good_chan_idx[dropped]] = niter + 1
        niter += 1
    converged = not active_chans.any()

    # embed the solution back into the full (antenna, channel) grid
    full_gains = np.full((nants, nfreqs), np.nan, dtype=complex)
    full_gains[np.ix_(sel_ants, good_chan_idx)] = gains

    conv_crit = _stationarity_residual(vis_ratio, ratio_wgts, full_gains,
                                       ant_i_idx, ant_j_idx)
    if not converged:
        raise RuntimeError(
            f'Per-channel gain refinement did not converge: '
            f'{int(active_chans.sum())} channels remain above '
            f'refine_tol={refine_tol} after {refine_maxiter} rounds '
            f'(max fixed-point residual {np.nanmax(conv_crit):.2e}). '
            'Do not proceed with unconverged gains.')
    return full_gains, {'iter': chan_iters, 'conv_crit': conv_crit}


def refine_gains(data_model_ratio, wgts, g0, ant_to_SNAP_dict=None,
                 refine_tol=1e-8, refine_maxiter=100,
                 sync_tol=1e-3, sync_maxiter=200, verbose=False):
    '''Solve for per-antenna, per-channel refined gains on cross baselines,
    given starting gains g0. The data/model ratio divided by g0_i g0_j^* should
    be ~1 up to per-antenna corrections beyond g0 — including any per-SNAP
    cross-correlation-only signal loss, which is the quantity this stage is
    designed to capture. If ant_to_SNAP_dict is given, only baselines between
    antennas on DIFFERENT SNAPs are used (intra-SNAP baselines are exempt from
    decoherence and would bias it toward zero).

    Arguments:
        data_model_ratio: DataContainer from build_data_model_ratio
        wgts: DataContainer of weights from build_data_model_ratio
        g0: dict mapping (ant, antpol) to starting gain waterfalls (e.g. the
            product of firstcal_gains and calibrate_abs_amp_from_autos gains)
        ant_to_SNAP_dict: optional dict mapping antenna numbers to SNAP IDs
            (any hashable). If given, EVERY antenna appearing in any cross
            baseline of data_model_ratio must be present, whether or not it
            ends up in the solve (raises ValueError otherwise); if None, all
            cross baselines participate.
        refine_tol, refine_maxiter, sync_tol, sync_maxiter:
            see _refine_gains_single_pol_time
        verbose: print statements if True

    Returns:
        refined_gains: dict mapping (ant, antpol) to complex (Ntimes, Nfreqs)
            gain waterfalls; np.nan where unsolved
        meta: dict with 'iter' and 'conv_crit', each a dict keyed by
            (time index, pol) of per-channel (Nfreqs,) arrays (see
            _refine_gains_single_pol_time)

    Raises:
        ValueError: if ant_to_SNAP_dict is given but missing antennas, if
            the flagging pattern is not uniform (see _shared_channel_flags),
            or if any unflagged cell has zero or non-finite data/model ratio
        RuntimeError: if the solve fails to converge for any (time, pol)
    '''
    bls = [bl for bl in data_model_ratio if bl[0] != bl[1]]
    all_ants = sorted({ant for bl in bls for ant in bl[:2]})
    if ant_to_SNAP_dict is not None:
        missing = [ant for ant in all_ants if ant not in ant_to_SNAP_dict]
        if len(missing) > 0:
            raise ValueError('ant_to_SNAP_dict is missing antennas that '
                             f'appear in the data: {missing}. All antennas '
                             'must be mapped to SNAPs if any are.')

    refined_gains = {}
    meta = {'iter': {}, 'conv_crit': {}}
    for pol in sorted({bl[2] for bl in bls}):
        antpol = utils.split_pol(pol)[0]
        bls_here = [bl for bl in bls if bl[2] == pol
                    and (bl[0], antpol) in g0 and (bl[1], antpol) in g0]
        if ant_to_SNAP_dict is not None:
            # intra-SNAP baselines are excluded from the solve by omission
            # (they are exempt from decoherence and would bias it low)
            n_cross = len(bls_here)
            bls_here = [bl for bl in bls_here
                        if ant_to_SNAP_dict[bl[0]] != ant_to_SNAP_dict[bl[1]]]
            utils.echo(f'{len(bls_here)} of {n_cross} {pol} baselines are '
                       'inter-SNAP and enter the solve', verbose=verbose)
        if len(bls_here) == 0:
            continue
        ratio, ant_i_idx, ant_j_idx, ants = _pack_baseline_arrays(
            data_model_ratio, bls_here)
        wgt_arr = np.asarray([wgts[bl] for bl in bls_here])
        g0_arr = np.asarray([g0[(ant, antpol)] for ant in ants])
        nants = len(ants)
        ntimes, nfreqs = ratio.shape[1], ratio.shape[2]

        for ant in ants:
            refined_gains[(ant, antpol)] = np.full((ntimes, nfreqs), np.nan,
                                                   dtype=complex)
        for tind in range(ntimes):
            gain_ij = g0_arr[ant_i_idx, tind] * np.conj(g0_arr[ant_j_idx, tind])
            with np.errstate(invalid='ignore', divide='ignore'):
                vis_ratio = np.where(
                    np.isfinite(gain_ij) & (np.abs(gain_ij) > 0),
                    ratio[:, tind] / gain_ij, np.nan)
            ratio_wgts = np.where(np.isfinite(vis_ratio),
                                  wgt_arr[:, tind] * np.abs(gain_ij)**2, 0)
            gains_here, meta_here = _refine_gains_single_pol_time(
                vis_ratio, ratio_wgts, ant_i_idx, ant_j_idx, nants,
                refine_tol=refine_tol,
                refine_maxiter=refine_maxiter, sync_tol=sync_tol,
                sync_maxiter=sync_maxiter, verbose=verbose)
            utils.echo(f't{tind} {pol}: {int(meta_here["iter"].max())} '
                       'rounds, max stationarity residual '
                       f'{np.nanmax(meta_here["conv_crit"]):.2e}',
                       verbose=verbose)
            for i, ant in enumerate(ants):
                refined_gains[(ant, antpol)][tind] = gains_here[i]
            for key in meta:
                meta[key][(tind, pol)] = meta_here[key]
    return refined_gains, meta


########################################################################
# Full staged calibration pipeline
########################################################################


def sky_calibrate(data, model, autos=None, data_flags=None, model_flags=None,
                  ant_flags=None, auto_flags=None, ant_to_SNAP_dict=None,
                  freqs=None, bls=None, dt=None, df=None,
                  refine_tol=1e-8, refine_maxiter=100, sync_tol=1e-3,
                  sync_maxiter=200, verbose=False):
    '''Run the full staged sky-model-based calibration: data/model ratio →
    firstcal delays and offsets → autocorrelation-based amplitudes → per-channel
    refinement on (inter-SNAP) cross baselines. Returns gains g = g0 * refined,
    where g0 (firstcal phases × auto-derived amplitudes) is constructed so that
    per-SNAP cross-correlation-only signal loss can land only in the refined
    gains.

    Arguments:
        data: DataContainer of visibilities, including autocorrelations unless
            autos is given separately
        model: DataContainer or RedDataContainer of model visibilities,
            time-aligned to data. Primarily intended to be LST-stacked,
            redundantly-averaged, and filtered visibilities from prior nights
            (see build_data_model_ratio).
        autos: optional DataContainer of autocorrelations (default: data)
        data_flags, model_flags: optional flag DataContainers
        ant_flags: optional dict mapping (ant, antpol) to flag waterfalls
        auto_flags: optional flags for the autocorrelations (see
            calibrate_abs_amp_from_autos)
        ant_to_SNAP_dict: optional dict mapping antennas to SNAP IDs; see
            refine_gains. The M&C lookup is the caller's responsibility.
        freqs: ndarray of frequencies in Hz (default: data.freqs)
        bls: optional list of baselines to calibrate with
        dt: integration time in seconds (default: inferred from autos)
        df: channel width in Hz (default: inferred from autos)
        refine_tol, refine_maxiter, sync_tol, sync_maxiter:
            see _refine_gains_single_pol_time
        verbose: print statements if True

    Returns:
        gains: dict mapping (ant, antpol) to complex (Ntimes, Nfreqs) gain
            waterfalls, g0 * refined_gains; np.nan where unsolved
        meta: dict of stage products and diagnostics: 'data_model_ratio',
            'wgts', 'dlys', 'offsets', 'abs_amp_gains', 'g0', 'refined_gains',
            and the refinement's 'iter' and 'conv_crit' dicts keyed by
            (time index, pol) of per-channel (Nfreqs,) arrays
    '''
    if freqs is None:
        freqs = data.freqs
    freqs = np.asarray(freqs)
    if autos is None:
        autos = data

    utils.echo('Building data/model ratio and weights...', verbose=verbose)
    data_model_ratio, wgts = build_data_model_ratio(
        data, model, autos=autos, data_flags=data_flags,
        model_flags=model_flags, ant_flags=ant_flags, bls=bls, dt=dt, df=df)

    utils.echo('Solving model-based firstcal delays and offsets...',
               verbose=verbose)
    dlys, offsets = model_based_firstcal(data_model_ratio, wgts, freqs,
                                         verbose=verbose)
    fc_gains = firstcal_gains(dlys, offsets, freqs)

    utils.echo('Calibrating amplitudes from autocorrelations...',
               verbose=verbose)
    abs_amp_gains = calibrate_abs_amp_from_autos(autos, auto_flags=auto_flags)
    g0 = merge_gains([abs_amp_gains, fc_gains])

    utils.echo('Refining gains per channel on cross baselines...',
               verbose=verbose)
    refined_gains, refine_meta = refine_gains(
        data_model_ratio, wgts, g0, ant_to_SNAP_dict=ant_to_SNAP_dict,
        refine_tol=refine_tol,
        refine_maxiter=refine_maxiter, sync_tol=sync_tol,
        sync_maxiter=sync_maxiter, verbose=verbose)

    # every antenna with starting gains must come through refinement — a gap
    # would otherwise be silently dropped by the key intersection in
    # merge_gains. Whole antennas should only ever be excluded by omission
    # from bls, so a gap here is an error, not a fallback.
    unrefined = sorted(set(g0) - set(refined_gains))
    if len(unrefined) > 0:
        raise ValueError('No baselines in the refinement solve for '
                         f'{unrefined} (e.g. all their partners are on the '
                         'same SNAP). Exclude these antennas from bls '
                         'instead.')
    gains = merge_gains([g0, refined_gains])
    meta = {'data_model_ratio': data_model_ratio, 'wgts': wgts, 'dlys': dlys,
            'offsets': offsets, 'abs_amp_gains': abs_amp_gains, 'g0': g0,
            'refined_gains': refined_gains, **refine_meta}
    return gains, meta


########################################################################
# Per-SNAP, per-X-engine-block decoherence estimation
########################################################################


def _block_design_matrix(nfreqs, nchans_per_block):
    '''Map channels to X-engine blocks. Returns (chan_to_block, design):
    chan_to_block[c] = c // nchans_per_block, and design is the
    (Nfreqs, Nblocks) 0/1 matrix with design[c, b] = 1 where
    chan_to_block[c] == b.'''
    chan_to_block = np.arange(nfreqs) // nchans_per_block
    nblocks = int(chan_to_block[-1]) + 1
    design = np.zeros((nfreqs, nblocks))
    design[np.arange(nfreqs), chan_to_block] = 1.0
    return chan_to_block, design


def _dpss_bases(freqs, band_slices, gain_smoothing_scale, eigenval_cutoff,
                verbose=False):
    '''One real DPSS basis per band (None for empty bands). Same convention
    as nucal.compute_spectral_filters, calling hera_filters.dspec directly
    so that this module does not inherit nucal's jax dependency chain. At
    the default eigenval_cutoff the retained transition modes make the
    effective smoothing scale exceed the nominal one; verbose prints it.'''
    bases = []
    for bi, band in enumerate(band_slices):
        if freqs[band].size == 0:
            bases.append(None)
            continue
        basis = np.asarray(dspec.dpss_operator(
            freqs[band], [0.0], [gain_smoothing_scale],
            eigenval_cutoff=[eigenval_cutoff])[0]).real
        bases.append(basis)
        bw = freqs[band].max() - freqs[band].min()
        utils.echo(f'band {bi}: {basis.shape[1]} DPSS modes '
                   f'(2BT = {2 * bw * gain_smoothing_scale:.1f}, effective '
                   f'scale ~{basis.shape[1] / (2 * bw) * 1e9:.0f} ns at '
                   f'eigenval_cutoff {eigenval_cutoff:g})', verbose=verbose)
    return bases


def _project_out_smooth(vals, wgts, band_slices, dpss_bases):
    '''Return the residual of vals after a weighted least-squares fit of the
    per-band DPSS bases — i.e. vals with everything spectrally smooth
    projected out, independently in each band.

    On unflagged channels this is identical to hera_filters'
    dspec.fourier_filter with mode='dpss_leastsq' (pinned by a unit test);
    it is inlined because each call here projects the spectrum AND all of
    the block-design columns through one shared factorization, inside the
    estimator's per-(SNAP, integration, FGLS-round) hot loop, and because
    the error propagation requires the data, normal matrices, and estimator
    kernel to be images of the exact same linear operator.

    Arguments:
        vals: (Nfreqs,) or (Nfreqs, k) ndarray to project
        wgts: (Nfreqs,) weights for the fit (0 excludes a channel)
        band_slices: list of slices into the frequency axis (bands are
            contiguous by definition)
        dpss_bases: list of per-band bases from _dpss_bases

    Returns: ndarray like vals; bands with no unflagged channels are
        set to 0.'''
    vals2 = vals.reshape(len(wgts), -1)
    resid = vals2.astype(float).copy()
    for band, basis in zip(band_slices, dpss_bases):
        if basis is None or not (wgts[band] > 0).any():
            resid[band] = 0
            continue
        sqrt_wgts = np.sqrt(wgts[band])
        coeffs, *_ = np.linalg.lstsq(basis * sqrt_wgts[:, None],
                                     vals2[band] * sqrt_wgts[:, None],
                                     rcond=None)
        resid[band] = vals2[band] - basis @ coeffs
    return resid.reshape(vals.shape)
