# -*- coding: utf-8 -*-
# Copyright 2026 the HERA Project
# Licensed under the MIT License

"""Staged sky-model-based calibration and per-SNAP decoherence estimation.

This module calibrates raw visibilities against a sky model in stages designed
so that any per-antenna, non-smooth amplitude effect that only appears on cross
correlations (e.g. per-SNAP signal loss in the correlator, a.k.a. "decoherence")
lands in exactly one place — the final per-channel refinement gains — where the
second half of the module then measures it:

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

Because of that staging, estimate_SNAP_decoherence can measure per-SNAP,
per-X-engine-block signal loss from the calibrated gains (from this
pipeline or any other algorithm whose gains retain the decoherence
signature, with matching ln|gain| inverse variances): each SNAP's
log-gain spectra are fit as a smooth per-antenna component plus a shared
nonnegative staircase on X-engine blocks, using a firm-threshold (MCP)
penalty so that blocks without significant evidence are exactly zero, with
errors derived from the residuals' own autocovariance (Newey-West-style
"HAC" errors; see estimate_SNAP_decoherence). One degeneracy to
keep in mind: within each band a suppression common to EVERY block is
indistinguishable from smooth structure, so estimates are relative to each
band's least-suppressed block (see the function docstring).

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
from . import flag_utils
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
        ant_i_idx: int ndarray of first-antenna indices into antnums, per baseline
        ant_j_idx: int ndarray of second-antenna indices into antnums, per baseline
        antnums: sorted list of antenna numbers appearing in bls
    '''
    antnums = sorted({antnum for bl in bls for antnum in bl[:2]})
    idx = {antnum: i for i, antnum in enumerate(antnums)}
    stacked = np.asarray([container[bl] for bl in bls])
    ant_i_idx = np.array([idx[bl[0]] for bl in bls], dtype=int)
    ant_j_idx = np.array([idx[bl[1]] for bl in bls], dtype=int)
    return stacked, ant_i_idx, ant_j_idx, antnums


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
                                              ant_j_idx, nants, mode='pinv'):
    '''Weighted least-squares solve of per-baseline differences for per-antenna
    values, i.e. vals[bl] ~ x[i] - x[j], with the mean over solved antennas
    fixed to 0 (an overall offset is unobservable in differences).

    Arguments:
        vals: ndarray of per-baseline measurements (e.g. delays tau_i - tau_j)
        wgts: ndarray of per-baseline weights (non-positive weights are ignored)
        ant_i_idx / ant_j_idx: int ndarrays indexing each baseline's antennas
        nants: total number of antennas indexed
        mode: linsolve.LinearSolver solve mode (default 'pinv', which gives
            the minimum-norm solution of the singular difference system)

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
        # the explicit mean subtraction after the solve removes the
        # degeneracy by fixing the mean to 0
        sol = linsolve.LinearSolver(ls_data, wgts=ls_wgts).solve(mode=mode)
        for i in range(nants):
            if f'x_{i}' in sol:
                x[i] = float(sol[f'x_{i}'])
        solved = np.isfinite(x)
        x[solved] -= x[solved].mean()
    return x


def model_based_firstcal(data_model_ratio, wgts, freqs, mode='pinv',
                         verbose=False):
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
        mode: linsolve.LinearSolver solve mode for the per-antenna solves
            (default 'pinv'; see _solve_per_antenna_weighted_least_squares)
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
        ratio, ant_i_idx, ant_j_idx, antnums = _pack_baseline_arrays(
            data_model_ratio, bls_here)
        wgt_arr = np.asarray([wgts[bl] for bl in bls_here])
        nants = len(antnums)
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
                nants, mode=mode)
            with np.errstate(invalid='ignore'):
                dly_phasor = np.exp(
                    -2j * np.pi * freqs[None, :]
                    * np.nan_to_num(ant_dlys[tind, ant_i_idx]
                                    - ant_dlys[tind, ant_j_idx])[:, None])
            resid_phases = np.angle(
                (wgtd_ratio[:, tind] * dly_phasor).sum(axis=1))
            ant_offsets[tind] = _solve_per_antenna_weighted_least_squares(
                resid_phases, solve_wgts[:, tind], ant_i_idx, ant_j_idx,
                nants, mode=mode)

        n_unsolved = int(np.sum(~np.isfinite(ant_dlys).any(axis=0)))
        if n_unsolved > 0:
            utils.echo(f'{n_unsolved} antennas in {pol} have no usable '
                       'cross-baseline data; setting delays/offsets to 0.',
                       verbose=verbose)
        antpol = utils.split_pol(pol)[0]
        for i, antnum in enumerate(antnums):
            dlys[(antnum, antpol)] = np.nan_to_num(ant_dlys[:, i])[:, None]
            offsets[(antnum, antpol)] = np.nan_to_num(ant_offsets[:, i])[:, None]
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


def _relative_chi2_gradient(vis_ratio, ratio_wgts, full_gains, ant_i_idx,
                            ant_j_idx):
    '''Compute the convergence criterion: the per-channel MAXIMUM of the
    relative gradient of chi^2 over all solved antennas. At the exact
    weighted least-squares optimum the gradient of chi^2 with respect to
    every gain vanishes — equivalently, every gain equals the weighted
    projection of the data onto the other antennas' gains,
    g_i = sum_j(w_ij * z_ij * g_j) / sum_j(w_ij * |g_j|^2) = U / D — so
    max |U/D - g| / |g|, the chi^2 gradient normalized by the local
    curvature D and gain scale, measures how far each cell is from the
    optimum, independently of the solver's own update sizes. Taking maxima
    (never a median or percentile) is deliberate: a median criterion can
    declare victory while an entire contiguous band remains unconverged.

    NOTE: this deliberately differs from redcal's conv_crit, which is the
    relative step between iterations. A step-size rule shows that the
    solver stopped moving (true distance ~ step / (1 - contraction rate)),
    not that the solution is near the optimum; the chi^2 gradient measures
    distance from the optimum directly and works for gains from any
    solver. The 'iter'/'conv_crit' meta keys are shared with redcal, but
    conv_crit's semantics differ as described here.

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
    exactly what _relative_chi2_gradient checks — independent of the
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
    same convention as redcal.remove_degen_gains. Convergence is verified
    via the maximum relative chi^2 gradient over ALL solved cells
    (_relative_chi2_gradient) and enforced with a RuntimeError.

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
            (max relative chi^2 gradient over antennas in each channel;
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

    conv_crit = _relative_chi2_gradient(vis_ratio, ratio_wgts, full_gains,
                                        ant_i_idx, ant_j_idx)
    if not converged:
        raise RuntimeError(
            f'Per-channel gain refinement did not converge: '
            f'{int(active_chans.sum())} channels remain above '
            f'refine_tol={refine_tol} after {refine_maxiter} rounds '
            f'(max relative chi^2 gradient {np.nanmax(conv_crit):.2e}). '
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
    all_antnums = sorted({antnum for bl in bls for antnum in bl[:2]})
    if ant_to_SNAP_dict is not None:
        missing = [antnum for antnum in all_antnums
                   if antnum not in ant_to_SNAP_dict]
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
        ratio, ant_i_idx, ant_j_idx, antnums = _pack_baseline_arrays(
            data_model_ratio, bls_here)
        wgt_arr = np.asarray([wgts[bl] for bl in bls_here])
        g0_arr = np.asarray([g0[(antnum, antpol)] for antnum in antnums])
        nants = len(antnums)
        ntimes, nfreqs = ratio.shape[1], ratio.shape[2]

        for antnum in antnums:
            refined_gains[(antnum, antpol)] = np.full((ntimes, nfreqs), np.nan,
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
                       'rounds, max relative chi^2 gradient '
                       f'{np.nanmax(meta_here["conv_crit"]):.2e}',
                       verbose=verbose)
            for i, antnum in enumerate(antnums):
                refined_gains[(antnum, antpol)][tind] = gains_here[i]
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
        if band is None or freqs[band].size == 0:
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
        set to 0.

    Raises: ValueError if a band has more than zero but no more unflagged
        channels than basis modes (an interpolatory fit with exactly zero
        residual — no information; such spectra must be flagged entirely
        or excluded upstream).'''
    vals2 = vals.reshape(len(wgts), -1)
    # channels outside every band slice (exterior flagged channels trimmed
    # by get_minimal_slices, or fully-flagged None bands) come back 0;
    # they carry zero weight in every downstream sum
    resid = np.zeros_like(vals2, dtype=float)
    for band, basis in zip(band_slices, dpss_bases):
        if band is None or basis is None:
            continue
        nunflagged = int((wgts[band] > 0).sum())
        if nunflagged == 0:
            continue
        # <= modes unflagged makes the fit interpolatory (zero residual,
        # no information): refuse rather than contribute silently
        if nunflagged <= basis.shape[1]:
            raise ValueError(
                f'A band has only {nunflagged} unflagged channels for '
                f'{basis.shape[1]} DPSS modes: the smooth fit would be '
                'interpolatory, leaving no residual information. Flag '
                'such spectra entirely or exclude them upstream.')
        # weighted least squares via the (Nmodes, Nmodes) normal equations
        # rather than an SVD of the full (Nchans, Nmodes) design — the same
        # strategy as hera_filters' 'dpss_solve'/'dpss_matrix' modes, ~5x
        # faster at these shapes. lstsq (rather than solve) tolerates a
        # numerically singular Gram matrix by returning minimum-norm
        # coefficients, which leave the residual on weighted channels
        # unaffected.
        wgt_band = wgts[band]
        gram = basis.T @ (basis * wgt_band[:, None])
        proj = basis.T @ (wgt_band[:, None] * vals2[band])
        coeffs, *_ = np.linalg.lstsq(gram, proj, rcond=None)
        resid[band] = vals2[band] - basis @ coeffs
    return resid.reshape(vals.shape)


def _mcp_penalized_nnls(normal_mat, rhs, zero_below, unbiased_above,
                        mask=None, start=None, tol=1e-10, maxiter=20000):
    '''Penalized nonnegative least squares solved by coordinate descent.

    Nonnegative least squares ("NNLS") is least squares with every fitted
    coefficient constrained to be >= 0
    (https://en.wikipedia.org/wiki/Non-negative_least_squares) —
    appropriate here because packet loss can only suppress gains. It is
    solved by cyclic coordinate descent: each coordinate in turn gets its
    closed-form single-coordinate optimum with all others held fixed,
    repeated until no coordinate moves.

    On top of the nonnegativity, each coordinate's update is mapped through
    a FIRM THRESHOLD: set to 0 below zero_below[b], linearly and partially
    shrunk between zero_below[b] and unbiased_above[b], and left exactly
    unpenalized above unbiased_above[b]. This is the coordinate-wise form
    of the minimax concave penalty ("MCP"; Zhang 2010, Annals of
    Statistics 38, 894, https://doi.org/10.1214/09-AOS729). Compared to
    the more familiar soft threshold (LASSO), which subtracts the full
    threshold from EVERY surviving coefficient and so biases large ones
    low, the firm threshold's penalty flattens out: coefficients well
    above the noise are recovered without shrinkage ("nearly unbiased"
    sparse estimation), while coefficients without significant evidence
    are still set exactly to 0. With zero_below = 0 and
    unbiased_above = inf this reduces to plain NNLS restricted to the
    coordinates in mask.

    Sign convention: the staircase enters the spectral model as -p (a
    suppression), so the coordinate update carries a leading minus; see
    estimate_SNAP_decoherence.

    Arguments:
        normal_mat: (Nblocks, Nblocks) normal matrix of the block system
        rhs: (Nblocks,) right-hand side
        zero_below: (Nblocks,) firm-threshold lower corners (>= 0)
        unbiased_above: (Nblocks,) upper corners; np.inf disables shrinkage
        mask: optional boolean array restricting which coordinates update
        start: optional starting point (updated coordinates only)
        tol: convergence threshold on the largest coordinate change
        maxiter: iteration cap

    Returns: (Nblocks,) nonnegative solution.

    Raises: RuntimeError if coordinate descent does not converge.'''
    fit = np.zeros(len(rhs)) if start is None else start.copy()
    updatable = np.diag(normal_mat) > 0
    if mask is not None:
        updatable &= mask
    upd_idx = np.where(updatable)[0]
    for _ in range(maxiter):
        dmax = 0.0
        for b in upd_idx:
            diag_b = normal_mat[b, b]
            # unconstrained single-coordinate optimum given all others
            pu = -(rhs[b] + normal_mat[b] @ fit - diag_b * fit[b]) / diag_b
            if pu <= zero_below[b]:
                new = 0.0
            elif np.isinf(unbiased_above[b]):
                new = pu - zero_below[b]
            elif (unbiased_above[b] <= zero_below[b]
                    or pu >= unbiased_above[b]):
                new = pu
            else:
                new = ((pu - zero_below[b]) * unbiased_above[b]
                       / (unbiased_above[b] - zero_below[b]))
            dmax = max(dmax, abs(new - fit[b]))
            fit[b] = new
        if dmax < tol:
            return fit
    raise RuntimeError('MCP coordinate descent did not converge in '
                       f'{maxiter} iterations (last max change {dmax:.2e}).')


def _SNAP_log_gain_spectra(ant_keys, tind, gains, logamp_wgts):
    '''ln|gain| spectra and weights for one SNAP at one integration: one
    (log_amp, wgts) pair per antenna-pol on the SNAP with any usable
    channels. NaN/zero-weight channels carry zero weight.'''
    spectra = []
    for key in ant_keys:
        gain = np.asarray(gains[key])[tind]
        with np.errstate(invalid='ignore', divide='ignore'):
            log_amp = np.log(np.abs(gain))
        wgt = np.where(np.isfinite(log_amp) & (logamp_wgts[key][tind] > 0),
                       logamp_wgts[key][tind], 0)
        if (wgt > 0).any():
            spectra.append((np.nan_to_num(log_amp), wgt))
    return spectra


def _fix_band_floors(block_vals, band_blocks):
    """If every covered block in a band is strictly positive, subtract the
    band's minimum (in place) so its least-suppressed block is exactly 0.
    Usually a no-op: the MCP solve's nonnegativity boundary already pins
    each band's minimum block to 0."""
    for in_band in band_blocks:
        if in_band and np.all(block_vals[in_band] > 0):
            block_vals[in_band] -= block_vals[in_band].min()


def estimate_SNAP_decoherence(gains, logamp_wgts, ant_to_SNAP_dict,
                              freqs, nchans_per_block=96,
                              gain_smoothing_scale=100e-9,
                              eigenval_cutoff=1e-12, detection_sigma=2.0,
                              full_sigma=3.0, band_split_freq=100e6,
                              min_block_coverage=0.05,
                              full_block_coverage=0.9, verbose=False):
    '''Estimate per-SNAP, per-X-engine-block signal loss ("decoherence")
    from the spectral structure of the refined gains.

    Model, per (SNAP, integration): the SNAP's antenna-pol log-gain spectra
    share
        ln|g_ap(nu)| = smooth_ap(nu) - log_suppression_b + n_ap(nu) + r(nu)
    where b indexes X-engine blocks (nchans_per_block channels each),
    log_suppression = -ln(1 - p) >= 0 for loss fraction p, n_ap is thermal
    noise with known variance 1 / logamp_wgts, and r is a stationary
    residual bandpass-error field (see Noise model below). The smooth
    per-antenna component (a DPSS basis per band) absorbs bandpass and
    calibration structure; the fitted staircase absorbs the
    block-discontinuous suppression that per-SNAP packet loss imprints on
    inter-SNAP cross-correlations.

    Fitting: nonnegative least squares on the smooth-projected block system
    (coefficients constrained >= 0, since packet loss can only suppress)
    with a firm-threshold minimax concave ("MCP") penalty INSIDE the fit —
    blocks with evidence below detection_sigma * sigma_b are set exactly
    to 0, partially shrunk up to full_sigma * sigma_b, and unbiased beyond.
    See _mcp_penalized_nnls for an explanation of both concepts and
    references.

    The smooth components and the staircase are fit SIMULTANEOUSLY, not
    alternately: for any given staircase the optimal smooth coefficients
    are a closed-form weighted projection, so they are profiled out
    analytically by building the block system from smooth-projected data
    and design columns. By the Frisch-Waugh-Lovell theorem
    (https://en.wikipedia.org/wiki/Frisch%E2%80%93Waugh%E2%80%93Lovell_theorem)
    this yields exactly the joint least-squares solution over (smooth,
    staircase) — and because the penalty acts only on the staircase
    coefficients, the equivalence carries over to the penalized fit. The
    only alternation in the algorithm is the feasible-GLS iteration
    between this joint fit and re-estimating the noise autocovariance
    (see Errors below).

    Noise model (estimated from the same data, alternating with the joint
    fit above): beyond the two deterministic components, each ln|gain|
    spectrum carries thermal noise with exactly known per-channel variance
    1 / logamp_wgts, PLUS a residual bandpass-error field that neither
    component captures (e.g. cable reflections, sky-model error). That
    residual field is modeled as a zero-mean, wide-sense stationary
    process within each band — the single statistical assumption — and is
    parameterized NONPARAMETRICALLY: its free parameters are the values of
    its autocovariance C(lag) at lags 0 through L = 2 * nchans_per_block
    channels (a window long enough to span correlations on the block scale
    itself); no functional form is assumed. Per band, C is measured from
    the empirical autocovariance of the weighted-mean residual of the
    current fit, with the known thermal variance subtracted at lag 0 so
    that only the EXCESS covariance remains, then tapered by the
    triangular Bartlett window, 1 - lag / (L + 1), whose implied
    covariance is positive semi-definite (no negative variance estimates).

    Errors: feasible generalized least squares, i.e. generalized least
    squares with the noise covariance estimated from the data themselves
    (https://en.wikipedia.org/wiki/Generalized_least_squares): sigma_b^2
    is the exact thermal part (from the normal-matrix diagonal) plus C
    propagated through the estimator's linear kernel — the
    heteroskedasticity-and-autocorrelation-consistent ("HAC") error
    construction of econometrics (Newey & West 1987,
    https://en.wikipedia.org/wiki/Newey%E2%80%93West_estimator), which
    keeps error bars honest when residuals are correlated from channel to
    channel. Each round re-measures C from the current (smooth, staircase)
    residual and refits the staircase with the updated thresholds,
    iterating until the detected support is stable.

    DEGENERACY (important): within each band, a suppression common to ALL
    covered blocks is indistinguishable from smooth structure, so the
    least-suppressed covered block is pinned to 0 — estimates are RELATIVE
    to each band's cleanest block, unlike diff-based estimates.

    Arguments:
        gains: dict mapping (ant, antpol) to complex (Ntimes, Nfreqs)
            waterfalls of TOTAL instrument gains, from any calibration
            algorithm — e.g. the gains returned by sky_calibrate, but
            nothing here assumes that provenance. The staircase is measured
            on ln|gains|, so the gains must retain the decoherence
            signature: gains derived only from autocorrelations or
            intra-SNAP baselines cannot show it.
        logamp_wgts: dict mapping (ant, antpol) to (Ntimes, Nfreqs)
            inverse-variance weights of ln|gains| (0 excludes a channel).
            EVERY key in gains must be present (ValueError otherwise).
            These calibrate sigma_b and therefore the detection thresholds,
            so they should reflect the actual error model of whatever
            algorithm produced the gains; for sky_calibrate gains, build
            them with log_gain_inverse_variance.
        ant_to_SNAP_dict: dict mapping antenna numbers to SNAP IDs. EVERY
            antenna appearing in gains must be present (ValueError
            otherwise).
        freqs: ndarray of frequencies in Hz
        nchans_per_block: channels per X-engine block; the block map is
            channel_index // nchans_per_block
        gain_smoothing_scale: DPSS half-width in seconds for the smooth
            component (same convention as nucal.compute_spectral_filters)
        eigenval_cutoff: DPSS eigenvalue cutoff (HERA's conventional 1e-12;
            note the effective smoothing scale then exceeds the nominal one)
        detection_sigma: firm-threshold lower corner in units of sigma_b;
            blocks below this evidence level are set exactly to 0
        full_sigma: firm-threshold upper corner; estimates are unbiased
            beyond this evidence level
        band_split_freq: frequency in Hz splitting the spectrum into the
            independently-fit low_band and high_band (default 100 MHz,
            inside the flagged FM band). Each band is trimmed to the
            minimal slice containing its not-always-flagged channels
            (flag_utils.get_minimal_slices), keeping exterior flagged
            channels from destabilizing the DPSS fits; a fully-flagged
            band is skipped.
        min_block_coverage: minimum unflagged fraction for a block to be fit
        full_block_coverage: blocks below this unflagged fraction (or at a
            band edge) are excluded from the noise-inflation diagnostic
        verbose: print statements if True

    Returns:
        decoherence: dict mapping SNAP ID to (Ntimes, Nblocks) ndarrays of
            the loss fraction p = 1 - exp(-log_suppression), np.nan where
            unfit — the physical product downstream corrections and
            flagging consume
        meta: dict of fit-domain products and diagnostics, all keyed by
            SNAP ID:
            'log_suppression': the fitted -ln(1 - p) >= 0 dicts themselves
                — the fit's native domain, where the MCP thresholds, the
                floor degeneracy, and the reported errors all live
            'log_suppression_refit': unpenalized refit on the detected
                active set (unbiased values for mapping/comparison)
            'log_suppression_sigma': HAC 1-sigma errors on active blocks
            'fgls_iterations': (Ntimes,) iterations to stable support
            'n_spectra_per_SNAP': (Ntimes,) contributing antenna-pol spectra
            'sigma_over_thermal': (Ntimes, Nbands) median noise inflation
                from the residual ACF over interior blocks
            plus 'covered_blocks' (Nblocks bool), 'edge_blocks' (sorted
            list), 'chan_to_block' (Nfreqs int), and 'band_slices' (list
            of slices into the frequency axis giving each band's trimmed
            extent, None for a fully-flagged band), shared across SNAPs

    Raises:
        ValueError: if ant_to_SNAP_dict is missing antennas in gains, or if
            logamp_wgts is missing keys in gains
    '''
    antnums = sorted({key[0] for key in gains})
    missing = [antnum for antnum in antnums
               if antnum not in ant_to_SNAP_dict]
    if len(missing) > 0:
        raise ValueError('ant_to_SNAP_dict is missing antennas that appear '
                         f'in gains: {missing}. All antennas must be '
                         'mapped to SNAPs.')
    missing_wgts = sorted(set(gains) - set(logamp_wgts))
    if len(missing_wgts) > 0:
        raise ValueError('logamp_wgts is missing keys that appear in '
                         f'gains: {missing_wgts}.')
    freqs = np.asarray(freqs)
    nfreqs = len(freqs)
    ntimes = np.asarray(next(iter(gains.values()))).shape[0]
    # bands are contiguous by definition, so they are carried as slices.
    # Each band is trimmed to the minimal slice containing its
    # not-always-flagged channels (flags here = zero weight for every
    # antenna), so that exterior flagged channels do not destabilize the
    # DPSS fits — same practice as elsewhere in the pipeline. A fully
    # flagged band comes back as None and is skipped throughout.
    flag_wf = np.ones((ntimes, nfreqs), dtype=bool)
    for key in gains:
        flag_wf &= ~(np.asarray(logamp_wgts[key]) > 0)
    _, band_slices = flag_utils.get_minimal_slices(
        flag_wf, freqs=freqs, freq_cuts=[band_split_freq])
    nbands = len(band_slices)
    chan_to_block, block_design = _block_design_matrix(nfreqs,
                                                       nchans_per_block)
    nblocks = block_design.shape[1]
    dpss_bases = _dpss_bases(freqs, band_slices, gain_smoothing_scale,
                             eigenval_cutoff, verbose=verbose)

    # block coverage: which blocks have enough unflagged channels to fit,
    # and which sit at band edges or are partially covered (excluded
    # from the noise-inflation diagnostic, where edge effects dominate)
    chan_ok = np.zeros(nfreqs, dtype=bool)
    for key in gains:
        chan_ok |= (np.asarray(logamp_wgts[key]) > 0).any(axis=0)
    coverage = np.array([(chan_ok & (chan_to_block == b)).sum()
                         / nchans_per_block for b in range(nblocks)])
    covered_blocks = coverage > min_block_coverage
    edge_blocks = set()
    band_blocks = []
    for band in band_slices:
        if band is None:
            band_blocks.append([])
            continue
        in_band = [b for b in range(nblocks) if covered_blocks[b]
                   and np.any(chan_to_block[band] == b)]
        band_blocks.append(in_band)
        if in_band:
            edge_blocks |= {in_band[0], in_band[-1]}
        edge_blocks |= {b for b in in_band
                        if coverage[b] < full_block_coverage}

    # HAC ("heteroskedasticity and autocorrelation consistent", the
    # Newey-West error construction — see the docstring) settings:
    # residual autocovariances out to a two-block lag window enter
    # sigma_b, tapered by the triangular Bartlett window 1 - lag/(L + 1),
    # which keeps the implied covariance positive semi-definite
    hac_nlags = 2 * nchans_per_block
    bartlett = 1 - np.arange(hac_nlags + 1) / (hac_nlags + 1)
    no_upper = np.full(nblocks, np.inf)

    SNAPs = sorted({ant_to_SNAP_dict[antnum] for antnum in antnums})
    SNAP_keys = {SNAP: [key for key in sorted(gains)
                        if ant_to_SNAP_dict[key[0]] == SNAP]
                 for SNAP in SNAPs}
    log_suppression = {s: np.full((ntimes, nblocks), np.nan) for s in SNAPs}
    refit_out = {s: np.full((ntimes, nblocks), np.nan) for s in SNAPs}
    sigma_out = {s: np.full((ntimes, nblocks), np.nan) for s in SNAPs}
    iters_out = {s: np.zeros(ntimes, dtype=int) for s in SNAPs}
    nspectra_out = {s: np.zeros(ntimes, dtype=int) for s in SNAPs}
    inflation_out = {s: np.full((ntimes, nbands), np.nan) for s in SNAPs}

    def _lag_covariances(a, b, nlags):
        '''Unnormalized lag cross-covariances sum_n a[n] * b[n + lag] for
        lag = 0..nlags — the ingredients of the HAC sums below. Returns an
        (nlags + 1,) ndarray.'''
        return np.array([(a[:len(a) - lag] * b[lag:]).sum()
                        for lag in range(nlags + 1)])

    for SNAP in SNAPs:
        utils.echo(f'Fitting SNAP {SNAP}...', verbose=verbose)
        for tind in range(ntimes):
            spectra = _SNAP_log_gain_spectra(SNAP_keys[SNAP], tind,
                                             gains, logamp_wgts)
            nspectra_out[SNAP][tind] = len(spectra)
            if len(spectra) == 0:
                continue

            # accumulate the block normal system on the smooth-projected
            # design: columns [log_amp, block_design] projected together so
            # each spectrum's own weights shape its projection
            normal_mat = np.zeros((nblocks, nblocks))
            rhs = np.zeros(nblocks)
            total_wgts = np.zeros(nfreqs)
            for log_amp, wgt in spectra:
                resid = _project_out_smooth(
                    np.column_stack([log_amp, block_design]), wgt,
                    band_slices, dpss_bases)
                normal_mat += resid[:, 1:].T @ (wgt[:, None] * resid[:, 1:])
                rhs += resid[:, 1:].T @ (wgt * resid[:, 0])
                total_wgts += wgt
            normal_diag = np.diag(normal_mat).copy()
            # a block is fit only if it has data AND meets the coverage
            # threshold (matching the min_block_coverage documentation);
            # sparse blocks would otherwise get under-inflated errors,
            # since their HAC sigma cannot be measured reliably
            ok_blocks = (normal_diag > 0) & covered_blocks
            if not ok_blocks.any():
                continue
            # estimator kernel: the weighted, smooth-projected block design
            # (residual ACF propagates through this into sigma_b)
            kernel = _project_out_smooth(block_design, total_wgts,
                                         band_slices, dpss_bases
                                         ) * total_wgts[:, None]

            # feasible-GLS loop: fit -> residual ACF -> sigma_b ->
            # thresholds -> refit, iterated until the detected support of
            # the staircase is stable
            fit = np.zeros(nblocks)
            prev_support = None
            acf_by_band = {}
            with np.errstate(invalid='ignore', divide='ignore'):
                variances = np.where(ok_blocks,
                                     1 / np.maximum(normal_diag, 1e-300),
                                     np.nan)
            for fgls_iter in range(8):
                staircase = block_design @ (-fit)
                resid_field = np.zeros(nfreqs)
                for log_amp, wgt in spectra:
                    smooth = (log_amp - staircase
                              - _project_out_smooth(log_amp - staircase,
                                                    wgt, band_slices,
                                                    dpss_bases))
                    resid_field += wgt * (log_amp - smooth - staircase)
                with np.errstate(invalid='ignore', divide='ignore'):
                    mean_resid = np.where(
                        total_wgts > 0,
                        resid_field / np.maximum(total_wgts, 1e-300), 0.0)
                for bi, band in enumerate(band_slices):
                    if band is None or not band_blocks[bi]:
                        continue
                    band_wgts = total_wgts[band]
                    band_ok = band_wgts > 0
                    if not band_ok.any():
                        continue
                    vals = np.where(band_ok, mean_resid[band], 0.0)
                    band_mask = band_ok.astype(float)
                    npairs = np.maximum(
                        _lag_covariances(band_mask, band_mask, hac_nlags), 1)
                    acf = _lag_covariances(vals, vals, hac_nlags) / npairs
                    tapered = bartlett * acf
                    # subtract the (exactly known) thermal part at lag 0 so
                    # only the excess bandpass-error covariance propagates
                    tapered[0] = max(0.0, acf[0]
                                     - (1 / band_wgts[band_ok]).mean())
                    acf_by_band[bi] = tapered
                    for b in band_blocks[bi]:
                        if not ok_blocks[b]:
                            continue
                        kern_b = (kernel[:, b] / normal_diag[b])[band]
                        auto = _lag_covariances(kern_b, kern_b, hac_nlags)
                        ripple = (tapered[0] * auto[0]
                                  + 2 * (tapered[1:] * auto[1:]).sum())
                        variances[b] = (1 / normal_diag[b]
                                        + max(0.0, ripple))
                sigmas = np.sqrt(variances)
                zero_below = np.where(ok_blocks, detection_sigma * sigmas,
                                      0.0)
                unbiased_above = np.where(ok_blocks, full_sigma * sigmas,
                                          0.0)
                fit = _mcp_penalized_nnls(normal_mat, rhs, zero_below,
                                          unbiased_above, mask=ok_blocks,
                                          start=fit)
                # support is deliberately pre-floor: the floor is a
                # reporting convention, not a model constraint
                support = fit > 0
                _fix_band_floors(fit, band_blocks)
                if (prev_support is not None
                        and np.array_equal(support, prev_support)):
                    break
                prev_support = support
            iters_out[SNAP][tind] = fgls_iter + 1
            for bi in range(nbands):
                interior = [b for b in band_blocks[bi]
                            if ok_blocks[b] and b not in edge_blocks]
                if interior:
                    inflation_out[SNAP][tind, bi] = np.median(
                        np.sqrt(variances[interior]
                                * normal_diag[interior]))

            # unpenalized refit on the detected support (unbiased values),
            # with the same per-band floor degeneracy fixing
            refit = _mcp_penalized_nnls(normal_mat, rhs, np.zeros(nblocks),
                                        no_upper, mask=support, start=fit)
            _fix_band_floors(refit, band_blocks)

            # HAC covariance of the refit on the active set: thermal
            # (inverse normal matrix) plus the residual ACF propagated
            # through pairs of estimator-kernel columns
            sigma_fit = np.full(nblocks, np.nan)
            active_idx = np.where(support)[0]
            if len(active_idx) > 0:
                inv_active = np.linalg.pinv(
                    normal_mat[np.ix_(active_idx, active_idx)])
                ripple_mat = np.zeros((len(active_idx), len(active_idx)))
                for bi, band in enumerate(band_slices):
                    tapered = acf_by_band.get(bi)
                    in_band_pos = [i for i, b in enumerate(active_idx)
                                   if b in set(band_blocks[bi])]
                    if tapered is None or not in_band_pos:
                        continue
                    for ai in in_band_pos:
                        for aj in in_band_pos:
                            if aj < ai:
                                continue
                            kern_a = kernel[:, active_idx[ai]][band]
                            kern_b = kernel[:, active_idx[aj]][band]
                            cov_ab = _lag_covariances(kern_a, kern_b,
                                                      hac_nlags)
                            cov_ba = _lag_covariances(kern_b, kern_a,
                                                      hac_nlags)
                            val = (tapered[0] * cov_ab[0]
                                   + (tapered[1:] * (cov_ab[1:]
                                                     + cov_ba[1:])).sum())
                            ripple_mat[ai, aj] = ripple_mat[aj, ai] = val
                full_cov = (inv_active
                            + inv_active @ ripple_mat @ inv_active)
                sigma_fit[active_idx] = np.sqrt(
                    np.maximum(np.diag(full_cov), 0))

            # blocks with no constraining data are unmeasured, not zero
            log_suppression[SNAP][tind] = np.where(ok_blocks, fit, np.nan)
            refit_out[SNAP][tind] = np.where(ok_blocks, refit, np.nan)
            sigma_out[SNAP][tind] = sigma_fit

    with np.errstate(invalid='ignore'):
        decoherence = {SNAP: 1 - np.exp(-log_suppression[SNAP])
                       for SNAP in SNAPs}
    meta = {'log_suppression': log_suppression,
            'log_suppression_refit': refit_out,
            'log_suppression_sigma': sigma_out,
            'fgls_iterations': iters_out,
            'n_spectra_per_SNAP': nspectra_out,
            'sigma_over_thermal': inflation_out,
            'covered_blocks': covered_blocks,
            'edge_blocks': sorted(edge_blocks),
            'chan_to_block': chan_to_block,
            'band_slices': band_slices}
    return decoherence, meta


def log_gain_inverse_variance(wgts, g0, refined_gains, ant_to_SNAP_dict):
    '''Per-antenna inverse variance of ln|gain| implied by this module's
    refinement solve — the thermal Fisher information the solve actually
    delivered about each antenna's log-amplitude, per (time, channel).
    For antenna a it evaluates to
        2 |refined_a|^2 * sum over partners j of
            wgts_aj |g0_a g0_j|^2 |refined_j|^2,
    summed over inter-SNAP baselines only.

    Derivation, term by term:
    1. The refinement fit vis_ratio ~ g_i * conj(g_j), where vis_ratio is
       the data/model ratio DIVIDED by g0_i * conj(g0_j). The ratio itself
       has inverse variance wgts (from build_data_model_ratio); dividing a
       measurement by a known constant divides its standard deviation by
       that constant's magnitude, so the divided quantity's inverse
       variance is wgts_ij |g0_i g0_j|^2 — the same effective weights the
       refinement applies internally.
    2. Perturbing g_i -> g_i * exp(eta_i) and linearizing,
       vis_ratio / (g_i conj(g_j)) ~ 1 + (eta_i + eta_j)
                                       + 1j * (phi_i - phi_j),
       so the REAL part carries the log-amplitude equation. The complex
       noise splits its variance evenly between real and imaginary parts,
       and the division by g_i * conj(g_j) scales it by 1/|g_i g_j|^2, so
       each baseline contributes information
       2 * wgts_ij |g0_i g0_j|^2 * |refined_i|^2 |refined_j|^2 about
       eta_i + eta_j (|refined| appears because the g0 division already
       happened). The explicit factor of 2 is the real-part-only factor:
       it is why this is TWICE the diagonal of the refinement's
       log-amplitude normal matrix — inside the solver a uniform factor
       cancels in the normal equations, but it does not cancel in
       variances.
    3. Summing over every baseline containing antenna a gives the DIAGONAL
       Fisher information: partners' gains are treated as known, which
       neglects the O(1/Nants) anti-correlations between antennas (the
       standard per-antenna approximation). This biases the predicted
       scatter slightly LOW (an ensemble test measures ~13% in sigma at 10
       antennas); the deficit is harmless downstream because the HAC noise
       model measures the total residual variance and subtracts only the
       thermal part claimed here, so anything missed re-enters sigma_b as
       measured excess.
    4. The exclusions mirror the solve: intra-SNAP baselines are skipped
       because the refinement excluded them — these weights must not claim
       information the solve never used. Antennas absent from
       refined_gains, non-finite g0 products, and NaN refined gains all
       contribute nothing.

    Note this is deliberately THERMAL-ONLY: correlated bandpass
    systematics are absent by design, because estimate_SNAP_decoherence
    measures that excess covariance empirically (the HAC noise model) and
    adds it on top. These weights need only set the thermal floor of
    sigma_b and the relative inverse-variance weighting of spectra within
    the staircase fit.

    This is the companion to estimate_SNAP_decoherence for gains produced
    by sky_calibrate: pass the result as its logamp_wgts argument. Gains
    from any other algorithm need their own ln|gain| inverse variances.

    Arguments:
        wgts: DataContainer of data/model-ratio weights (build_data_model_
            ratio convention), keyed by cross baselines, e.g. from
            sky_calibrate meta['wgts']
        g0: dict of starting gains keyed (ant, antpol), e.g. from
            sky_calibrate meta['g0']
        refined_gains: dict of refined gains keyed (ant, antpol), e.g. from
            sky_calibrate meta['refined_gains']
        ant_to_SNAP_dict: dict mapping antenna numbers to SNAP IDs

    Returns: dict mapping (ant, antpol) to (Ntimes, Nfreqs) inverse-variance
        waterfalls (0 where an antenna has no usable inter-SNAP data).'''
    inv_var = {key: np.zeros(np.asarray(refined_gains[key]).shape)
               for key in refined_gains}
    for bl in wgts:
        if bl[0] == bl[1]:
            continue
        if ant_to_SNAP_dict[bl[0]] == ant_to_SNAP_dict[bl[1]]:
            continue
        key_i, key_j = utils.split_bl(bl)
        if key_i not in refined_gains or key_j not in refined_gains:
            continue
        g0_ij = g0[key_i] * np.conj(g0[key_j])
        ref_i2 = np.abs(np.nan_to_num(refined_gains[key_i]))**2
        ref_j2 = np.abs(np.nan_to_num(refined_gains[key_j]))**2
        wgt_bl = np.where(np.isfinite(g0_ij),
                          np.asarray(wgts[bl]) * np.abs(g0_ij)**2, 0)
        inv_var[key_i] += 2 * wgt_bl * ref_j2 * ref_i2
        inv_var[key_j] += 2 * wgt_bl * ref_i2 * ref_j2
    return inv_var
