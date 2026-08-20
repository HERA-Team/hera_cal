# -*- coding: utf-8 -*-
# Copyright 2019 the HERA Project
# Licensed under the MIT License

"""Module for applying calibration solutions to visibility data, both in memory and on disk."""

import numpy as np
import argparse
import copy
import warnings
from . import io
from . import utils
from . import redcal
from . import noise
from . import datacontainer
import pyuvdata.utils as uvutils
from pyuvdata import UVData


def _check_polarization_consistency(data, gains):
    '''This fucntion raises an error if all the gain keys are cardinal but none of the data keys are cardinal
    (e/n rather than x/y), or vice versa. In the mixed case, or if one is empty, no errors are raised.'''
    if (len(data) > 0) and (len(gains) > 0):
        data_keys_cardinal = [utils._is_cardinal(bl[2]) for bl in data.keys()]
        gain_keys_cardinal = [utils._is_cardinal(ant[1]) for ant in gains.keys()]
        if np.all(data_keys_cardinal) and not np.any(gain_keys_cardinal):
            raise KeyError("All the data keys are cardinal (e.g. 'nn' or 'ee'), but none of the gain keys are.")
        elif np.all(gain_keys_cardinal) and not np.any(data_keys_cardinal):
            raise KeyError("All the gain keys are cardinal (e.g. 'Jnn' or 'Jee'), but none of the data keys are.")


def calibrate_redundant_solution(data, data_flags, new_gains, new_flags, all_reds,
                                 old_gains=None, old_flags=None, gain_convention='divide'):
    '''Update the calibration of a redundant visibility solution (or redundantly averaged visibilities).
    This function averages together all gain ratios (old/new) within a redundant group (which should
    ideally all be the same) to figure out the proper gain to apply/unapply to the visibilities. If all
    gain ratios are flagged for a given time/frequency within a redundant group, the data_flags are
    updated. Typical use is to use absolute/smooth_calibrated gains as new_gains, omnical gains as
    old_gains, and omnical visibility solutions as data. NOTE: BDA not supported; gain and data shapes must match.

    Arguments:
        data: DataContainer containing baseline-pol complex visibility data. This is modified in place.
        data_flags: DataContainer containing data flags. They are updated based on the flags of the
            calibration solutions.
        new_gains: Dictionary of complex calibration gains to apply with keys like (1,'Jnn')
        new_flags: Dictionary with keys like (1,'Jnn') of per-antenna boolean flags to update data_flags
            if either antenna in a visibility is flagged. Must have all keys in new_gains.
        all_reds: list of lists of redundant baseline tuples, e.g. (0,1,'nn'). Must be a superset of
            the reds used for producing cal
        old_gains: Dictionary of complex calibration gains to take out with keys like (1,'Jnn').
            Default of None implies means that the "old" gains are all 1s. Must be either None or
            have all the same keys as new_gains.
        old_flags: Dictionary with keys like (1,'Jnn') of per-antenna boolean flags to update data_flags
            if either antenna in a visibility is flagged. Default of None all old_gains are unflagged.
            Must be either None or have all the same keys as new_flags.
        gain_convention: str, either 'divide' or 'multiply'. 'divide' means V_obs = gi gj* V_true,
            'multiply' means V_true = gi gj* V_obs. Assumed to be the same for new_gains and old_gains.
    '''

    _check_polarization_consistency(data, new_gains)
    _check_polarization_consistency(data_flags, new_flags)
    exponent = {'divide': 1, 'multiply': -1}[gain_convention]
    if old_gains is None:
        old_gains = {ant: np.ones_like(new_gains[ant]) for ant in new_gains}
    if old_flags is None:
        old_flags = {ant: np.zeros_like(new_flags[ant]) for ant in new_flags}

    # assert that all antennas in new_gains are also in new_flags, old_gains, and old_flags
    assert np.all([ant in new_flags for ant in new_gains])
    assert np.all([ant in old_gains for ant in new_gains])
    assert np.all([ant in old_flags for ant in new_gains])

    for red in all_reds:
        # skip if there's nothing to calibrate
        if np.all([bl not in data for bl in red]):
            continue

        # Fill in missing antennas with flags
        for bl in red:
            for ant in utils.split_bl(bl):
                if ant not in new_gains:
                    new_gains[ant] = np.ones_like(list(new_gains.values())[0])
                    new_flags[ant] = np.ones_like(list(new_flags.values())[0])
                if ant not in old_gains:
                    old_gains[ant] = np.ones_like(list(old_gains.values())[0])
                    old_flags[ant] = np.ones_like(list(old_flags.values())[0])

        # Compute all gain ratios within a redundant baseline, ensuring autocorrelations say real
        gain_ratios = [old_gains[i, utils.split_pol(pol)[0]] * np.conj(old_gains[j, utils.split_pol(pol)[1]])
                       / new_gains[i, utils.split_pol(pol)[0]] / np.conj(new_gains[j, utils.split_pol(pol)[1]])
                       if not ((i == j) and (utils.split_pol(pol)[0] == utils.split_pol(pol)[1]))
                       else np.abs(old_gains[i, utils.split_pol(pol)[0]])**2 / np.abs(new_gains[i, utils.split_pol(pol)[0]])**2
                       for (i, j, pol) in red]

        # Set flagged values to np.nan for those gain rations
        for n, bl in enumerate(red):
            ant1, ant2 = utils.split_bl(bl)
            gain_ratios[n][new_flags[ant1] | new_flags[ant2] | old_flags[ant1] | old_flags[ant2]] = np.nan

        # Average gain ratios using np.nanmean
        avg_gains = np.nanmean(gain_ratios, axis=0)
        avg_flags = ~np.isfinite(avg_gains)
        avg_gains[avg_flags] = 1

        # Apply average gains ratios and update flags
        for bl in red:
            if bl in data:
                data_flags[bl] |= avg_flags
                data[bl] *= avg_gains**exponent


def correct_SNAP_decoherence_in_place(data, decoherence, ant_to_SNAP_dict,
                                      data_flags=None, nchans_per_block=96):
    '''Correct visibilities in place for measured per-SNAP, per-X-engine-block
    signal loss ("decoherence"). The correction is baseline-CLASS dependent,
    which is why it cannot be folded into per-antenna gains:

        * cross-correlations between antennas on DIFFERENT SNAPs are divided
          by (1 - p_i) * (1 - p_j), where p_i is the loss fraction of the
          SNAP that antenna i is on (looked up via ant_to_SNAP_dict, per
          time and per block) — under the stale-packet model only
          current-times-current products correlate, so the observed
          visibility is suppressed by exactly that product of
          coherence factors;
        * baselines within a single SNAP — including all autocorrelations
          and cross-polarized "autos" — are EXEMPT (both antennas ride the
          same packet stream, so stale-times-stale is still coherent) and
          are left untouched. Folding the correction into gains would
          over-correct these by roughly 2p.

    The decoherence is per-SNAP and therefore polarization-common: the same
    correction applies to all visibility polarizations of a baseline.

    CAVEAT inherited from estimate_SNAP_decoherence: within each band the
    measured p is RELATIVE to that band's least-suppressed covered block, so
    this correction removes the spectral structure decoherence imprints but
    not any band-common suppression shared by every block.

    NaN contract: np.nan in decoherence means "unmeasured". Unmeasured
    decoherence overlapping UNFLAGGED inter-SNAP data is an error — it
    would silently leave uncorrected suppression in data marked good.
    Unmeasured decoherence over flagged data is fine: the unmeasured SNAP
    contributes no correction there (coherence factor treated as 1), though
    a measured partner SNAP's correction still applies; flags are never
    modified by this function. With
    data_flags=None, nothing is treated as flagged, making this check its
    strictest.

    Arguments:
        data: DataContainer of visibilities, modified in place
        decoherence: dict mapping SNAP ID to (Ntimes, Nblocks) ndarrays of
            loss fraction p (np.nan where unmeasured), as returned by
            skycal.estimate_SNAP_decoherence. Ntimes must match the data
            (no BDA up/downsampling support).
        ant_to_SNAP_dict: dict mapping antenna numbers to SNAP IDs. EVERY
            antenna appearing in data must be present, and every SNAP with
            an antenna in the data must appear in decoherence (ValueError
            otherwise).
        data_flags: optional DataContainer of boolean flag waterfalls with
            the same keys as data, used ONLY to evaluate the NaN contract
            (never modified). Default None treats all data as unflagged.
        nchans_per_block: channels per X-engine block; the block map is
            channel_index // nchans_per_block, matching
            estimate_SNAP_decoherence. Nfreqs / nchans_per_block (rounded
            up) must match the decoherence arrays\' Nblocks.

    Raises:
        ValueError: if ant_to_SNAP_dict is missing antennas in data; if a
            SNAP in the data is missing from decoherence; if decoherence
            shapes
            do not match the data; or if unmeasured (NaN) decoherence
            overlaps unflagged inter-SNAP data.
    '''
    antnums_in_data = sorted({antnum for bl in data.keys()
                              for antnum in bl[:2]})
    missing = [antnum for antnum in antnums_in_data
               if antnum not in ant_to_SNAP_dict]
    if len(missing) > 0:
        raise ValueError('ant_to_SNAP_dict is missing antennas that appear '
                         f'in the data: {missing}. All antennas must be '
                         'mapped to SNAPs.')

    # expand each needed SNAP\'s (Ntimes, Nblocks) p into an
    # (Ntimes, Nfreqs) coherence-factor waterfall 1 - p via the block map
    ntimes, nfreqs = data[next(iter(data.keys()))].shape
    chan_to_block = np.arange(nfreqs) // nchans_per_block
    nblocks = int(chan_to_block[-1]) + 1
    SNAPs_in_data = sorted({ant_to_SNAP_dict[antnum]
                            for antnum in antnums_in_data})
    missing_SNAPs = [SNAP for SNAP in SNAPs_in_data
                     if SNAP not in decoherence]
    if len(missing_SNAPs) > 0:
        raise ValueError('decoherence is missing SNAPs that appear in the '
                         f'data: {missing_SNAPs}.')
    coherence_factor, unmeasured = {}, {}
    for SNAP in SNAPs_in_data:
        p = np.asarray(decoherence[SNAP])
        if p.shape != (ntimes, nblocks):
            raise ValueError(f'decoherence[{SNAP!r}] has shape {p.shape} '
                             f'but the data implies ({ntimes}, {nblocks}) '
                             f'with nchans_per_block={nchans_per_block}.')
        coherence_factor[SNAP] = 1 - np.nan_to_num(p)[:, chan_to_block]
        unmeasured[SNAP] = np.isnan(p)[:, chan_to_block]

    # NaN contract: unmeasured decoherence must not overlap unflagged
    # inter-SNAP data
    for bl in data.keys():
        i, j, pol = bl
        if ant_to_SNAP_dict[i] == ant_to_SNAP_dict[j]:
            continue
        unm = (unmeasured[ant_to_SNAP_dict[i]]
               | unmeasured[ant_to_SNAP_dict[j]])
        if data_flags is not None:
            unm = unm & ~data_flags[bl]
        if unm.any():
            tinds, chans = np.nonzero(unm)
            raise ValueError('Unmeasured (NaN) decoherence overlaps '
                             f'unflagged data, e.g. on {bl} at time index '
                             f'{tinds[0]}, block {chan_to_block[chans[0]]}. '
                             'Flag that data or restrict to measured '
                             'blocks before correcting.')

    # apply the correction to inter-SNAP cross-correlations only
    for bl in data.keys():
        i, j, pol = bl
        if ant_to_SNAP_dict[i] == ant_to_SNAP_dict[j]:
            continue
        data[bl] /= (coherence_factor[ant_to_SNAP_dict[i]]
                     * coherence_factor[ant_to_SNAP_dict[j]])


def calibrate_and_red_avg(data, gains, reds, ant_flags=None, ex_ants=None, data_flags=None,
                          snap_decoherence=None, dt=None, df=None, compute_chisq=True,
                          effective_nsamples=True):
    '''Calibrate visibilities and redundantly average them with inverse-variance noise
    weights, one group at a time (a full-size calibrated copy of the data is never
    materialized). data must include co-polarized autocorrelations: they set each
    cross-correlation's noise variance, sigma^2 = A_i * A_j / (dt * df), and their
    per-polarization averages over unflagged antennas (binary weights, since
    inverse-variance weighting would bias this noise-prediction statistic low) are
    always included in the results. Cross-polarized autocorrelations are averaged the
    same way if their groups are listed, though they are never used for noise weights.
    What is averaged is governed entirely by reds.

    By default the returned nsamples are EFFECTIVE nsamples, defined so that the
    standard predictor sigma^2 = Abar_i * Abar_j / (dt * df * nsamples), evaluated with
    the averaged autocorrelations returned here, is exact for every averaged product:
    Abar_i * Abar_j * sum(w) / (dt * df) for cross groups (above the count, since the
    average leans on quieter-than-typical pairs) and n^2 * Abar_i * Abar_j / (the sum
    over antennas of their co-polarized auto products) for the autocorrelations.
    Effective nsamples is spectrally smooth (auto structure common to all antennas
    cancels), so it can be sensibly inpainted across flagging gaps.

    If snap_decoherence is given, gains are first cleaned of the fitted suppression
    staircase (SNAPDecoherence.correct_gains; autocorrelations and intra-SNAP baselines
    are exempt) and inter-SNAP cross-correlations instead get the exact correction
    (correct_SNAP_decoherence_in_place), with unmeasured (np.nan) blocks flagged. The
    stored antenna -> SNAP mapping is used throughout and must cover every antenna in
    gains.

    Chi^2 (co-polarized only): each group's mean is the only fit parameter, so a
    participating baseline's weighted scatter about it has expectation 1 - w / sum(w)
    per pixel. Antennas in ex_ants never enter the averages, but those with usable data
    still get chi^2 against the good-antenna group means, attributed only to themselves,
    with expectation 1 + w / sum(w_good).

    Arguments:
        data: DataContainer of visibilities, including co-polarized autocorrelations
            (ValueError otherwise). Baselines lacking gains or autos are omitted.
        gains: dict mapping (ant, antpol) e.g. (0, 'Jee') to (Ntimes, Nfreqs) complex
            gain waterfalls. Nonfinite gains are treated as flagged.
        reds: list of lists of redundant groups (antpairpol tuples) to average, e.g.
            from redcal.get_reds(antpos, pols=pols, include_autos=True). Their
            polarizations govern what is averaged: include 'en'/'ne' groups to average
            cross-polarized cross-correlations (calibrated by e.g. g_Jee * conj(g_Jnn)).
            MUST include a co-polarized autocorrelation group for every polarization
            used (ValueError otherwise). reds is passed to the returned
            RedDataContainers verbatim, so every listed baseline -- including flagged
            or excluded antennas' -- resolves to its group's average.
        ant_flags: optional dict mapping (ant, antpol) to boolean flag waterfalls.
        ex_ants: optional iterable of (ant, antpol) tuples excluded from all averages.
        data_flags: optional DataContainer of boolean flag waterfalls with the same keys
            as data. Flagged cells get zero weight in averages, chi^2, and effective
            nsamples. Flags on an autocorrelation affect only its average, never the
            noise weights of cross-correlations using that antenna -- flagging an
            antenna everywhere is ant_flags' job. Unnecessary in most current HERA
            analyses, where flags are carried per-antenna (or array-wide) and belong
            in ant_flags.
        snap_decoherence: optional io.SNAPDecoherence, e.g. from
            SNAPDecoherence.from_estimate. Default None: no decoherence handling.
        dt: integration time in seconds. Default None infers from data's times.
        df: channel width in Hz. Default None infers from data's freqs.
        compute_chisq: if True (default), accumulate DoF-normalized chi^2.
        effective_nsamples: if True (default), return effective nsamples; else counts.

    Returns:
        red_avg_data: RedDataContainer of weighted group averages, keyed by each group's
            first contributing baseline and including the averaged autocorrelations
        red_avg_flags: RedDataContainer, True where a group has no unflagged members
        red_avg_nsamples: RedDataContainer of effective nsamples (or counts)
        meta: {'chisq_per_ant': (ant, antpol) -> chi^2 waterfalls (np.nan where nothing
            was accumulated), 'total_chisq': antpol -> per-polarization totals} if
            compute_chisq, else {}
    '''
    ant_flags = ({} if ant_flags is None else ant_flags)
    ex_ants = set([] if ex_ants is None else ex_ants)
    if dt is None:
        dt = noise.infer_dt(data.times_by_bl, next(iter(data))) * 24.0 * 3600.0
    if df is None:
        df = np.median(np.ediff1d(data.freqs))
    if not any(utils.join_bl(ant, ant) in data for ant in gains):
        raise ValueError('data must include co-polarized autocorrelations: they set the noise '
                         'weights and the averaged autocorrelations that anchor effective nsamples.')
    listed_auto_antpols = {utils.split_pol(red[0][2])[0] for red in reds if red[0][0] == red[0][1]
                           and utils.split_pol(red[0][2])[0] == utils.split_pol(red[0][2])[1]}
    missing_antpols = ({antpol for red in reds for antpol in utils.split_pol(red[0][2])}
                       - listed_auto_antpols)
    if len(missing_antpols) > 0:
        raise ValueError('reds must include a co-polarized autocorrelation group for every '
                         f'polarization it uses (e.g. via redcal.get_reds with include_autos=True), '
                         f'but is missing {sorted(missing_antpols)}.')

    # per-antenna flags (excluded antennas are handled by membership, not flags). if decoherence
    # is given, correct the staircase out of the gains and precompute per-SNAP suppression and
    # unmeasured-block waterfalls
    gain_flags = {ant: (~np.isfinite(g) | ant_flags.get(ant, False)) for ant, g in gains.items()}
    finite_gains = {ant: np.where(np.isfinite(g), g, 1) for ant, g in gains.items()}
    if snap_decoherence is not None:
        finite_gains = snap_decoherence.correct_gains(finite_gains)
        ant_to_SNAP = snap_decoherence.ant_to_SNAP_dict
        nchans_per_block = snap_decoherence.block_freqs.shape[1]
        log_supp = {SNAP: np.repeat(np.nan_to_num(ls), nchans_per_block, axis=1)
                    for SNAP, ls in snap_decoherence._log_suppression.items()}
        unmeasured = {SNAP: np.repeat(np.isnan(p), nchans_per_block, axis=1)
                      for SNAP, p in snap_decoherence.decoherence.items()}

        def _is_inter_SNAP(bl):
            return ant_to_SNAP.get(bl[0]) != ant_to_SNAP.get(bl[1])

    # calibrated autocorrelations (small: per-antenna) set every baseline's noise variance
    cal_autos = {}
    for ant in finite_gains:
        auto_bl = utils.join_bl(ant, ant)
        if auto_bl in data:
            with np.errstate(divide='ignore', invalid='ignore'):
                cal_autos[auto_bl] = np.abs(data[auto_bl]) / np.abs(finite_gains[ant])**2
    cal_autos = datacontainer.DataContainer(cal_autos)

    def _bl_flags(bl):
        '''Both antennas' gain_flags, plus data_flags and unmeasured decoherence blocks
        for inter-SNAP baselines.'''
        ant_i, ant_j = utils.split_bl(bl)
        flags = gain_flags[ant_i] | gain_flags[ant_j]
        if data_flags is not None:
            flags = flags | data_flags[bl]
        if snap_decoherence is not None and _is_inter_SNAP(bl):
            # .get defaults: SNAPs without stored results (possible for excluded antennas) add no flags
            flags = flags | unmeasured.get(ant_to_SNAP.get(bl[0]), False) | unmeasured.get(ant_to_SNAP.get(bl[1]), False)
        return flags

    def _noise_wgts(bl, flags):
        '''Inverse noise variance from the calibrated autos, zeroed where flagged.'''
        with np.errstate(all='ignore'):
            sigma2 = noise.predict_noise_variance_from_autos(bl, cal_autos, dt=dt, df=df)
            return np.where(flags | ~(sigma2 > 0), 0, 1 / np.where(sigma2 > 0, sigma2, 1))

    def _usable_bl(bl, exclusion_chisq=False):
        '''True if bl has the gains and co-polarized autos it needs and the right number of
        excluded antennas: none for the averages, exactly one for the excluded-antenna chi^2.'''
        ant_i, ant_j = utils.split_bl(bl)
        if ant_i not in finite_gains or ant_j not in finite_gains:
            return False
        if (ant_i in ex_ants) + (ant_j in ex_ants) != (1 if exclusion_chisq else 0):
            return False
        return utils.join_bl(ant_i, ant_i) in cal_autos and utils.join_bl(ant_j, ant_j) in cal_autos

    # average reds' autocorrelation groups (co-polarized first, then cross-polarized, whose
    # effective nsamples need the co-polarized averages) with binary weights: every listed
    # antenna's auto key resolves to the average via the returned RedDataContainers, but
    # only usable antennas contribute
    red_avg_data, red_avg_flags, red_avg_nsamples = {}, {}, {}
    avg_autos = {}
    auto_reds = sorted((red for red in reds if red[0][0] == red[0][1]),
                       key=lambda red: (utils.split_pol(red[0][2])[0] != utils.split_pol(red[0][2])[1],
                                        red[0][2]))
    for red in auto_reds:
        pol = red[0][2]
        members = [bl for bl in red if bl in data and _usable_bl(bl)]
        if len(members) == 0:
            continue
        cal_group = datacontainer.DataContainer({bl: data[bl].copy() for bl in members})
        with np.errstate(divide='ignore', invalid='ignore'):
            calibrate_in_place(cal_group, finite_gains)
        binary = [(~_bl_flags(bl)).astype(float) for bl in members]
        wgt_sum = np.sum(binary, axis=0)
        with np.errstate(all='ignore'):
            avg = np.sum([b * cal_group[bl] for b, bl in zip(binary, members)], axis=0) \
                  / np.where(wgt_sum > 0, wgt_sum, 1)
        avg = np.where(wgt_sum > 0, avg, 0)
        key = members[0]
        red_avg_data[key] = avg
        red_avg_flags[key] = wgt_sum == 0
        antpol_i, antpol_j = utils.split_pol(pol)
        if antpol_i == antpol_j:
            avg_autos[antpol_i] = np.abs(avg)
        if effective_nsamples:
            # uniform weighting of per-antenna auto noise: Var = sum(P_k) / (n^2 dt df) with P_k
            # each antenna's co-polarized auto product, so n_eff = n^2 Abar_i Abar_j / sum(P_k)
            product_sum = np.sum([b * np.abs(cal_autos[utils.join_bl((bl[0], antpol_i), (bl[0], antpol_i))]
                                             * cal_autos[utils.join_bl((bl[0], antpol_j), (bl[0], antpol_j))])
                                  for b, bl in zip(binary, members)], axis=0)
            with np.errstate(all='ignore'):
                red_avg_nsamples[key] = np.where(product_sum > 0,
                                                 wgt_sum**2 * avg_autos[antpol_i] * avg_autos[antpol_j]
                                                 / np.where(product_sum > 0, product_sum, 1), 0)
        else:
            red_avg_nsamples[key] = wgt_sum

    chisq_num, chisq_dof, total_num, total_dof = {}, {}, {}, {}
    for red in reds:
        if red[0][0] == red[0][1]:
            continue  # autocorrelations are always averaged separately, above
        # members with usable gains and autocorrelations and at least one unflagged cell
        group_wgts, group_flags = {}, {}
        for bl in red:
            if bl in data and _usable_bl(bl):
                flags = _bl_flags(bl)
                wgt = _noise_wgts(bl, flags)
                if np.any(wgt > 0):
                    group_wgts[bl], group_flags[bl] = wgt, flags
        group_bls = list(group_wgts)
        if len(group_bls) == 0:
            continue

        # calibrate this group's raw data (a full-size calibrated copy is never materialized)
        group_data = datacontainer.DataContainer({bl: data[bl].copy() for bl in group_bls})
        with np.errstate(divide='ignore', invalid='ignore'):
            calibrate_in_place(group_data, finite_gains)
        if snap_decoherence is not None:
            # exact class-aware correction: inter-SNAP crosses divided by (1 - p_i)(1 - p_j)
            correct_SNAP_decoherence_in_place(group_data, snap_decoherence.decoherence, ant_to_SNAP,
                                              data_flags=datacontainer.DataContainer(group_flags),
                                              nchans_per_block=nchans_per_block)

        # inverse-variance weighted average of this group
        wgt_sum = np.sum([group_wgts[bl] for bl in group_bls], axis=0)
        with np.errstate(all='ignore'):
            avg = np.sum([group_wgts[bl] * group_data[bl] for bl in group_bls], axis=0) \
                  / np.where(wgt_sum > 0, wgt_sum, 1)
        avg = np.where(wgt_sum > 0, avg, 0)
        key = group_bls[0]
        red_avg_data[key] = avg
        red_avg_flags[key] = wgt_sum == 0
        if effective_nsamples:
            antpol_i, antpol_j = utils.split_pol(key[2])
            with np.errstate(all='ignore'):
                red_avg_nsamples[key] = avg_autos[antpol_i] * avg_autos[antpol_j] * wgt_sum / (dt * df)
        else:
            red_avg_nsamples[key] = np.sum([group_wgts[bl] > 0 for bl in group_bls], axis=0).astype(float)

        # accumulate redundant-baseline chi^2 over co-polarized cross-correlations.
        # NOTE: utils.chisq / redcal.normalized_chisq are deliberately not used here: their DoF
        # normalization (redcal.predict_chisq_per_ant) assumes omnical, where gains are fit from
        # redundancy. Here only each group's mean is fit, so the expected chi^2 is accumulated
        # directly: 1 - w / sum(w) per participating baseline (and 1 + w / sum(w) for excluded
        # antennas below, whose baselines did not participate in the mean).
        if compute_chisq and utils.split_pol(red[0][2])[0] == utils.split_pol(red[0][2])[1]:
            antpol = utils.split_bl(key)[0][1]
            for bl in group_bls:
                with np.errstate(all='ignore'):
                    z2 = np.where(group_wgts[bl] > 0, group_wgts[bl] * np.abs(group_data[bl] - avg)**2, 0)
                    dof = np.where(group_wgts[bl] > 0, 1 - group_wgts[bl] / np.where(wgt_sum > 0, wgt_sum, 1), 0)
                for ant in utils.split_bl(bl):
                    chisq_num[ant] = chisq_num.get(ant, 0) + z2
                    chisq_dof[ant] = chisq_dof.get(ant, 0) + dof
                total_num[antpol] = total_num.get(antpol, 0) + z2
                total_dof[antpol] = total_dof.get(antpol, 0) + dof

            # excluded antennas with usable data: chi^2 against the good-antenna group mean,
            # attributed only to the excluded antenna (never to its partners or the totals)
            if len(ex_ants) > 0 and np.any(wgt_sum > 0):
                excl_wgts = {}
                for bl in red:
                    if bl in data and bl not in group_wgts and _usable_bl(bl, exclusion_chisq=True):
                        wgt = _noise_wgts(bl, _bl_flags(bl))
                        if np.any(wgt > 0):
                            excl_wgts[bl] = wgt
                if len(excl_wgts) > 0:
                    excl_data = datacontainer.DataContainer({bl: data[bl].copy() for bl in excl_wgts})
                    with np.errstate(divide='ignore', invalid='ignore'):
                        calibrate_in_place(excl_data, finite_gains)
                    for bl in excl_wgts:
                        ant_i, ant_j = utils.split_bl(bl)
                        if snap_decoherence is not None and _is_inter_SNAP(bl):
                            # the exact correction, applied inline since an excluded antenna's
                            # SNAP may lack stored results (.get defaults to no correction)
                            excl_data[bl] *= np.exp(log_supp.get(ant_to_SNAP.get(bl[0]), 0)
                                                    + log_supp.get(ant_to_SNAP.get(bl[1]), 0))
                        usable = (excl_wgts[bl] > 0) & (wgt_sum > 0)
                        with np.errstate(all='ignore'):
                            z2 = np.where(usable, excl_wgts[bl] * np.abs(excl_data[bl] - avg)**2, 0)
                            expectation = np.where(usable, 1 + excl_wgts[bl] / np.where(wgt_sum > 0, wgt_sum, 1), 0)
                        excluded_ant = (ant_i if ant_i in ex_ants else ant_j)
                        chisq_num[excluded_ant] = chisq_num.get(excluded_ant, 0) + z2
                        chisq_dof[excluded_ant] = chisq_dof.get(excluded_ant, 0) + expectation

    meta = {}
    if compute_chisq:
        with np.errstate(all='ignore'):
            meta['chisq_per_ant'] = {ant: np.where(chisq_dof[ant] > 0,
                                                   chisq_num[ant] / np.where(chisq_dof[ant] > 0, chisq_dof[ant], 1),
                                                   np.nan) for ant in chisq_num}
            meta['total_chisq'] = {antpol: np.where(total_dof[antpol] > 0,
                                                    total_num[antpol] / np.where(total_dof[antpol] > 0, total_dof[antpol], 1),
                                                    np.nan) for antpol in total_num}
    return (datacontainer.RedDataContainer(red_avg_data, reds=reds),
            datacontainer.RedDataContainer(red_avg_flags, reds=reds),
            datacontainer.RedDataContainer(red_avg_nsamples, reds=reds),
            meta)


def build_gains_by_cadences(data, gains, cal_flags=None, flags_are_wgts=False):
    ''' Builds dictionaries that map gains to the various cadences in potentially BDA data.
        As necessary, will upsample gains/flags by duplication and downsample gains/flags by
        (weighted) averaging. When downsampling, flags are ORed and weights are averaged.
        Assumes that the all cadences in the data are a power-of-two multiple of the slowest cadence.

    Arguments:
        data: DataContainer containing baseline-pol complex visibility data. Only used
            to figure out the various waterfall shapes.
        gains: Dictionary mapping antenna tuples to complex gains to upsample/downsample as needed.
        cal_flags: Dictionary mapping antenna tuples to boolean flags (or float weights).
        flags_are_wgts: if True, treat data_flags as weights where 0s represent flags and
            non-zero weights are unflagged data.

    Returns:
        gains_by_Nt: dictionary mapping numbers of integration to gain dictionaries
        cal_flags_by_Nt: dictionary mapping numbers of integration to flag/weight dictionaries.
            If cal_flags is None, this will be None as well.
    '''
    # get all cadences (unique shapes of the time dimension in the data)
    data_Nts = sorted(list(set([wf.shape[0] for wf in data.values()])))

    # Warn the user if the data doesn't conform to the expectation that all BDA is by a power of 2
    for Nt in data_Nts:
        power_of_2 = np.log(Nt / np.min(data_Nts)) / np.log(2)
        if not np.isclose(power_of_2, np.round(power_of_2)):
            warnings.warn(f'Data with {Nt} integrations is inconsistent with BDA by powers of 2 '
                          f'when the slowest cadence has {np.min(data_Nts)} integrations.')

    # initialize results dictionaries, handling the case where there are None and/or empty dicts
    # and also the case where gains/flags are scalars, which then get recast as 2D arrays
    if gains == {}:
        gains_by_Nt = {Nt: {} for Nt in data_Nts}
    else:
        if np.isscalar(list(gains.values())[0]):
            gains_by_Nt = {1: {ant: np.array([[gain]]) for ant, gain in gains.items()}}
        else:
            gains_by_Nt = {list(gains.values())[0].shape[0]: gains}
    cal_flags_by_Nt = None
    if cal_flags is not None:
        if cal_flags == {}:
            cal_flags_by_Nt = {Nt: {} for Nt in data_Nts}
        else:
            if np.isscalar(list(cal_flags.values())[0]):
                cal_flags_by_Nt = {1: {ant: np.array([[cf]]) for ant, cf in cal_flags.items()}}
            else:
                cal_flags_by_Nt = {list(cal_flags.values())[0].shape[0]: cal_flags}

    # Handle the case where gains/flags have a single integration (and are thus trivially broadcastable)
    if 1 in gains_by_Nt:
        for Nt in data_Nts:
            gains_by_Nt[Nt] = gains_by_Nt[1]
    if cal_flags_by_Nt is not None and 1 in cal_flags_by_Nt:
        for Nt in data_Nts:
            cal_flags_by_Nt[Nt] = cal_flags_by_Nt[1]

    # If necessary, upsample gains (and flags) by repeating them
    while True:
        max_gain_Nt = np.max(list(gains_by_Nt.keys()))
        if max_gain_Nt >= np.max(list(data_Nts)):
            break
        gains_by_Nt[max_gain_Nt * 2] = {ant: gains_by_Nt[max_gain_Nt][ant].repeat(2, axis=0)
                                        for ant in gains_by_Nt[max_gain_Nt]}
        if cal_flags_by_Nt is not None:
            cal_flags_by_Nt[max_gain_Nt * 2] = {ant: cal_flags_by_Nt[max_gain_Nt][ant].repeat(2, axis=0)
                                                for ant in cal_flags_by_Nt[max_gain_Nt]}

    # If necessary, downsample gains (and flags) by (flag-weigted) averaging (ORing) them
    while True:
        min_gain_Nt = np.min(list(gains_by_Nt.keys()))
        if min_gain_Nt <= np.min(list(data_Nts)):
            break
        gains_by_Nt[min_gain_Nt // 2] = {}
        if cal_flags_by_Nt is not None:
            cal_flags_by_Nt[min_gain_Nt // 2] = {}
        for ant, gain in gains_by_Nt[min_gain_Nt].items():
            # break gains and flags into even and odd times to average together
            even_gains = gain[0::2, :]
            odd_gains = gain[1::2, :]
            if cal_flags_by_Nt is not None:
                # use flags/weights to perform a weighted average
                even_flags = cal_flags_by_Nt[min_gain_Nt][ant][0::2, :]
                odd_flags = cal_flags_by_Nt[min_gain_Nt][ant][1::2, :]
                if flags_are_wgts:
                    weights = [even_flags, odd_flags]
                    # average weights
                    cal_flags_by_Nt[min_gain_Nt // 2][ant] = (even_flags + odd_flags) / 2
                else:
                    weights = [(~even_flags).astype(float), (~odd_flags).astype(float)]
                    # OR flags
                    cal_flags_by_Nt[min_gain_Nt // 2][ant] = even_flags | odd_flags
                # average with mask array to robustly handle case where weights sum to 0
                gains_by_Nt[min_gain_Nt // 2][ant] = np.ma.average([even_gains, odd_gains], axis=0, weights=weights).data
            else:
                # just do a straight average
                gains_by_Nt[min_gain_Nt // 2][ant] = np.average([even_gains, odd_gains], axis=0)

    # Warn if there cadences in the data that are missing that still aren't in gains_by_Nt
    for Nt in data_Nts:
        if Nt not in gains_by_Nt:
            warnings.warn(f'Data with {Nt} integrations cannot be calibrated with any of gain cadences: {list(gains_by_Nt.keys())}')

    return gains_by_Nt, cal_flags_by_Nt


def calibrate_in_place(data, new_gains, data_flags=None, cal_flags=None, old_gains=None,
                       gain_convention='divide', flags_are_wgts=False):
    '''Update data and data_flags in place, taking out old calibration solutions, putting in new calibration
    solutions, and updating flags from those calibration solutions. Previously flagged data is modified, but
    left flagged. Missing antennas from either the new gains, the cal_flags, or (if it's not None) the old
    gains are automatically flagged in the data's visibilities that involves those antennas. Data and gain
    shapes should always match in the frequency direction. Can apply Ntimes=1 gains by broadcasting. Can
    also up/downsample gains with Ntimes differing from those in the data by a power of 2, which is useful
    when the data is BDA and has Ntimes of multiple different powers of 2.

    Arguments:
        data: DataContainer containing baseline-pol complex visibility data. This is modified in place.
        new_gains: Dictionary of complex calibration gains to apply with keys like (1,'Jnn')
        data_flags: DataContainer containing data flags. This is modified in place if its not None.
        cal_flags: Dictionary with keys like (1,'Jnn') of per-antenna boolean flags to update data_flags
            if either antenna in a visibility is flagged. Any missing antennas are assumed to be totally
            flagged, so leaving this as None will result in input data_flags becoming totally flagged.
        old_gains: Dictionary of complex calibration gains to take out with keys like (1,'Jnn').
            Default of None implies that the data is raw (i.e. uncalibrated).
        gain_convention: str, either 'divide' or 'multiply'. 'divide' means V_obs = gi gj* V_true,
            'multiply' means V_true = gi gj* V_obs. Assumed to be the same for new_gains and old_gains.
        flags_are_weights: bool, if True, treat data_flags as weights where 0s represent flags and
            non-zero weights are unflagged data.
    '''

    _check_polarization_consistency(data, new_gains)
    exponent = {'divide': 1, 'multiply': -1}[gain_convention]

    # build dictionary of all necessary gain shapes to account for calibration of BDA data
    new_gains_by_Nt, cal_flags_by_Nt = build_gains_by_cadences(data, new_gains, cal_flags=cal_flags, flags_are_wgts=flags_are_wgts)
    if old_gains is not None:
        old_gains_by_Nt, _ = build_gains_by_cadences(data, old_gains)

    # loop over baselines in data
    for (i, j, pol) in data.keys():

        ap1, ap2 = utils.split_pol(pol)
        flag_all = False

        # get relevant shaped gains for this data waterfall
        Nt = data[(i, j, pol)].shape[0]
        try:
            new_gains_here = new_gains_by_Nt[Nt]
        except KeyError:
            raise ValueError(f'new_gains with {list(new_gains.values())[0].shape[0]} integrations are incompatible with data with {Nt} integrations.')
        cal_flags_here = None
        if cal_flags_by_Nt is not None:
            try:
                cal_flags_here = cal_flags_by_Nt[Nt]
            except KeyError:
                raise ValueError(f'cal_flags with {list(cal_flags.values())[0].shape[0]} integrations are incompatible with data with {Nt} integrations.')
        old_gains_here = None
        if old_gains is not None:
            try:
                old_gains_here = old_gains_by_Nt[Nt]
            except KeyError:
                raise ValueError(f'old_gains with {list(old_gains.values())[0].shape[0]} integrations are incompatible with data with {Nt} integrations.')

        # handle autocorrelations separately to keep them real
        if (i == j) & (ap1 == ap2):
            try:
                data[(i, j, pol)] /= (np.abs(new_gains_here[(i, ap1)])**2)**exponent
            except KeyError:
                flag_all = True
            if old_gains is not None:
                try:
                    data[(i, j, pol)] *= (np.abs(old_gains_here[(i, ap1)])**2)**exponent
                except KeyError:
                    flag_all = True
        else:
            # apply new gains for antennas i and j. If either is missing, flag the whole baseline
            try:
                data[(i, j, pol)] /= (new_gains_here[(i, ap1)])**exponent
            except KeyError:
                flag_all = True
            try:
                data[(i, j, pol)] /= np.conj(new_gains_here[(j, ap2)])**exponent
            except KeyError:
                flag_all = True
            # unapply old gains for antennas i and j. If either is missing, flag the whole baseline
            if old_gains is not None:
                try:
                    data[(i, j, pol)] *= (old_gains_here[(i, ap1)])**exponent
                except KeyError:
                    flag_all = True
                try:
                    data[(i, j, pol)] *= np.conj(old_gains_here[(j, ap2)])**exponent
                except KeyError:
                    flag_all = True

        if data_flags is not None:
            if cal_flags is None:
                # when data_flags is provided but cal_flags is not, flag everything
                flag_all = True
            else:
                # update data_flags in the case where flags are weights, flag all if cal_flags are missing
                if flags_are_wgts:
                    try:
                        data_flags[(i, j, pol)] *= (~cal_flags_here[(i, ap1)]).astype(float)
                        data_flags[(i, j, pol)] *= (~cal_flags_here[(j, ap2)]).astype(float)
                    except KeyError:
                        flag_all = True
                # update data_flags in the case where flags are booleans, flag all if cal_flags are missing
                else:
                    try:
                        data_flags[(i, j, pol)] += cal_flags_here[(i, ap1)]
                        data_flags[(i, j, pol)] += cal_flags_here[(j, ap2)]
                    except KeyError:
                        flag_all = True

            # if the flag object is given, update it for this baseline to be totally flagged
            if flag_all:
                if flags_are_wgts:
                    data_flags[(i, j, pol)] = np.zeros_like(data[(i, j, pol)], dtype=float)
                else:
                    data_flags[(i, j, pol)] = np.ones_like(data[(i, j, pol)], dtype=bool)


def apply_cal(data_infilename, data_outfilename, new_calibration, old_calibration=None, flag_file=None,
              flag_filetype='h5', a_priori_flags_yaml=None, flag_nchan_low=0, flag_nchan_high=0, filetype_in='uvh5', filetype_out='uvh5',
              nbl_per_load=None, gain_convention='divide', upsample=False, downsample=False, redundant_solution=False, bl_error_tol=1.0,
              add_to_history='', clobber=False, redundant_average=False, redundant_weights=None,
              freq_atol=1., redundant_groups=1, dont_red_average_flagged_data=False, spw_range=None,
              exclude_from_redundant_mode="data", vis_units=None, **kwargs):
    '''Update the calibration solution and flags on the data, writing to a new file. Takes out old calibration
    and puts in new calibration solution, including its flags. Also enables appending to history.

    Arguments:
        data_infilename: filename of the data to be calibrated.
        data_outfilename: filename of the resultant data file with the new calibration and flags.
        new_calibration: filename of the calfits file (or a list of filenames) for the calibration
            to be applied, along with its new flags (if any).
        old_calibration: filename of the calfits file (or a list of filenames) for the calibration
            to be unapplied. Default None means that the input data is raw (i.e. uncalibrated).
        flag_file: optional path to file containing flags to be ORed with flags in input data. Must have
            the same shape as the data.
        flag_filetype: filetype of flag_file to pass into io.load_flags. Either 'h5' (default) or legacy 'npz'.
        a_priori_flags_yaml : path to YAML with antenna frequency and time flags in the YAML.
            Flags are combined with ant_metrics's xants and ex_ants. If any
            polarization is flagged for an antenna, all polarizations are flagged.
            see hera_qm.metrics_io.read_a_priori_chan_flags (for freq flag format),
            hera_qm.metrics_io.read_a_priori_int_flags (for time flag format),
            hera_qm.metrics_io.read_a_priori_ant_flags (for antenna flag format).
        flag_nchan_low: integer number of channels at the low frequency end of the band to always flag (default 0)
        flag_nchan_high: integer number of channels at the high frequency end of the band to always flag (default 0)
        filetype_in: type of data infile. Supports 'miriad', 'uvfits', and 'uvh5'.
        filetype_out: type of data outfile. Supports 'miriad', 'uvfits', and 'uvh5'.
        nbl_per_load: maximum number of baselines to load at once. Default (None) is to load the whole file at once.
            Enables partial reading and writing, but only for uvh5 to uvh5.
            nbl_per_load is only supported if filetype_in is .uvh5.
        gain_convention: str, either 'divide' or 'multiply'. 'divide' means V_obs = gi gj* V_true,
            'multiply' means V_true = gi gj* V_obs. Assumed to be the same for new_gains and old_gains.
        upsample: if True, upsample baseline-dependent-averaged data file to the highest temporal resolution
        downsample: if True, downsample baseline-dependent-averaged data file to the lowest temporal resolution
        redundant_solution: If True, average gain ratios in redundant groups to recalibrate e.g. redcal solutions.
            NOTE: BDA data is not supported in this mode. Gain shapes must be made to match data samples using upsample/downsample.
        bl_error_tol: the largest allowable difference between baselines in a redundant group
            (in the same units as antpos). Normally, this is up to 4x the largest antenna position error.
        add_to_history: appends a string to the history of the output file. This will preceed combined histories
            of flag_file (if applicable), new_calibration and, old_calibration (if applicable).
        clobber: if True, overwrites existing file at outfilename
        redundant_average : bool, optional
            If True, redundantly average calibrated data and save to <data_outfilename>.red_avg.<filetype_out>
        redundant_weights : datacontainer, optional.
            Datacontainer containing weights to use in redundant averaging.
            only used if redundant_average is True.
            Default is None. If None is passed, then nsamples are used as the redundant weights.
        tol_factor: float, optional
            Float specifying the tolerance (as a fraction of channel width) within which cal frequencies must be matched in calibration solution to apply.
        redundant_groups : int, optional.
            Integer specifying how many different subsets of each redundant group to write to an independent file.
            If more then one redundant subgroup is specified, then output files will have label .uvh5 -> .n.uvh5
            redundant_groups>1 not supported with partial I/O yet.
        dont_red_average_flagged_data : bool, optional.
            If True, baselines within a redundant group with all pols flagged do not count towards the number of baselines
            in that group above the number of groups to output. This lets us throw away groups that in principal have greater
            then the minimum number of baselines to allow for a split into different output groups but could result in one of
            the subgroups being entirely flagged. This option is only used when redundant_groups > 1.
            Not supported for partial I/O.
        spw_range : 2-tuple specifying range of channels to select and redundantly average.
        exclude_from_redundant_mode: str, optional
            specify whether to use entirely flagged data, 'data', or ex_ants from an external yaml file 'yaml' to determine
            baselines to exclude from redundant average.
        vis_units : str, optional
            string specifying units of calibrated visibility. Overrides gain_scale in calibration file.
            Default is None -> calibration gain_scale is used to set vis_units in calibrated file.
        kwargs: dictionary mapping updated UVData attributes to their new values.
            See pyuvdata.UVData documentation for more info.
    '''
    # UPDATE CAL FLAGS WITH EX_ANTS INSTEAD OF FILTERING BASELINES.
    # optionally load external flags
    if exclude_from_redundant_mode not in ['yaml', 'data']:
        raise ValueError("exclude_from_redundant_mode must be 'yaml' or 'data'.")
    if flag_file is not None:
        ext_flags, flag_meta = io.load_flags(flag_file, filetype=flag_filetype, return_meta=True)
        add_to_history += '\nFLAGS_HISTORY: ' + str(flag_meta['history']) + '\n'

    # load new calibration solution
    hc = io.HERACal(new_calibration)
    new_gains, new_flags, _, _ = hc.read()
    if a_priori_flags_yaml is not None:
        from hera_qm.utils import apply_yaml_flags
        from hera_qm.metrics_io import read_a_priori_ant_flags
        # flag hc
        hc = apply_yaml_flags(hc, a_priori_flags_yaml,
                              ant_indices_only=True)
        # and rebuild data containers.
        new_gains, new_flags, _, _ = hc.build_calcontainers()
        ex_ants = read_a_priori_ant_flags(a_priori_flags_yaml, ant_indices_only=True)
    else:
        ex_ants = None
    add_to_history += '\nNEW_CALFITS_HISTORY: ' + hc.history + '\n'

    # load old calibration solution
    if old_calibration is not None:
        old_hc = io.HERACal(old_calibration)
        old_hc.read()
        # determine frequencies to load in old_hc that are close to hc
        freqs_to_load = []
        for f in old_hc.freqs:
            # set atol to be 1/10th of a channel
            if np.any(np.isclose(hc.freqs, f, rtol=0., atol=freq_atol)):
                freqs_to_load.append(f)
        if spw_range is not None:
            freqs_to_load = freqs_to_load[spw_range[0]:spw_range[1]]
        old_hc.select(frequencies=np.asarray(freqs_to_load))  # match up frequencies with hc.freqs
        old_gains, old_flags, _, _ = old_hc.build_calcontainers()
        add_to_history += '\nOLD_CALFITS_HISTORY: ' + old_hc.history + '\n'
    else:
        old_gains, old_flags = None, None
    hd = io.HERAData(data_infilename, filetype=filetype_in, upsample=upsample, downsample=downsample)
    if spw_range is None:
        spw_range = (0, hd.Nfreqs)
    else:
        if filetype_in != 'uvh5':
            raise NotImplementedError("spw only implemented for uvh5 files.")
    if filetype_in == 'uvh5':
        freqs_to_load = []
        for f in hd.freqs[spw_range[0]:spw_range[1]]:
            if np.any(np.isclose(hc.freqs, f, rtol=0., atol=freq_atol)):
                freqs_to_load.append(f)
    else:
        freqs_to_load = None
    # reselect cals to match hd freqs_to_load
    if freqs_to_load is not None:
        calfreqs = []
        calfreqsold = []
        for f in hc.freqs:
            if np.any(np.isclose(freqs_to_load, f)):
                calfreqs.append(f)
            if old_calibration is not None and np.any(np.isclose(old_hc.freqs, f)):
                calfreqsold.append(f)
        hc.select(frequencies=calfreqs)
        new_gains, new_flags, _, _ = hc.build_calcontainers()
        if old_calibration is not None:
            old_hc.select(frequencies=calfreqsold)
            old_gains, old_flags, _, _ = old_hc.build_calcontainers()

    add_to_history = utils.history_string(add_to_history)
    no_red_weights = redundant_weights is None
    if nbl_per_load is not None:
        if not ((filetype_in == 'uvh5') and (filetype_out == 'uvh5')):
            raise NotImplementedError('Partial writing is not implemented for non-uvh5 I/O.')
        if not redundant_groups == 1:
            raise NotImplementedError("Splitting redundant groups into subgroups is not yet implemented for partial I/O!")
        if dont_red_average_flagged_data:
            raise NotImplementedError("Completely skipping flagged data in redundantly averaged data not implemented for partial I/O!")
        for attribute, value in kwargs.items():
            hd.__setattr__(attribute, value)
        if redundant_average or redundant_solution:
            all_reds = redcal.get_reds(hd.data_antpos, pols=hd.pols, bl_error_tol=bl_error_tol, include_autos=True)
        else:
            all_reds = []
        if redundant_average:
            # initialize a redunantly averaged HERAData on disk
            # first copy the original HERAData
            all_red_antpairs = [[bl[:2] for bl in grp] for grp in all_reds if grp[-1][-1] == hd.pols[0]]
            hd_red = io.HERAData(data_infilename, upsample=upsample, downsample=downsample)
            # go through all redundant groups and remove the groups that do not
            # have baselines in the data. Each group is still labeled by the
            # first baseline of each group regardless if that baseline is in
            # the data file.
            reds_data = redcal.filter_reds(all_reds, bls=hd.bls)
            reds_data_bls = []
            for grp in reds_data:
                reds_data_bls.append(grp[0])
            # couldn't get a system working where we just read in the outputs one at a time.
            # so unfortunately, we have to load one baseline per redundant group.
            hd_red.read(bls=reds_data_bls, frequencies=freqs_to_load)

        # consider calucate reds here instead and pass in (to avoid computing it multiple times)
        # I'll look into generators and whether the reds calc is being repeated.
        for data, data_flags, data_nsamples in hd.iterate_over_bls(Nbls=nbl_per_load, chunk_by_redundant_group=redundant_average,
                                                                   reds=all_reds, frequencies=freqs_to_load):
            for bl in data_flags.keys():
                # apply band edge flags
                data_flags[bl][:, 0:flag_nchan_low] = True
                data_flags[bl][:, data_flags[bl].shape[1] - flag_nchan_high:] = True
                # apply external flags
                if flag_file is not None:
                    data_flags[bl] = np.logical_or(data_flags[bl], ext_flags[bl])
            if redundant_solution:
                calibrate_redundant_solution(data, data_flags, new_gains, new_flags, all_reds, old_gains=old_gains,
                                             old_flags=old_flags, gain_convention=gain_convention)
            else:
                calibrate_in_place(data, new_gains, data_flags=data_flags, cal_flags=new_flags,
                                   old_gains=old_gains, gain_convention=gain_convention)
            hd.update(data=data, flags=data_flags)

            if redundant_average:
                # by default, weight by nsamples (but not flags). This prevents spectral structure from being introduced
                # and also allows us to compute redundant averaged vis in flagged channels (in case flags are spurious).
                if no_red_weights:
                    redundant_weights = copy.deepcopy(data_nsamples)
                    for bl in data_flags:
                        if exclude_from_redundant_mode == 'data':
                            if np.all(data_flags[bl]):
                                redundant_weights[bl][:] = 0.
                        elif exclude_from_redundant_mode == 'yaml' and ex_ants is not None:
                            if bl[0] in ex_ants or bl[1] in ex_ants:
                                redundant_weights[bl][:] = 0.
                # redundantly average
                utils.red_average(data=data, flags=data_flags, nsamples=data_nsamples,
                                  reds=all_red_antpairs, wgts=redundant_weights, inplace=True,
                                  propagate_flags=True)
                # update redundant data. Don't partial write.
                hd_red.update(nsamples=data_nsamples, flags=data_flags, data=data)
            else:
                if vis_units is None:
                    if hasattr(hc, 'gain_scale') and hc.gain_scale is not None:
                        if hd.vis_units is not None and hc.gain_scale.lower() != "uncalib" and hd.vis_units.lower() != hc.gain_scale.lower():
                            warnings.warn(f"Replacing original data vis_units of {hd.vis_units}"
                                          f" with calibration vis_units of {hc.gain_scale}", RuntimeWarning)
                        vis_units = hc.gain_scale
                    else:
                        vis_units = hd.vis_units
                    # partial write works for no redundant averaging.
                hd.partial_write(data_outfilename, inplace=True, clobber=clobber, add_to_history=add_to_history, vis_units=vis_units, **kwargs)

        if redundant_average:
            # if we did redundant averaging, just write the redundant dataset out in the end at once.
            if hasattr(hc, 'gain_scale') and hc.gain_scale is not None:
                if hd.vis_units is not None and hc.gain_scale.lower() != "uncalib" and hd.vis_units.lower() != hc.gain_scale.lower():
                    warnings.warn(f"Replacing original data vis_units of {hd.vis_units}"
                                  f" with calibration vis_units of {hc.gain_scale}", RuntimeWarning)
                hd_red.vis_units = hc.gain_scale
            if vis_units is not None:
                hd_red.vis_units = vis_units
            hd_red.write_uvh5(data_outfilename, clobber=clobber)
    # full data loading and writing
    else:
        data, data_flags, data_nsamples = hd.read(frequencies=freqs_to_load)
        data_antpos = hd.get_metadata_dict()['data_antpos']
        pols = hd.get_metadata_dict()['pols']
        if redundant_average or redundant_solution:
            all_reds = redcal.get_reds(data_antpos, pols=pols, bl_error_tol=bl_error_tol, include_autos=True)
        else:
            all_reds = []
        if redundant_average:
            all_red_antpairs = [[bl[:2] for bl in grp] for grp in all_reds if grp[-1][-1] == pols[0]]
            data_antpairs = hd.get_antpairs()
            reds_data = [[bl for bl in blg if bl in data_antpairs] for blg in all_red_antpairs]
            reds_data = [blg for blg in reds_data if len(blg) > 0]
        for bl in data_flags.keys():
            # apply band edge flags
            data_flags[bl][:, 0:flag_nchan_low] = True
            data_flags[bl][:, data_flags[bl].shape[1] - flag_nchan_high:] = True
            # apply external flags
            if flag_file is not None:
                data_flags[bl] = np.logical_or(data_flags[bl], ext_flags[bl])
        if redundant_solution:
            calibrate_redundant_solution(data, data_flags, new_gains, new_flags, all_reds, old_gains=old_gains,
                                         old_flags=old_flags, gain_convention=gain_convention)
        else:
            calibrate_in_place(data, new_gains, data_flags=data_flags, cal_flags=new_flags,
                               old_gains=old_gains, gain_convention=gain_convention)
        if not redundant_average:
            if vis_units is None:
                if hasattr(hc, 'gain_scale') and hc.gain_scale is not None:
                    if hd.vis_units is not None and hd.vis_units.lower() != "uncalib" and hd.vis_units.lower() != hc.gain_scale.lower():
                        warnings.warn(f"Replacing original data vis_units of {hd.vis_units}"
                                      " with calibration vis_units of {hc.gain_scale}", RuntimeWarning)
                    vis_units = hc.gain_scale
            if vis_units is not None:
                kwargs['vis_units'] = vis_units
            io.update_uvdata(hd, data=data, flags=data_flags, add_to_history=add_to_history, **kwargs)
            io._write_HERAData_to_filetype(hd, data_outfilename, filetype_out=filetype_out, clobber=clobber)

        else:
            all_red_antpairs = [[bl[:2] for bl in grp] for grp in all_reds if grp[-1][-1] == hd.pols[0]]
            hd.update(data=data, flags=data_flags, nsamples=data_nsamples, **kwargs)
            # by default, weight by nsamples (but not flags). This prevents spectral structure from being introduced
            # and also allows us to compute redundant averaged vis in flagged channels (in case flags are spurious).
            if no_red_weights:
                redundant_weights = copy.deepcopy(data_nsamples)
                for bl in data_flags:
                    if np.all(data_flags[bl]):
                        if exclude_from_redundant_mode == 'data':
                            if np.all(data_flags[bl]):
                                redundant_weights[bl][:] = 0.
                        elif exclude_from_redundant_mode == 'yaml' and ex_ants is not None:
                            if bl[0] in ex_ants or bl[1] in ex_ants:
                                redundant_weights[bl][:] = 0.
            for red_chunk in range(redundant_groups):
                red_antpairs = []
                reds_data_bls = []
                for grp in reds_data:
                    # trim group to only include baselines with redundant weights not equal to zero.
                    grp0 = grp[0]
                    if dont_red_average_flagged_data and redundant_groups > 1:
                        grp = [ap for ap in grp if np.any(np.asarray([~np.isclose(redundant_weights[ap + (pol,)], 0.0) for pol in data_flags.pols()]))]
                    # only include groups with more elements then redundant groups!
                    if len(grp) >= redundant_groups:
                        red_antpairs.append(grp[red_chunk:: redundant_groups])
                        reds_data_bls.append(grp0)
                data_red, flags_red, nsamples_red = utils.red_average(data=data, flags=data_flags, nsamples=data_nsamples,
                                                                      reds=red_antpairs, red_bl_keys=reds_data_bls, wgts=redundant_weights, inplace=False,
                                                                      propagate_flags=True)
                # update redundant data. Don't partial write.
                hd_red = io.HERAData(data_infilename, upsample=upsample, downsample=downsample)
                if len(reds_data_bls) > 0:
                    hd_red.read(bls=reds_data_bls, frequencies=freqs_to_load)
                    # update redundant data. Don't partial write.
                    hd_red.update(nsamples=nsamples_red, flags=flags_red, data=data_red)
                    hd_red.update(nsamples=nsamples_red, flags=flags_red, data=data_red)
                    if redundant_groups > 1:
                        outfile = data_outfilename.replace('.uvh5', f'.{red_chunk}.uvh5')
                    else:
                        outfile = data_outfilename
                    if filetype_out == 'uvh5':
                        if hasattr(hc, 'gain_scale') and hc.gain_scale is not None:
                            if hd_red.vis_units is not None and hd_red.vis_units.lower() != "uncalib" and hd_red.vis_units.lower() != hc.gain_scale.lower():
                                warnings.warn(f"Replacing original data vis_units of {hd.vis_units}"
                                              " with calibration vis_units of {hc.gain_scale}", RuntimeWarning)
                            hd_red.vis_units = hc.gain_scale
                        if vis_units is not None:
                            hd_red.vis_units = vis_units
                        hd_red.write_uvh5(outfile, clobber=clobber)
                    else:
                        raise NotImplementedError("redundant averaging only supported for uvh5 outputs.")
                else:
                    warnings.warn("No unflagged data so no calibration or outputs produced.")


def apply_cal_argparser():
    '''Arg parser for commandline operation of apply_cal.'''
    a = argparse.ArgumentParser(description="Apply (and optionally, also unapply) a calfits file to visibility file.")
    a.add_argument("infilename", type=str, help="path to visibility data file to calibrate")
    a.add_argument("outfilename", type=str, help="path to new visibility results file")
    a.add_argument("--new_cal", type=str, default=None, nargs="+", help="path to new calibration calfits file (or files for cross-pol)")
    a.add_argument("--old_cal", type=str, default=None, nargs="+", help="path to old calibration calfits file to unapply (or files for cross-pol)")
    a.add_argument("--flag_file", type=str, default=None, help="path to file of flags to OR with data flags")
    a.add_argument("--flag_filetype", type=str, default='h5', help="filetype of flag_file (either 'h5' or legacy 'npz'")
    a.add_argument("--flag_nchan_low", type=int, default=0, help="integer number of channels at the low frequency end of the band to always flag (default 0)")
    a.add_argument("--flag_nchan_high", type=int, default=0, help="integer number of channels at the high frequency end of the band to always flag (default 0)")
    a.add_argument("--filetype_in", type=str, default='uvh5', help='filetype of input data files')
    a.add_argument("--filetype_out", type=str, default='uvh5', help='filetype of output data files')
    a.add_argument("--nbl_per_load", type=str, default=None, help="Maximum number of baselines to load at once. uvh5 to uvh5 only."
                                                                  "Default loads the whole file. If 'none' is provided, also loads whole file.")
    a.add_argument("--redundant_groups", type=int, default=1, help="Number of subgroups to split each redundant baseline into for cross power spectra. ")
    a.add_argument("--gain_convention", type=str, default='divide',
                   help="'divide' means V_obs = gi gj* V_true, 'multiply' means V_true = gi gj* V_obs.")
    a.add_argument("--upsample", default=False, action="store_true", help="Upsample BDA files to the highest temporal resolution.")
    a.add_argument("--downsample", default=False, action="store_true", help="Downsample BDA files to the highest temporal resolution.")
    a.add_argument("--redundant_solution", default=False, action="store_true",
                   help="If True, average gain ratios in redundant groups to recalibrate e.g. redcal solutions.")
    a.add_argument("--clobber", default=False, action="store_true", help='overwrites existing file at outfile')
    a.add_argument("--vis_units", default=None, type=str, help="String to insert into vis_units attribute of output visibility file.")
    a.add_argument("--redundant_average", default=False, action="store_true", help="Redundantly average calibrated data.")
    a.add_argument("--dont_red_average_flagged_data", default=False, action="store_true", help="Do not include flagged data in redundant averages. Prevents redundant groups where one subgroup is flagged.")
    a.add_argument("--spw_range", default=None, type=int, nargs=2, help="specify spw range to load.")
    a.add_argument("--exclude_from_redundant_mode", default='data', type=str, help="exclude visibilities from redundant average based on whether entire waterfall is flagged ,'data'"
                                                                                   ", or whether its antennas are present in a yaml file.")
    a.add_argument("--a_priori_flags_yaml", type=str, default=None, help="path to yaml file to use in apriori flags.")
    return a
