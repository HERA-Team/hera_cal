# -*- coding: utf-8 -*-
# Copyright 2019 the HERA Project
# Licensed under the MIT License

import numpy as np
from hera_filters import dspec

from . import utils
from scipy.interpolate import interp1d
from .datacontainer import DataContainer
from .vis_clean import VisClean
from pyuvdata import UVData, UVFlag, UVBeam
import argparse
from . import io
from . import vis_clean
import warnings
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, ITRS
from astropy import units
import astropy.constants as const
from astropy_healpix import HEALPix
from astropy.time import Time
from pyuvdata import utils as uvutils
from . import utils
import copy
from .utils import echo
import datetime
import astropy.constants as const
from . import redcal

SPEED_OF_LIGHT = const.c.si.value
SDAY_KSEC = units.sday.to("ks")


def _get_key_reds(antpos, keys):
    """
    Private function to get redundant groups in list of antpairpol keys.

    Parameters
    ----------
    antpos: dict
        dictionary with integer keys mapping to length 3 numpy array ENU antenna positions.
    keys:
        list of antpairpols to split into redundant groups.

    Returns
    -------
    reds: list of lists of tuples
        list of lists where each sublist is a redundant group
        and each element of each redundant group is a redundant antpairpol
        in keys.
    """
    # get redundancies (will only compute fr-profile once for each red group).
    unique_pols = set()
    for bl in keys:
        if bl[-1] not in unique_pols:
            unique_pols.add(bl[-1])
    reds = redcal.get_reds(antpos, pols=list(unique_pols), include_autos=True)
    reds = redcal.filter_reds(reds, bls=keys)
    return reds


def select_tophat_frates(blvecs, case, uvd=None, keys=None, frate_standoff=0.0, frate_width_multiplier=1.0,
                         min_frate_half_width=0.025, max_frate_half_width=np.inf,
                         max_frate_coeffs=None, uvb=None, frate_profiles=None, percentile_low=5.0, percentile_high=95.0,
                         fr_freq_skip=1, nfr=None, dfr=None, verbose=False):
    """
    Select fringe-rates to filter given a filtering case.

    Parameters
    ----------
    blvecs: dict with int 2-tuple keys and 3-vector numpy array values.
        dictionary mapping antpair tuples to numpy 3-vector of baseline in ENU coords.
    case: str
        Filtering case ['max_frate_coeffs', 'uvbeam', 'frate_profiles', 'sky']
        If case == 'max_frate_coeffs', then determine fringe rate centers
        and half-widths based on the max_frate_coeffs arg (see below).
        If case == 'uvbeam', then determine fringe rate centers and half widths
        from histogram of main-beam wrt instantaneous sky fringe rates
        If case == 'frate_profiles': then use user-supplied fringe-rate profiles
        and compute cutoffs based on percentile_low and percentile_high
    uvd: UVData object, optional
        uvdata containing data and metadata to which tophat fringe rate filter will be applied.
    frate_standoff: float, optional
        additional fringe-rate to add to min and max of computed fringe-rate bounds [mHz]
        to add to analytic calculation of fringe-rate bounds for emission on the sky.
        default = 0.0.
    frate_width_multiplier: float, optional
        fraction of horizon to fringe-rate filter.
        default is 1.0
    min_frate_half_width: float, optional
        minimum half-width of fringe-rate filter, regardless of baseline length in mHz.
        Default is 0.025
    max_frate_half_width: float, optional
        maximum half-width of fringe-rate filter, regardless of baseline length in mHz.
        Default is np.inf, i.e. no limit
    keys: list of visibilities to filter in the (i,j,pol) format.
        If None (the default), all visibilities are filtered.
    max_frate_coeffs, 2-tuple float
        Maximum fringe-rate coefficients for the model max_frate [mHz] = x1 * | EW_bl_len | [ m ] + x2."
        These are used only if case == 'max_frate_coeffs'.
    uvb: UVBeam object, optional
        UVBeam object with model of the primary beam. Only used if case=='uvbeam'
    percentile_low: float, optional
        if uvb is provided, filter fringe rates that are larger then this percentile
        in the histogram of beam squared powers for instantaneous fringe rates.
        Only used if case == 'uvbeam' or case == 'frate_profiles'
    percentile_high: float, optional
        if uvb is provided, filter fringe rates that are smaller then this percentile
        in the histogram of beam squared powers for instantaneous fringe rates.
        only used if case == 'uvbeam' or case == 'frate_profiles'
    fr_freq_skip: int, optional
        bin fringe rates from every freq_skip channels.
        default is 1 -> takes a long time. We recommend setting this to be larger.
        only used if case == 'uvbeam'
    dfr: float, optional.
        spacing of fringe-rate grid used to perform binning and percentile calc
        in units of mHz.
        default is None -> set to 1 / (dtime * Ntime) of uvd.
        only used of case == 'uvbeam' or case == 'frate_profiles'
    nfr: float, optional.
        number of points on fringe-rate grid to perform binning and percentile calc.
        default is None -> set to uvd.Ntimes.
        only used of case == 'uvbeam' or case == 'frate_profiles'
    verbose: bool, optional, lots of outputs!
    frate_profiles: dict object, optional
        Dictionary mapping antpairpol keys to fringe-rate profiles.
        centered on a grid of length nfr spaced by dfr centered at 0.
        only used if case == 'frate_profiles'
    filter_kwargs: see fourier_filter for a full list of filter_specific arguments.

    Returns
    -------
    frate_centers: dict object,
        Dictionary with the center fringe-rate of each baseline in to_filter in units of mHz.
        keys are antpairpols.
    frate_half_widths: dict object
        Dictionary with the half widths of each fringe-rate window around the frate_centers in units of mHz.
        keys are antpairpols
    """
    if case != 'max_frate_coeffs':
        assert uvd is not None, "Must provide uvd if case != max_frate_coeffs."
    if keys is None:
        keys = list(uvd.get_antpairpols())
    if case == 'sky':
        # Fringe-rate filter all modes that could be occupied by sky emission.
        frate_centers, frate_half_widths = sky_frates(uvd, keys=keys, frate_standoff=frate_standoff,
                                                      frate_width_multiplier=frate_width_multiplier,
                                                      min_frate_half_width=min_frate_half_width,
                                                      max_frate_half_width=max_frate_half_width)
    elif case == 'max_frate_coeffs':
        # Fringe-rate filter based on max_frate_coeffs
        assert max_frate_coeffs is not None, "max_frate_coeffs must be provided if case='max_frate_coeffs'."
        frate_half_widths = {k: np.max([max_frate_coeffs[0] * np.abs(blvecs[k[:2]][0]) + max_frate_coeffs[1], 0.0]) for k in keys}
        # impose min_frate_half_width and max_frate_half_width
        frate_half_widths = {k: np.max([min_frate_half_width, np.min([max_frate_half_width, frate_half_widths[k]])]) for k in keys}
        frate_centers = {k: 0.0 for k in keys}
    elif case == 'frate_profiles':
        # Fringe-rate filter based on frate_profiles
        assert frate_profiles is not None, "frate_profiles must be provided if case='frate_profiles'"
        frate_centers, frate_half_widths = get_fringe_rate_limits(uvd, nfr=nfr, dfr=dfr, percentile_low=percentile_low,
                                                                  percentile_high=percentile_high, keys=keys, verbose=verbose, frate_standoff=frate_standoff,
                                                                  fr_freq_skip=fr_freq_skip, frate_width_multiplier=frate_width_multiplier,
                                                                  min_frate_half_width=min_frate_half_width, max_frate_half_width=max_frate_half_width,
                                                                  frate_profiles=frate_profiles)
    elif case == 'uvbeam':
        # Fringe-rate filter based on the instantanous fringe-rates on uvbeam object.
        assert uvb is not None, "uvb must be provided iv case='uvbeam'."
        frate_centers, frate_half_widths = get_fringe_rate_limits(uvd, uvb, nfr=nfr, dfr=dfr,
                                                                  percentile_low=percentile_low,
                                                                  percentile_high=percentile_high,
                                                                  keys=keys, verbose=verbose,
                                                                  frate_standoff=frate_standoff, fr_freq_skip=fr_freq_skip,
                                                                  frate_width_multiplier=frate_width_multiplier,
                                                                  min_frate_half_width=min_frate_half_width,
                                                                  max_frate_half_width=max_frate_half_width,
                                                                  frate_profiles=frate_profiles)
    return frate_centers, frate_half_widths


def sky_frates(uvd, keys=None, frate_standoff=0.0, frate_width_multiplier=1.0,
               min_frate_half_width=0.025, max_frate_half_width=np.inf):
    """Compute sky fringe-rate ranges based on baselines, telescope location, and frequencies in uvdata.

    Parameters
    ----------
    uvd: UVData object
        uvdata object of data to compute sky-frate limits for.
    keys: list of antpairpol tuples, optional
        list of antpairpols to generate sky fringe-rate centers and widths for.
        Default is None -> use all keys in uvd
    frate_standoff: float, optional
        additional fringe-rate to add to min and max of computed fringe-rate bounds [mHz]
        to add to analytic calculation of fringe-rate bounds for emission on the sky.
        default = 0.0.
    frate_width_multiplier: float, optional
        fraction of width of range of fringe-rates associated with sky emission to filter.
        default is 1.0
    min_frate_half_width: float, optional
        minimum half-width of fringe-rate filter, regardless of baseline length in mHz.
        Default is 0.025
    max_frate_half_width: float, optional
        maximum half-width of fringe-rate filter, regardless of baseline length in mHz.
        Default is np.inf, i.e. no limit

    Returns
    -------
    frate_centers: dict object,
        Dictionary with the center fringe-rate of each baseline in to_filter in units of mHz.
        These are antpairpol format [e.g. (0, 1, 'ee')]
    frate_half_widths: dict object
        Dictionary with the half widths of each fringe-rate window around the center_frates in units of mHz.
        These are antpairpol format [e.g. (0, 1, 'ee')]
    """
    if keys is None:
        keys = uvd.get_antpairpols()
    antpos, antnums = uvd.get_ENU_antpos()
    sinlat = np.sin(np.abs(uvd.telescope_location_lat_lon_alt[0]))
    frate_centers = {}
    frate_half_widths = {}

    # compute maximum fringe rate dict based on baseline lengths.
    # see derivation in https://www.overleaf.com/read/chgpxydbhfhk
    # which starts with expressions in
    # https://ui.adsabs.harvard.edu/abs/2016ApJ...820...51P/exportcitation
    for k in keys:
        ind1 = np.where(antnums == k[0])[0][0]
        ind2 = np.where(antnums == k[1])[0][0]
        blvec = antpos[ind1] - antpos[ind2]
        blcos = blvec[0] / np.linalg.norm(blvec[:2])
        if np.isfinite(blcos):
            frateamp_df = np.linalg.norm(blvec[:2]) / SDAY_KSEC / SPEED_OF_LIGHT * 2 * np.pi
            # set autocorrs to have blcose of 0.0

            if blcos >= 0:
                max_frate_df = frateamp_df * np.sqrt(sinlat ** 2. + blcos ** 2. * (1 - sinlat ** 2.))
                min_frate_df = -frateamp_df * sinlat
            else:
                min_frate_df = -frateamp_df * np.sqrt(sinlat ** 2. + blcos ** 2. * (1 - sinlat ** 2.))
                max_frate_df = frateamp_df * sinlat

            min_frate = np.min([f0 * min_frate_df for f0 in uvd.freq_array])
            max_frate = np.max([f0 * max_frate_df for f0 in uvd.freq_array])
        else:
            max_frate = 0.0
            min_frate = 0.0

        frate_centers[k] = (max_frate + min_frate) / 2.
        frate_centers[utils.reverse_bl(k)] = -frate_centers[k]

        frate_half_widths[k] = np.abs(max_frate - min_frate) / 2. * frate_width_multiplier + frate_standoff
        frate_half_widths[k] = np.max([frate_half_widths[k], min_frate_half_width])  # Don't allow frates smaller then min_frate
        frate_half_widths[k] = np.min([frate_half_widths[k], max_frate_half_width])  # Don't allow frates larger then max_frate
        frate_half_widths[utils.reverse_bl(k)] = frate_half_widths[k]

    return frate_centers, frate_half_widths


def build_fringe_rate_profiles(uvd, uvb, keys=None, normed=True, combine_pols=True, nfr=None, dfr=None,
                               taper='none', fr_freq_skip=1, verbose=False):
    """
    Calculate fringe-rate profiles to either directly apply as an FIR filter or set a range to filter.


    Produces a dictionary of numpy arrays. Each array is the optimal weight
    (SNR squared) to multiply data in each fringe-rate bin for an isotropic sky using the
    instantaneous fringe-rates inside
    of the fringe-rate kernel in equation  (4) in Parsons et al. 2016
    https://ui.adsabs.harvard.edu/abs/2016ApJ...820...51P/abstract
    The fringe-rate profile is integrated over all frequencies with some
    unflagged data so we recommend only using this function on subbands
    of <~ 10-20 MHz or frequency intervals
    over which the primary beam does not evolve substantially.

    Parameters
    ----------
    uvd: UVData object
        UVData holding baselines for which we will build fringe-rate profiles.
    uvb: UVBeam object
        UVBeam object holding beams that we will build fr profiles for.
        Profiles are generated from power beams. If an efield beam is provided
        a copy will be converted to the required power format. This function
        also requires healpix formatted beams so this hidden copy will also
        be converted to healpix if it was not already in this format.
    keys: list of antpairpol tuples
        list of antpairpol tuples of baselines to calculate fringe-rate limits for.
        default = None -> compute profiles for all antpairpols in uvd.
    normed: bool, optional
        if True, normalize profiles by sum.
    combine_pols: bool, optional
        if True, set profiles for all pols equal to the sum of the profiles of individual pols.
    dfr: float, optional.
        spacing of fringe-rate grid used to perform binning and percentile calc
        in units of mHz.
        default is None -> set to 1 / (dtime * Ntime) of uvd.
    nfr: float, optional.
        number of points on fringe-rate grid to perform binning and percentile calc.
        default is None -> set to uvd.Ntimes.
    taper: str, optional
        taper expected for power spectrum calculations. Fringe-rates from different frequencies
        Valid taper options can be found in hera_filters.dspec.gen_window
        Located here
        https://github.com/HERA-Team/hera_filters/blob/3601a6a707ee0b10c6219f9a82f0f90dbab1f15c/hera_filters/dspec.py#L1155
    fr_freq_skip: int, optional
        bin fringe rates from every freq_skip channels.
        default is 1 -> takes a long time. We recommend setting this to be larger.
    verbose: bool, optional
        lots of text output.
    Returns
    -------
    fr_grid: np.ndarray
        length nfr grid of fringe-rates spaced by dfr centered at zero.
        units are whatever the units of dfr are.

    profiles: Dictionary object. Maps antpairpol tuples to numpy.ndarray object
              with the sum of the beam squared in all directions falling into
              a particular fringe-rate bin.

    """

    uvb = copy.deepcopy(uvb)
    # convert to power and healpix if necesssary
    if uvb.beam_type == 'efield':
        uvb.efield_to_power()
    try:
        uvb.to_healpix()
    except ValueError as err:
        warnings.warn("UVBeam object already in healpix format...")
    if keys is None:
        keys = uvd.get_antpairpols()

    antpos_trf = uvd.antenna_positions  # earth centered antenna positions
    antnums = uvd.antenna_numbers  # antenna numbers.

    lat, lon, alt = uvd.telescope_location_lat_lon_alt_degrees
    location = EarthLocation(lon=lon * units.deg, lat=lat * units.deg, height=alt * units.m)

    # get topocentricl AzEl Beam coordinates.
    hp = HEALPix(nside=uvb.nside, order=uvb.ordering)
    az, alt = hp.healpix_to_lonlat(range(uvb.Npixels))
    # zero out beam below the horizon
    uvb.data_array[0, :, :, alt <= 0 * units.radian] = 0.
    # Covert AltAz coordinates of UVBeam pixels to barycentric coordinates.
    obstime = Time(np.median(np.unique(uvd.time_array)), format='jd')
    altaz = AltAz(obstime=obstime, location=location)
    # coordinates of beam pixels in topocentric frame.
    altaz_coords = SkyCoord(alt=alt, az=az, frame=altaz)
    # transform beam pixels from topocentric to ITRS
    eq_coords = altaz_coords.transform_to(ITRS())
    # get cartesian xyz unit vectors in the direction of each beam pixle.
    eq_xyz = np.vstack([eq_coords.x, eq_coords.y, eq_coords.z])

    # generate fringe_rate grid in mHz.
    dt = SDAY_KSEC * np.median(np.diff(np.unique(uvd.time_array)))  # times in kSec
    if nfr is None:
        nfr = uvd.Ntimes
    if dfr is None:
        # if no dfr provided, set to 1 / (ntimes * dt)
        dfr = 1. / (dt * nfr)

    # build grid.
    min_frate = - dfr * nfr / 2
    if np.mod(nfr, 2) != 0:
        min_frate += dfr / 2
    fr_grid = np.arange(nfr) * dfr + min_frate
    # fringe rate bin edges including upper edge of rightmost bin.
    frate_bins = np.hstack([fr_grid - dfr / 2., [fr_grid.max() + dfr / 2.]])

    # frequency tapering function expected for power spectra.
    # square b/c for power spectrum.
    ftaper = dspec.gen_window(taper, uvd.Nfreqs) ** 2.
    if keys is None:
        keys = uvd.get_antpairpols()

    profiles = {}  # store profiles here.
    ap_blkeys = {}
    # keeps track of different polarizations for each antenna pair
    # for if we are going to sum over polarizations.
    # get redundancies (will only compute fr-profile once for each red group).
    reds = _get_key_reds(dict(zip(*uvd.get_ENU_antpos()[::-1])), keys)

    for redgrp in reds:
        # only explicitly calculate fr profile for the first vis in each redgroup.
        bl = redgrp[0]
        echo(f"Generating FR-Profile of {bl} at {str(datetime.datetime.now())}", verbose=verbose)
        # sum beams from all frequencies
        # get polarization index
        polindex = np.where(uvutils.polstr2num(bl[-1], x_orientation=uvb.x_orientation) == uvb.polarization_array)[0][0]
        # get baseline vector in equitorial coordinates.
        blvec = antpos_trf[antnums == bl[1]] - antpos_trf[antnums == bl[0]]
        # initialize binned power.
        # we will bin frate power together for all frequencies, weighted by taper.
        binned_power = np.zeros_like(fr_grid)
        binned_power_conj = np.zeros_like(fr_grid)
        # iterate over each frequency and ftaper weighting.
        # use linspace to make sure we get first and last frequencies.
        unflagged_chans = ~np.all(np.all(uvd.flag_array[:, :, :].squeeze(), axis=0), axis=-1)
        chans_to_use = np.arange(uvd.Nfreqs).astype(int)[unflagged_chans][::fr_freq_skip]
        frate_coeff = 2 * np.pi / SPEED_OF_LIGHT / SDAY_KSEC
        frate_over_freq = np.dot(np.cross(np.array([0, 0, 1.]), blvec), eq_xyz) * frate_coeff
        # histogram all frequencies together in one step.
        frates = np.hstack([frate_over_freq * freq for freq in uvd.freq_array[chans_to_use]]).squeeze()
        # beam squared values weighted by the taper function.
        bsq = np.hstack([np.abs(uvb.data_array[0, polindex, np.argmin(np.abs(freq - uvb.freq_array[0])), :].squeeze()) ** 2. * freqweight ** 2. for freq, freqweight in zip(uvd.freq_array[chans_to_use], ftaper[chans_to_use])])
        # histogram fringe rates weighted by beam square values.
        binned_power = np.histogram(frates, bins=frate_bins, weights=bsq)[0]
        binned_power_conj = np.histogram(-frates, bins=frate_bins, weights=bsq)[0]

        # iterate over redgrp and set profiles for each baseline key.
        for blk in redgrp:
            profiles[blk] = binned_power
            profiles[utils.reverse_bl(blk)] = binned_power_conj
            if blk[:2] not in ap_blkeys:
                ap_blkeys[blk[:2]] = [blk]
                ap_blkeys[utils.reverse_bl(blk)[:2]] = [utils.reverse_bl(blk)]
            else:
                ap_blkeys[blk[:2]].append(blk)  # append antpairpols
                ap_blkeys[utils.reverse_bl(blk)[:2]].append(utils.reverse_bl(blk))

    # combine polarizations by summing over all profiles for each antenna-pair.
    if combine_pols:
        for ap in ap_blkeys:
            profile_summed_over_pols = np.sum([profiles[bl] for bl in ap_blkeys[ap]], axis=0)
            for bl in ap_blkeys[ap]:
                profiles[bl] = profile_summed_over_pols
    # normalize if desired
    if normed:
        for bl in profiles:
            profiles[bl] /= np.sum(profiles[bl])
    return fr_grid, profiles


def get_fringe_rate_limits(uvd, uvb=None, frate_profiles=None, percentile_low=5., percentile_high=95., keys=None,
                           dfr=None, nfr=None, taper='none', frate_standoff=0.0, frate_width_multiplier=1.0,
                           min_frate_half_width=0.025, max_frate_half_width=np.inf,
                           fr_freq_skip=1, verbose=False):
    """
    Get bounding fringe-rates for isotropic emission for a UVBeam object across all frequencies.

    Parameters
    ----------
    uvd: UVData object
        UVData holding baselines for which we will build fringe-rate profiles.
    uvb: UVBeam object, optional
        UVBeam object holding beams that we will build fr profiles for.
        default is None -> use pre-computed frate_profiles.
        User must provide either frate_profiles or uvb. Otherwise an error
        will be raised.
    frate_profiles: dict object, optional
        Dictionary mapping antpairpol keys to fringe-rate profiles.
        centered on a grid of length nfr spaced by dfr centered at 0.
        default is None -> automatically calculate profiles using uvb.
        User must provide either frate_profiles or uvb. Otherwise an error
        will be raised.
    percentile_low: float, optional
        Percent of beam-squared power below lower fringe rate.
    percentile_high: float, optional
        Percent of beam-squared power above upper fringe rate.
    keys: dict, optional
        antpairpol keys to compute fringe-rate limits for.
    dfr: float, optional.
        spacing of fringe-rate grid used to perform binning and percentile calc
        in units of mHz.
        default is None -> set to 1 / (dtime * Ntime) of uvd.
    nfr: float, optional.
        number of points on fringe-rate grid to perform binning and percentile calc.
        default is None -> set to uvd.Ntimes.
    taper: str, optional
        taper expected for power spectrum calculations. Fringe-rate profiles from different frequencies
        will be summed into the total fringe-rate profile with the weight of this taper.
        Valid taper options can be found in hera_filters.dspec.gen_window
        Located here
        https://github.com/HERA-Team/hera_filters/blob/3601a6a707ee0b10c6219f9a82f0f90dbab1f15c/hera_filters/dspec.py#L1155
    frate_standoff: float, optional
        Additional fringe-rate standoff in mHz to add to Omega_E b_{EW} nu/c to add
        to limits computed from beam-squared histogram.
        default = 0.0.
    frate_width_multiplier: float, optional
        fraction of horizon to fringe-rate filter.
        default is 1.0
    min_frate_half_width: float, optional
        minimum fringe-rate width to filter, regardless of baseline length in mHz.
        Default is 0.025
    max_frate_half_width: float, optional
        maximum half-width of fringe-rate filter, regardless of baseline length in mHz.
        Default is np.inf, i.e. no limit
    fr_freq_skip: int, optional
        bin fringe rates from every freq_skip channels.
        default is 1 -> takes a long time. We recommend setting this to be larger.
    verbose: bool, optional
        lots of text output, default is False.

    Returns
    -------
    frate_centers: dict object,
        Dictionary with the center fringe-rate of each baseline in to_filter in units of mHz.
        keys are antpairpols.
    frate_half_widths: dict object
        Dictionary with the half widths of each fringe-rate window around the frate_centers in units of mHz.
        keys are antpairpols
    """
    frate_centers = {}
    frate_half_widths = {}
    if keys is None:
        keys = uvd.get_antpairpols()

    if frate_profiles is None:
        if uvb is not None:
            fr_grid, frate_profiles = build_fringe_rate_profiles(uvd=uvd, uvb=uvb, keys=keys, normed=True, nfr=nfr, dfr=dfr,
                                                                 taper=taper, fr_freq_skip=fr_freq_skip, verbose=verbose)
        else:
            raise ValueError("Must either supply uvb or frate_profiles!")
    elif frate_profiles is not None and uvb is not None:
        raise ValueError("Must provide either uvb or frate_profiles but not both!")
    else:
        if nfr is None:
            nfr = uvd.Ntimes
        if dfr is None:
            dfr = 1. / (nfr * np.mean(np.diff(np.unique(uvd.time_array))) * SDAY_KSEC)
        fr_grid = np.arange(-nfr // 2, nfr // 2) * dfr

    reds = _get_key_reds(dict(zip(*uvd.get_ENU_antpos()[::-1])), keys)

    for redgrp in reds:
        bl0 = redgrp[0]
        frlows = []
        frhighs = []
        for bl in [bl0, utils.reverse_bl(bl0)]:
            binned_power = frate_profiles[bl]
            # normalize to sum to 100.
            binned_power /= np.sum(binned_power)
            binned_power *= 100.
            # get CDF as function of fringe-rate bin.
            cspower = np.cumsum(binned_power)
            # add dfr to ends of grid and set to zero and 100 to avoid
            # out of bounds errors.
            nfr = len(fr_grid)
            dfr = np.median(np.diff(fr_grid))
            cspower_interp = np.hstack([[0], cspower, [100.]])
            fr_grid_interp = np.hstack([[fr_grid.min() - dfr], fr_grid, [fr_grid.max() + dfr]])
            cspower_func = interp1d(cspower_interp, fr_grid_interp)
            # find low and high bins containing mass between percentile_low and percentile_high.
            frlows.append(cspower_func(percentile_low))
            frhighs.append(cspower_func(percentile_high))
        # iterate through redundant group and assign centers / half-widths.
        for blt in redgrp:
            # save low and high fringe rates for bl and its conjugate
            for cnum, bl in enumerate([blt, utils.reverse_bl(blt)]):
                frate_centers[bl] = .5 * (frlows[cnum] + frhighs[cnum])
                frate_half_widths[bl] = .5 * np.abs(frlows[cnum] - frhighs[cnum]) * frate_width_multiplier + frate_standoff
                frate_half_widths[bl] = np.max([frate_half_widths[bl], min_frate_half_width])
                frate_half_widths[bl] = np.min([frate_half_widths[bl], max_frate_half_width])

    return frate_centers, frate_half_widths


def timeavg_waterfall(data, Navg, flags=None, nsamples=None, wgt_by_nsample=True,
                      wgt_by_favg_nsample=False, rephase=False, lsts=None, freqs=None,
                      bl_vec=None, lat=-30.72152, extra_arrays={}, verbose=True):
    """
    Calculate the time average of a visibility waterfall. The average is optionally
    weighted by a boolean flag array (flags) and also optionally by an Nsample array (nsample),
    such that, for a single frequency channel, the time average is constructed as

    avg_data = sum( data * flag_wgt * nsample ) / sum( flag_wgt * nsample )

    where flag_wgt is constructed as (~flags).astype(float).

    Additionally, one can rephase each integration in the averaging window to the LST of the
    window-center before taking their average. This assumes the
    input data are drift-scan phased. See hera_cal.utils.lst_rephase
    for details on the rephasing algorithm. By feeding an nsample array,
    the averaged nsample for each averaging window is computed and returned.

    Parameters
    ----------
    data : ndarray
        2D complex ndarray of complex visibility with shape=(Ntimes, Nfreqs)
        The rows of data are assumed to be ordered chronologically.

    Navg : int
        Number of time samples to average together, with the condition
        that 0 < Navg <= Ntimes. Navg = 1 is no averaging. Navg = Ntimes
        is complete averaging.

    flags : ndarray
        2D boolean ndarray containing data flags with matching shape of data.
        Flagged pixels are True, otherwise False.

    nsamples : ndarray, optional
        2D float ndarray containing the number of pre-averages behind each pixel
        in data. Default is to assume unity for all pixels.

    wgt_by_nsample : bool, optional
        If True, perform time average weighted by nsample, otherwise perform uniform
        average. Default is True.

    wgt_by_favg_nsample : bool, optional
        If True, perform time average weighting by averaging the number of samples across
        frequency for each integration. Mutually exclusive with wgt_by_nsample. Default False.

    rephase : boolean, optional
        If True, phase each integration to the LST of the averaging window-center
        before averaging. Need to feed lsts, freqs and bl_vec if True.

    lsts : ndarray, optional
        1D float array holding the LST [radians] of each time integration in
        data. Shape=(Ntimes,)

    freqs : ndarray, optional
        1D float array holding the starting frequency of each frequency bin [Hz]
        in data. Shape=(Nfreqs,)

    bl_vec : ndarray, optional
        3D float ndarray containing the visibility baseline vector in meters
        in the ENU (TOPO) frame.

    lat : float, optional
        Latitude of observer in degrees North. Default is HERA coordinates.

    extra_arrays : dict, optional
        Dictionary of extra 1D arrays with shape=(Ntimes,) to push through
        averaging windows. For example, a time_array, or
        anything that has length Ntimes.

    verbose : bool, optional
        If True, report feedback to standard output.

    Returns
    -------
    avg_data : ndarray
        2D complex array with time-average spectrum, shape=(Navg_times, Nfreqs)

    win_flags : ndarray
        2D boolean array with OR of flags in averaging window, shape=(Navg_times, Nfreqs)

    avg_nsamples : ndarray
        2D array containing the sum of nsamples of each averaging window, weighted
        by the input flags, if fed. Shape=(Navg_times, Nfreqs)

    avg_lsts : ndarray
        1D float array holding the center LST of each averaging window, if
        lsts was fed. Shape=(Navg_times,).

    avg_extra_arrays : dict
        Dictionary of 1D arrays holding average of input extra_arrays for
        each averaging window, shape=(Navg_times,).
    """
    # type check
    assert isinstance(data, np.ndarray), "data must be fed as an ndarray"
    if rephase:
        assert lsts is not None and freqs is not None and bl_vec is not None, "" \
            "If rephase is True, must feed lsts, freqs and bl_vec."
    if (wgt_by_nsample and wgt_by_favg_nsample):
        raise ValueError('wgt_by_nsample and wgt_by_favg_nsample cannot both be True.')

    # unwrap lsts if fed
    if lsts is not None:
        lsts = np.unwrap(lsts)

    # form flags if None
    if flags is None:
        flags = np.zeros_like(data, dtype=bool)
    assert isinstance(flags, np.ndarray), "flags must be fed as an ndarray"

    # turn flags into weights
    flagw = (~flags).astype(float)

    # form nsamples if None
    if nsamples is None:
        nsamples = np.ones_like(data, dtype=float)
    assert isinstance(nsamples, np.ndarray), "nsamples must be fed as an ndarray"

    # assert Navg makes sense
    Ntimes = data.shape[0]
    assert Navg <= Ntimes and Navg > 0, "Navg must satisfy 0 < Navg <= Ntimes"

    # calculate Navg_times, the number of remaining time samples after averaging
    Navg_times = float(Ntimes) / Navg
    if Navg_times % 1 > 1e-10:
        if verbose:
            print("Warning: Ntimes is not evenly divisible by Navg, "
                  "meaning the last output time sample will be noisier "
                  "than the others.")
    Navg_times = int(np.ceil(Navg_times))

    # form output avg list
    avg_data = []
    win_flags = []
    avg_lsts = []
    avg_nsamples = []
    avg_extra_arrays = dict([('avg_{}'.format(a), []) for a in extra_arrays])

    # iterate through Navg_times
    for i in range(Navg_times):
        # get starting and stopping indices
        start = i * Navg
        end = (i + 1) * Navg
        d = data[start:end, :]
        f = flags[start:end, :]
        fw = flagw[start:end, :]
        n = nsamples[start:end, :]

        # calculate mean_l and l, if lsts was fed
        if lsts is not None:
            lst = lsts[start:end]
            mean_l = np.mean(lst)
            avg_lsts.append(mean_l)

        # rephase data if desired
        if rephase:
            # get dlst and rephase
            dlst = mean_l - lst
            d = utils.lst_rephase(d, bl_vec, freqs, dlst, lat=lat, inplace=False, array=True)

        # form data weights
        if wgt_by_nsample:
            w = fw * n
        elif wgt_by_favg_nsample:
            w = fw * np.mean(n, axis=1, keepdims=True)
        else:
            w = fw
        w_sum = np.sum(w, axis=0, keepdims=False).clip(1e-10, np.inf)

        # perfom weighted average of data along time
        ad = np.sum(d * w, keepdims=False, axis=0) / w_sum
        an = np.sum(n * fw, keepdims=False, axis=0)

        # append to data lists
        avg_data.append(ad)
        win_flags.append(np.min(f, axis=0, keepdims=False))
        avg_nsamples.append(an)

        # average arrays in extra_arrays
        for a in extra_arrays:
            avg_extra_arrays['avg_{}'.format(a)].append(np.mean(extra_arrays[a][start:end]))

    avg_data = np.asarray(avg_data, complex)
    win_flags = np.asarray(win_flags, bool)
    avg_nsamples = np.asarray(avg_nsamples, float)
    avg_lsts = np.asarray(avg_lsts, float)

    # wrap lsts
    avg_lsts = avg_lsts % (2 * np.pi)

    return avg_data, win_flags, avg_nsamples, avg_lsts, avg_extra_arrays


def apply_fir(data, fir, wgts=None, axis=0):
    """
    Convolves an FIR filter with visibility data.

    Args:
        data : complex ndarray of shape (Ntimes, Nfreqs)
        fir : complex 2d array of shape (Ntimes, Nfreqs)
            holding FIR filter to convolve against data
        wgts : float ndarray of shape (Ntimes, Nfreqs)
            Default is all ones.
        axis : int
            data axis along which to apply FIR

    Returns:
        new_data : complex ndarray of shape (Ntimes, Nfreqs)
            Contains data convolved with fir across
            time for each frequency channel independently.
    """
    # shape checks
    shape = list(data.shape)
    Ntimes, Nfreqs = shape
    assert isinstance(fir, np.ndarray), "fir must be an ndarray"
    if fir.ndim == 1:
        # try to broadcast given axis
        if axis == 0:
            fir = np.repeat(fir[:, None], Nfreqs, axis=1)
        elif axis == 1:
            fir = np.repeat(fir[None, :], Ntimes, axis=0)

    assert (Ntimes, Nfreqs) == fir.shape, "fir shape must match input data along time and frequency"

    # get weights
    if wgts is None:
        wgts = np.ones_like(data, dtype=float)

    new_data = np.empty_like(data, dtype=complex)

    shape.pop(axis)
    for i in range(shape[0]):
        slices = [i, i]
        slices[axis] = slice(None)
        slices = tuple(slices)
        new_data[slices] = np.convolve(data[slices] * wgts[slices], fir[slices], mode='same')

    return new_data


def frp_to_fir(frp, delta_bin=None, axis=0, undo=False):
    '''
    Transform a fourier profile to an FIR, or vice versa.

    This function assumes the convention of fft for real->fourier space and ifft
    for fourier->real space. The input fourier profile must have monotonically increasing fourier bins.

    Args:
        frp : 1D or 2D ndarray of the fourier profile.
        delta_bin : frp bin width along axis of fourier transform.
        axis : int, axis of frp along which to take fourier transform
        undo : bool, if True converts an fir to frp, else converts frp to fir

    Returns:
        fir : ndarray of the FIR filter, else undo == True then ndarray of frp
        frbins : 1D ndarray of fourier bins [1/delta_bin] if delta_bin is provided, else is None.
    '''
    # generate fir
    frp = np.fft.ifftshift(frp, axes=axis)
    if undo:
        fir = np.fft.fft(frp, axis=axis)
    else:
        fir = np.fft.ifft(frp, axis=axis)
    fir = np.fft.fftshift(fir, axes=axis)

    # generate frbins
    if delta_bin is None:
        frbins = None
    else:
        frbins = np.fft.fftshift(np.fft.fftfreq(len(frp), delta_bin), axes=axis)

    return fir, frbins


def fr_tavg(frp, noise_amp=None, axis=0):
    """
    Calculate the attenuation induced by fourier filtering a noise signal.

    See Ali et al. 2015 Eqn (9)

    Args:
        frp : A 1D or 2D fourier profile
        noise_amp : The noise amplitude (stand dev. not variance) in frate space
            with shape matching input frp
        axis : int, axis of frp along which filtering is done

    Returns:
        t_ratio : ndarray, effective integration ratio t_after / t_before
    """
    if noise_amp is None:
        noise_amp = np.ones_like(frp, dtype=float)

    t_ratio = np.sum(np.abs(noise_amp)**2, axis=axis, keepdims=True) / np.sum(np.abs(frp)**2 * np.abs(noise_amp)**2, axis=axis, keepdims=True).clip(1e-10, np.inf)

    return t_ratio


class FRFilter(VisClean):
    """
    FRFilter object. See hera_cal.vis_clean.VisClean.__init__ for instantiation options.
    """
    def timeavg_data(self, data, times, lsts, t_avg, flags=None, nsamples=None,
                     wgt_by_nsample=True, wgt_by_favg_nsample=False, rephase=False,
                     verbose=True, output_prefix='avg', keys=None, overwrite=False):
        """
        Time average data attached to object given a averaging time-scale t_avg [seconds].
        The resultant averaged data, flags, time arrays, etc. are attached to self
        with the name "{}_data".format(output_prefix), etc

        Note: The t_avg provided will be rounded to the nearest time that makes Navg
            an integer, and is stored as self.t_avg and self.Navg.

        Note: Time-averaging data with differing time-dependent flags per freq channel
            can create artificial spectral structure in the averaged data products.
            One can mitigate this by factorizing the flags into time-freq separable masks,
            see self.factorize_flags.

        Args :
            data : DataContainer
                data to time average, must be consistent with self.lsts and self.freqs
            times : 1D array
                Holds Julian Date time array for input data
            lsts : 1D array
                Holds LST time array for input data
            t_avg : float
                Width of time-averaging window in seconds.
            flags : DataContainer
                flags to use in averaging. Default is None.
                Must be consistent with self.lsts, self.freqs, etc.
            nsamples : DataContainer
                nsamples to use in averaging. Default is None.
                Must be consistent with self.lsts, self.freqs, etc.
            wgt_by_nsample : bool
                If True, perform time average weighted by nsample, otherwise perform
                uniform average. Default is True.
            wgt_by_favg_nsample : bool
                If True, perform time average weighting by averaging the number of samples across
                frequency for each integration. Mutually exclusive with wgt_by_nsample. Default False.
            rephase : bool
                If True, rephase data in averaging window to the window-center.
            keys : list of len-3 antpair-pol tuples
                List of data keys to operate on.
            overwrite : bool
                If True, overwrite existing keys in output DataContainers.
        """
        # turn t_avg into Navg
        Ntimes = len(times)
        dtime = np.median(np.abs(np.diff(times))) * 24 * 3600
        Navg = int(np.round((t_avg / dtime)))
        assert Navg > 0, "A t_avg of {:0.5f} makes Navg=0, which is too small.".format(t_avg)
        if Navg > Ntimes:
            Navg = Ntimes
        old_t_avg = t_avg
        t_avg = Navg * dtime

        if verbose:
            print("The t_avg provided of {:.3f} has been shifted to {:.3f} to make Navg = {:d}".format(
                old_t_avg, t_avg, Navg))

        # setup containers
        for n in ['data', 'flags', 'nsamples']:
            name = "{}_{}".format(output_prefix, n)
            if not hasattr(self, name):
                setattr(self, name, DataContainer({}))
            if n == 'data':
                avg_data = getattr(self, name)
            elif n == 'flags':
                avg_flags = getattr(self, name)
            elif n == 'nsamples':
                avg_nsamples = getattr(self, name)

        # setup averaging quantities
        if flags is None:
            flags = DataContainer(dict([(k, np.zeros_like(data[k], bool)) for k in data]))
        if nsamples is None:
            nsamples = DataContainer(dict([(k, np.ones_like(data[k], float)) for k in data]))

        if keys is None:
            keys = data.keys()

        # iterate over keys
        al = None
        at = None
        for i, k in enumerate(keys):
            if k in avg_data and not overwrite:
                utils.echo("{} exists in output DataContainer and overwrite == False, skipping...".format(k), verbose=verbose)
                continue
            (ad, af, an, al,
             ea) = timeavg_waterfall(data[k], Navg, flags=flags[k], nsamples=nsamples[k],
                                     rephase=rephase, lsts=lsts, freqs=self.freqs, bl_vec=self.blvecs[k[:2]],
                                     lat=self.lat, extra_arrays=dict(times=times), wgt_by_nsample=wgt_by_nsample,
                                     wgt_by_favg_nsample=wgt_by_favg_nsample, verbose=verbose)
            avg_data[k] = ad
            avg_flags[k] = af
            avg_nsamples[k] = an
            at = ea['avg_times']

        setattr(self, "{}_times".format(output_prefix), np.asarray(at))
        setattr(self, "{}_lsts".format(output_prefix), np.asarray(al))
        self.t_avg = t_avg
        self.Navg = Navg

    def filter_data(self, data, frps, flags=None, nsamples=None,
                    output_prefix='filt', keys=None, overwrite=False,
                    edgecut_low=0, edgecut_hi=0, axis=0, verbose=True):
        """
        Apply an FIR filter to data.

        Args :
            data : DataContainer
                data to time average, must be consistent with self.lsts and self.freqs
            frps : DataContainer
                DataContainer holding 2D fringe-rate profiles for each key in data,
                with values the same shape as data.
            flags : DataContainer
                flags to use in averaging. Default is None.
                Must be consistent with self.lsts, self.freqs, etc.
            nsamples : DataContainer
                nsamples to use in averaging. Default is None.
                Must be consistent with self.lsts, self.freqs, etc.
            keys : list of len-3 antpair-pol tuples
                List of data keys to operate on.
            overwrite : bool
                If True, overwrite existing keys in output DataContainers.
            edgecut_low : int, number of bins to flag on low side of axis
            edgecut_hi : int, number of bins to flag on high side of axis
        """
        # setup containers
        for n in ['data', 'flags', 'nsamples']:
            name = "{}_{}".format(output_prefix, n)
            if not hasattr(self, name):
                setattr(self, name, DataContainer({}))
            if n == 'data':
                filt_data = getattr(self, name)
            elif n == 'flags':
                filt_flags = getattr(self, name)
            elif n == 'nsamples':
                filt_nsamples = getattr(self, name)

        # setup averaging quantities
        if flags is None:
            flags = DataContainer(dict([(k, np.zeros_like(data[k], bool)) for k in data]))
        if nsamples is None:
            nsamples = DataContainer(dict([(k, np.ones_like(data[k], float)) for k in data]))

        if keys is None:
            keys = data.keys()

        # iterate over keys
        for i, k in enumerate(keys):
            if k in filt_data and not overwrite:
                utils.echo("{} exists in ouput DataContainer and overwrite == False, skipping...".format(k), verbose=verbose)
                continue

            # get wgts
            w = (~flags[k]).astype(float)
            shape = [1, 1]
            shape[axis] = -1
            w *= dspec.gen_window('none', w.shape[axis], edgecut_low=edgecut_low, edgecut_hi=edgecut_hi).reshape(tuple(shape))
            f = np.isclose(w, 0.0)

            # calculate effective nsamples
            eff_nsamples = np.zeros_like(nsamples[k])
            eff_nsamples += np.sum(nsamples[k] * w, axis=axis, keepdims=True) / np.sum(w, axis=axis, keepdims=True).clip(1e-10, np.inf)
            eff_nsamples *= fr_tavg(frps[k], axis=axis) * np.sum(w, axis=axis, keepdims=True).clip(1e-10, np.inf) / w.shape[axis]

            # setup FIR
            fir, _ = frp_to_fir(frps[k], axis=axis, undo=False)

            # apply fir
            dfilt = apply_fir(data[k], fir, wgts=w, axis=axis)

            # append
            filt_data[k] = dfilt
            filt_flags[k] = f
            filt_nsamples[k] = eff_nsamples

    def tophat_frfilter(self, frate_centers, frate_half_widths, keys=None, wgts=None, mode='clean',
                        skip_wgt=0.1, tol=1e-9, verbose=False, cache_dir=None, read_cache=False,
                        write_cache=False, pre_filter_modes_between_lobe_minimum_and_zero=False, **filter_kwargs):
        '''
        A wrapper around VisClean.fourier_filter specifically for
        filtering along the time axis with uniform fringe-rate weighting.

        Parameters
        ----------
        frate_centers: dict object,
            Dictionary with the center fringe-rate of each baseline in to_filter in units of mHz.
            keys are antpairpols.
        frate_half_widths: dict object
            Dictionary with the half widths of each fringe-rate window around the frate_centers in units of mHz.
            keys are antpairpols
        keys: list of visibilities to filter in the (i,j,pol) format.
          If None (the default), all visibilities are filtered.
        wgts: dictionary or DataContainer with all the same keys as self.data.
            Linear multiplicative weights to use for the fr filter. Default, use np.logical_not
            of self.flags. hera_filters.dspec.fourier_filter will renormalize to compensate.
        mode: string specifying filtering mode. See fourier_filter or hera_filters.dspec.fourier_filter for supported modes.
        skip_wgt: skips filtering rows with very low total weight (unflagged fraction ~< skip_wgt).
            Model is left as 0s, residual is left as data, and info is {'skipped': True} for that
            time. Skipped channels are then flagged in self.flags.
            Only works properly when all weights are all between 0 and 1.
        tol : float, optional. To what level are foregrounds subtracted.
        verbose: If True print feedback to stdout
        cache_dir: string, optional, path to cache file that contains pre-computed dayenu matrices.
         see hera_filters.dspec.dayenu_filter for key formats.
        read_cache: bool, If true, read existing cache files in cache_dir before running.
        write_cache: bool. If true, create new cache file with precomputed matrices
            that were not in previously loaded cache files.
        pre_filter_modes_between_lobe_minimum_and_zero: bool, optional
            Subtract power between the main-lobe and zero before applying main-lobe filter.
            This is to avoid having the main-lobe filter respond to any exceedingly high power
            at low fringe rates which can happen during Galaxy set.
            or if there is egregious cross-talk / ground pickup.
            This arg is only used if beamfitsfile is not None.
            Default is False.
        filter_kwargs: see fourier_filter for a full list of filter_specific arguments.

        Returns
        -------
        N/A

        Results are stored in:
          self.clean_resid: DataContainer formatted like self.data with only high-fringe-rate components
          self.clean_model: DataContainer formatted like self.data with only low-fringe-rate components
          self.clean_info: Dictionary of info from hera_filters.dspec.fourier_filter with the same keys as self.data
        '''
        if keys is None:
            keys = list(self.data.keys())
        # read in cache
        if not mode == 'clean':
            if read_cache:
                filter_cache = io.read_filter_cache_scratch(cache_dir)
            else:
                filter_cache = {}
            keys_before = list(filter_cache.keys())
        else:
            filter_cache = None

        if wgts is None:
            wgts = io.DataContainer({k: (~self.flags[k]).astype(float) for k in self.flags})
        if pre_filter_modes_between_lobe_minimum_and_zero:
            self.pre_filter_resid = DataContainer({})
            self.pre_filter_resid_flags = DataContainer({})
            filter_kwargs_no_data = copy.deepcopy(filter_kwargs)
            if 'data' in filter_kwargs_no_data:
                del filter_kwargs_no_data['data']
        for k in keys:
            if mode != 'clean':
                filter_kwargs['suppression_factors'] = [tol]
            else:
                filter_kwargs['tol'] = tol

            if pre_filter_modes_between_lobe_minimum_and_zero:
                min_frate_abs = np.min([np.abs(frate_centers[k] + frate_half_widths[k]),
                                        np.abs(frate_centers[k] - frate_half_widths[k])])
                # only pre-filter if main-lobe does not include zero fringe-rates.
                if not (frate_centers[k] + frate_half_widths[k] >= 0. and frate_centers[k] - frate_half_widths[k] <= 0.):
                    self.fourier_filter(keys=[k], filter_centers=[0.], filter_half_widths=[min_frate_abs],
                                        mode=mode, x=self.times * SDAY_KSEC,
                                        wgts=wgts, output_prefix='pre_filter',
                                        ax='time', cache=filter_cache, skip_wgt=skip_wgt, verbose=verbose, **filter_kwargs)
                else:
                    if 'data' not in filter_kwargs:
                        self.pre_filter_resid[k] = self.data[k]
                    else:
                        self.pre_filter_resid[k] = filter_kwargs['data'][k]
                    if 'flags' not in filter_kwargs:
                        self.pre_filter_resid_flags[k] = self.flags[k]
                    else:
                        self.pre_filter_flags[k] = filter_kwargs['flags'][k]
                # center at zero fringe-rate since some fourier_filter modes are
                # high pass only.
                phasor = np.exp(2j * np.pi * self.times * SDAY_KSEC * frate_centers[k])
                self.pre_filter_resid[k] /= phasor[:, None]
                filter_center_to_use = 0.0
                self.fourier_filter(keys=[k], filter_centers=[filter_center_to_use], filter_half_widths=[frate_half_widths[k]],
                                    mode=mode, x=self.times * SDAY_KSEC,
                                    wgts=wgts, data=self.pre_filter_resid, flags=self.pre_filter_resid_flags,
                                    ax='time', cache=filter_cache, skip_wgt=skip_wgt, verbose=verbose, **filter_kwargs_no_data)
            else:
                # center sky-modes at zero fringe-rate.
                phasor = np.exp(2j * np.pi * self.times * SDAY_KSEC * frate_centers[k])
                if 'data' in filter_kwargs:
                    input_data = filter_kwargs['data']
                else:
                    input_data = self.data
                input_data[k] /= phasor[:, None]
                filter_center_to_use = 0.0
                self.fourier_filter(keys=[k], filter_centers=[filter_center_to_use], filter_half_widths=[frate_half_widths[k]],
                                    mode=mode, x=self.times * SDAY_KSEC,
                                    flags=self.flags, wgts=wgts,
                                    ax='time', cache=filter_cache, skip_wgt=skip_wgt, verbose=verbose, **filter_kwargs)

            # recenter data in fringe-rate by multiplying back the phaser.
            if 'output_prefix' in filter_kwargs:
                filtered_data = getattr(self, filter_kwargs['output_prefix'] + '_data')
                filtered_model = getattr(self, filter_kwargs['output_prefix'] + '_model')
                filtered_resid = getattr(self, filter_kwargs['output_prefix'] + '_resid')
            else:
                filtered_data = self.clean_data
                filtered_model = self.clean_model
                filtered_resid = self.clean_resid
            filtered_data[k] *= phasor[:, None]
            filtered_model[k] *= phasor[:, None]
            filtered_resid[k] *= phasor[:, None]
            if 'data' in filter_kwargs:
                input_data = filter_kwargs['data']
            else:
                input_data = self.data
            input_data[k] *= phasor[:, None]
        if not mode == 'clean':
            if write_cache:
                filter_cache = io.write_filter_cache_scratch(filter_cache, cache_dir, skip_keys=keys_before)


def time_avg_data_and_write(input_data_list, output_data, t_avg, baseline_list=None,
                            wgt_by_nsample=True, wgt_by_favg_nsample=False, rephase=False,
                            filetype='uvh5', verbose=False, clobber=False, flag_output=None, **read_kwargs):
    """Time-averaging with a baseline cornerturn


    Parameters
    ----------
    intput_data_list: list of strings.
        list of names of input data file to read baselines across.
    baseline_list: list
        list of antpolpairs or antpairs
    output_data: str
        name of output data file.
    t_avg: float
        width of time-averaging interval in seconds.
    wgt_by_nsample: bool, optional
        weight by nsamples in time average
        default is True
    wgt_by_favg_nsample : bool
        If True, perform time average weighting by averaging the number of samples across
        frequency for each integration. Mutually exclusive with wgt_by_nsample. Default False.
    rephase: bool, optional
        rephase each time bin to central lst.
    filetype : str, optional
        specify if uvh5, miriad, ect...
        default is uvh5.
    verbose: bool, optional
        if true, more outputs.
        default is False
    clobber: bool, optional
        if true, overwrite output ata if it already exists.
        default is False
    flag_output: str, optional
        string to write flag output. Optional.
    read_kwargs: kwargs dict
        additional kwargs for for io.HERAData.read()
    Returns
    -------
    None
    """
    if baseline_list is not None and len(baseline_list) == 0:
        warnings.warn("Length of baseline list is zero."
                      "This can happen under normal circumstances when there are more files in datafile_list then baselines."
                      "in your dataset. Exiting without writing any output.", RuntimeWarning)
    else:
        fr = FRFilter(input_data_list, filetype=filetype)
        fr.read(bls=baseline_list, **read_kwargs)

        fr.timeavg_data(fr.data, fr.times, fr.lsts, t_avg, flags=fr.flags, nsamples=fr.nsamples,
                        wgt_by_nsample=wgt_by_nsample, wgt_by_favg_nsample=wgt_by_favg_nsample, rephase=rephase)
        fr.write_data(fr.avg_data, output_data, overwrite=clobber, flags=fr.avg_flags, filetype=filetype,
                      nsamples=fr.avg_nsamples, times=fr.avg_times, lsts=fr.avg_lsts)
        if flag_output is not None:
            uv_avg = UVData()
            uv_avg.read(output_data)
            uv_avg.use_future_array_shapes()
            uvf = UVFlag(uv_avg, mode='flag', copy_flags=True)
            uvf.to_waterfall(keep_pol=False, method='and')
            uvf.write(flag_output, clobber=clobber)


def tophat_frfilter_argparser(mode='clean'):
    '''Arg parser for commandline operation of tophat fr-filters.

    Parameters
    ----------
    mode : string, optional.
        Determines sets of arguments to load.
        Can be 'clean', 'dayenu', or 'dpss_leastsq'.

    Returns
    -------
    argparser
        argparser for tophat fringe-rate (time-domain) filtering for specified filtering mode

    '''
    ap = vis_clean._filter_argparser()
    filt_options = ap.add_argument_group(title='Options for the fr-filter')
    filt_options.add_argument("--frate_width_multiplier", type=float, default=1.0, help="Fraction of maximum sky-fringe-rate to interpolate / filter. "
                                                                                        "Used if select_mainlobe is False and max_frate_coeffs not specified.")
    filt_options.add_argument("--frate_standoff", type=float, default=0.0, help="Standoff in fringe-rate to filter [mHz]. "
                                                                                "Used of select_mainlobe is False and max_frate_coeffs not specified.")
    filt_options.add_argument("--min_frate_half_width", type=float, default=0.025, help="minimum half-width of fringe-rate filter, regardless of baseline length in mHz. "
                                                                                        "Default is 0.025.")
    filt_options.add_argument("--max_frate_half_width", type=float, default=np.inf, help="maximum half-width of fringe-rate filter, regardless of baseline length in mHz. "
                                                                                         "Default is np.inf, i.e. no limit.")
    filt_options.add_argument("--max_frate_coeffs", type=float, default=None, nargs=2, help="Maximum fringe-rate coefficients for the model max_frate [mHz] = x1 * EW_bl_len [ m ] + x2. "
                                                                                            "Providing these overrides the sky-based fringe-rate determination! Default is None.")
    filt_options.add_argument("--skip_autos", default=False, action="store_true", help="Exclude autos from filtering.")
    filt_options.add_argument("--beamfitsfile", default=None, type=str, help="Path to UVBeam beamfits file to use for determining isotropic sky fringe-rates to filter.")
    filt_options.add_argument("--percentile_low", default=5.0, type=float, help="Reject fringe-rates with beam power below this percentile if beamfitsfile is provided.")
    filt_options.add_argument("--percentile_high", default=95.0, type=float, help="Reject fringe-rates with beam power above this percentile if beamfitsfile is provided.")
    filt_options.add_argument("--taper", default='none', type=str, help="Weight fringe-rates at different frequencies by the square of this taper if beamfitsfile is provided.")
    filt_options.add_argument("--fr_freq_skip", default=1, type=int, help="fr_freq_skip: int, optional "
                                                                          "bin fringe rates from every freq_skip channels. "
                                                                          "default is 1 -> takes a long time. We recommend setting this to be larger.")
    filt_options.add_argument("--pre_filter_modes_between_lobe_minimum_and_zero", action="store_true", default=False, help="Subtract emission between the main-lobe fringe-rate region and zero "
                                                                                                                           "before applying main-lobe fringe rate filter. This is to prevent "
                                                                                                                           "the main-lobe filter to responding to overwhelmingly bright emission "
                                                                                                                           "centered at zero fringe-rate, which can happen if we have lots of cross-talk.")
    filt_options.add_argument("--wgt_by_nsample", default=False, action="store_true", help="Use weights proportional to nsamples during FRF. Default is to just use flags by nsamples.")
    from .smooth_cal import _pair
    filt_options.add_argument("--lst_blacklists", type=_pair, default=[], nargs='+', help="space-separated list of dash-separted pairs of LSTs in hours bounding (inclusively) \
                                                                                           blacklisted LSTs assigned zero weight (by default) during FRF, e.g. '3-4 10-12 23-.5'")
    filt_options.add_argument("--blacklist_wgt", type=float, default=0.0, help="Relative weight to assign to blacklisted lsts compared to 1.0. Default 0.0 \
                                                                                means no weight. Note that 0.0 will create problems for DPSS at edge times and frequencies.")

    desc = ("Filtering case ['max_frate_coeffs', 'uvbeam', 'sky']",
            "If case == 'max_frate_coeffs', then determine fringe rate centers",
            "and half-widths based on the max_frate_coeffs arg (see below).",
            "If case == 'uvbeam', then determine fringe rate centers and half widths",
            "from histogram of main-beam wrt instantaneous sky fringe rates.",
            "If case == 'sky': then use fringe-rates corresponding to range of ",
            "instantanous fringe-rates that include sky emission.")
    filt_options.add_argument("--case", default="sky", help=' '.join(desc), type=str)
    return ap


def load_tophat_frfilter_and_write(datafile_list, case, baseline_list=None, calfile_list=None,
                                   Nbls_per_load=None, spw_range=None, external_flags=None,
                                   factorize_flags=False, time_thresh=0.05, wgt_by_nsample=False,
                                   lst_blacklists=None, blacklist_wgt=0.0,
                                   res_outfilename=None, CLEAN_outfilename=None, filled_outfilename=None,
                                   clobber=False, add_to_history='', avg_red_bllens=False, polarizations=None,
                                   overwrite_flags=False,
                                   flag_yaml=None, skip_autos=False, beamfitsfile=None, verbose=False,
                                   read_axis=None,
                                   percentile_low=5., percentile_high=95.,
                                   frate_standoff=0.0, frate_width_multiplier=1.0,
                                   min_frate_half_width=0.025, max_frate_half_width=np.inf,
                                   max_frate_coeffs=None, fr_freq_skip=1,
                                   **filter_kwargs):
    '''
    A tophat fr-filtering method that only simultaneously loads and writes user-provided
    list of baselines. This is to support parallelization over baseline (rather then time) if baseline_list is specified.

    Arguments:
        datafile_list: list of data files to perform cross-talk filtering on
        case: str
            Filtering case ['max_frate_coeffs', 'uvbeam', 'frate_profiles', 'sky']
            If case == 'max_frate_coeffs', then determine fringe rate centers
            and half-widths based on the max_frate_coeffs arg (see below).
            If case == 'uvbeam', then determine fringe rate centers and half widths
            from histogram of main-beam wrt instantaneous sky fringe rates
            If case == 'sky': then use fringe-rates corresponding to range of
            instantanous fringe-rates that include sky emission.
        baseline_list: list of antenna-pair 2-tuples to filter and write out from the datafile_list.
                       If None, load all baselines in files. Default is None.
        calfile_list: optional list of calibration files to apply to data before fr filtering
        Nbls_per_load: int, the number of baselines to load at once.
            If None, load all baselines at once. default : None.
        spw_range: 2-tuple or 2-list, spw_range of data to filter.
        factorize_flags: bool, optional
            If True, factorize flags before running fr filter. See vis_clean.factorize_flags.
        time_thresh : float, optional
            Fractional threshold of flagged pixels across time needed to flag all times
            per freq channel. It is not recommend to set this greater than 0.5.
            Fully flagged integrations do not count towards triggering time_thresh.
        wgt_by_nsample : bool, optional
            If True, perform FRF with weighting proportional to nsamples (flags still get 0 weight).
            Default False, meaning use only ~flags as weights.
        lst_blacklists : list of 2-tuples, optional
            List of LST ranges (in hours, inclusive) to assign zero weight (by default) when performing FRF (regardless
            of whether or not they are flagged). Ranges crossing the 24h brach cut, e.g. [(23, 1)], are allowed.
        blacklist_wgt : float, optional
            Relative weight to assign to blacklisted lsts compared to 1.0. Default 0.0 means no weight.
            Note that 0.0 will create problems for DPSS at edge times and frequencies.
        res_outfilename: path for writing the filtered visibilities with flags
        CLEAN_outfilename: path for writing the CLEAN model visibilities (with the same flags)
        filled_outfilename: path for writing the original data but with flags unflagged and replaced
            with CLEAN models wherever possible
        clobber: if True, overwrites existing file at the outfilename
        add_to_history: string appended to the history of the output file
        avg_red_bllens: bool, if True, round baseline lengths to redundant average. Default is False.
        polarizations : list of polarizations to process (and write out). Default None operates on all polarizations in data.
        overwrite_flags : bool, if true reset data flags to False except for flagged antennas.
        flag_yaml: path to manual flagging text file.
        skip_autos: bool, if true, exclude autocorrelations from filtering. Default is False.
                 autos will still be saved in the resides as zeros, as the models as the data (with original flags).
        beamfitsfile: str, optional.
            path to UVBeam object for calculating frate bounds.
            default is None -> use other filter kwargs to determine fringe rate bounds.
        verbose: bool, optional
            Helpful text outputs.
        read_axis: str, optional
            optional parameter to pass as the axis arg to io.HERAData.read()
        percentile_low: float, optional
            if uvb is provided, filter fringe rates that are larger then this percentile
            in the histogram of beam squared powers for instantaneous fringe rates.
        percentile_high: float, optional
            if uvb is provided, filter fringe rates that are smaller then this percentile
            in the histogram of beam squared powers for instantaneous fringe rates.
        frate_standoff: float, optional
            additional fringe-rate to add to min and max of computed fringe-rate bounds [mHz]
            to add to analytic calculation of fringe-rate bounds for emission on the sky.
            default = 0.0.
        frate_width_multiplier: float, optional
            fraction of horizon to fringe-rate filter.
            default is 1.0
        min_frate_half_width: float, optional
            minimum half-width of fringe-rate filter, regardless of baseline length in mHz.
            Default is 0.025
        max_frate_half_width: float, optional
            maximum half-width of fringe-rate filter, regardless of baseline length in mHz.
            Default is np.inf, i.e. no limit
        max_frate_coeffs, 2-tuple float
            Maximum fringe-rate coefficients for the model max_frate [mHz] = x1 * | EW_bl_len | [ m ] + x2."
            Providing these overrides the sky-based fringe-rate determination! Default is None.
        fr_freq_skip: int, optional
            bin fringe rates from every freq_skip channels.
            default is 1 -> takes a long time. We recommend setting this to be larger.
            only used if case == 'uvbeam'
        filter_kwargs: additional keyword arguments to be passed to FRFilter.tophat_frfilter()
    '''
    if baseline_list is not None and Nbls_per_load is not None:
        raise NotImplementedError("baseline loading and partial i/o not yet implemented.")
    hd = io.HERAData(datafile_list, filetype='uvh5')
    if baseline_list is not None and len(baseline_list) == 0:
        warnings.warn("Length of baseline list is zero."
                      "This can happen under normal circumstances when there are more files in datafile_list then baselines."
                      "in your dataset. Exiting without writing any output.", RuntimeWarning)
    else:
        if baseline_list is None:
            if len(hd.filepaths) > 1:
                baseline_list = list(hd.antpairs.values())[0]
            else:
                baseline_list = hd.antpairs
        if spw_range is None:
            spw_range = [0, hd.Nfreqs]
        freqs = hd.freq_array.flatten()[spw_range[0]:spw_range[1]]
        baseline_antennas = []
        for blpolpair in baseline_list:
            baseline_antennas += list(blpolpair[:2])
        baseline_antennas = np.unique(baseline_antennas).astype(int)
        if calfile_list is not None:
            cals = io.HERACal(calfile_list)
            cals.read(antenna_nums=baseline_antennas, frequencies=freqs)
        else:
            cals = None
        if polarizations is None:
            if len(hd.filepaths) > 1:
                polarizations = list(hd.pols.values())[0]
            else:
                polarizations = hd.pols
        if Nbls_per_load is None:
            Nbls_per_load = len(baseline_list)
        for i in range(0, len(baseline_list), Nbls_per_load):
            frfil = FRFilter(hd, input_cal=cals)
            frfil.read(bls=baseline_list[i:i + Nbls_per_load],
                       frequencies=freqs, polarizations=polarizations, axis=read_axis)
            if avg_red_bllens:
                frfil.avg_red_baseline_vectors()
            if external_flags is not None:
                frfil.apply_flags(external_flags, overwrite_flags=overwrite_flags)
            if flag_yaml is not None:
                frfil.apply_flags(flag_yaml, overwrite_flags=overwrite_flags, filetype='yaml')
            if factorize_flags:
                frfil.factorize_flags(time_thresh=time_thresh, inplace=True)
            keys = frfil.data.keys()
            if skip_autos:
                keys = [bl for bl in keys if bl[0] != bl[1]]
            if beamfitsfile is not None:
                uvb = UVBeam()
                uvb.read_beamfits(beamfitsfile)
                uvb.use_future_array_shapes()
            else:
                uvb = None
            if len(keys) > 0:
                # figure out frige rate centers and half-widths
                assert case in ['sky', 'max_frate_coeffs', 'uvbeam'], f'case={case} is not valid.'
                frate_centers, frate_half_widths = select_tophat_frates(uvd=frfil.hd, blvecs=frfil.blvecs,
                                                                        case=case, keys=keys, uvb=uvb,
                                                                        frate_standoff=frate_standoff,
                                                                        frate_width_multiplier=frate_width_multiplier,
                                                                        min_frate_half_width=min_frate_half_width,
                                                                        max_frate_half_width=max_frate_half_width,
                                                                        max_frate_coeffs=max_frate_coeffs,
                                                                        percentile_low=percentile_low,
                                                                        percentile_high=percentile_high,
                                                                        fr_freq_skip=fr_freq_skip,
                                                                        verbose=verbose)

                # Build weights using flags, nsamples, and exlcuded lsts
                wgts = io.DataContainer({k: (~frfil.flags[k]).astype(float) for k in frfil.flags})
                for k in wgts:
                    if wgt_by_nsample:
                        wgts[k] *= frfil.nsamples[k]
                    if lst_blacklists is not None:
                        for lb in lst_blacklists:
                            if lb[0] < lb[1]:
                                is_blacklisted = (frfil.lsts >= lb[0] * np.pi / 12) & (frfil.lsts <= lb[1] * np.pi / 12)
                            else:
                                is_blacklisted = (frfil.lsts >= lb[0] * np.pi / 12) | (frfil.lsts <= lb[1] * np.pi / 12)
                            wgts[k][is_blacklisted, :] = wgts[k][is_blacklisted, :] * blacklist_wgt

                # run tophat filter
                frfil.tophat_frfilter(frate_centers=frate_centers, frate_half_widths=frate_half_widths,
                                      keys=keys, verbose=verbose, wgts=wgts, **filter_kwargs)
            else:
                frfil.clean_data = DataContainer({})
                frfil.clean_flags = DataContainer({})
                frfil.clean_resid = DataContainer({})
                frfil.clean_resid_flags = DataContainer({})
                frfil.clean_model = DataContainer({})
            # put autocorr data into filtered data containers if skip_autos = True.
            # so that it can be written out into the filtered files.
            if skip_autos:
                for bl in frfil.data.keys():
                    if bl[0] == bl[1]:
                        frfil.clean_data[bl] = frfil.data[bl]
                        frfil.clean_flags[bl] = frfil.flags[bl]
                        frfil.clean_resid[bl] = frfil.data[bl]
                        frfil.clean_model[bl] = np.zeros_like(frfil.data[bl])
                        frfil.clean_resid_flags[bl] = frfil.flags[bl]

            frfil.write_filtered_data(res_outfilename=res_outfilename, CLEAN_outfilename=CLEAN_outfilename,
                                      filled_outfilename=filled_outfilename, partial_write=Nbls_per_load < len(baseline_list),
                                      clobber=clobber, add_to_history=add_to_history,
                                      extra_attrs={'Nfreqs': frfil.hd.Nfreqs, 'freq_array': frfil.hd.freq_array, 'channel_width': frfil.hd.channel_width})
            frfil.hd.data_array = None  # this forces a reload in the next loop


def time_average_argparser():
    """
    Define an argument parser for time averaging data.

    Parameters
    ----------
    None

    Returns
    -------
    ap : ArgumentParser object
        An instance of an `ArgumentParser` that has the relevant options defined.
    """
    ap = argparse.ArgumentParser(description="Time-average data.")
    ap.add_argument("input_data_list", type=str, nargs="+", help="list of data files to use for determining baseline chunk if performing cornerturn.")
    ap.add_argument("output_data", type=str, help="name of data file to write out time-average.")
    ap.add_argument("--cornerturnfile", type=str, help="name of data file to determine baselines based on posotion in input_data_list."
                                                       "If provided, will perform cornerturn from time to baselines.")
    ap.add_argument("--t_avg", type=float, help="number of seconds to average over.", default=None)
    ap.add_argument("--rephase", default=False, action="store_true", help="rephase to averaging window center.")
    ap.add_argument("--dont_wgt_by_nsample", default=False, action="store_true", help="don't weight averages by nsample. Default is to wgt by nsamples.")
    ap.add_argument("--wgt_by_favg_nsample", default=False, action="store_true", help="weight each integration by frequency-averaged nsamples.")
    ap.add_argument("--clobber", default=False, action="store_true", help="Overwrite output files.")
    ap.add_argument("--verbose", default=False, action="store_true", help="verbose output.")
    ap.add_argument("--flag_output", default=None, type=str, help="optional filename to save a separate copy of the time-averaged flags as a uvflag object.")
    ap.add_argument("--filetype", default="uvh5", type=str, help="optional filetype specifier. Default is 'uvh5'. Set to 'miriad' if reading miriad files etc...")

    return ap
