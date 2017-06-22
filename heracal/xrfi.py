'''Module for all things Radio Frequency Interference Flagging'''
import numpy as np
from scipy.signal import medfilt


def medmin(d):
    """Calculate the median minus median statistic of array.

    Args:
        d (array): 2D data array

    Returns:
        (array): array with the statistic applied.
    """
    # return np.median(np.min(chisq,axis=0))
    mn = np.min(d, axis=0)
    return 2 * np.median(mn) - np.min(mn)


def medminfilt(d, K=8):
    """Filter an array on scales of K indexes with medmin.

    Args:
        d (array): 2D data array.
        K (int, optional): integer representing box size to apply statistic.

    Returns:
        array: filtered array. Same shape as input array.
    """
    d_sm = np.empty_like(d)
    for i in xrange(d.shape[0]):
        for j in xrange(d.shape[1]):
            i0, j0 = max(0, i - K), max(0, j - K)
            i1, j1 = min(d.shape[0], i + K), min(d.shape[1], j + K)
            d_sm[i, j] = medmin(d[i0:i1, j0:j1])
    return d_sm


def watershed_flag(d, f=None, sig_init=6, sig_adj=2):
    '''Generates a mask for flags using a watershed algorithm.

    Returns a watershed flagging of an array that is in units of standard
    deviation (i.e. how many sigma the datapoint is from the center).

    Args:
        d (array): 2D array to perform watershed on.
            d should be in units of standard deviations.
        f (array, optional): input flags. Same size as d.
        sig_init (int): number of sigma to flag above, initially.
        sig_adj (int): number of sigma to flag above for points
            near flagged points.

    Returns:
        bool array: Array of mask values for d.
    '''
    # mask off any points above 'sig' sigma and nan's.
    f1 = np.ma.array(d, mask=np.where(d > sig_init, 1, 0))
    f1.mask |= np.isnan(f1)
    if not f is None:
        f1.mask |= f

    # Loop over flagged points and examine adjacent points to see if they exceed sig_adj
    # Start the watershed
    prevx, prevy = 0, 0
    x, y = np.where(f1.mask)
    while x.size != prevx and y.size != prevy:
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            prevx, prevy = x.size, y.size
            xp, yp = (x + dx).clip(0,
                                   f1.shape[0] - 1), (y + dy).clip(0, f1.shape[1] - 1)
            i = np.where(f1[xp, yp] > sig_adj)[0]  # if sigma > 'sigl'
            f1.mask[xp[i], yp[i]] = 1
            x, y = np.where(f1.mask)
    return f1.mask


def xrfi_simple(d, f=None, nsig_df=6, nsig_dt=6, nsig_all=0):
    '''Flag RFI using derivatives in time and frequency.

    Args:
        d (array): 2D data array to flag on with first axis being times
            and second axis being frequencies.
        f (array, optional): input flags, defaults to zeros
        nsig_df (float, optional): number of sigma above median to flag in frequency direction
        nsig_dt (float, optional): number of sigma above median to flag in time direction
        nsig_all (float, optional): overall flag above some sigma. Skip if 0.

    Returns:
        bool array: mask array for flagging.

    '''
    if f is None:
        f = np.zeros(d.shape, dtype=np.bool)
    if nsig_df > 0:
        d_df = d[:, 1:-1] - .5 * (d[:, :-2] + d[:, 2:])
        d_df2 = np.abs(d_df)**2
        sig2 = np.median(d_df2, axis=1)
        sig2.shape = (-1, 1)
        f[:, 0] = 1
        f[:, -1] = 1
        f[:, 1:-1] = np.where(d_df2 / sig2 > nsig_df**2, 1, f[:, 1:-1])
    if nsig_dt > 0:
        d_dt = d[1:-1, :] - .5 * (d[:-2, :] + d[2:, :])
        d_dt2 = np.abs(d_dt)**2
        sig2 = np.median(d_dt2, axis=0)
        sig2.shape = (1, -1)
        f[0, :] = 1
        f[-1, :] = 1
        f[1:-1, :] = np.where(d_dt2 / sig2 > nsig_dt**2, 1, f[1:-1, :])
    if nsig_all > 0:
        ad = np.abs(d)
        med = np.median(ad)
        sig = np.sqrt(np.median(np.abs(ad - med)**2))
        f = np.where(ad > med + nsig_all * sig, 1, f)
    return f


def detrend_deriv(d, dt=True, df=True):
    '''XXX This only works ok on sparse RFI.'''
    if df:
        d_df = np.empty_like(d)
        d_df[:, 1:-1] = (d[:, 1:-1] - .5 *
                         (d[:, :-2] + d[:, 2:])) / np.sqrt(1.5)
        d_df[:, 0] = (d[:, 0] - d[:, 1]) / np.sqrt(2)
        d_df[:, -1] = (d[:, -1] - d[:, -2]) / np.sqrt(2)
    else:
        d_df = d
    if dt:
        d_dt = np.empty_like(d_df)
        d_dt[1:-1] = (d_df[1:-1] - .5 * (d_df[:-2] + d_df[2:])) / np.sqrt(1.5)
        d_dt[0] = (d_df[0] - d_df[1]) / np.sqrt(2)
        d_dt[-1] = (d_df[-1] - d_df[-2]) / np.sqrt(2)
    else:
        d_d = d_df
    d2 = np.abs(d_dt)**2
    # model sig as separable function of 2 axes
    sig_f = np.median(d2, axis=0)
    sig_f.shape = (1, -1)
    sig_t = np.median(d2, axis=1)
    sig_t.shape = (-1, 1)
    sig = np.sqrt(sig_f * sig_t / np.median(sig_t))
    return d_dt / sig


def detrend_medminfilt(d, K=8):
    """Detrend array using medminfilt statistic. See medminfilt.

    Args:
        d (array): data array to detrend.
        K (int): box size to apply medminfilt over

    Returns:
        bool array: boolean array of flags
    """
    d_sm = medminfilt(np.abs(d), 2 * K + 1)
    d_rs = d - d_sm
    d_sq = np.abs(d_rs)**2
    # puts minmed on same scale as average
    sig = np.sqrt(medminfilt(d_sq, 2 * K + 1)) * (K / .64)
    f = d_rs / sig
    return f


def detrend_medfilt(d, K=8):
    """Detrend array using a median filter.

    Args:
        d (array): data array to detrend.
        K (int, optional): box size to apply medminfilt over

    Returns:
        bool array: boolean array of flags
    """
    d = np.concatenate([d[K - 1::-1], d, d[:-K - 1:-1]], axis=0)
    d = np.concatenate([d[:, K - 1::-1], d, d[:, :-K - 1:-1]], axis=1)
    d_sm = medfilt(d, 2 * K + 1)
    d_rs = d - d_sm
    d_sq = np.abs(d_rs)**2
    # puts median on same scale as average
    sig = np.sqrt(medfilt(d_sq, 2 * K + 1) / .456)
    f = d_rs / sig
    return f[K:-K, K:-K]


def xrfi(d, f=None, K=8, sig_init=6, sig_adj=2):
    """Run best rfi exciion we have. Uses detrending and watershed algorithms above.
    Args: 
        d (array): 2D of data array.
        f (array, optional): input flag array
        K (int, optional): Box size for detrend
        sig_init (float, optional): initial sigma to flag.
        sig_adj (float, optional): number of sigma to flag adjacent to flagged data (sig_init) 

    Returns:
        bool array: array of flags
    """
    nsig = detrend_medfilt(d, K=K)
    f = watershed_flag(np.abs(nsig), f=f, sig_init=sig_init, sig_adj=sig_adj)
    return f

# XXX split off median filter as one type of flagger
