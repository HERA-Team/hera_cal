from copy import deepcopy
import argparse

def filter_argparser():
    "Core Arg parser for commandline operation of hera_cal.delay_filter and hera_cal.xtalk_filter"
    a = argparse.ArgumentParser(description="Perform delay filter of visibility data.")
    a.add_argument("infilename", type=str, help="path to visibility data file to delay filter")
    a.add_argument("--filetype_in", type=str, default='uvh5', help='filetype of input data files (default "uvh5")')
    a.add_argument("--filetype_out", type=str, default='uvh5', help='filetype for output data files (default "uvh5")')
    a.add_argument("--calfile", default=None, type=str, help="optional string path to calibration file to apply to data before delay filtering")
    a.add_argument("--partial_load_Nbls", default=None, type=int, help="the number of baselines to load at once (default None means load full data")
    a.add_argument("--res_outfilename", default=None, type=str, help="path for writing the filtered visibilities with flags")
    a.add_argument("--clobber", default=False, action="store_true", help='overwrites existing file at outfile')
    a.add_argument("--spw_range", type=int, default=None, nargs=2, help="spectral window of data to foreground filter.")
    return a

#------------------------------------------
# Here are arg-parsers clean filtering
#------------------------------------------

def clean_argparser():
    '''Arg parser for CLEAN.'''
    a = filter_argparser()

    a.add_argument("--CLEAN_outfilename", default=None, type=str, help="path for writing the filtered model visibilities (with the same flags)")
    a.add_argument("--filled_outfilename", default=None, type=str, help="path for writing the original data but with flags unflagged and replaced with filtered models wherever possible")
    clean_options = a.add_argument_group(title='Options for CLEAN')
    clean_options.add_argument("--window", type=str, default='blackman-harris', help='window function for frequency filtering (default "blackman-harris",\
                              see uvtools.dspec.gen_window for options')
    clean_options.add_argument("--skip_wgt", type=float, default=0.1, help='skips filtering and flags times with unflagged fraction ~< skip_wgt (default 0.1)')
    clean_options.add_argument("--maxiter", type=int, default=100, help='maximum iterations for aipy.deconv.clean to converge (default 100)')
    clean_options.add_argument("--edgecut_low", default=0, type=int, help="Number of channels to flag on lower band edge and exclude from window function.")
    clean_options.add_argument("--edgecut_hi", default=0, type=int, help="Number of channels to flag on upper band edge and exclude from window function.")
    clean_options.add_argument("--gain", type=float, default=0.1, help="Fraction of residual to use in each iteration.")
    clean_options.add_argument("--alpha", type=float, default=.5, help="If window='tukey', use this alpha parameter (default .5).")
    return a

#------------------------------------------
# Here are are parsers for linear filters.
#------------------------------------------

def linear_argparser():
    '''Arg parser for commandline operation of hera_cal.delay_filter in various linear modes.'''
    a = delay_filter_argparser()
    cache_options = a.add_argument_group(title='Options for caching')
    a.add_argument("--write_cache", default=False, action="store_true", help="if True, writes newly computed filter matrices to cache.")
    a.add_argument("--cache_dir", type=str, default=None, help="directory to store cached filtering matrices in.")
    a.add_argument("--read_cache", default=False, action="store_true", help="If true, read in cache files in directory specified by cache_dir.")
    return a

#----------------------------------------
# Arg-parser for delay-filtering.
#---------------------------------------

def delay_filter_argparser(mode='clean'):
    '''Core Arg parser for commandline operation of delay filters.'''
    if mode == 'clean':
        a = clean_argparser()
    elif mode in ['linear', 'dayenu', 'dpss_leastsq']:
        a = linear_argparser()
    filt_options = a.add_argument_group(title='Options for the delay filter')
    filt_options.add_argument("--standoff", type=float, default=15.0, help='fixed additional delay beyond the horizon (default 15 ns)')
    filt_options.add_argument("--horizon", type=float, default=1.0, help='proportionality constant for bl_len where 1.0 (default) is the horizon\
                              (full light travel time)')
    filt_options.add_argument("--tol", type=float, default=1e-9, help='Threshold for foreground subtraction (default 1e-9)')
    filt_options.add_argument("--min_dly", type=float, default=0.0, help="A minimum delay threshold [ns] used for filtering.")
    return a

#------------------------------------------
# Here are arg-parsers for xtalk-filtering.
#------------------------------------------

def xtalk_filter_argparser(mode='clean'):
    '''Core Arg parser for commandline operation of delay filters.'''
    if mode == 'clean':
        a = clean_argparser()
    elif mode in ['linear', 'dayenu', 'dpss_leastsq']:
        a = linear_argparser()
    filt_options = a.add_argument_group(title='Options for the cross-talk filter')
    a.add_argument("--max_frate_coeffs", type=float, default=None, nargs=2, help="Maximum fringe-rate coefficients for the model max_frate [mHz] = x1 * EW_bl_len [ m ] + x2.")
    return a
