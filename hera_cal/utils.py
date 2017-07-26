import os
def get_HERA_aa(freqs,calfile='hera_cm',
                locations_file=None,
                array_epoch_jd=None):
    # create an aipy AntennaArray object with the HERA configuration as it
    # was on array_epoch_jd.
    #
    # A default file is supplied but will soon be out of date.
    #   - create a new one with hera_mc/scripts/XXX.py
    #
    # NOTE, if you didn't specify a array_epoch_jd, one will be chosen for you
    # by the cal file.
    #
    # Also supports loading of old style cal files, just set calfile away from default
    #

    #sometimes the default option is None, but we really mean no input
    if calfile is None:
        calfile='hera_cm'
        #set defaults that hera_cm needs.
    if locations_file is None:
        locations_file = os.path.join(os.path.dirname(__file__), 'data/hera_ant_locs_05_16_2017.csv')
    if array_epoch_jd is None:
        array_epoch_jd = 2457458  #need a date


    exec('from {calfile} import get_aa'.format(calfile=calfile))
    if calfile!='hera_cm': #load an alternate cal file
        #WARNING, array position is being set by whatever happens to be in your cal file
        #IT IS PROBABLY NOT UP TO DATE
        return get_aa(freqs)
    else:
        return get_aa(freqs,locations_file,array_epoch_jd)
