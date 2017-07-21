import omni
import firstcal
import redcal

def get_HERA_aa(freqs,calfile='hera_cm',
                locations_file='data/hera_ant_locs_05_16_2017.csv',
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

    exec('from {calfile} import get_aa'.format(calfile=calfile))
    if calfile!='hera_cm': #load an alternate cal file
        #WARNING, array position is being set by whatever happens to be in your cal file
        #IT IS PROBABLY NOT UP TO DATE
        return get_aa(freqs)
    else:

        exec('from {calfile} import get_aa'.format(calfile=calfile))
        return get_aa(freqs,locations_file,array_epoch_jd)
