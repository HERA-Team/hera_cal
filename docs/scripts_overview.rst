Scripts Overview
===============
These scripts constitute the bulk of the HERA calibration pipeline in the order they are normally run.


get_bad_ants.py
---------------
Reads in a miriad uv raw data file to determine which antennas appear dead or otherwise bad. Output is a text file that ends with ".badants.txt" and contains a comma-separated list of bad antennas.

Full usage instuctions available using ``python get_bad_ants.py -h`` in the scripts directory.


firstcal_run.py
---------------
Uses ratios of visibilities of nominally redundant baselines to determine the delay associated with each antenna (up to some overall delay). Reads in a miriad uv raw data file. Bad antennas are specified on the command line. Saves delays in the calfits format using the "cal_type = delay" mode (as opposed to a frequency-dependent quantity) to a file ends with ".first.calfits".

Full usage instuctions available using ``python firstcal.py -h`` in the scripts directory.

omni_run.py
---------------
Runs the hera_cal adapation of the omnical package. Includes logcal, lincal, and remove_degen. Reads in both the miriad uv raw data file and the ".first.calfits" file from firstcal. Outputs three results files. ".vis.uvfits" has the omnical model visibilities (one per unique baseline); ".xtalk.uvfits" has the time-averaged visibilities used for cross-talk estimation (one per baseline, but only one time sample); and ".omni.calfits" which includes the combined first-cal and omnical best-guess gains and chi^2 per antenna.

Full usage instuctions available using ``python omni_run.py -h`` in the scripts directory.

omni_xrfi.py
---------------
Uses a watershed algorithm to identify outliers. Takes in omnical result ".omni.calfits." Currently uses an average chi^2 over all antennas, but will eventually use the global chi^2 determined by omnical. Creates a new calfits file ending in ".omni.xrfi.calfits" that includes both omnical gains and chi^2 per antenna but has updated flags to reflect RFI excision.

Full usage instuctions available using ``python xrfi.py -h`` in the scripts directory.

omni_apply.py
---------------
Applies a calfits file to a miriad uv raw data file to produce a calibrated miriad uv data file with a specified string appended to the end of the filename (e.g. ".uvcF" or ".uvcO" or ".uvcOR").

Full usage instuctions available using ``python omni_apply.py -h`` in the scripts directory.
