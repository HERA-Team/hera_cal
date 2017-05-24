Scripts Overview
===============
These scripts constitute the bulk of the HERA calibration pipeline.


get_bad_ants.py
---------------
Reads in a miriad uv data file to determine which antennas appear dead or otherwise bad. Output is a text file that ends with ".badants.txt" and contains a comma-separated list of bad antennas.

Full usage instuctions available using ``python get_bad_ants.py -h`` in the scripts directory.


firstcal.py
---------------
Uses ratios of visibilities of nominally redundant baselines to determine the delay associated with each antenna (up to some overall delay). Saves delays as frequency-dependent gains with magnitude unity in a calfits file, as defined in pyuvdata, that ends with ".first.calfits".

Full usage instuctions available using ``python firstcal.py -h`` in the scripts directory.

omni_run.py
---------------
XXX:write stuff

Full usage instuctions available using ``python omni_run.py -h`` in the scripts directory.

XXX xrfi.py
---------------
XXX:write stuff

Full usage instuctions available using ``python xrfi.py -h`` in the scripts directory.

omni_apply.py
---------------
XXX:write stuff

Full usage instuctions available using ``python omni_apply.py -h`` in the scripts directory.
