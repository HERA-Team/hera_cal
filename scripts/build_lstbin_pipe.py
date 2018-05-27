#!/usr/bin/env python2.7
"""
build_lstbin_pipe.py
--------------------

Parse a lstbin_params.yaml file and
build a PBS script for running hera_cal.lstbin
on a cluster.
"""
import os
import argparse
import yaml
from hera_cal import lstbin
import glob
import copy

# setup argparser
args = argparse.ArgumentParser(description="Build a PBS script for hera_cal.lstbin")
args.add_argument("paramfile", type=str, help="path to YAML lstbin parameter file. See hera_cal/scripts/lstbin_params.yaml for an example.")
args.add_argument("--pbsfile", type=str, default="lstbin_pipe.sh", help="path to output PBS batch script")

# parse args
a = args.parse_args()

# load yaml file
print "...loading {}".format(a.paramfile)
with open(a.paramfile, 'r') as f:
    yaml_params = yaml.load(f)

# setup default parameters
default_params = {
'pbsfile'   : 'lstbin_pipe.sh',
'rephase'   : '',
'sig_clip'  : '',
'sigma'     : 5.0,
'min_N'     : 5,
'overwrite' : False,
'dlst'      : None,
'lst_start' : 0.0,
'outdir'    : None,
'vis_units' : 'Jy',
'ntimes_per_file' : 60,
'file_ext'  : "{}.{}.{:7.5f}.uv",
'queue'     : 'hera',
'outfile'   : 'lstbin.out',
'nodes'     : 1,
'ppn'       : 1,
'vmem'      : '32gb',
'walltime'  : '24:00:00',
'arrayjob'  : False,
'cwd'       : os.getcwd()
}

# update default parameters with parameter file
default_params.update(yaml_params)

# setup PBS file
pbs = "" \
"#!/bin/bash\n" \
"#PBS -q {queue}\n" \
"#PBS -j oe\n" \
"#PBS -o {outfile}\n" \
"#PBS -l nodes={nodes}:ppn={ppn}\n" \
"#PBS -l walltime={walltime}\n" \
"#PBS -l vmem={vmem}\n" \
"#PBS -V\n" \
"{arrayjob}\n\n" \
"echo start: $(date)\n" \
"cd {cwd}\n" \
"lstbin_run.py --dlst {dlst} --lst_start {lst_start} --ntimes_per_file {ntimes_per_file} " \
"--file_ext {file_ext} --outdir {outdir} {overwrite} {output_file_select} --vis_units {vis_units} {sig_clip} --sigma {sigma} --min_N {min_N} {rephase} {data_files}\n" \
"echo end: $(date)"

# parse special kwargs
if isinstance(default_params['rephase'], (bool)):
    if default_params['rephase']:
        default_params['rephase'] = '--rephase'
    else:
        default_params['rephase'] = ''
if isinstance(default_params['sig_clip'], (bool)):
    if default_params['sig_clip']:
        default_params['sig_clip'] = '--sig_clip'
    else:
        default_params['sig_clip'] = ''
if default_params['overwrite']:
    default_params['overwrite'] = '--overwrite'
else:
    default_params['overwrite'] = ''

# Loop over polarizations
for i, pol in enumerate(default_params['pols']):
    params = copy.deepcopy(default_params)

    # configure datafiles
    params['data_files'] = map(lambda df: df.format(pol=pol), params['data_files'])

    # setup arrayjob if desired
    if params['arrayjob']:
        # parse datafiles
        _datafiles = map(lambda df: sorted(glob.glob(df)), params['data_files'])

        # run config_lst_bin_files to get output files
        output = lstbin.config_lst_bin_files(_datafiles, dlst=params['dlst'], lst_start=params['lst_start'],
                                             ntimes_per_file=params['ntimes_per_file'])

        nfiles = len(output[3])
        params['arrayjob'] = "#PBS -t 0-{}%5".format(nfiles-1)
        params['output_file_select'] = "--output_file_select ${PBS_ARRAYID}"

    else:
        params['arrayjob'] = ''
        params['output_file_select'] = ''

    # add quotations to datafiles
    params['data_files'] = map(lambda df: "'{}'".format(df), params['data_files'])

    # configure data files, which should be fed as a list of search strings
    params['data_files'] = " ".join(params['data_files'])

    # format params that may be polarization specific
    params['outfile'] = params['outfile'].format(pol=pol)
    params['pbsfile'] = params['pbsfile'].format(pol=pol)

    # format string
    pbs_file = pbs.format(**params)

    # write to file
    print "...writing {}".format(params['pbsfile'])
    with open(params['pbsfile'], 'w') as f:
        f.write(pbs_file)

