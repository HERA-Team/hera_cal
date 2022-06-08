from setuptools import setup

import os
import sys
import json

sys.path.append("hera_cal")
import version  # noqa

data = [version.git_origin, version.git_hash, version.git_description, version.git_branch]
with open(os.path.join('hera_cal', 'GIT_INFO'), 'w') as outfile:
    json.dump(data, outfile)


def package_files(package_dir, subdirectory):
    # walk the input package_dir/subdirectory
    # return a package_data list
    paths = []
    directory = os.path.join(package_dir, subdirectory)
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            path = path.replace(package_dir + '/', '')
            paths.append(os.path.join(path, filename))
    return paths


data_files = package_files('hera_cal', 'data') + package_files('hera_cal', 'calibrations')

setup_args = {
    'name': 'hera-calibration',
    'author': 'HERA Team',
    'url': 'https://github.com/HERA-Team/hera_cal',
    'license': 'BSD',
    'description': 'collection of calibration routines to run on the HERA instrument.',
    'package_dir': {'hera_cal': 'hera_cal'},
    'packages': ['hera_cal'],
    'include_package_data': True,
    'scripts': ['scripts/extract_hh.py', 'scripts/post_redcal_abscal_run.py',
                'scripts/apply_cal.py', 'scripts/delay_filter_run.py',
                'scripts/lstbin_run.py', 'scripts/extract_autos.py',
                'scripts/smooth_cal_run.py', 'scripts/redcal_run.py',
                'scripts/auto_reflection_run.py', 'scripts/noise_from_autos.py',
                'scripts/query_ex_ants.py', 'scripts/red_average.py',
                'scripts/time_average.py', 'scripts/tophat_frfilter_run.py', 'scripts/model_calibration_run.py',
                'scripts/time_chunk_from_baseline_chunks_run.py', 'scripts/chunk_files.py', 'scripts/transfer_flags.py',
                'scripts/flag_all.py', 'scripts/throw_away_flagged_antennas.py', 'scripts/select_spw_ranges.py',
                'scripts/multiply_gains.py'],
    'version': version.version,
    'package_data': {'hera_cal': data_files},
    'install_requires': [
        'numpy>=1.10',
        'scipy',
        'astropy',
        'astropy-healpix',
        'pyuvdata',
        'linsolve',
        'hera_qm',
        'scikit-learn'
    ],
    'extras_require': {
        "all": [
            'aipy>=3.0',
            'uvtools',
        ]
    },
    'zip_safe': False,
}


if __name__ == '__main__':
    setup(*(), **setup_args)
