from setuptools import setup

import os
import sys
import json
from pathlib import Path

sys.path.append("hera_cal")


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
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup_args = {
    'name': 'hera-calibration',
    'author': 'HERA Team',
    'url': 'https://github.com/HERA-Team/hera_cal',
    'license': 'BSD',
    'description': 'collection of calibration routines to run on the HERA instrument.',
    'long_description': long_description,
    'long_description_content_type': 'text/markdown',
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
    'package_data': {'hera_cal': data_files},
    'install_requires': [
        'numpy>=1.10',
        'scipy',
        'h5py',
        'hdf5plugin',
        'astropy',
        'astropy-healpix',
        'pyuvdata<=2.2.12',
        'linsolve',
        'hera_qm',
        'scikit-learn',
        'hera_filters',
        'aipy',
    ],
    # 'extras_require': {
    #     "all": [
    #         'aipy @ git+https://github.com/hera-team/aipy'
    #     ]
    # },
    'zip_safe': False,
}


if __name__ == '__main__':
    setup(*(), **setup_args)
