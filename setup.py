from setuptools import setup
import glob
import os
import sys
from hera_cal import version
import json

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
    'name': 'hera_cal',
    'author': 'HERA Team',
    'url': 'https://github.com/HERA-Team/hera_cal',
    'license': 'BSD',
    'description': 'collection of calibration routines to run on the HERA instrument.',
    'package_dir': {'hera_cal': 'hera_cal'},
    'packages': ['hera_cal'],
    'include_package_data': True,
    'scripts': ['scripts/extract_hh.py', 'scripts/omni_abscal_run.py',
                'scripts/abscal_run.py', 'scripts/apply_cal.py',
                'scripts/delay_filter_run.py',
                'scripts/lstbin_run.py', 'scripts/omni_abscal_run.py',
                'scripts/smooth_cal_run.py', 'scripts/redcal_run.py'],
    'version': version.version,
    'package_data': {'hera_cal': data_files},
    'zip_safe': False,
}


if __name__ == '__main__':
    setup(*(), **setup_args)
