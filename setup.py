from setuptools import setup

import os
import sys
import json

sys.path.append("hera_cal")
import version # noqa

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
    'scripts': ['scripts/extract_hh.py', 'scripts/post_redcal_abscal_run.py',
                'scripts/apply_cal.py', 'scripts/delay_filter_run.py',
                'scripts/lstbin_run.py', 'scripts/extract_autos.py',
                'scripts/smooth_cal_run.py', 'scripts/redcal_run.py',
                'scripts/auto_reflection_run.py', 'scripts/noise_from_autos.py'],
    'version': version.version,
    'package_data': {'hera_cal': data_files},
    'install_requires': [
        'numpy>=1.10',
        'scipy',
        'astropy',
        'pyuvdata',
        'aipy>=3.0rc2',
        'pyephem',
        'uvtools @ git+git://github.com/HERA-Team/uvtools',
        'linsolve @ git+git://github.com/HERA-Team/linsolve',
        'hera_qm @ git+git://github.com/HERA-Team/hera_qm',
        'scikit-learn'
    ],
    'zip_safe': False,
}


if __name__ == '__main__':
    setup(*(), **setup_args)
