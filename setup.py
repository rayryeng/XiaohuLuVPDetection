#!/usr/bin/env python
"""
setup.py file for installing the VP detection package.

Heavily borrowed from Kenneth Reitz -
https://github.com/kennethreitz/setup.py
"""

import io
import os
import sys

from setuptools import find_packages, setup

# Package meta-data.
NAME = 'lu_vp_detect'
DESCRIPTION = "Xiaohu Lu's Vanishing Point Detection algorithm"
URL = 'https://github.com/rayryeng/XiaohuLuVPDetection'
EMAIL = 'rphan@ryerson.ca'
AUTHOR = 'Ray Phan'
REQUIRES_PYTHON = '>=3.5.0'
VERSION = None

# Required packages
REQUIRED = ['numpy', 'opencv-contrib-python==4.0.0.21']

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Load the package's __version__.py module as a dictionary.
about = {}
if not VERSION:
    project_slug = NAME.lower().replace("-", "_").replace(" ", "_")
    with open(os.path.join(here, project_slug, '__version__.py')) as f:
        exec(f.read(), about)
else:
    about['__version__'] = VERSION

# Where the magic happens:
setup(
    name=NAME,
    version=about['__version__'],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(
        exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    # If your package is a single module, use this instead of 'packages':
    #py_modules=['lu_vp_detect'],
    entry_points={
        'console_scripts': ['run_vp_detect=lu_vp_detect.run_vp_detect:main'],
    },
    install_requires=REQUIRED,
    include_package_data=True,
    license='MIT',
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy'
    ],
)
