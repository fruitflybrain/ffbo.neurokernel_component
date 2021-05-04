#!/usr/bin/env python

import sys, os
from glob import glob

# Install setuptools if it isn't available:
try:
    import setuptools
except ImportError:
    from ez_setup import use_setuptools
    use_setuptools()

from distutils.command.install_headers import install_headers
from setuptools import find_packages
from setuptools import setup

NAME =               'neurokernel_component'
VERSION =            '0.3.0'
AUTHOR =             'Chung-Heng Yeh, Yiyin Zhou'
AUTHOR_EMAIL =       'c.yeh@columbia.edu, yiyin@ee.columbia.edu'
MAINTAINER =         'Yiyin Zhou'
MAINTAINER_EMAIL =   'yiyin@ee.columbia.edu'
DESCRIPTION =        'Server Component for Neurokernel Execution Engine'
URL =                'https://github.com/fruitflybrain/ffbo.neurokernel_component'
LONG_DESCRIPTION =   DESCRIPTION
DOWNLOAD_URL =       URL
LICENSE =            'BSD'
CLASSIFIERS = [
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: BSD License',
    'Operating System :: OS Independent',
    'Programming Language :: Python',
    'Topic :: Scientific/Engineering',
    'Topic :: Software Development']

# Explicitly switch to parent directory of setup.py in case it
# is run from elsewhere:
os.chdir(os.path.dirname(os.path.realpath(__file__)))
PACKAGES =           find_packages()

if __name__ == "__main__":
    if os.path.exists('MANIFEST'):
        os.remove('MANIFEST')

    setup(
        name = NAME,
        version = VERSION,
        author = AUTHOR,
        author_email = AUTHOR_EMAIL,
        license = LICENSE,
        classifiers = CLASSIFIERS,
        description = DESCRIPTION,
        long_description = LONG_DESCRIPTION,
        url = URL,
        maintainer = MAINTAINER,
        maintainer_email = MAINTAINER_EMAIL,
        packages = PACKAGES,
        include_package_data = True,
        install_requires = [
            'autobahn[twisted]',
            'simplejson',
            'configparser',
            'networkx',
            'neurokernel',
            'neurodriver',
            'neuroarch',
            'msgpack > 1.0',
            'msgpack-numpy',
            'pyopenssl',
            'txaio',
        ],
        )
