#!/usr/bin/env python2
# coding=utf-8

# Authors: Santi Villalba <sdvillal@gmail.com>
# Licence: BSD 3 clause

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
    name='strawlab_examples',
    license='BSD 3 clause',
    description='Data analysis examples from the Straw lab',
    long_description=open('README.rst').read().replace('|Build Status| |Coverage Status| |Scrutinizer Status|', ''),
    version='1.0.0-dev0',
    url='https://github.com/strawlab/strawlab-examples',
    author='Santi Villalba',
    author_email='sdvillal@gmail.com',
    packages=['strawlab_examples',
              'strawlab_examples.benchmarks',
              'strawlab_examples.euroscipy'],
    classifiers=[
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Operating System :: Unix',
    ],
    install_requires=['whatami', 'jagged', 'future', 'humanize', 'matplotlib', 'seaborn', 'requests'],
    tests_require=['pytest', 'pytest-cov', 'pytest-pep8'],
    platforms=['Any'],
)
