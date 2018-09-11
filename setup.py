#!/usr/bin/env python

'''
Run this one if you just want a python package.
'''
from setuptools import setup
from setuptools import find_packages


setup(name='costar_hyper',
      version='0.1.0',
      description='CoSTAR HyperTree Architecture Search Deep Learning Robotics Library',
      author='Andrew Hundt',
      author_email='ATHundt@gmail.com',
      url='https://github.com/cpaxton/costar_plan',
      license='MIT',
      install_requires=['keras>=2.2.2', 'h5py', 'scikit-image>=0.14', 'shapely', 'tqdm', 'keras_tqdm', 'pyquaternion'],
      extras_require={
          'h5py': ['h5py'],
          'visualize': ['pydot>=1.2.0'],
          'tests': ['pytest',
                    'pytest-pep8',
                    'pytest-xdist',
                    'pytest-cov'],
      },
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Developers',
          'Intended Audience :: Education',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: Apache v2',
          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.6',
          'Topic :: Software Development :: Libraries',
          'Topic :: Software Development :: Libraries :: Python Modules'
      ],
      packages=find_packages())