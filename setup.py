from setuptools import find_packages, setup

setup(
   name='dl4cv',
   version='1.0',
   packages=find_packages(exclude=('datasets', 'datasets.*', 'saves', 'saves.*'))
)
