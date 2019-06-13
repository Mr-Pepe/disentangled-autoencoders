from setuptools import setup, find_packages

setup(
   name='dl4cv',
   version='1.0',
   packages=find_packages(exclude=('datasets', 'datasets.*', 'saves', 'saves.*'))
)
