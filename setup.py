"""Setup for decision tree module."""
from setuptools import setup


extra_packages = {
    'testing': ['pytest', 'pytest-cov', 'numpy']
}


setup(
    name='Decision Tree',
    description='Provides an implementation of a crude decision tree.',
    version=0.0,
    author='Kurt Maurer',
    author_email='kurtrm@gmail.com',
    extras_require=extra_packages
)
