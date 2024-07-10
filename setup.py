#!usr/bin/env python
__author__ = "Colin Togashi"
__email__ = "colin.togashi@gmail.com"
__copyright__ = "Copyright 2021 RoMeLa"
__date__ = "June 28, 2021"

__version__ = "0.0.1"
__status__ = "Prototype"

from setuptools import setup

# with open("README.md", "r") as fh:
#     long_description = fh.read()

setup(
    name='highlevel',
    version='0.0.1',
    author='Colin Togashi',
    author_email="colin.togashi@gmail.com",
    description="",
    # long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=["highlevel"],
    install_requires=[
        'numba',
        'numpy',
        'matplotlib',
        'scipy'
    ],
)