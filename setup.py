#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 12:18:53 2018

@author: Sven serneels, Ponalytics
"""

from setuptools import setup, find_packages
import re
import os

# Read version/author/license from __init__.py without importing the package,
# which would fail because dependencies (numpy, etc.) aren't installed yet.
_init_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "twoblock", "__init__.py")
_metadata = {}
with open(_init_path) as f:
    for line in f:
        for key in ("__version__", "__author__", "__license__"):
            if line.startswith(key):
                _metadata[key] = line.split("=", 1)[1].strip().strip("\"'")

__version__ = _metadata["__version__"]
__author__ = _metadata["__author__"]
__license__ = _metadata["__license__"]

readme_file = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "README.md"
)
try:
    from m2r import parse_from_file

    readme = parse_from_file(readme_file)
except ImportError:
    # m2r may not be installed in user environment
    with open(readme_file) as f:
        readme = f.read()

setup(
    name="twoblock",
    version=__version__,
    author=__author__,
    author_email="svenserneels@gmail.com",
    description="A Scikit-Learn Compatible Library for Simultaneous Two-Block Sufficient Dimension Reduction Methods",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/SvenSerneels/twoblock",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=find_packages("src"),  # include all packages under src
    package_dir={"": "src"},  # tell distutils packages are under src
    include_package_data=True,
    install_requires=[
        "numpy>=1.22.0",
        "scipy>=1.8.0",
        "scikit-learn>=1.3.0",
        "pandas>=1.4.0",
    ],
    extras_require={
        "gpu": ["cupy-cuda12x>=14.0"],
        "robpca": ["robpy"],  # For n <= p case in spadimo
    },
)
