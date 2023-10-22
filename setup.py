#!/usr/bin/env python3
"""
Setup script for MatNexus
© Lei Zhang, Markus Stricker, 2023
"""

from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

setup(
    name="MatNexus",
    version="0.1",
    description="Library for natural language processing for scientific " "papers",
    url="https://gitlab.ruhr-uni-bochum.de/icams-mids/text_mining_tools",
    author="Lei Zhang, Markus Stricker",
    author_email="Lei.Zhang-w2i@rub.de, markus.stricker@rub.de",
    license="GNU GPL v3",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
    keywords="physics, material science, natural language processing",
    packages=find_packages(),
    include_package_data=True,
)
