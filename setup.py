#!/usr/bin/env python

import pathlib
from setuptools import setup

# Parent directory
HERE = pathlib.Path(__file__).parent

# The readme file
README = (HERE / "README.md").read_text()

setup(
    name="fourth_day",
    version="1.0.1",
    description="Bioluminescence modeling for deep-sea experiments",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Stephan Meighen-Berger",
    author_email="stephan.meighenberger@gmail.com",
    url='https://github.com/MeighenBergerS/fourth_day',
    license="MIT",
    install_requires=[
        "PyYAML",
        "numpy",
        "scipy",
        "pandas",
        "pyDataverse",
        "tqdm"
    ],
    extras_require={
        "interactive": ["nbstripout", "matplotlib", "jupyter"],
    },
    packages=["fourth_day"],
    package_data={'fourth_day': [
        "data/*.pkl",
        "data/life/light/*.txt",
        "data/life/movement/*.txt"
    ]},
    include_package_data=True
)
