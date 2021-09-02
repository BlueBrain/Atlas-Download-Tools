# The package atldld is a tool to download atlas data.
#
# Copyright (C) 2021 EPFL/Blue Brain Project
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""The setup script."""
from setuptools import find_packages, setup

python_requires = ">=3.6"
install_requires = [
    "Pillow",
    "appdirs",
    "click",
    "dataclasses; python_version < '3.7'",
    "matplotlib",
    "numpy",
    "opencv-python",
    "pandas",
    "requests",
    "scikit-image",
]
extras_require = {
    "dev": [
        "bandit",
        "black",
        "flake8",
        "isort",
        "mypy",
        "pydocstyle",
        "pytest",
        "pytest-cov",
        "pytest-mock",
        "responses",
        "tox",
    ],
    "docs": ["sphinx>=1.3", "sphinx-bluebrain-theme"],
}

description = "Search, download, and prepare atlas data."
long_description = """
Among different sources of data, Allen Brain Institute
hosts a rich database of gene expression images, Nissl volumes, and annotation atlases.
The Atlas-Download-Tools library can help you to download single section images
and entire datasets, as well as the corresponding metadata. It can further
pre-process the image data to place it in the standard reference space.
"""

setup(
    name="atldld",
    author="Blue Brain Project, EPFL",
    url="https://github.com/BlueBrain/Atlas-Download-Tools",
    use_scm_version={
        "write_to": "src/atldld/version.py",
        "write_to_template": '"""The package version."""\n__version__ = "{version}"\n',
        "local_scheme": "no-local-version",
    },
    package_dir={"": "src"},
    packages=find_packages("src"),
    python_requires=python_requires,
    install_requires=install_requires,
    extras_require=extras_require,
    entry_points={"console_scripts": ["atldld = atldld.cli:root"]},
    description=description,
    long_description=long_description,
    long_description_content_type="text/x-rst",
    license="LGPLv3",
    classifiers=[
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
        "Operating System :: Unix",
        "Operating System :: MacOS",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3 :: Only",
    ],
)
