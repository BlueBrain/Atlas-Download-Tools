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
        "pydocstyle",
        "pytest",
        "pytest-cov",
        "pytest-mock",
        "tox",
    ],
}

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
)
