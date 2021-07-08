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
