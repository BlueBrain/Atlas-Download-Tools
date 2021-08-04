.. _installation:

Installation
============
It is highly recommended to install the project into a new virtual environment.


Installation from PyPI
----------------------
The :code:`atldld` package can be easily installed from PyPI.


.. code-block:: bash

    pip install atldld

Installation from source
------------------------
As an alternative to installing from PyPI, if you want to try the latest version
you can also install from source.

.. code-block:: bash

    pip install git+https://github.com/BlueBrain/Atlas-Download-Tools

Development installation
------------------------
For development installation one needs additional dependencies grouped in :code:`extras_requires` in the
following way:

- **dev** - pytest + plugins, flake8, pydocstyle, tox
- **docs** - sphinx

.. code-block:: bash

    git clone https://github.com/BlueBrain/Atlas-Download-Tools
    cd Atlas-Download-Tools
    pip install -e '.[dev,docs]'


Generating documentation
------------------------
To generate the documentation make sure you have dependencies from :code:`extras_requires` - :code:`docs`.

.. code-block:: bash

    cd docs
    make clean && make html

One can view the docs by opening :code:`docs/_build/html/index.html` in a browser.
