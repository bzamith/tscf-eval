Installation
============

Requirements
------------

- Python 3.10-3.13
- NumPy >= 1.24.0
- pandas >= 2.0.0
- scikit-learn >= 1.3.0
- aeon >= 1.3.0
- tqdm >= 4.65.0

From PyPI
---------

The recommended way to install tscf-eval is via pip:

.. code-block:: bash

   pip install tscf-eval

Optional Dependencies
---------------------

tscf-eval has several optional dependency groups:

**DTW Support** (for Dynamic Time Warping distance):

.. code-block:: bash

   pip install tscf-eval[dtw]

**Full Installation** (all optional features):

.. code-block:: bash

   pip install tscf-eval[full]

**Development** (testing, linting, etc.):

.. code-block:: bash

   pip install tscf-eval[dev]

**Documentation** (Sphinx and plugins):

.. code-block:: bash

   pip install tscf-eval[docs]

From Source
-----------

To install from source for development:

.. code-block:: bash

   git clone https://github.com/bzamith/tscf-eval.git
   cd tscf-eval
   pip install -e ".[dev]"

Verifying Installation
----------------------

You can verify the installation by importing the package:

.. code-block:: python

   import tscf_eval
   print(tscf_eval.__version__)

Or by running the test suite:

.. code-block:: bash

   pytest tests/ -v
