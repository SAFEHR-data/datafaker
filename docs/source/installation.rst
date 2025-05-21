.. _page-installation:

Installation
============

To use SqlSynthGen, first install it.

Make sure you have pipx installed. To do this on Windows:

.. code-block:: console
   $ python -m pip install pipx
   $ python -m pipx ensurepath

Then close your command shell and open another. Now you can use ``pipx``.

.. code-block:: console

   $ pipx install git+https://github.com/tim-band/sqlsynthgen

Check that you can view the help message with:

.. code-block:: console

   $ sqlsynthgen --help

It can also be used directly within a Docker container by downloading image ``timband/ssg``.
See the :ref:`quickstart guide <page-quickstart>` for more information.
