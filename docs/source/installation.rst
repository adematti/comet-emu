.. _installation:

Installation
============

Installing COMET is as easy as

::

  pip install comet-emu


.. note::

  COMET requires a Python 3 environment with version ``>= 3.8``\ ; it moreover depends on the following packages:

  * numpy
  * matplotlib
  * scipy
  * astropy
  * scikit-learn
  * numba

  The installation process will automatically install these packages if they are not already present.


One of COMET's submodules makes use of an external library, ``libgrid.so``\ , which will be compiled with the pip install command above. This requires a C++ compiler with standard >=C++11 and the following two additional libraries:

* OpenMP
* Boost

The automatic compilation may fail if these libraries do not exist or cannot be found, but COMET will be installed successfully nonetheless. If this is the case, importing COMET will prompt the message

    "Warning! 'libgrid.so' not found, bispectrum binning options will not be available."

It is possible to compile the ``libgrid.so`` library manually, following the steps below. First, make a local clone of the ``comet-emu`` repository:

::

  git clone git@gitlab.com:aegge/comet-emu.git
  cd comet-emu
  pip install -e .


Now, compile and link the library, making sure to specify the include path to the OpenMP and Boost libraries, for instance (replacing ``/path/to/header/files``\ ):

::

  cd discreteness
  g++-14 -O3 -c -fPIC grid.cpp -o grid.o -I/path/to/header/files -fopenmp
  g++-14 -O3 -shared -o libgrid.so grid.o -I/path/to/header/files -fopenmp


Once installed, make sure to check out the :ref:`Tutorial pages<examples>`
for examples on how to use the code and its various options.
