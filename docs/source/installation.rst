.. _installation:

Installation
============

Install the code is as easy as

::

  pip install comet


Then you can follow the `Jupyter Notebook <https://gitlab.com/aegge/pt-emulator/-/tree/main/notebooks>`_
for a small example in how to make predictions, compare with data and estimate
the :math:`\chi^2` of you model.

**Developer version**

If you want to modify the code and play around with it, we provide a developer
version so that you can make it and test it. Also, could be possible that you
have your own theoretical predictions and you wish to train the emulator
with your own computations. You can install the developer
version as follow.

::

  git clone git@gitlab.com:aegge/pt-emulator.git
  cd pt-emulator
  pip install -e .


Then you can follow the `Jupyter Notebook <https://gitlab.com/aegge/pt-emulator/-/tree/main/notebooks>`_
to learn how to train the *COMET* and make predictions.

.. warning::
   the comet emulator only works in a
   Python 3 environment; the data file at its core cannot
   be unpickled by Python 2.x; in case your ``pip``
   command doesn't link to a Python 3 pip executable, please
   modify the line above accordingly (e.g. with ``pip3`` instead of ``pip``)

.. note::
  The comet emulator depends on the following external packages:

  * numpy
  * matplotlib
  * scipy
  * astropy
  * GPy
  * pyDOE

  The installation process will automatically try to install them if they are not already present.
