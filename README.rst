mvcmi
=====
Multivariate Conditional Mutual Information

Installation
~~~~~~~~~~~~
We recommend the `Anaconda Python distribution <https://www.anaconda.com/products/individual>`_.

To install ``mvcmi``, you can install the current version of the code (nightly) with::

   $ pip install --upgrade https://api.github.com/repos/mvcmi/mvcmi/zipball/main

To make an editable install, please clone the repository and make an editable install::

   $ git clone https://github.com/mvcmi/mvcmi.git
   $ cd mvcmi/
   $ pip install -e .

To check if everything worked fine, you can do::

   $ python -c 'import mvcmi'

and it should not give any error messages.

Cite
~~~~
If you use mvcmi in your work, please cite the following publication::

   Sundaram, P., Luessi, M., Bianciardi, M., Stufflebeam, S., Hämäläinen, M. and Solo, V., 2019.
   Individual resting-state brain networks enabled by massive multivariate conditional mutual
   information. IEEE transactions on medical imaging, 39(6), pp.1957-1966.
