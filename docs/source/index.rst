Sensie: Probing the sensitivity of neural networks
==================================================

Introduction
^^^^^^^^^^^^

Sensie's goal is to quickly interrogate a trained machine learning model, determining the sensitivity to a parameter or perturbation of the data.

Sensie probes the sensitivity of a network to inputs with a particular property, *p. This property can be a feature of the data; an otherwise known property of a test or training set that is not provided explicitly in training; or a function that can artificially vary this property for a supplied test set. The effect of a particular property is measured according to the variance it introduces to the correct output of the network, such as the score for the correct class . Quantitatively, we would like to know the function *mean_score = f(p)* for some property p; Sensie can calculate a linear approximation to this unknown function.

For more information and examples, see the GitHub repository at https://github.com/coljac/sensie.

Installation
^^^^^^^^^^^^

Check out the repository and install with::

    pip install . 

(or add the sensie directory to your PYTHONPATH.)

Dependencies are listed in ``requirements.txt`` included in the repository. Sensie requires python 3.6 or above.

Optionally, install pytest with ``pip install pytest``, then run the tests with ``pytest test`` from the repository root.


.. toctree::
   :maxdepth: 2
   :caption: API documentation:

   sensie


Issues, Questions and Contributions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Any problems or questions? Email colin@coljac.net, or open an issue on GitHub at https://github.com/coljac/sensie.

Contributions are welcome and encouraged. Fork the GitHub repository to your own machine, make some changes, and push your work back up to the fork and open a `pull request <https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/about-pull-requests>`_ so that I can review and incorporate the changes.

Index
=====

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


