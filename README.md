# Sensie version 0.1

Sensie is toolset for probing the sensitivity of a deep neural network model to various properties of the data. 

## Overview

An active area of research (and not a little hand-wringing) in deep learning at present is better understanding how to interpret the key features learned by DNNs, particularly in order to better understand and predict failure modes. Various algorithms and toolsets exist for interpreting DNNs; in the computer vision arena, saliency maps are a key technique for understanding the key features of the images used by the network in order to make its decisions. However, these maps are not informative in all applications, for instance in some scientific applications where inputs are not RGB images.

Sensie probes the sensitivity of a network to inputs with a particular property, <img src="/tex/0d19b0a4827a28ecffa01dfedf5f5f2c.svg?invert_in_darkmode&sanitize=true" align=middle width=12.92146679999999pt height=14.15524440000002pt/>. This property can be a feature of the data; an otherwise known property of the test data that is not provided explicitly in training; or a function that can 'artificially' vary the property for a supplied test set. The effect of a particular property is measured according to the variance it introduces to the correct output of the network, such as the score for the correct class <img src="/tex/e92bd1a512cbe7d9721e846312bfe3fc.svg?invert_in_darkmode&sanitize=true" align=middle width=14.523852749999989pt height=19.415200200000008pt/>. Quantitatively, we seek the output <img src="/tex/f6818419a9b2a0402d0b9bc468cc9189.svg?invert_in_darkmode&sanitize=true" align=middle width=20.966980649999996pt height=32.29212359999999pt/> for a supplied property, <img src="/tex/0d19b0a4827a28ecffa01dfedf5f5f2c.svg?invert_in_darkmode&sanitize=true" align=middle width=12.92146679999999pt height=14.15524440000002pt/>.

Although the algorithm is simple, it has the potential to highlight and quantify gaps in the training set or areas of increasing unrealiability in a trained model.  

Sensie can operate on models trained for discrete (classifier) or ~~continuous (regression) outputs~~ (coming soon).

## Requirements

Sensie assumes a Keras-like interface but with a user supplied predictor function, any framework should be applicable. Prequisites are listed in `requirements.txt`. The pymc3 probabilistic programming framework is required for the production of credible intervals for feature sensitivities. This can be used to assess the significance of a sensitivity analysis.

## Usage

- Create a `sensie.Probe` instance to wrap a pre-trained model.
- Pass the probe a test set, ground truth, and either a vector/tensor containing the property to test, or a function that mutates a training example, taking a scalar parameter indicating the size of the effect.
- Sensie will return a `sensie.SensitivityMeasure` object, a collection of the results of each test as `sensie.SingleTest` objects.
- Examine a plot of each test, view the `summary` to quantify the size of the effect.
- The sensitivity, i.e. the gradient of mean correct score with property <img src="/tex/0d19b0a4827a28ecffa01dfedf5f5f2c.svg?invert_in_darkmode&sanitize=true" align=middle width=12.92146679999999pt height=14.15524440000002pt/>, is determined using ordinary linear regression (`sensie.Probe.get_gradient`) or using Bayesian inference with pymc3; this 
- Where the relationship is non-linear, polynomial fits can also be visualised in order to identify regions of the parameter space where the network is most sensitive to the supplied property.

Sensie assumes that model.predict returns a XX. If this is not the case, supply a predictor function at instantiation time that does: `probe = sensie.Probe(model, predictor)` where `predictor` is a function with the signature `predictor(model, x_test)` and returns a tensor of dimensions `(N, n_classes)`.

Documentation can be accessed at readthedocs.org

## Example

How sensitive is a model trained on MNIST digits to the orientation of the digits?
```
def rotate(x, angle):
  # rotates the image by angle degrees
  pass

model = ... # trained model

sensie_mnist = sensie.Probe(model)
sensie....
```
![MNIST rotation sensitivity](examples/sensie1.png)

For this and more complex examples, see the notebooks in the `examples` directory.

### References

