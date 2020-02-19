import sys
# import os
# import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import scipy
from scipy import stats
from sklearn.linear_model import BayesianRidge, LinearRegression, Ridge
# from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import sys

# TODOS
# Nice plot of significance
# Package it up properly
# Document the categorical convention
# Tests

#### For version 0.0.1
# Arviz PDF [ ]
# Linear fit plot [X]
# Polynomial fits [X]
# Fit all properties at once [ ]
# Enhanced summary [x]
# Plot labelling [ ]
# Plot font size [ ]
# Simple class analysis method [X]
# Refactor API slightly [-]
# TESTS
# DOCS
# Show pdf
# Save results in an efficient format [ ]
def progbar(current, to, width=40, show=True, message=None, stderr=False):
    percent = float(current) / float(to)
    length = int(width * percent)
    if show:
        count = " (%d/%d)    " % (current, to)
    else:
        count = ""
    if message:
        count += message
    outstream = sys.stderr if stderr else sys.stdout
    outstream.write(("\r[" + ("#" * length) + " " * (width - length) +
                     "] %0d" % (percent * 100)) + "%" + count)
    outstream.flush()



class SingleTest(object):
    """
    Encapsulates the results of a single signficance test.
    """
    def __init__(self, property_name, p_vals, means, stds, p_points=None, y_vals=None):
        """
        p_vals: ndarray
        """
        self.property_name = property_name
        self.p_vals = p_vals
        self.p_points = p_points
        self.y_vals = y_vals
        self.means = means
        self.stds = stds

        self.beta = None
        self.intercept = None
        self.ci_95_low  = None
        self.ci_95_high = None
        self.ci_50_low  = None
        self.ci_50_high = None
        self.pos = None

    def set_fit(self, beta, intercept, ci_95_low=np.nan, ci_95_high=np.nan, 
                                    ci_50_low=np.nan, ci_50_high=np.nan, pos=None):
        self.beta = beta
        self.intercept = intercept
        self.ci_95_low = ci_95_low
        self.ci_95_high = ci_95_high
        self.ci_50_low = ci_50_low
        self.ci_50_high = ci_50_high
        self.pos = pos

    def sort_and_reorder(self, labels=None):
        y = self.means
        s = self.stds
        class_ordering = np.argsort(y)
        self.means = y[class_ordering]
        self.stds = s[class_ordering]
        if labels is not None:
            return np.array(labels)[class_ordering]

    def get_gradient(self):
        ys = self.means
        xs = self.p_vals
        fit = self._fit_line(xs, ys)  
        self.set_fit(*fit)
        return fit

    def _fit_line(self, xs, ys):
        ols = LinearRegression()
        xx = xs.reshape(-1, 1)
        ols.fit(xx, ys)
        return ols.coef_, ols.intercept_

    def _run_inference(self, xs, ys, samples=4000, alpha_sd0=2, beta_sd0=1,
            epsilon_0=3, tune=None):
        """ Uses Bayesian inference courtesy of pymc3 to estimate the sensitivity
        coefficient (i.e. the gradient of correct score as a function of a given 
        property and provide a 95% credible interval. (If zero is in this
        interval we interpret the gradient as not very significant.) """

        # Don't require pymc3 unless we are using this method
        import pymc3 as pm
        import arviz as az
        if tune is None:
            tune = int(samples/2)
        with pm.Model() as model_g:
            alpha = pm.Normal('alpha', mu=0, sd=alpha_sd0)
            beta = pm.Normal('beta', mu=0, sd=beta_sd0)
            epsilon = pm.HalfCauchy('epsilon', epsilon_0)

            mu = pm.Deterministic('mu', alpha + beta * xs)
            y_pred = pm.Normal('y_pred', mu=mu, sd=epsilon, observed=ys)

            trace_g = pm.sample(samples, tune=tune)

            alpha_m = trace_g['alpha'].mean()
            beta_m = trace_g['beta'].mean()

        return beta_m, pm.stats.hpd(trace_g['beta'], alpha=0.05), pm.stats.hpd(trace_g['beta'], alpha=0.5), trace_g

    def get_significance(self):
        if self.beta is None:
            return None
        if np.isnan(self.ci_95_low):
            sig = (self.means[0] - self.means[-1])/self.stds.mean()
            if np.abs(sig) > 0:
                return "high"
            else:
                return "low"
        else:
            if (self.ci_95_low > 0 and self.ci_95_high > 0) or (self.ci_95_low < 0 and self.ci_95_high < 0):
                return "high"
            elif (self.ci_50_low > 0 and self.ci_50_high > 0) or (self.ci_50_low < 0 and self.ci_50_high < 0):
                return "medium"
            else:
                return "low"


    def get_credible_interval(self, means_only=False, tune=None, samples=400):
        ys = self.y_vals
        if means_only or ys is None:
            ys = self.means
            xs = self.p_vals
        else:
            xs = self.p_points

        results = self._run_inference(xs, ys, samples=samples, tune=tune)
        self.set_fit(results[0], results[3]['alpha'].mean(), ci_95_low=results[1][0], ci_95_high=results[1][1], ci_50_low=results[2][0], 
                    ci_50_high=results[2][1], pos=results[3])
        return results


    def summary(self):
        """ Show the result (gradient) of score sensitivity to this property, 
            optionally with credible intervals.

            Returns:
            A pandas DataFrame with the results of the test.
        """
        if self.beta is None:
            try:
                self.get_gradient()
            except:
                self.beta = np.nan # Failed to fit

        result = pd.DataFrame(
                {"property": self.property_name,
                    "sensitivity": self.beta,
                    "significance": self.get_significance(),
                    "sens_50_low": self.ci_50_low,
                    "sens_50_high": self.ci_50_high,
                    "sens_95_low": self.ci_95_low,
                    "sens_95_high": self.ci_95_high,
                    }, index=[0]
                )
        result = result.set_index("property")
        return result 

    def __str__(self):
        if self.beta is not None:
            return f"Test {self.property_name}: beta = {self.beta:.2f}"
        else:
            return f"Test {self.property_name}"

class SensitivityMeasure(object):
    """
    """
    def __init__(self, x_test, y_test, rightscores):
        """
        """
        # self.x_test = x_test
        self.y_test = y_test
        self.rightscores = rightscores
        self.tests = {}

    def append(self, label, ps, means, stds, p_points=None, y_vals=None):
        """
        """
        self.tests[label] = SingleTest(label, ps, means, stds, p_points=p_points, y_vals=y_vals)

    def summary(self):
        """
        Returns:
            A pandas DataFrame with a row for each test performed.
        """
        result = None
        for test in self.tests:
            if result is None:
                result = self.tests[test].summary()
            else:
                result.append(self.tests[test].summary())
        return result

    def get_credible_intervals(self):
        for label, test in self.tests.items():
            test.get_credible_interval()


class Probe(object):
    """A class that wraps a pre-trained model and provides methods for testing its robustness
       and sensitivity to various properties."""

    def __init__(self, model, predict_function=None):
        """model: A pretrained model object.
        predict_function: a function that takes a tensor of inputs and returns a vector of scores. By default,
         assumes a Keras.Model obkect and calls the model.predict() method."""

        self.model = model
        if predict_function is None:
            self.predict_function = lambda x, y: x.predict(y) # get_predictions_keras
        else: 
            self.predict_function = predict_function

    def predict_and_measure(self, x_test, y_test, p_test, prop=None, continuous=False, bins=20,
            label=None, plot=False, propnames=None, batch_size=256) -> SensitivityMeasure:
        """Scores the provided x_test and returns a SensitivityMeasure object with measured
        values and for plotting.

        Parameters
        ----------
        x_test: numpy.ndarray
            Tensor of examples for testing

        y_test: numpy.ndarray
            Vector of ground-truth classes

        p_test: numpy.ndarray or pandas.DataFrame
            Tensor or DataFrame containing the property/properties for testing.

        prop: int or str
            (Optional) A numerical or string index into p_test, returning a vector or Series of the property
        in question. If this is None, will attempt for all columns in p_test

        label: str
            (Optional) An string label for the property/properties in question; used for plotting.

        plot: bool
            If True, produce and display a plot of the results.

        Returns
        -------
        SensitivityMeasure
            An object containing summary information about the analysis.
        """

        # scores = self.predict_function(self.model, x_test, batch_size=batch_size)
        # rightscores = scores[np.arange(scores.shape[0]), y_test]
        rightscores = self._run_prediction(x_test, y_test, batch_size=batch_size)

        if propnames is None:
            propnames = []
        
            if prop is not None:
                propnames = [prop]
            else:
                if type(p_test) == pd.DataFrame:
                    propnames = p_test.columns
                else:
                    # propnames = range(p_test.shape[1]) # Why 1?
                    propnames = [str(x) for x in sorted(np.unique(p_test))]


        results = SensitivityMeasure(x_test, y_test, rightscores)
        for idx, propname in enumerate(propnames):
            progbar(idx+1, len(propnames), message = propname + "     ")

            if type(p_test) == pd.DataFrame:
                p_values = p_test.loc[:, propname]
            else:
                if np.ndim(p_test) == 1:
                    p_values = p_test
                else:
                    p_values = p_test[:, propname]

            p_bin_values = np.unique(p_values)

            if continuous:
                # bin up by the specified property
                if len(p_bin_values) > bins:
                    p_bin_values = np.linspace(p_values.min(), p_values.max(), bins)


            x, means, std_devs = self._bin_and_measure(rightscores, p_values, p_bin_values=p_bin_values, binup=continuous)

            results.append(propname, x, means, std_devs, p_points=p_values, y_vals=rightscores)

        if plot:
            # if ci:
                # results.tests[label].get_credible_interval(means_only=False)
            # else:
            results.tests[propname].get_gradient()
            self.plot_property(results.tests[propname], label=label)

        return results

    def _run_prediction(self, x_test, y_test, batch_size=256):
        i = 0
        rightscores = np.zeros(x_test.shape[0], dtype=np.float16)
        while i < x_test.shape[0]:
            # TODO REMOVE
            # ct.progbar(int(i/batch_size), int(x_test.shape[0]/batch_size))
            scores = self.predict_function(self.model, x_test[i:i+batch_size])
            rightscores[i:i+batch_size] = scores[np.arange(scores.shape[0]), y_test[i:i+batch_size]]


            i += batch_size
        return rightscores

    def _bin_and_measure(self, rightscores, p_test_values, p_bin_values=None, binup=False):
        """ Bin up the by property value (by class if discrete, or by bin if continuous)
        and return the mean correct class score and std dev for each bin. """
        if p_bin_values is not None:
            n_bins = len(p_bin_values)
            p_values = p_bin_values
        else:
            n_bins = len(p_test_values)
            p_values = p_test_values

        x = p_values
        y = np.zeros(n_bins)
        s = np.zeros(n_bins)

        if binup:
            bin_indices = np.digitize(p_test_values, p_bin_values)

        for i in range(n_bins):
            pval = p_values[i]
            if binup:
                in_bin = rightscores[np.where(bin_indices == i)]
                if len(in_bin > 0):
                    mean_score = in_bin.mean()
                    std_dev = in_bin.std()
                else:
                    mean_score = np.nan
                    std_dev = np.nan
            else:
                mean_score = rightscores[np.where(p_test_values == pval)].mean()
                std_dev = rightscores[np.where(p_test_values == pval)].std()

            y[i] = mean_score
            s[i] = std_dev
        return x, y, s


    def predict_and_measure_perturbed(self, x_test, y_test, perturber, p_values=None,
                                      p_min=0, p_max=1, steps=10, label=None,
                                      plot=False, ci=False) -> SensitivityMeasure:
        """Scores the provided x_test as altered by the supplied perturber function, and returns
        a SensitivityMeasure object with measured values and for plotting.

        Parameters
        ----------
        x_test: numpy.ndarray
            Tensor of examples for testing

        y_test: numpy.ndarray
            Vector of ground-truth classes

        perturber: function
            A function, f(x_test, p), which alters (perturbs) the test set by an amount or scale p.

        p_values: list or ndarray
            An iterable list of p_values to be passed to the perturber function and measured. If not supplied,
            numpy.linspace(p_low, p_high, steps) will be used instead.

        p_min: int
            The minimum, and first, value for p to be passed to the perturber function.

        p_max: int
            The maximum, and last, value for p to be passed to the perturber function.

        steps:
            The number of steps from p_min to p_max to be passed to the perturber function.

        label: str
            (Optional) An string label for the property/properties in question; used for plotting.

        plot: bool
            If True, produce and display a plot of the results.

        Returns
        -------
        SensitivityMeasure
            an object containing summary information about the analysis.
        """

        if label is None:
            label = "perturber"
        if p_values is None:
            p_values = np.linspace(p_min, p_max, steps)
        elif type(p_values) != np.ndarray:
            p_values = np.array(p_values)

        # p_scores = np.asarray([y_test] * p_values.size) * 0.00
        p_scores = np.zeros(y_test.shape[0] * p_values.size)
        p_test_values = np.zeros(y_test.shape[0] * p_values.size)
        #TODO: progbar
        #TODO: Do in batches, for now do all at once

        for i, p_val in enumerate(p_values):
            # This needs to go in batches onto separate processors
            # ########
            # Batch
            # ########
            progbar(i+1, len(p_values), message=f"{p_val:.2f}  ")
            if True:
                batch_size = 1000
                num_batches = int(x_test.shape[0]/batch_size) + 1
                for b in range(num_batches):
                    s = b*batch_size
                    f = min((b+1) * batch_size, x_test.shape[0])
                    if s >= x_test.shape[0]:
                        break

                    scores = self.predict_function(self.model, perturber(x_test[s:f], p_val))
                    rightscores = scores[np.arange(f-s), y_test[s:f]]
                    
                    p_scores[(i*y_test.size) + s:(i*y_test.size) + f] = rightscores
                    # p_scores[i*y_test.size:(i+1)*y_test.size] = rightscores

            else:
                scores = self.predict_function(self.model, perturber(x_test, p_val))
                rightscores = scores[np.arange(scores.shape[0]), y_test]
                p_scores[i*y_test.size:(i+1)*y_test.size] = rightscores
                # p_test_values[i*y_test.size:(i+1)*y_test.size] = p_val

            p_test_values[i*y_test.size:(i+1)*y_test.size] = p_val

        results = SensitivityMeasure(x_test, y_test, p_scores)

        p_bin_values = np.unique(p_values)

        x, means, std_devs = self._bin_and_measure(p_scores, p_test_values, p_bin_values=p_bin_values,
                binup=False)

                    # SingleTest(ps, means, stds, p_points=p_points, y_vals=y_vals)
                    # def append(self, label, ps, means, stds, p_points=None, y_vals=None):
        results.append(label, x, means, std_devs, p_points=p_test_values, y_vals=p_scores)

        if plot:
            if ci:
                results.tests[label].get_credible_interval(means_only=False)
            else:
                results.tests[label].get_gradient()
            self.plot_property(results.tests[label], label=label)

        for var, obj in locals().items():
            sz = sys.getsizeof(obj)
            if sz > 100000:
                print(var, sz/1e6)

        return results

    def test_class_sensitivity(self, x_test, y_test, plot=False):
        results = self.predict_and_measure(x_test, y_test, y_test, prop=None, label="class", propnames=["class"])
        labels = [str(x) for x in range(len(np.unique(y_test)))]
        labels = results.tests['class'].sort_and_reorder(labels)
        if plot:
            self.plot_property(results.tests['class'], label="class", ticklabels=labels)
        return results


    def plot_significance(self, test, label=None):
        beta = test.pos['beta']
        summary = test.summary().iloc[0]
        min_, max_, mean_ = beta.min(), beta.max(), beta.mean()
        plt.hist(beta, bins=50)

        plt.xlim(min(min_, -0.05), max(max_, 0.05))
        plt.axvline(0, ls="--", color="black")
        plt.axvline(summary.sens_50_low, ls="--", color="red")
        plt.axvline(summary.sens_50_high, ls="--", color="red")
        plt.axvline(summary.sens_95_low, ls="--", color="green")
        plt.axvline(summary.sens_95_high, ls="--", color="green")
        if label is not None:
            plt.title(f"{label} significance: {summary.significance}");

    def plot_all(self, measure):
        # TODO
        pass


    def plot_property(self, test, label="property", show_fit=False, fit="line", annotate=False, save_to=None, ticklabels=None, errorbars=True, fitorder=2):
        # import seaborn as sns
        # sns.set(style="darkgrid")

        if errorbars:
            plt.errorbar(test.p_vals, test.means, yerr=test.stds, marker="o", fmt='-o')
        else:
            plt.plot(test.p_vals, test.means, marker="o")

        if show_fit:
            if fit == "line":
                if test.beta is None:
                    test.get_gradient()
                xx = np.linspace(test.p_vals.min(), test.p_vals.max())
                plt.plot(xx, test.intercept + test.beta*xx, "k--")
            elif fit == "polynomial":
                xx = test.p_vals#.reshape(-1, 1)
                pmodel = np.polyfit(xx, test.means, fitorder)
                pmodel = np.poly1d(pmodel)
                plt.plot(xx, pmodel(xx), "--", color="green")
                # pmodel = make_pipeline(PolynomialFeatures(fitorder), Ridge())
                # pmodel.fit(xx, test.means)
                # plt.plot(test.p_vals, pmodel.predict(xx), "--", color="green")

        plt.axis([None, None, -0.1, 1.1])
        plt.title('Sensitivity to ' + label);
        plt.xlabel(label)
        plt.ylabel("mean score $\overline{y}_c$")
        if ticklabels is not None:
            plt.xticks(range(len(ticklabels)), ticklabels)
        plt.show()
        if save_to is not None:
            plt.savefig(save_to, bbox_inches="tight")


