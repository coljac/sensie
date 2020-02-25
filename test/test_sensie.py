import numpy as np
import sys, os

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../sensie")
import sensie


class FakeModel(object):
    
    def __init__(self, function):
        self.function = function
    
    def predict(self, x):
        return self.function(x)


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def fake_dist(true_class, bonus=1.0):
    true_class = true_class.astype(int)
    fake_dist = np.random.random(size=(true_class.shape[0], 10))
    fake_dist[np.arange(true_class.shape[0]), true_class[:, 0]] += bonus
    return np.apply_along_axis(softmax, 1, fake_dist)

def fake_dist_w_bias(x):
    true_class = x[:, 0].astype(int)
    fake_dist = np.random.random(size=(x.shape[0], 10))
    fake_dist[np.arange(true_class.shape[0]), true_class] += x[:, 1]
    return np.apply_along_axis(softmax, 1, fake_dist)   

def perturb(x, p):
    x[:, 1] /= p/3
    return x

def perturb_dont(x, p):
    return x 


# Test 1: Fake model shows no significant correlation
def test_no_correlation():
    N = 1000
    classes = 10
    b = 3

    model = FakeModel(lambda x: fake_dist(x, b))
    X_test = np.random.random(size=(N,1))*classes
    y_test = X_test.astype(int)[:, 0]
    y = model.predict(X_test)
    yc = np.argmax(y, axis=1)
    fake_p = np.random.randint(10, size=1000)
    probe = sensie.Probe(model)
    result = probe.predict_and_measure(X_test, y_test, fake_p, plot=False, propnames=["fake"]);
    result.get_credible_intervals()
    assert np.abs(result.tests['fake'].beta) < 0.002 and (result.tests['fake'].get_significance() == "low")

# Test 2: Fake model with significant correlation
def test_high_correlation():
    N = 1000
    classes = 10
    # bonus = 0.5 # Accuracy parameter = low
    bonus = 3 # Accuracy parameter = high


    model = FakeModel(lambda x: fake_dist_w_bias(x))
    X_test = np.zeros((N, 2))
    X_test[:, 0] = (10*np.arange(N)/N).astype(np.int)
    X_test[:, 1] = bonus*np.arange(N)/N
    y_test = X_test.astype(int)[:, 0]
    y = model.predict(X_test)

    fake_p = (10*np.arange(N)/N).astype(int)

    probe = sensie.Probe(model)
    result = probe.predict_and_measure(X_test, y_test, fake_p, plot=False, propnames=["fake"]);
    result.get_credible_intervals()
    assert np.abs(result.tests['fake'].beta) > 0.05 and (result.tests['fake'].get_significance() == "high")


# Test 3: Fake model with class disparity
def test_class_disparity():
    N = 1000
    classes = 10

    model = FakeModel(lambda x: fake_dist_w_bias(x))
    X_test = np.random.random(size=(N,2))*classes
    X_test[:, 1] = X_test[:, 0]/3
    y_test = X_test.astype(int)[:, 0]
    y = model.predict(X_test)
    yc = np.argmax(y, axis=1)

    X_test[800:900, 1] = X_test[0:100, 1]

    fake_p = (10*np.arange(N)/N).astype(int)

    probe = sensie.Probe(model)
    result = probe.predict_and_measure(X_test, y_test, fake_p, plot=False, propnames=["fake"]);

    result = probe.test_class_sensitivity(X_test, y_test, plot=False)
    assert  result.tests['class'].means[-1]/result.tests['class'].means[0] > 3

def test_perturber_low():
    N = 1000
    classes = 10
    # bonus = 0.5 # Accuracy parameter = low
    bonus = 3 # Accuracy parameter = high


    model = FakeModel(lambda x: fake_dist_w_bias(x))
    X_test = np.zeros((N, 2))
    X_test[:, 0] = (10*np.arange(N)/N).astype(np.int)
    # X_test = np.random.random(size=(N,2))*classes
    X_test[:, 1] = bonus*np.arange(N)/N
    y_test = X_test.astype(int)[:, 0]
    y = model.predict(X_test)
    # yc = np.argmax(y, axis=1)

    fake_p = (10*np.arange(N)/N).astype(int)

    probe = sensie.Probe(model)
    result = probe.predict_and_measure_perturbed(X_test, y_test, perturb, p_min=1, p_max=10, steps=10, 
                                                 plot=False, label="fake")
    result.get_credible_intervals()
    assert np.abs(result.tests['fake'].beta) > 0.05 and (result.tests['fake'].get_significance() == "high")



def test_perturber_high():
    N = 1000
    classes = 10
    # bonus = 0.5 # Accuracy parameter = low
    bonus = 3 # Accuracy parameter = high


    model = FakeModel(lambda x: fake_dist_w_bias(x))
    X_test = np.zeros((N, 2))
    X_test[:, 0] = (10*np.arange(N)/N).astype(np.int)
    # X_test = np.random.random(size=(N,2))*classes
    X_test[:, 1] = bonus*np.arange(N)/N
    y_test = X_test.astype(int)[:, 0]
    y = model.predict(X_test)
    # yc = np.argmax(y, axis=1)

    fake_p = (10*np.arange(N)/N).astype(int)

    probe = sensie.Probe(model)
    result = probe.predict_and_measure_perturbed(X_test, y_test, perturb_dont, p_min=1, p_max=10, steps=10, 
                                                 plot=False, label="fake")
    result.get_credible_intervals()
    assert np.abs(result.tests['fake'].beta) < 0.002 and (result.tests['fake'].get_significance() == "low")


if __name__=="__main__":
    print("Run these tests with pytest.")
