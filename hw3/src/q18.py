import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def sign(x):
    return (x >= 0) + (-1) * (x < 0)

def theta(s):
    # sigmoid function
    return 1 / (1 + np.exp(-s))
# Error
def E_in(w, X, Y):
    # X: (N, d), w: (d, ), Y:(N,)
    assert Y.shape[0] == X.shape[0]
    assert w.shape[0] == X.shape[1]
    # (N, ) * (N, )
    return np.mean(np.log(1 + np.exp(-Y * X.dot(w))))

def E_in_zero_one(w, X, Y):
    return E_out(w, X, Y)
# 0/1 error
def E_out(w, X, Y):
    # (N, ) == (N, )
    assert Y.shape[0] == X.shape[0]
    assert w.shape[0] == X.shape[1]
    assert sign(X.dot(w)).shape == Y.shape
    return np.mean(sign(X.dot(w)) != Y)

def gradient_E_in(w, X, Y):
    # (N, 1) * (N, d) -> (N, d)
    # mean over examples
    # return a (d, ) vector
    assert Y.shape[0] == X.shape[0]
    assert w.shape[0] == X.shape[1]
    # (N, 1) * (N, 1)
    assert np.expand_dims(theta(-Y*X.dot(w)), axis=1).shape == (X.shape[0], 1)
    assert (np.expand_dims(Y, 1) * X).shape == X.shape
    return np.mean(np.expand_dims(theta(-Y*X.dot(w)), axis=1) * (-np.expand_dims(Y, 1) * X), axis=0)

def load(filename):
    X, Y = [], []
    with open(filename) as f:
        for line in f:
            split = line.split(" ")[1:]
            split = list(map(lambda x: float(x), split))
            X.append(split[:-1])
            Y.append(split[-1])
            assert split[-1] == 1 or split[-1] == -1
    
    # X: (N, d), Y: (N)
    return np.array(X), np.array(Y) 

def plot_GDandSGD(E_ins, E_outs, eta):
    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.set_size_inches(18, 18)
    ax1.set_xlabel('t')
    ax1.set_ylabel('E_in')
    ax1.set_title('E_in(w_t) as a function of t')
    ax1.grid(b=True)
    ax1.plot(E_ins["SGD"], "b-", label='SGD')
    ax1.plot(E_ins["GD"], "r-", label='GD')
    ax1.legend()
    # 
    ax2.set_xlabel('t')
    ax2.set_ylabel('E_out')
    ax2.set_title('E_out(w_t) as a function of t')
    ax2.grid(b=True)
    ax2.plot(E_outs["SGD"], "b-", label="SGD")
    ax2.plot(E_outs["GD"], "r-", label="GD")
    ax2.legend()
    fig.savefig("mlhw3_{}.png".format(eta))


def logistic_regression(eta, T):
    train_X, train_Y = load("hw3_train.dat")
    test_X, test_Y = load("hw3_test.dat")
    # Feature dimension should be the same
    assert train_X.shape[1] == test_X.shape[1] 
    feature_dim = train_X.shape[1]
    E_ins, E_outs = {"SGD": [], "GD": []}, {"SGD": [], "GD": []}
    w_SGD, w_GD = np.zeros((train_X.shape[1], )), np.zeros((train_X.shape[1], ))
    # 
    for i in range(T):
        E_ins["SGD"].append(E_in_zero_one(w_SGD, train_X, train_Y))
        E_ins["GD"].append(E_in_zero_one(w_GD, train_X, train_Y))
        E_outs["SGD"].append(E_out(w_SGD, test_X, test_Y))
        E_outs["GD"].append(E_out(w_GD, test_X, test_Y))
        #
        N = train_X.shape[0]

        sgd_gradient = gradient_E_in(w_SGD, train_X[i % N: (i % N) + 1], train_Y[i % N: (i % N) + 1])
        gd_gradient = gradient_E_in(w_GD, train_X, train_Y)
        assert sgd_gradient.shape[0] == feature_dim
        assert gd_gradient.shape[0] == feature_dim

        w_SGD = w_SGD - eta * sgd_gradient
        w_GD = w_GD - eta * gd_gradient

    # plot result
    plot_GDandSGD(E_ins, E_outs, eta)

def main():
    logistic_regression(0.001, 2000)
    logistic_regression(0.01, 2000)
    
    

if __name__ == "__main__":
    main()