import numpy as np
import copy 
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

def loadData(fname):
    with open(fname) as f:
        X = []
        Y = []
        for line in f:
            splited_line = line.strip(" ").split(" ")
            splited_line = [float(e) for e in splited_line]
            X.append(splited_line[:-1])
            Y.append(int(splited_line[-1]))

        X = np.array(X) # (N, 9)
        Y = np.array(Y) # (N, )
    
    return X, Y


def sign(x):
    return (x >= 0).astype(float) + (-(x < 0).astype(float))

def generate_h(s, theta):
    return lambda x: s * sign(x - theta)

def compute_E(data, h):
    X = [x for (x, y) in data]
    Y = [y for (x, y) in data]
    return np.mean(h(X) != Y)

def generateData(size=20):
    data = np.random.uniform(-1, 1, size=size)
    target = []
    for x in data:
        if np.random.uniform() < 0.2:
            # wrong target
            y = -sign(x)
        else:
            y = sign(x)
        target.append(y)
    return list(zip(data, target))


def decision_stump_algorithm(train):
    '''
        Fit one dimension decision stump
    '''
    train = copy.deepcopy(train) # Avoid in-place sort function
    # enumerate
    train.sort(key=lambda x: x[0]) # sort over X
    best_parameter = []
    best_Ein = float("inf")
    # 2N dichotomy
    for s in [-1, 1]:
        # N-1 interval
        for i in range(len(train)-1):
            mid_theta = (train[i][0] + train[i+1][0]) / 2
            E_in = compute_E(train, h=generate_h(s, mid_theta))
            if E_in < best_Ein:
                best_Ein = E_in
                best_parameter.clear()
                best_parameter.append((s, mid_theta))
            elif E_in == best_Ein:
                best_parameter.append((s, mid_theta))
        
        # Check edge point
        mid_theta = (-1 + train[0][0]) / 2
        E_in = compute_E(train, generate_h(s, mid_theta))
        
        if E_in < best_Ein:
            best_Ein = E_in
            best_parameter.clear()
            best_parameter.append((s, mid_theta))
        elif E_in == best_Ein:
            best_parameter.append((s, mid_theta))
        # Check 
        mid_theta = (1 + train[-1][0]) / 2
        E_in = compute_E(train, generate_h(s, mid_theta))

        if E_in < best_Ein:
            best_Ein = E_in
            best_parameter.clear()
            best_parameter.append((s, mid_theta))
        elif E_in == best_Ein:
            best_parameter.append((s, mid_theta))
        
    # break ties
    s, theta = best_parameter[np.random.choice(len(best_parameter))]
    return s, theta

def run():
    # generate data
    train = generateData()
    test = generateData()
    #
    s, theta = decision_stump_algorithm(train)
    h = generate_h(s, theta)
    # Test E out
    E_in = compute_E(train, h)
    E_out = compute_E(test, h)
    return E_in, E_out, s, theta

def run_multidim():
    X, Y = loadData("hw2_train.dat")
    featureDim = X.shape[1]
    ss, thetas = [], []
    E_ins = []
    # Run over each dimension
    for i in range(featureDim):
        tmpdata = list(zip(X[:, i], Y))
        s, theta = decision_stump_algorithm(tmpdata)
        E_in = compute_E(tmpdata, generate_h(s, theta))
        ss.append(s)
        thetas.append(theta)
        E_ins.append(E_in)
    
    # return best of best
    best_i = None
    best_E_in = float("inf")
    for i in range(len(E_ins)):
        if E_ins[i] < best_E_in:
            best_E_in = E_ins[i]
            best_i = i
    
    best_s, best_theta = ss[best_i], thetas[best_i]
    h = generate_h(best_s, best_theta)
    # 
    test_X, test_Y = loadData("hw2_test.dat")
    E_out = compute_E(list(zip(test_X[:, best_i], test_Y)), h)

    return best_E_in, E_out 


def plotHistogram(Ein_Eout_diffs, fname="Diffs_Histogram"):
    plt.xlabel("E_in - E_out")
    plt.ylabel("probability density function")
    plt.title("Histogram")
    plt.hist(Ein_Eout_diffs, bins=10, density=True)
    plt.savefig(fname + ".png")


def main():
    avg_Ein = 0.
    avg_Eout = 0.
    avg_Eout_true = 0.
    E_diffs = []
    E_sample_diffs = []
    for i in range(1, 1001):
        E_in, E_out, s, theta = run()
        avg_Ein = (avg_Ein * (i-1) + E_in) / i
        avg_Eout = (avg_Eout * (i-1) + E_out) / i
        E_out_true = 0.5 + 0.3 * s * (abs(theta) - 1)
        avg_Eout_true = (avg_Eout_true * (i-1) + E_out_true) / i
        E_diffs.append(E_in-E_out_true)
        E_sample_diffs.append(E_in-E_out)

    # Why sample E_out is so different from E_out expected value???
    print("Avg Eout true: {}".format(avg_Eout_true))
    print("Avg Ein: {}, Avg Eout: {}".format(avg_Ein, avg_Eout))
    plotHistogram(E_diffs)


if __name__ == "__main__":
    main()