import argparse
from pathlib import Path
import numpy as np
import random
from tqdm import tqdm
import copy
import collections
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt



RANDOM_SEEDS = [i for i in range(3000)]
RADNOME_SEEDS_POINTER = [0]

class Dataset():
    def __init__(self, X, Y):
       self.X = X
       self.Y = Y 
    
    def create_data(self, method="naive"):
        X, Y = copy.deepcopy(self.X), copy.deepcopy(self.Y)
        # Size
        N = X.shape[0]
        if method == "naive":
            pass
        elif method == "random":
            shuffle_indices = np.arange(N)
            np.random.shuffle(shuffle_indices)
            X = np.take(X, shuffle_indices, axis=0)
            Y = np.take(Y, shuffle_indices, axis=0)
        return X, Y
    
    def random_sample(self):
        N = self.X.shape[0]
        sample_index = np.random.choice(N)
        return self.X[sample_index], self.Y[sample_index]

class NaivePLA():
    def __init__(self):
        # Number of update
        self.number_of_update = 0
        # weight
        self.w = np.zeros((5, ))
        # Stopping index
        self.stop_idx = 10000000000
        # Verbose interval
        self.verbose_interval = 500

    @staticmethod
    def sign(x):
        '''
            sign function
            sign(0) = -1
        '''
        if isinstance(x, np.ndarray):
            positive = (x > 0).astype(int)
            negative = -(x <= 0).astype(int)
            output = positive + negative
        else:
            output = None
            if x > 0:
                output = 1
            else:
                output = -1
        return output
    
    def update_counter(self):
        self.number_of_update += 1

    def update_weight(self, y_n_t, x_n_t):
        self.w = self.w + y_n_t * x_n_t
        # record update
        self.update_counter()

    def check_stop_condition(self, idx, has_error):
        '''
            Return if done
        '''
        done = False
        # If idx bigger than stop idx, then done
        if idx >= self.stop_idx:
            done = True
        # If error does not exist, then is done
        elif has_error == False:
            done = True

        return done

    def loop_data(self, X, Y):
        '''
            Input:
                X, Y
            Output:
                has_error
            
            Loop whole data once
        '''
        has_error = False
        for _, (x_n_t, y_n_t) in enumerate(zip(X, Y)):
            #    
            if self.sign(self.w.dot(x_n_t)) != y_n_t:
                # Update
                self.update_weight(y_n_t, x_n_t)
                # 
                has_error = True
            else:
                # Keep looping
                continue

        return has_error

    def run_pla(self, X, Y):
        idx = 0
        while True:
            # Return if error
            has_error = self.loop_data(X, Y)
            # Stopping condition
            done = self.check_stop_condition(idx, has_error)
            if done:
                break
            # Add one
            idx += 1
        
        return 

    def fit(self, dataset):
        X, Y = dataset.create_data(method="naive")
        self.run_pla(X, Y)
        # Report
        self.report()
        return 

    def report(self):
        print("Number of updates is %d" % (self.number_of_update))

class PredeterminedRandomCyclePLA(NaivePLA):
    def __init__(self):
        super().__init__()
        self.avg_number_of_update = 0

    def reset(self):
        self.number_of_update = 0
        # weight
        self.w = np.zeros((5, ))

    def report(self):
        print("Average number of updates is %f" % (self.avg_number_of_update) )
    
    def plot(self, update_time_list):
        plt.xlabel("Number of updates")
        plt.ylabel("Frequency of the number")
        plt.title("PLA Update Frequency")
        plt.hist(update_time_list, density=True, facecolor='r', alpha=0.8)
        plt.savefig("PLAUpdateFrequency.png")

    def fit(self, dataset):
        experiment_times = 1126
        update_time_list = []
        for idx in tqdm(range(experiment_times)):
            # Reset
            self.reset()
            # Create dataset
            X, Y = dataset.create_data(method="random")
            self.run_pla(X, Y)
            # Record number of updates into list
            update_time_list.append(self.number_of_update)
            # 
            self.avg_number_of_update = self.avg_number_of_update * (idx / (idx + 1)) + self.number_of_update / (idx + 1)
            # 
        self.plot(update_time_list)
        self.report()

class PredeterminedRandomCyclePLAwithLearningRate(PredeterminedRandomCyclePLA):
    def __init__(self):
        super().__init__()
    
    def update_weight(self, y_n_t, x_n_t):
        self.w = self.w + 0.5 * y_n_t * x_n_t
        # record update
        self.update_counter()

class PocketPLA(PredeterminedRandomCyclePLA):
    def __init__(self):
        super().__init__()
        self.w_pocket = np.zeros((5, ))
        self.avg_test_error = 0.

    def loop_data(self, dataset):
        '''
            Input:
                dataset
            Output:
                has_error
            
            Loop whole data once
        '''
        # Initial error
        min_error = self.verify(dataset, self.w_pocket)
        for _ in range(self.stop_idx):
            # If already update 50 times, then break
            if self.number_of_update == 100:
                break
            
            x_n_t, y_n_t = dataset.random_sample()
            #    
            if self.sign(self.w.dot(x_n_t)) != y_n_t:
                # Update
                self.update_weight(y_n_t, x_n_t)
                # Verify training set
                error = self.verify(dataset, self.w)
                # Now self.w is better than pocket's w
                if error < min_error:
                    min_error = error
                    # Save to pocket
                    self.w_pocket = copy.deepcopy(self.w)

            else:
                # Keep looping
                continue

        return 

    def reset(self):
        self.number_of_update = 0
        # weight
        self.w = np.zeros((5, ))
        # 
        self.w_pocket = np.zeros((5, ))


    def verify(self, dataset, w):
        X, Y = dataset.create_data()
        # X: (N, 5), w:(5) -> (N, )
        error_rate = (self.sign(X.dot(w)) != Y).mean()
        return error_rate

    def run_pla(self, dataset, test_dataset, method):
        # Run 50 updates on dataset
        self.loop_data(dataset)
        # Verify test data
        if method == "pocket":
            error_rate = self.verify(test_dataset, self.w_pocket)
        elif method == "original":
            error_rate = self.verify(test_dataset, self.w)
        return error_rate
    
    def report(self):
        print("Average test error rate is %f" % (self.avg_test_error) )

    def fit(self, dataset, test_dataset):
        experiment_times = 2000
        for idx in tqdm(range(experiment_times)):
            # Reset
            self.reset()
            # Run pla and evaluate the testing error
            error_rate = self.run_pla(dataset, test_dataset, method="pocket")
            # 
            self.avg_test_error = self.avg_test_error * (idx / (idx + 1)) + error_rate / (idx + 1)

        self.report()


class Init():
    def __init__(self):
        # Parse args
        parser = argparse.ArgumentParser(description="PLA algorithm")
        parser.add_argument("dataPath", type=Path)
        parser.add_argument("modelType", type=str)
        parser.add_argument("--testdataPath", type=Path, default=None)
        self.args = parser.parse_args()
        
    def prepare(self):
        '''
            Output:
                dataset
        '''
        assert self.args.dataPath != self.args.testdataPath
        # Load X, Y
        X, Y = self._load(self.args.dataPath)
        X = self._add_bias_feature(X)
        if self.args.testdataPath is not None:
            # Test load
            test_X, test_Y = self._load(self.args.testdataPath)
            test_X = self._add_bias_feature(test_X)
        # Create dataset
        dataset = Dataset(X, Y)
        if self.args.testdataPath is not None:
            test_dataset = Dataset(test_X, test_Y)
        else:
            test_dataset = None
        return dataset, test_dataset, self.args.modelType

    def _load(self, dataPath):
        # Load text
        data = []
        with dataPath.open() as f:
            for line in f:
                line = line.strip("\n")
                # Replace tab as whitespace
                line = line.replace("\t", " ")
                splited = line.split()
                assert len(splited) == 5
                data.append(line.split())
        # 
        data = np.array(data)
        X = data[:, :-1].astype(float) # (N, 4)
        Y = data[:, -1].astype(int) # (N, )
        return X, Y

    def _add_bias_feature(self, X):
        '''
            Input: 
                X: (N, 4)
            Output:
                X: (N, 5)
        '''
        N = X.shape[0]
        bias_one = np.ones((N, 1))
        X = np.concatenate([bias_one, X], axis=-1)
        assert X.shape[1] == 5
        return X



def main():
    init = Init()
    # Read dataset
    dataset, test_dataset, modelType = init.prepare()
    # 
    if modelType == "NaivePLA":
        model = NaivePLA()
    elif modelType == "PredeterminedRandomCyclePLA":
        model = PredeterminedRandomCyclePLA()
        model.fit(dataset)
    elif modelType == "PredeterminedRandomCyclePLAwithLearningRate":
        model = PredeterminedRandomCyclePLAwithLearningRate()
    elif modelType == "PocketPLA":
        model = PocketPLA()
        # model fit
        model.fit(dataset, test_dataset)

if __name__ == "__main__":
    main()