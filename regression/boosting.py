# The base class with the weak learner 
from regression.tree import Tree 

# Data wrangling 
import pandas as pd 

# Directory traversal 
import os 
import shutil

# Python infinity
from math import inf

# Ploting 
import matplotlib.pyplot as plt 
import imageio


class RegressionGB():
    """
    Class that implements the regression gradient boosting algorithm 
    """
    def __init__(
        self, 
        d: pd.DataFrame,
        y_var: str,
        x_vars: list,
        max_depth: int = 4,
        min_sample_leaf: int = 2,
        learning_rate: float = 0.4,
    ):
        # Saving the names of y variable and X features
        self.y_var = y_var 
        self.features = x_vars

        # Saving the node data to memory 
        self.d = d[[y_var] + x_vars].copy()

        # Saving the data to the node 
        self.Y = d[y_var].values.tolist()

        # Saving the number of observations in data
        self.n = len(d)

        # Saving the tree hyper parameters
        self.max_depth = max_depth
        self.min_sample_leaf = min_sample_leaf

        # Saving the learning rate 
        self.learning_rate = learning_rate

        # Weak learner list 
        self.weak_learners = []

        # Setting the current iteration m to 0
        self.cur_m = 0

        # Saving the mean of y
        self.y_mean = self.get_mean(self.Y)

        # Saving the y_mean as the most recent prediction 
        self._predictions = [self.y_mean] * self.n

    @staticmethod
    def get_mean(x: list) -> float:
        """
        Calculates the mean over a list of float elements
        """
        # Initiating the sum counter 
        _sum = 0 

        # Infering the lenght of list 
        _n = len(x)

        if _n == 0:
            return inf

        # Iterating through the y values
        for _x in x:
            _sum += _x

        # Returning the mean 
        return _sum / _n

    def fit(
        self, 
        m: int = 10
        ):
        """
        Applies the iterative algorithm 
        """
        # Converting the X to suitable inputs
        _inputs = self.d[self.features].to_dict('records')

        # Saving the gamma list to memory 
        self.gamma = []

        # Iterating over the number of estimators
        for _ in range(self.cur_m, self.cur_m + m):
            # Calculating the residuals
            _residuals = [self.Y[i] - self._predictions[i] for i in range(self.n)]

            # Saving the current iterations residuals to the original dataframe 
            _r_name = f"residuals"
            self.d[_r_name] = _residuals

            # Creating a weak learner 
            _weak_learner = Tree(
                d = self.d.copy(), 
                y_var = _r_name,
                x_vars = self.features,
                max_depth = self.max_depth,
                min_sample_leaf = self.min_sample_leaf,
            )

            # Growing the tree on the residuals
            _weak_learner.fit()

            # Appending the weak learner to the list
            self.weak_learners.append(_weak_learner)

            # Getting the weak learner predictions
            _predictions_wl = [_weak_learner.predict(_x) for _x in _inputs] 

            # Updating the current predictions
            self._predictions = [self._predictions[i] + self.learning_rate * _predictions_wl[i] for i in range(self.n)]

        # Incrementing the current iteration 
        self.cur_m += m

    def predict(self, x: dict) -> float:
        """
        Given the dictionary, predict the value of the y variable
        """
        # Starting from the mean
        yhat = self.y_mean

        for _m in range(self.cur_m):
            yhat += self.learning_rate * self.weak_learners[_m].predict(x)

        return yhat

if __name__ == '__main__':
    # Infering the file location 
    _current_dir = os.path.dirname(os.path.abspath(__file__))

    # Defining the data directory 
    _data_dir = os.path.join(_current_dir, 'data')

    # Reading the data
    d = pd.read_csv(os.path.join(_data_dir, 'auto-mpg.csv'))

    # Encoding the correct types for categorical variables
    d['origin'] = d['origin'].astype('category')

    # Defining the y_var 
    _y_var = 'mpg'

    # Defining the features
    _x_vars = ['weight']

    # Creating a tmp directory for gifs 
    _tmp_dir = os.path.join(_current_dir, 'tmp')
    
    # Deleting all the previous runs
    if os.path.exists(_tmp_dir):
        shutil.rmtree(_tmp_dir)

    # Creating
    os.mkdir(_tmp_dir) 

    # Initiating the object 
    _reg_gb = RegressionGB(
        d = d,
        y_var = _y_var,
        x_vars = _x_vars,
        max_depth = 2,
        min_sample_leaf = 2,
        learning_rate = 0.3,
    )

    # Number of iterations 
    _n = 30

    for i in range(_n):
        _filename = os.path.join(_tmp_dir, f"frame_{i}.png")
        # Ploting the initial points and predictions
        plt.figure(figsize=(12, 8))
        if i > 0:
            _reg_gb.fit(m=1)
        plt.plot(d[_x_vars].values, d[_y_var].values, 'o', label='original', alpha=0.85)
        plt.scatter(d[_x_vars].values, _reg_gb._predictions, edgecolors='black', label=f'predictions - iteration {i}', color='orange')
        plt.xlabel('weight')
        plt.ylabel('mpg')
        plt.title(f'Gradient boosting results')
        plt.legend()
        plt.show()
        plt.savefig(_filename)

    # Saving the gif
    _gif_dir = os.path.join(_current_dir, 'gif')
    if not os.path.exists(_gif_dir):
        os.mkdir(_gif_dir)

    with imageio.get_writer(os.path.join(_gif_dir, 'RGB.gif'), mode='I', duration=0.4) as writer:
        _files = [os.path.join(_tmp_dir, x) for x in os.listdir(_tmp_dir)]
        _files.sort(key=lambda x: os.path.getmtime(x))
        for filename in _files:
            image = imageio.imread(os.path.join(_tmp_dir, filename))
            writer.append_data(image)